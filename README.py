import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
BATCH_SIZE = 2          # configurable batch size for prompts
NEW_TOKENS = 128        # number of new tokens to generate per input
PEAK_THRESHOLD = 0.5    # threshold for "peak" attention weight (e.g., >50% on one token)
DEVICE = torch.device("cpu")  # assume CPU-only; use torch.device("cuda") for GPU
DTYPE = torch.float32   # use FP32 as specified (could use bfloat16 for lower memory then float())
# ---------------------

print(f"Loading model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load with remote code for custom MoE, and force to CPU memory (do not use device_map="auto" for now)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
model.to(DEVICE)
model.eval()
model.generation_config = model.generation_config or {}  # ensure generation_config exists
# Set pad_token_id if not set (some CausalLM models don't have one by default)
if getattr(model.generation_config, "pad_token_id", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.eos_token_id

# Prepare AG News dataset prompts
print("Loading AG News dataset and preparing prompts...")
dataset = load_dataset("ag_news", split="test[:{}]".format(BATCH_SIZE))  # take first BATCH_SIZE samples from test
texts = dataset["text"][:BATCH_SIZE]
# Optionally truncate long articles to a reasonable prompt length to save time
MAX_PROMPT_LEN = 128  # max tokens in prompt to use
for i, txt in enumerate(texts):
    # Truncate text to first MAX_PROMPT_LEN tokens for efficiency
    tokens = tokenizer.tokenize(txt)
    if len(tokens) > MAX_PROMPT_LEN:
        tokens = tokens[:MAX_PROMPT_LEN]
        texts[i] = tokenizer.convert_tokens_to_string(tokens)

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(DEVICE)

print(f"Prepared {len(texts)} prompts. Generating {NEW_TOKENS} new tokens each...")

# Data structures to accumulate metrics
num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.n_layer
num_heads = model.config.num_attention_heads

# Initialize accumulators for entropy and peak usage
head_entropy_sum = [[0.0 for _ in range(num_heads)] for __ in range(num_layers)]
head_peak_count = [[0   for _ in range(num_heads)] for __ in range(num_layers)]
head_token_count = 0  # total number of token generation steps processed (for normalization)

# Define hook function to capture attention weights from each layer's attention module
def attentions_hook(module, input, output):
    """
    This hook captures the attention weights and computes per-head entropy and peak usage for the last query token.
    It assumes `output` is a tuple: (attn_output, attn_weights, past_key_value).
    """
    # Unpack the output tuple from the attention forward
    attn_output, attn_weights, past = output
    if attn_weights is None:
        return  # no attention weights to process (this might happen if output_attentions=False)
    # module.layer_idx gives the index of this layer's attention
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        # If layer_idx is not set, we cannot attribute to a specific layer; skip
        return

    # attn_weights shape: (batch, num_heads, query_len, key_len)
    # We only want the **last query token's** attention distribution for generation:
    # If query_len > 1 (likely only for the first forward pass with the full prompt),
    # we take only the last query (the last token in the sequence).
    # Otherwise (query_len == 1), we take that single query.
    batch_size, n_heads, q_len, k_len = attn_weights.shape
    # Determine index of the query to use (last one)
    query_index = q_len - 1  # last query token index
    # Extract the attention distribution for that query: shape (batch, num_heads, key_len)
    # This is the softmaxed attention probabilities over all key positions for the last token's query
    attn_probs = attn_weights[:, :, query_index, :]  # size: (batch_size, num_heads, key_len)

    # Compute entropy and peak for each head
    # Small epsilon to avoid log(0)
    eps = 1e-20
    # We operate per head, but can vectorize over batch and heads.
    # attn_probs[b, h, :] is the distribution for batch b, head h.
    # Calculate entropy: -sum(p * log p)
    # We'll average over batch if batch_size > 1
    probs = attn_probs.detach()  # detach from graph, we only need values
    # Ensure numerical stability: add eps before log
    log_probs = torch.log(probs + eps)
    ent = -torch.sum(probs * log_probs, dim=-1)  # shape: (batch_size, num_heads)
    # Identify if head has a peak > threshold
    # max_prob shape: (batch_size, num_heads)
    max_prob, _ = torch.max(probs, dim=-1)

    # Accumulate metrics for each head
    for b in range(batch_size):
        for h in range(n_heads):
            H = ent[b, h].item()
            head_entropy_sum[layer_idx][h] += H
            # Count a peak usage if max attention > PEAK_THRESHOLD
            if max_prob[b, h].item() > PEAK_THRESHOLD:
                head_peak_count[layer_idx][h] += 1

    # Increment total token count (per batch) by batch_size for each query processed
    # We do this outside the hook to avoid multiple additions per layer.
    # (We'll increment in the generate loop below after calling the model for each step.)

# Register the hook on each attention layer of the model
# DeepSeek model uses a decoder architecture with self-attention in each layer.
# We find the modules corresponding to multi-head attention.
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Module) and hasattr(module, "forward"):
        # Identify the attention modules by class name or attribute
        # In DeepSeek's code, the class is likely DeepseekAttention
        if module.__class__.__name__.endswith("Attention"):
            module.register_forward_hook(attentions_hook)

# We will use manual generation loop to have control over each step, because with hooks 
# we need to call the model step by step to capture attentions.
# (Alternatively, we could use model.generate with return_dict_in_generate=True and output_attentions=True,
#  then parse the attention outputs, but hooking allows on-the-fly processing and is more flexible.)
print("Starting generation with attention introspection...")
input_ids_gen = input_ids  # this will hold the current input ids for each step (initially the prompt)
past_key_values = None

# We will generate NEW_TOKENS in a loop, one token at a time (greedy decoding for simplicity).
generated_sequences = [[] for _ in range(input_ids_gen.size(0))]  # to store newly generated tokens for each batch item

for step in range(NEW_TOKENS):
    # Forward pass for the current step
    # We use use_cache=True to get past_key_values for next iteration.
    # output_attentions=False because we use hooks to capture them (our hooks still get called).
    with torch.no_grad():
        outputs = model(input_ids=input_ids_gen, attention_mask=attention_mask, use_cache=True, output_attentions=False, past_key_values=past_key_values)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
    past_key_values = outputs.past_key_values  # update past for next token
    # Only the last token's logits matter for next token prediction
    next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
    # Greedy decoding: choose the highest logit token
    next_tokens = torch.argmax(next_token_logits, dim=-1)  # (batch_size,)
    # Append to generated sequences and prepare for next iteration
    for i, token in enumerate(next_tokens.tolist()):
        generated_sequences[i].append(token)
    # Set the next input_ids as the predicted token (unsqueeze to shape (batch,1) for next step)
    input_ids_gen = next_tokens.unsqueeze(1)
    # Update attention mask if applicable (extend by 1 of ones)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=DEVICE)], dim=1)
    # Increment token count for each head by batch_size (we processed one new token per batch item)
    head_token_count += input_ids_gen.size(0)

print("Generation complete.\n")

# Now we have accumulated entropy sums and peak counts for each head across all generated tokens.
# Compute average entropy and peak usage frequency for each head.
head_entropy_avg = [[0.0 for _ in range(num_heads)] for __ in range(num_layers)]
head_peak_frac = [[0.0 for _ in range(num_heads)] for __ in range(num_layers)]
for layer in range(num_layers):
    for head in range(num_heads):
        if head_token_count > 0:
            head_entropy_avg[layer][head] = head_entropy_sum[layer][head] / head_token_count
            head_peak_frac[layer][head] = head_peak_count[layer][head] / head_token_count

# Rank heads by combined importance score (higher = more important)
# We define a simple combined metric: score = (1 - (entropy / log(N))) + peak_frac, averaged or weighted.
# For simplicity, we use normalized entropy (relative to its maximum possible).
import math
head_scores = [[0.0 for _ in range(num_heads)] for __ in range(num_layers)]
for layer in range(num_layers):
    for head in range(num_heads):
        # Normalize entropy by log of average key count (approximate by prompt length + generated length)
        # We can approximate key count by prompt_length + current generated length (this is rough).
        # Alternatively, we could have tracked average key_len per head, but we'll estimate using final sequence length.
        avg_key_len = input_ids.size(1) + NEW_TOKENS  # original prompt length + total generated
        max_entropy = math.log(avg_key_len + 1e-9)  # log of key count
        entropy = head_entropy_avg[layer][head]
        entropy_score = 0.0
        if max_entropy > 0:
            entropy_score = 1.0 - (entropy / max_entropy)  # 1 = very focused (low entropy), 0 = very diffuse (high entropy)
        # Combined with peak fraction (already 0 to 1)
        peak_score = head_peak_frac[layer][head]
        head_scores[layer][head] = (entropy_score + peak_score) / 2.0  # simple average of two metrics

# Determine hot vs cold heads. For demonstration, let's define:
# - "Hot heads": score above a certain percentile or threshold
# Here we pick top 30% of heads as hot (you can adjust criterion).
all_scores = [head_scores[l][h] for l in range(num_layers) for h in range(num_heads)]
if len(all_scores) == 0:
    threshold_score = 0.0
else:
    cutoff_index = int(len(all_scores) * 0.7)  # 70th percentile as cutoff for hot (top 30% are hot)
    cutoff_index = min(len(all_scores)-1, max(0, cutoff_index))
    sorted_scores = sorted(all_scores)
    threshold_score = sorted_scores[cutoff_index]

hot_heads = []   # list of (layer, head, score)
cold_heads = []  # list of (layer, head, score)
for layer in range(num_layers):
    for head in range(num_heads):
        score = head_scores[layer][head]
        if score >= threshold_score:
            hot_heads.append((layer, head, score))
        else:
            cold_heads.append((layer, head, score))

# Sort hot heads by score (descending) for reporting
hot_heads.sort(key=lambda x: x[2], reverse=True)
cold_heads.sort(key=lambda x: x[2], reverse=True)

# Summarize per-layer: how many hot heads per layer
layer_hot_count = [0]*num_layers
for (layer, head, score) in hot_heads:
    layer_hot_count[layer] += 1

# Print out a report
print("==== Attention Head Importance Analysis ====")
total_heads = num_layers * num_heads
print(f"Total layers: {num_layers}, heads per layer: {num_heads}, total heads: {total_heads}")
print(f"Threshold score for 'hot' heads: {threshold_score:.3f}")
print(f"Identified {len(hot_heads)} hot heads and {len(cold_heads)} cold heads (out of {total_heads}).\n")

print("Hot heads (retain in HBM):")
for layer, head, score in hot_heads[:min(10, len(hot_heads))]:
    print(f" - Layer {layer} Head {head}: combined score {score:.3f} (entropy ~{head_entropy_avg[layer][head]:.3f}, peak frac {head_peak_frac[layer][head]:.2f})")
if len(hot_heads) > 10:
    print(f" ... and {len(hot_heads)-10} more hot heads.")

print("\nCold heads (candidate for offloading):")
for layer, head, score in cold_heads[-min(10, len(cold_heads)):]:
    print(f" - Layer {layer} Head {head}: combined score {score:.3f} (entropy ~{head_entropy_avg[layer][head]:.3f}, peak frac {head_peak_frac[layer][head]:.2f})")
if len(cold_heads) > 10:
    print(f" ... and {len(cold_heads)-10} more cold heads.")

# Layer-level suggestions
print("\nLayer-wise summary of hot/cold heads:")
for layer in range(num_layers):
    hot_count = layer_hot_count[layer]
    if hot_count == 0:
        print(f" - Layer {layer}: ALL {num_heads} heads cold -> **offload entire layer**")
    elif hot_count < num_heads:
        print(f" - Layer {layer}: {hot_count}/{num_heads} heads are hot -> **partially offload** (keep hot heads in HBM, offload {num_heads-hot_count} cold heads)")
    else:
        print(f" - Layer {layer}: All heads hot -> keep entire layer in HBM")
