import math
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# =========================
# Config
# =========================
MODEL_NAME       = "deepseek-ai/deepseek-moe-16b-base"
DEVICE           = torch.device("cpu")        # CPU as requested
DTYPE            = torch.float32              # FP32 as requested
BATCH_SIZE       = 2                          # any batch size
WARMUP_NEW_TOK   = 64                         # short warm-up for profiling
MAIN_NEW_TOK     = 128                        # main generation length
LONG_RANGE_W     = 256                        # "older than W" threshold for frequency
TOP_FRAC_HOT     = 0.34                       # top ~1/3 as HOT (tune)
MID_FRAC_WARM    = 0.33                       # middle ~1/3 as WARM
# windows for tiers (change per your memory)
WIN_HOT          = 2048
WIN_WARM         = 1024
WIN_COLD         = 256

# weights for LayerScore (sum to 1.0)
W_FOCUS    = 0.35   # 1 - normalized entropy
W_SHIFT    = 0.20   # temporal shift (variance across steps)
W_REPCHG   = 0.25   # representation change on prefill
W_LR_RATE  = 0.20   # long-range usage rate

# =========================
# Load model/tokenizer
# =========================
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
model.to(DEVICE).eval()

# Make sure we have a pad token id for batching causal models
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# =========================
# Helper: cosine rep change per layer (prefill)
# =========================
@torch.no_grad()
def layer_rep_change_prefill(model, tok, texts, max_prompt_tokens=256):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens).to(DEVICE)
    # We need hidden states per layer. For causal decoders, output_hidden_states=True returns:
    # hidden_states[0]    -> embeddings
    # hidden_states[1..L] -> outputs after each layer
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple length L+1
    # For layer ℓ, "input to layer" ~ hs[ℓ], "output of layer" ~ hs[ℓ+1].
    rep_chg = []
    # Average cosine across batch and tokens (avoid padding positions)
    attn_mask = enc.get("attention_mask", None)
    for L in range(num_layers):
        h_in  = hs[L]     # (B, T, H)
        h_out = hs[L+1]   # (B, T, H)
        # flatten batch/tokens while masking pads
        if attn_mask is not None:
            mask = attn_mask.bool()  # (B, T)
            # Avoid zero vectors: compute cosine per position then mask
            coss = []
            for b in range(h_in.size(0)):
                valid = mask[b]
                if valid.any():
                    x = h_in[b, valid]   # (Tv, H)
                    y = h_out[b, valid]  # (Tv, H)
                    # cosine_similarity over H, reduce mean over valid positions
                    cs = cosine_similarity(x, y, dim=-1).mean().item()
                    coss.append(cs)
            if len(coss) == 0:
                coss_avg = 1.0
            else:
                coss_avg = sum(coss)/len(coss)
        else:
            x = h_in.reshape(-1, h_in.size(-1))
            y = h_out.reshape(-1, h_out.size(-1))
            coss_avg = cosine_similarity(x, y, dim=-1).mean().item()
        rep_chg.append(1.0 - float(coss_avg))  # 1 - cosine
    return rep_chg  # list length = num_layers

# =========================
# Helper: attention-based metrics during warm-up generation
# =========================
@torch.no_grad()
def layer_metrics_warmup(model, tok, texts, max_new_tokens=WARMUP_NEW_TOK, long_range_W=LONG_RANGE_W):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    # Ask generate to return attentions per step:
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
        use_cache=True
    )
    # out.attentions: list over steps; each item is list over layers -> tensor (B, H, Tq, Tk)
    # We aggregate per layer by averaging over heads for the *last query* (Tq-1) each step.
    L = num_layers
    eps = 1e-12
    ent_sum    = [0.0] * L
    ent_count  = [0]   * L
    shift_vals = [[]   for _ in range(L)]
    lr_hits    = [0    for _ in range(L)]
    step_counts= 0

    for step_attn in out.attentions:           # per generated token
        for layer_idx, A in enumerate(step_attn):
            # A: (B, H, Tq, Tk). We’ll use the distribution for the last query.
            B, H, Tq, Tk = A.shape
            probs = A[:, :, -1, :] + eps       # (B, H, Tk)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # entropy per head (normalized by log Tk)
            Hmax = math.log(Tk + 1e-9)
            ent = -(probs * probs.log()).sum(dim=-1) / Hmax   # (B, H)
            ent_sum[layer_idx] += ent.mean().item()
            ent_count[layer_idx] += 1

            # temporal center-of-mass for attention index (0..Tk-1)
            idx = torch.arange(Tk, device=probs.device).float()
            com = (probs * idx).sum(dim=-1)    # (B, H)
            # record mean over batch & heads for this step
            shift_vals[layer_idx].append(com.mean().item())

            # long-range usage: mass on keys older than W
            if long_range_W > 0:
                cutoff = max(0, Tk - long_range_W)
                if cutoff > 0:
                    far_mass = probs[..., :cutoff].sum(dim=-1)  # (B, H)
                    # hit if avg far_mass >= 0.30
                    if far_mass.mean().item() >= 0.30:
                        lr_hits[layer_idx] += 1
        step_counts += 1

    # reduce
    mean_entropy = []
    temporal_shift = []
    long_range_rate = []
    for l in range(L):
        me = (ent_sum[l] / max(ent_count[l], 1)) if ent_count[l] else 0.0           # 0..1 (lower=better)
        # temporal shift = variance of CoM across steps (normalized later)
        if len(shift_vals[l]) >= 2:
            s = torch.tensor(shift_vals[l])
            ts = float(s.var(unbiased=False).item())
        else:
            ts = 0.0
        lrr = (lr_hits[l] / max(step_counts, 1)) if step_counts else 0.0            # 0..1
        mean_entropy.append(me)
        temporal_shift.append(ts)
        long_range_rate.append(lrr)

    # normalize temporal_shift into 0..1 robustly using per-run min/max
    ts_min, ts_max = (min(temporal_shift), max(temporal_shift)) if temporal_shift else (0.0, 1.0)
    if ts_max > ts_min:
        temporal_shift_norm = [(t - ts_min) / (ts_max - ts_min) for t in temporal_shift]
    else:
        temporal_shift_norm = [0.0 for _ in temporal_shift]

    # focus = 1 - entropy
    focus = [max(0.0, 1.0 - e) for e in mean_entropy]  # higher is better
    return {
        "mean_entropy": mean_entropy,            # 0..1 (lower better)
        "focus": focus,                          # 0..1 (higher better)
        "temporal_shift": temporal_shift_norm,   # 0..1
        "long_range_rate": long_range_rate       # 0..1
    }

# =========================
# Combine metrics -> LayerScore -> tiers
# =========================
def layer_scores_and_tiers(rep_change, attn_metrics):
    focus   = attn_metrics["focus"]
    tshift  = attn_metrics["temporal_shift"]
    lrr     = attn_metrics["long_range_rate"]

    # z-normalize rep_change across layers for fairness
    rep_mean = sum(rep_change)/len(rep_change)
    rep_std  = (sum((x - rep_mean)**2 for x in rep_change)/len(rep_change))**0.5 or 1.0
    rep_norm = [min(1.0, max(0.0, 0.5 + 0.2*((x - rep_mean)/rep_std))) for x in rep_change]  # squashed

    scores = []
    for l in range(num_layers):
        score = (W_FOCUS * focus[l] +
                 W_SHIFT * tshift[l] +
                 W_REPCHG * rep_norm[l] +
                 W_LR_RATE * lrr[l])
        scores.append(score)

    # rank and bucket
    order = sorted(range(num_layers), key=lambda i: scores[i], reverse=True)
    n_hot  = max(1, int(round(num_layers * TOP_FRAC_HOT)))
    n_warm = max(0, int(round(num_layers * MID_FRAC_WARM)))
    hot_idx   = set(order[:n_hot])
    warm_idx  = set(order[n_hot:n_hot+n_warm])
    cold_idx  = set(order[n_hot+n_warm:])

    # map windows
    layer_window = {}
    layer_tier   = {}
    for l in range(num_layers):
        if l in hot_idx:
            layer_window[l] = WIN_HOT
            layer_tier[l]   = "HOT"
        elif l in warm_idx:
            layer_window[l] = WIN_WARM
            layer_tier[l]   = "WARM"
        else:
            layer_window[l] = WIN_COLD
            layer_tier[l]   = "COLD"
    return scores, layer_tier, layer_window, order

# =========================
# Prune legacy past_key_values (tuple) per layer to window size
# =========================
def prune_past_legacy(past_key_values, layer_window):
    """
    past_key_values: tuple length L, each item = (k, v, [maybe more])
      Typical k/v shape = (B, num_heads, T, head_dim)
    We keep only last W_l tokens along the sequence axis per layer.
    Returns a pruned tuple of same structure.
    """
    if not isinstance(past_key_values, (tuple, list)):
        # Unknown cache type, skip pruning
        return past_key_values, False

    pruned_layers = []
    for l, layer_cache in enumerate(past_key_values):
        if layer_cache is None:
            pruned_layers.append(layer_cache)
            continue

        # layer_cache might be 2-tuple (k,v) or more (e.g., with k_pe/v_pe).
        # We only slice the KV tensors.
        new_items = []
        W = layer_window.get(l, None)
        for idx, item in enumerate(layer_cache):
            tensor = item
            if not torch.is_tensor(tensor):
                new_items.append(tensor)
                continue
            if W is None:
                new_items.append(tensor)
                continue
            # Identify sequence axis heuristically:
            # Try (B, H, T, Dh)
            if tensor.dim() == 4 and tensor.size(2) >= W:
                new_items.append(tensor[:, :, -W:, :])
            # Else try (B, T, H, Dh)
            elif tensor.dim() == 4 and tensor.size(1) >= W:
                new_items.append(tensor[:, -W:, :, :])
            else:
                # If tensor shorter than W or unexpected shape, keep as-is
                new_items.append(tensor)
        pruned_layers.append(tuple(new_items))
    return tuple(pruned_layers), True

# =========================
# Main
# =========================
if __name__ == "__main__":
    # ---------- Build prompts from AG News ----------
    ds = load_dataset("ag_news", split="test[:{}]".format(max(8, BATCH_SIZE*2)))
    texts = ds["text"][:max(8, BATCH_SIZE*2)]

    # ---------- Warm-up profiling ----------
    print("Profiling layer representation change (prefill)...")
    rep_change = layer_rep_change_prefill(model, tok, texts)

    print("Profiling attention metrics (warm-up generate)...")
    attn_metrics = layer_metrics_warmup(model, tok, texts, max_new_tokens=WARMUP_NEW_TOK, long_range_W=LONG_RANGE_W)

    # ---------- Combine into tiers ----------
    scores, layer_tier, layer_window, rank = layer_scores_and_tiers(rep_change, attn_metrics)

    print("\n=== Layer scores & tiers (highest first) ===")
    for idx in rank:
        print(f"Layer {idx:02d} | score={scores[idx]:.3f} | tier={layer_tier[idx]} | win={layer_window[idx]}"
              f" | focus={attn_metrics['focus'][idx]:.2f} | LRrate={attn_metrics['long_range_rate'][idx]:.2f} | repΔ={rep_change[idx]:.3f}")

    # ---------- Main generation with per-layer KV pruning ----------
    # Batch a few prompts and generate with pruning
    batch = tok(texts[:BATCH_SIZE], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    print("\nRunning main generation with tiered KV windows per layer...")
    past_kv = None
    generated = [[] for _ in range(input_ids.size(0))]

    # Prime: pass the full prompt once (prefill)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_attentions=False)
    logits = out.logits
    past_kv = out.past_key_values

    # Try pruning once after prefill
    past_kv, ok = prune_past_legacy(past_kv, layer_window)
    if not ok:
        print("⚠️ Cache is not legacy tuple; pruning was skipped. (Generation will still work.)")

    # Decode NEW tokens step-by-step (greedy) with pruning per step
    next_in = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # first next token from prefill
    for step in range(MAIN_NEW_TOK):
        with torch.no_grad():
            out = model(input_ids=next_in, attention_mask=None, use_cache=True,
                        past_key_values=past_kv, output_attentions=False)
        logits = out.logits
        past_kv = out.past_key_values

        # prune per-layer cache to its window
        past_kv, _ = prune_past_legacy(past_kv, layer_window)

        next_in = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        for b, tok_id in enumerate(next_in.squeeze(1).tolist()):
            generated[b].append(tok_id)

    # Decode text (optional)
    decoded = [tok.decode(seq, skip_special_tokens=True) for seq in generated]
    print("\n=== Sample generated continuations (truncated) ===")
    for i, s in enumerate(decoded):
        print(f"[{i}] {s[:160]}{'...' if len(s)>160 else ''}")

    print("\nDone. KV was tiered per layer with windows:", layer_window)
