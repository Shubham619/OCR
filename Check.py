# print_layer_importance.py
import sys, torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------- Args ---------
model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
prompt     = sys.argv[2] if len(sys.argv) > 2 else "Summarize the impact of memory bandwidth on LLM inference."
device     = "cuda" if torch.cuda.is_available() else "cpu"
dtype      = torch.float16 if device == "cuda" else torch.float32

# --------- Load ---------
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()

num_layers = model.config.num_hidden_layers

# --------- Importance via SqueezeAttention metric (prefill only) ---------
@torch.no_grad()
def layer_importance(texts, max_len=512):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # [emb, layer1, layer2, ..., layerL]
    imp = []

    # mask pads
    am = enc.get("attention_mask", None)
    for L in range(num_layers):
        h_in, h_out = hs[L], hs[L+1]           # (B, T, H)
        if am is not None:
            vals = []
            for b in range(h_in.size(0)):
                m = am[b].bool()
                if m.any():
                    vals.append(cosine_similarity(h_in[b, m], h_out[b, m], dim=-1).mean().item())
            cs = sum(vals)/len(vals) if len(vals) else 1.0
        else:
            cs = cosine_similarity(
                h_in.reshape(-1, h_in.size(-1)),
                h_out.reshape(-1, h_out.size(-1)), dim=-1
            ).mean().item()
        imp.append(1.0 - float(cs))            # importance = 1 - cosine
    return imp

scores = layer_importance([prompt])
print("\n=== Layer-wise importance (SqueezeAttention metric: 1 - cosine) ===")
for i, s in enumerate(scores):
    print(f"Layer {i:02d}: {s:.4f}")
