# OCR
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True)
tok = AutoTokenizer.from_pretrained("gpt2")

text = "The Eiffel Tower is in Paris. Where is the Eiffel Tower located?"
inputs = tok(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # attentions is a list: [layer1, layer2, ..., layerN]
    # each layer: (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions  

# Example: entropy per head in last layer
import torch.nn.functional as F
last_layer_attn = attentions[-1][0]  # shape: (num_heads, seq_len, seq_len)

head_scores = []
for h in range(last_layer_attn.size(0)):
    probs = last_layer_attn[h]  # (seq_len, seq_len)
    entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
    head_scores.append(entropy)

print("Head importance (lower entropy = sharper focus, more important):")
print(head_scores)
# kv_cache_offloading.py
import torch, psutil, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass
class Stats:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tokens: int = 0
    gpu_peak_gb: float = 0.0
    cpu_peak_gb: float = 0.0

class DRAMKVCache:
    """
    Layer-indexed KV store that lives in system DRAM.
    One layer at a time is prefetched to GPU.
    """
    def __init__(self):
        self.store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.thread = ThreadPoolExecutor(max_workers=2)
        self.stats = Stats()

    # ---------- cache lifecycle ----------
    def save(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        self.store[layer] = (k.detach().cpu(), v.detach().cpu())
        del k, v
        torch.cuda.empty_cache()

    def prefetch(self, layer: int, device: torch.device):
        if layer not in self.store: return
        k_cpu, v_cpu = self.store[layer]
        def _copy():
            self.store[layer] = (k_cpu.to(device, non_blocking=True),
                                 v_cpu.to(device, non_blocking=True))
        self.thread.submit(_copy)

    def fetch(self, layer: int, device: torch.device):
        k, v = self.store[layer]
        if not k.is_cuda:
            k, v = k.to(device), v.to(device)
            self.store[layer] = (k, v)
        return k, v

    # ---------- metric helpers ----------
    def finalize(self, tokens: int, ttft_s: float, total_s: float):
        self.stats.ttft_ms  = ttft_s * 1e3
        self.stats.tpot_ms  = (total_s - ttft_s) / max(1, tokens-1) * 1e3
        self.stats.tokens   = tokens
        gpu = torch.cuda.memory_stats()
        self.stats.gpu_peak_gb = gpu["reserved_bytes.all.peak"]/1024**3
        self.stats.cpu_peak_gb = psutil.virtual_memory().used/1024**3
        return self.stats


# offloaded_attention.py
import torch, torch.nn as nn
from kv_cache_offloading import DRAMKVCache

class OffloadedMHA(nn.Module):
    def __init__(self, hidden, heads, cache: DRAMKVCache, layer_idx: int):
        super().__init__()
        self.h, self.d = heads, hidden // heads
        self.layer = layer_idx
        self.cache = cache
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, use_cache=True):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.h, self.d).transpose(1,2)
        k_new = self.k(x).view(b, s, self.h, self.d).transpose(1,2)
        v_new = self.v(x).view(b, s, self.h, self.d).transpose(1,2)

        if use_cache:
            # 1️⃣ prefetch current layer KV
            self.cache.prefetch(self.layer, x.device)
            # 2️⃣ concatenate past + present
            k_hist, v_hist = self.cache.fetch(self.layer, x.device)              if self.layer in self.cache.store else (None, None)
            k_cat = torch.cat([k_hist, k_new], dim=2) if k_hist is not None else k_new
            v_cat = torch.cat([v_hist, v_new], dim=2) if v_hist is not None else v_new
            # 3️⃣ save back to DRAM
            self.cache.save(self.layer, k_cat, v_cat)
        else:
            k_cat, v_cat = k_new, v_new

        attn = (q @ k_cat.transpose(-1,-2)) / (self.d ** 0.5)
        y = (attn.softmax(dim=-1) @ v_cat).transpose(1,2).contiguous().view(b,s,-1)
        return self.o(y)


# benchmark_offload.py
import torch, time, json, psutil, gc
from transformers import LlamaTokenizer, LlamaForCausalLM
from kv_cache_offloading import DRAMKVCache
from patch_openllama import patch_openllama

PROMPT = "The Solar System consists of the Sun and the objects that orbit it. " * 64
NEW_TOKENS = 128
CTX_LENS   = [512, 1024, 2048, 4096]

def run(model, tokenizer, offload=False):
    device = model.device
    cache  = DRAMKVCache() if offload else None
    if offload: model = patch_openllama(model, cache)  # inject custom cache

    rows = []
    for ctx in CTX_LENS:
        gc.collect(); torch.cuda.empty_cache()
        text = PROMPT[:ctx]
        ids  = tokenizer(text, return_tensors="pt").to(device)

        torch.cuda.synchronize(); t0 = time.time()
        out = model.generate(**ids,
                             max_new_tokens=NEW_TOKENS,
                             use_cache=True,
                             cache_implementation=None if offload else "dynamic",
                             do_sample=False)
        torch.cuda.synchronize(); total = time.time() - t0
        stats = cache.finalize(NEW_TOKENS, cache.stats.ttft_ms/1e3 if offload else 0,
                               total) if offload else None

        rows.append({
            "ctx": ctx,
            "ttft_ms": stats.ttft_ms if offload else total*1e3,
            "tpot_ms": stats.tpot_ms if offload else 0,
            "gpu_gb":  stats.gpu_peak_gb if offload else
                       torch.cuda.max_memory_allocated()/1024**3,
            "cpu_gb":  stats.cpu_peak_gb if offload else
                       psutil.Process().memory_info().rss/1024**3})
    return rows

def main():
    tok = LlamaTokenizer.from_pretrained("openlm-research/open_llama_7b_v2",
                                         use_fast=False)
    mdl = LlamaForCausalLM.from_pretrained(
            "openlm-research/open_llama_7b_v2",
            torch_dtype=torch.float16,
            device_map="auto")
    base = run(mdl, tok, offload=False)
    offl = run(mdl, tok, offload=True)

    print("\n=== RESULTS ===")
    print(json.dumps({"baseline": base, "offloaded": offl}, indent=2))

if __name__ == "__main__":
    
from kv_cache_offloading import DRAMKVCache

def patch_openllama(model, cache):
    for idx, block in enumerate(model.model.layers):
        hidden = model.config.hidden_size
        heads  = model.config.num_attention_heads
        block.self_attn = OffloadedMHA(hidden, heads, cache, idx).to(model.device)
    return model
