#!/usr/bin/env python3
"""Init-only throughput + memory benchmark on the 0.8B: native Qwen GDN
(GDN2-family) vs original GDN3 vs KMD-2 (r=1 / r=4 / r=4 iso-state).

Measures (no training):
  * model params: total, and linear-attn-stack subtotal (the swapped part)
  * recurrent-state "KV-cache" analog per linear layer, analytic, from module
    dims (+ conv cache where the variant has one); full-attn layers' true KV
    cache is identical across variants and reported once for context
  * prefill throughput: full-model forward, T=512, batch 1, fp32, tokens/sec
  * peak CUDA memory for that forward

NOTE: throughput here is the CURRENT torch implementations (no fla kernels, no
decode cache in the drop-ins). It answers "what will the heal cost today", NOT
"what could optimized MIMO kernels do at decode" — that claim needs kernels.

Usage: bench_gdn_memory_throughput.py --variant native|gdn3|kmd2_r1|kmd2_r4|kmd2_r4_iso
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
                choices=["native", "gdn3", "kmd2_r1", "kmd2_r4", "kmd2_r4_iso"])
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--out", default="")
args = ap.parse_args()

if args.variant.startswith("kmd2"):
    os.environ["GDN3_KMD2"] = "1"
    os.environ["GDN3_KMD2_R"] = "1" if args.variant == "kmd2_r1" else "4"
    if args.variant == "kmd2_r4_iso":       # iso-state with native: dk=dv=128
        os.environ["GDN3_KMD2_DK"] = "128"
        os.environ["GDN3_KMD2_DV"] = "128"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/dev/gdn3_fable")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
dev = torch.device(args.device)

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
cfg = model.config
layer_types = cfg.to_dict().get("layer_types", [])
n_lin = sum(t == "linear_attention" for t in layer_types)
n_full = sum(t == "full_attention" for t in layer_types)

if args.variant != "native":
    from gdn3.gdn3_upgrade import GDN3UpgradeManager
    mgr = GDN3UpgradeManager(model)
    mgr.apply_upgrade()

model.config.use_cache = False
model.to(dev).eval()

# ---------------- params ----------------
total_params = sum(p.numel() for p in model.parameters())
lin_params = 0
for i, t in enumerate(layer_types):
    if t == "linear_attention":
        lin_params += sum(p.numel() for p in model.model.layers[i].linear_attn.parameters())

# ---------------- analytic recurrent-state size (per linear layer) ----------------
lyr = model.model.layers[[i for i, t in enumerate(layer_types) if t == "linear_attention"][0]].linear_attn
state_elems = conv_elems = 0
desc = ""
if args.variant == "native":
    H = cfg.linear_num_key_heads
    K = cfg.linear_key_head_dim
    V = cfg.linear_value_head_dim
    ck = cfg.linear_conv_kernel_dim
    state_elems = H * K * V
    conv_elems = (ck - 1) * (H * 3 * K)          # causal conv rolling cache
    desc = f"dense S[H={H},{K}x{V}] + conv cache"
elif args.variant == "gdn3":
    H, M, R, P = lyr.H, lyr.M, lyr.R, lyr.P
    per_lane = (R * lyr.a_v * lyr.a_k) + (R * lyr.b_v * lyr.b_k) + (lyr.V * P) + (lyr.K * P)
    state_elems = H * M * per_lane
    conv_elems = (lyr.conv_kernel_size - 1) * (H * 3 * lyr.K)
    desc = (f"Kron A+Bk + residual UV [H={H},M={M},R={R},P={P}] "
            f"({per_lane} elems/lane) + conv cache")
else:
    H, dk, dv = lyr.H, lyr.dk, lyr.dv
    state_elems = H * dv * dk                    # rank-r writes do NOT grow state
    conv_elems = 0                               # no conv in drop-in yet (retrofit adds Qwen's)
    desc = f"dense S[H={H},{dv}x{dk}] (r={lyr.r}; state size independent of r)"

# full-attn true KV cache, identical across variants (context only)
kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
full_kv_per_tok = 2 * kvh * hd                    # elems per token per full layer

# ---------------- prefill throughput ----------------
text = "The code for amber-lantern is 4521. " * 200
ids = tok(text, return_tensors="pt").input_ids[:, :args.seq].to(dev)
with torch.no_grad():
    for _ in range(2):
        model(input_ids=ids)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time(); iters = 5
    for _ in range(iters):
        model(input_ids=ids)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9

result = {
    "variant": args.variant,
    "params_total_M": round(total_params / 1e6, 2),
    "params_linear_stack_M": round(lin_params / 1e6, 2),
    "state_desc": desc,
    "state_elems_per_linear_layer": state_elems,
    "conv_cache_elems_per_linear_layer": conv_elems,
    "recurrent_state_total_MB_fp32": round((state_elems + conv_elems) * n_lin * 4 / 1e6, 2),
    "recurrent_state_total_MB_bf16": round((state_elems + conv_elems) * n_lin * 2 / 1e6, 2),
    "full_attn_kv_MB_per_1k_tokens_bf16": round(full_kv_per_tok * n_full * 1000 * 2 / 1e6, 2),
    "prefill_s_per_512tok_fwd": round(dt, 3),
    "prefill_tok_per_s": round(args.seq / dt, 1),
    "peak_fwd_mem_GB_fp32": round(peak_gb, 2),
    "n_linear_layers": n_lin, "n_full_layers": n_full,
}
print(json.dumps(result, indent=2))
if args.out:
    json.dump(result, open(args.out, "w"), indent=2)
