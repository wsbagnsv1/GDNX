"""Capture REAL _scan inputs from a live native-heal forward on an actual RULER
sample, then compare the frozen ref scan vs the fast (chunk-parallel bf16) scan on
those exact tensors. Diagnoses why the fast scan gives 0.00 recall despite passing
the benign-input bench. Also reports the decay cumulative-product range (the
suspected culprit: kDn = k/gcumF blows up when gcumF underflows)."""
import os, sys, random
os.environ["GDN3_KMD2_NATIVE"] = "1"; os.environ["GDN3_KMD2_ROUT"] = "4"
os.environ["GDN3_FAST_SCAN"] = "0"   # capture with the reference path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/dev/gdn3_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research/runs_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research/kernel_ab")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
from ruler_kmd2 import SNAP, build_sample
import ref_scan
from gdn3.kmd2_fast_scan import _scan_impl as fast_scan   # EAGER (skip compile)

DEV = "cuda:0"
tok = AutoTokenizer.from_pretrained(SNAP)
m = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(m); mgr.apply_upgrade()
sd = torch.load("/home/dev/gdn3_fable/runs/kmd2_native_heal/final/gdn3_layers.pt", map_location="cpu")
m.load_state_dict(sd, strict=False); m.to(DEV).eval()

# capture _scan args from the first upgraded layer
cap = {}
layer = None
for lyr in m.model.layers:
    if hasattr(lyr, "linear_attn") and type(lyr.linear_attn).__name__ == "KMD2NativeAttn":
        layer = lyr.linear_attn; break
orig = layer._scan
def spy(q, k, v, g, beta_e, beta_w):
    if "q" not in cap:
        cap.update(q=q.detach(), k=k.detach(), v=v.detach(), g=g.detach(),
                   be=beta_e.detach(), bw=beta_w.detach(),
                   om=layer.out_mix.detach() if layer.r_out > 1 else None)
    return orig(q, k, v, g, beta_e, beta_w)
layer._scan = spy

rng = random.Random(0)
ids, _ = build_sample(tok, 512, 16, 4, rng)
with torch.no_grad():
    m.model(input_ids=torch.tensor([ids], device=DEV), use_cache=False)

q, k, v, g = cap["q"], cap["k"], cap["v"], cap["g"]
be, bw, om = cap["be"], cap["bw"], cap["om"]
print(f"captured real _scan inputs: q{tuple(q.shape)} k{tuple(k.shape)} g{tuple(g.shape)}")
print(f"  decay g: min {g.min().item():.4f} max {g.max().item():.4f} mean {g.mean().item():.4f}")
# cumulative product over a chunk of 128 (what the ratio trick divides by)
C = 128
gc = g[:, :C].float().cumprod(dim=1)   # [B,C,H,dk]
print(f"  gcumF over C=128: min {gc.min().item():.3e} max {gc.max().item():.3e}")
gmin = gc.min().item()
amp = "inf (underflow to 0)" if gmin <= 0 else f"{1.0/gmin:.3e}x"
print(f"  => kDn=k/gcumF amplification up to {amp}  (ratio trick divides by this)")

with torch.no_grad():
    y_ref = ref_scan.scan(q, k, v, g, be, bw, om)
    y_fast = fast_scan(q, k, v, g, be, bw, om)
rel = ((y_fast.float() - y_ref).pow(2).mean() / y_ref.pow(2).mean()).item()
print(f"\nreal-input fwd relMSE fast-vs-ref: {rel:.4e}  (bench benign-input was ~1.4e-5)")
print(f"  y_ref  absmax {y_ref.abs().max().item():.3e}")
print(f"  y_fast absmax {y_fast.abs().max().item():.3e}  finite={torch.isfinite(y_fast).all().item()}")
