"""Measure the real decay g and its within-chunk cumulative product gcumF at
several chunk sizes C, to choose the repair. Reports whether 1/gcumF stays within
fp32 range (max 3.4e38) per C, and the g distribution."""
import os, sys, random
os.environ["GDN3_KMD2_NATIVE"] = "1"; os.environ["GDN3_KMD2_ROUT"] = "4"
os.environ["GDN3_FAST_SCAN"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/dev/gdn3_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research/runs_fable")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
from ruler_kmd2 import SNAP, build_sample

DEV = "cuda:0"
tok = AutoTokenizer.from_pretrained(SNAP)
m = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(m); mgr.apply_upgrade()
sd = torch.load("/home/dev/gdn3_fable/runs/kmd2_native_heal/final/gdn3_layers.pt", map_location="cpu")
m.load_state_dict(sd, strict=False); m.to(DEV).eval()

gcap = []
def spy_factory(orig):
    def spy(q, k, v, g, beta_e, beta_w):
        gcap.append(g.detach().float())
        return orig(q, k, v, g, beta_e, beta_w)
    return spy
for lyr in m.model.layers:
    if hasattr(lyr, "linear_attn") and type(lyr.linear_attn).__name__ == "KMD2NativeAttn":
        lyr.linear_attn._scan = spy_factory(lyr.linear_attn._scan)

rng = random.Random(0)
ids, _ = build_sample(tok, 512, 16, 4, rng)
with torch.no_grad():
    m.model(input_ids=torch.tensor([ids], device=DEV), use_cache=False)

g = torch.cat([x.reshape(-1, x.shape[-1]) for x in gcap], 0)  # [tokens*heads*layers, dk]
print(f"g samples {tuple(g.shape)}")
qs = torch.quantile(g.flatten()[:1_000_000], torch.tensor([0.0, .001, .01, .1, .5], device=DEV))
print(f"g quantiles [min,.1%,1%,10%,50%]: {[f'{v:.3e}' for v in qs.tolist()]}")
print(f"g exact zeros: {(g==0).float().mean().item()*100:.4f}%   g<1e-4: {(g<1e-4).float().mean().item()*100:.3f}%")

T = gcap[0].shape[1]
print(f"\nper-chunk gcumF underflow (fp32 min normal ~1.2e-38, so 1/gcumF overflows if gcumF<2.9e-39):")
for C in (16, 32, 64, 128):
    # worst-case within any aligned chunk across the real sequence
    gg = gcap[0][0, :(T//C)*C].reshape(-1, C, gcap[0].shape[2], gcap[0].shape[3])  # [nC,C,H,dk]
    fc = gg.float().cumprod(dim=1)                          # [nC,C,H,dk]
    fmin = fc.min().item()
    inv_max = (1.0 / fc.clamp_min(1e-45)).max().item()
    frac_uf = (fc < 2.9e-39).float().mean().item() * 100
    print(f"  C={C:3d}: gcumF min {fmin:.2e} | 1/gcumF max {inv_max:.2e} | "
          f"{frac_uf:.2f}% underflow fp32 | {'OK' if inv_max < 3e38 else 'OVERFLOWS'}")
