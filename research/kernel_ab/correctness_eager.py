import os, sys, torch, importlib.util
sys.path.insert(0, "/home/dev/gdn3_fable"); sys.path.insert(0, "/home/dev/gdn3_fable/research/kernel_ab")
import ref_scan
spec = importlib.util.spec_from_file_location("fk", "/home/dev/gdn3_fable/gdn3/kmd2_fast_scan.py")
fk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)
dev = "cuda:0"

def mk(B, T, H, r, dk, dv, seed):
    gG = torch.Generator(device=dev).manual_seed(seed); rr = lambda *s: torch.randn(*s, device=dev, generator=gG)
    q = rr(B, T, H, r, dk) * (dk ** -.5); k = torch.nn.functional.normalize(rr(B, T, H, dk), dim=-1); v = rr(B, T, H, dv)
    decay = torch.exp(-torch.nn.functional.softplus(rr(B, T, H, dk) * 1.2 - 0.6))  # realistic decay
    be = torch.sigmoid(rr(B, T, H)); bw = torch.sigmoid(rr(B, T, H)); om = rr(H, r)
    return q, k, v, decay, be, bw, om

for name, (B, T, H, r, dk, dv) in {"train": (2, 512, 16, 4, 128, 128), "eval": (1, 2048, 16, 4, 128, 128)}.items():
    ins = mk(B, T, H, r, dk, dv, hash(name) & 0xffff)
    yr = ref_scan.scan(*ins); yf = fk._scan_impl(*ins)  # EAGER, no compile
    rel = ((yf.float() - yr).pow(2).mean() / yr.pow(2).mean()).item()
    print(f"  {name}: eager fwd relMSE {rel:.3e}  finite={torch.isfinite(yf).all().item()}  (tol 2e-3)")
print(f"  C = {fk.C}")
