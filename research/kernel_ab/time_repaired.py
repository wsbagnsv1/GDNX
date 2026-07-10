import sys, time, torch, importlib.util
sys.path.insert(0, "/home/dev/gdn3_fable"); sys.path.insert(0, "/home/dev/gdn3_fable/research/kernel_ab")
import ref_scan
spec = importlib.util.spec_from_file_location("fk", "/home/dev/gdn3_fable/gdn3/kmd2_fast_scan.py")
fk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
B, T, H, r, dk, dv = 2, 512, 16, 4, 128, 128   # train config (heal uses seq_len 512)

def mk(seed=7):
    gG = torch.Generator(device=dev).manual_seed(seed); rr = lambda *s: torch.randn(*s, device=dev, generator=gG)
    q = rr(B, T, H, r, dk) * (dk ** -.5); k = torch.nn.functional.normalize(rr(B, T, H, dk), dim=-1); v = rr(B, T, H, dv)
    decay = torch.exp(-torch.nn.functional.softplus(rr(B, T, H, dk) * 1.2 - 0.6))
    be = torch.sigmoid(rr(B, T, H)); bw = torch.sigmoid(rr(B, T, H)); om = rr(H, r)
    return [q, k, v, decay, be, bw, om]

def fb(fn, ins, ups):
    xs = [t.clone().requires_grad_(t.dtype.is_floating_point) for t in ins]
    y = fn(*xs); (y.float() * ups).sum().backward()
    return y

def timeit(fn, label, nwarm=3, nit=8):
    ins = mk(); ups = torch.randn(B, T, H, dv, device=dev)
    for _ in range(nwarm): fb(fn, ins, ups)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(nit): fb(fn, ins, ups)
    torch.cuda.synchronize(); dt = (time.time() - t0) / nit
    print(f"  {label:28s} fwd+bwd {dt*1e3:7.2f} ms   {B*T/dt:8.0f} tok/s", flush=True); return dt

print(f"C={fk.C}  train B={B} T={T}", flush=True)
t_ref = timeit(ref_scan.scan, "reference (python loop)")
t_eag = timeit(fk._scan_impl, "repaired EAGER")
print(f"  -> eager speedup {t_ref/t_eag:.1f}x", flush=True)
print("  compiling (default mode)...", flush=True)
comp = torch.compile(fk._scan_impl)
t_c = timeit(comp, "repaired torch.compile")
print(f"  -> compiled speedup {t_ref/t_c:.1f}x", flush=True)
