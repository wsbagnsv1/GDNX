"""Time a single teacher-forced forward of the native heal at several lengths,
to decide the feasibility of the long-context RULER sweep."""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/dev/gdn3_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research/runs_fable")
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from ruler_kmd2 import load_student, _SDPA_BACKENDS

DEV = "cuda:0"
m = load_student("/home/dev/gdn3_fable/runs/kmd2_native_heal/final", 4, DEV, native=True)
for L in (16384, 32768):
    ids = torch.randint(0, 100000, (1, L), device=DEV)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), sdpa_kernel(_SDPA_BACKENDS):
        m.model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated(DEV) / 1e9
    print(f"L={L:6d}  {dt:7.2f}s/fwd  peakmem {mem:.1f}G", flush=True)
    torch.cuda.reset_peak_memory_stats(DEV)
