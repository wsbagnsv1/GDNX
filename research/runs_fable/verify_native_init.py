"""Verification gates for the GDN-2-point warm start. MUST pass before training:
  gate 1: per-layer output MSE (student KMD-2 native layer vs native layer),
          same inputs, ~0 (rotation at init contributes ~1e-3 relative).
  gate 2: end-to-end KL(teacher||student) on real text ~ 0 (target < 0.05).
  gate 3: RULER 16:1 @1024 recall at init ~ teacher (target > 0.8).
"""
import os, sys, random
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/dev/gdn3_fable")
os.environ["GDN3_KMD2_NATIVE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
DEV = "cuda:0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager

tok = AutoTokenizer.from_pretrained(SNAP)
teacher = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32,
                                               low_cpu_mem_usage=True).to(DEV).eval()
student = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32,
                                               low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(student); mgr.apply_upgrade(); upg = mgr.upgraded_layers
student.to(DEV).eval()

text = ("The quick brown fox jumps over the lazy dog. " * 30 +
        "One of the special magic numbers for kupola is: 4531207. " +
        "The grass is green and the sky is blue. " * 10 +
        "What is the special magic number for kupola? The number is: 4531207")
ids = tok(text, return_tensors="pt").input_ids[:, :512].to(DEV)

# gate 1: per-layer output match on identical inputs (hook teacher inputs, feed both)
t_in, t_out, s_out = {}, {}, {}
def hook_t(i):
    def h(mod, args, kwargs, out):
        x = args[0] if args else kwargs["hidden_states"]
        t_in[i], t_out[i] = x.detach(), out.detach()
    return h
hooks = [teacher.model.layers[i].linear_attn.register_forward_hook(hook_t(i), with_kwargs=True)
         for i in upg]
with torch.no_grad():
    t_logits = teacher(input_ids=ids).logits
for h in hooks: h.remove()
print("gate 1: per-layer output relMSE (same inputs):")
worst = 0.0
with torch.no_grad():
    for i in upg:
        so = student.model.layers[i].linear_attn(t_in[i])
        rel = (so - t_out[i]).pow(2).mean().item() / t_out[i].pow(2).mean().item()
        worst = max(worst, rel)
        if i in (upg[0], upg[-1]):
            print(f"  layer {i:2d}: relMSE {rel:.2e}")
print(f"  worst layer relMSE {worst:.2e}  -> {'PASS' if worst < 1e-2 else 'FAIL'}")

# gate 2: end-to-end KL
with torch.no_grad():
    s_logits = student(input_ids=ids).logits
kl = F.kl_div(F.log_softmax(s_logits.float(), -1), F.log_softmax(t_logits.float(), -1),
              reduction="batchmean", log_target=True).item() / ids.shape[1]
print(f"gate 2: end-to-end KL/token {kl:.4f}  -> {'PASS' if kl < 0.05 else 'FAIL'}")

# gate 3: quick RULER (16 needles, 1 query, ctx 1024, 4 samples)
sys.path.insert(0, "/home/dev/gdn3_fable/research/runs_fable")
from ruler_kmd2 import build_sample, score
rng = random.Random(1234)
got = tot = 0
for _ in range(4):
    sids, spans = build_sample(tok, 1024, 16, 1, rng)
    c, t = score(student, sids, spans, DEV)
    got += c; tot += t
print(f"gate 3: init RULER 16:1@1024 recall {got}/{tot}  -> "
      f"{'PASS' if got / max(1, tot) > 0.8 else 'FAIL'}")
