# KMD-2 Scan Kernel Speed Search — Operating Brief (read every turn)

You are an autonomous optimization loop. **Goal: make the KMD-2 recurrence scan as
fast as possible (forward + backward) while staying numerically correct**, so a
distillation heal that currently runs at ~18 s/step (a pure-Python per-token loop)
becomes fast enough to train long-context on 2×B300. Algorithmic wins here (chunking,
fusion, kernels) transfer to the B300; you develop on a local RTX 5060 Ti (Blackwell
sm_120, torch 2.12, triton 3.7, CUDA 12.8).

## What you optimize
The single function `scan(q, k, v, g, beta_e, beta_w, out_mix=None)` in **your
workspace's `cand_scan.py`**. Its exact required semantics (the recurrence, shapes,
dtypes) are the frozen reference `../ref_scan.py` — read its docstring. The starting
`cand_scan.py` is a copy of the reference (1.0× baseline). Your job is to rewrite the
body to be faster and produce the same outputs.

The recurrence is a gated delta rule with per-channel decay, decoupled erase/write
gates, and r_out output-MIMO query slots. The Python `for t in range(T)` loop is the
bottleneck. High-value directions (form your own hypothesis each turn, don't just try
them in order):
- **Chunk-parallel form**: process the sequence in blocks of C tokens with matmuls
  (intra-chunk via masked attention-like products, inter-chunk via the carried state
  S). This is the standard fast form of linear-attention / gated-delta-rule scans and
  is usually the single biggest win.
- `torch.compile` on the loop or the chunked body.
- A **Triton** kernel for the recurrence (fused decay + rank-1 updates + readout).
- Fuse the per-channel decay `g` into the state update; batch the small bmms.
- Handle r_out cheaply (the slots share q; the out_mix reduction is linear).
Keep everything differentiable — training needs correct **gradients**, not just
forward.

## The loop (one turn = one concrete improvement)
1. Skim `<ws>/leaderboard.jsonl` (durable results, bench appends here) and
   `<ws>/notes.md` (what you've tried + why) so you don't repeat yourself.
2. Form one hypothesis. Edit **only** `<ws>/cand_scan.py`.
3. Bench it (this checks correctness vs ref, then times; appends one line to your
   leaderboard):
   ```
   CUDA_VISIBLE_DEVICES=<gpu> /home/dev/gdn3_qwen35_package/.venv/bin/python \
     research/kernel_ab/bench_scan.py --cand research/kernel_ab/<ws>/cand_scan.py \
     --leaderboard research/kernel_ab/<ws>/leaderboard.jsonl --device cuda:0 \
     --note "one-sentence description of this change"
   ```
4. Append 1–3 lines to `<ws>/notes.md`: what you changed, the result
   (`train_fb_toks` / `speedup_fb`, or DISQUALIFIED + the relMSE), and the next idea.
   Then **STOP** — one improvement per turn.

## Rules
- **Primary metric**: `train_fb_toks` (forward+backward tok/s at B=2,T=512), and it
  only counts when `correct=true`. Secondary: `eval_fwd_toks` (B=1,T=2048).
- A candidate with fwd relMSE ≥ 2e-3 or grad relMSE ≥ 1e-2 is **DISQUALIFIED** — a
  fast wrong kernel is worthless (it must drop into the trained checkpoint unchanged).
  If disqualified, fix correctness before chasing speed.
- Edit ONLY files inside your own workspace directory. **Never** edit `../ref_scan.py`,
  `../bench_scan.py`, this brief, or the other workspace. Never weaken the tolerances.
- If a whole approach dead-ends, record why in notes.md and pivot. Escalate
  (Python-chunked → torch.compile → Triton) rather than idling or repeating.
- Preserve `scan`'s signature and keep it importable (the winner drops straight into
  `gdn3/kmd2_native.py::_scan`).
