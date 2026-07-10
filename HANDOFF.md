# GDN-X Handoff — KMD-2 working heal + fast kernel → Continuum Memory System (CMS)

Entry point for the team continuing this work. The prior release doc (`README.md`)
describes the older two-timescale package and is kept for history; **this file is
the current state.**

## TL;DR — what works now
- **A working GDN with MIMO + compaction that beats the native baseline on
  retrieval** (within its training regime). The KMD-2 native drop-in is warm-started
  at the GDN-2 point (mathematically = native Qwen3.5 linear attention at init) with
  identity-init new DOF (short conv, data-dependent rotation, r_out=4 output-MIMO,
  decoupled erase/write, per-channel decay), then healed by layerwise + KL
  distillation.
- **Multi-query RULER: heal > teacher** for context ≤4k (e.g. 512/4q **0.96 vs
  0.76**; 512/8q 0.85 vs 0.70). Ties ~8k. Hard extrapolation cliff ≥16k — a pure
  seq_len-512 training artifact, not architecture (see `research/KMD2_STATUS.md`).
- **A repaired chunk-parallel scan kernel at 33× fwd+bwd** (100k tok/s vs the 3k
  reference), turning ~18 s/step into ~0.5 s/step. Env-gated `GDN3_FAST_SCAN=1`.

## Where everything is
| what | path |
|---|---|
| Final architecture (warm-start heal drop-in) | `gdn3/kmd2_native.py` |
| **Fast scan kernel** (repaired, `GDN3_FAST_SCAN=1`) | `gdn3/kmd2_fast_scan.py` |
| Frozen-proxy drop-in (r=1 fast path) | `gdn3/kmd2.py` |
| Upgrade manager (selects native/kmd2/gdn3) | `gdn3/gdn3_upgrade.py` |
| Trained checkpoint (the milestone) | `runs/kmd2_native_heal/final/` |
| Training (warm-start + layerwise distill) | `train/train_gdn3_distill.py` |
| **Master findings ledger** | `research/EXPERIMENT_LEDGER.md` |
| **KMD-2 status / all decisions** | `research/KMD2_STATUS.md` |
| **Kernel A/B postmortem + repair** | `research/kernel_ab/POSTMORTEM.md` |
| RULER benchmark | `research/runs_fable/ruler_kmd2.py` |
| Kernel fitness harness (correctness-gated) | `research/kernel_ab/bench_scan.py` + `ref_scan.py` |
| MQAR proxy / from-scratch testbed | `research/proxy_mqar.py`, `research/testbed_mqar.py` |
| KMD-2 proposal (spec) | `fable_idea.txt` |
| Milestone plots | `research/runs_fable/*.png`, `research/kernel_ab/ab_speedup_race.png` |
| Data pipeline (training corpus) | `data_pipeline/`, `data/` |

## Open levers (candidates for CMS)
1. **Two-level chunking kernel** — the repaired kernel uses C=16 for fp32 decay
   safety (the ratio trick underflows at larger C on the real decay). Outer C=64/128
   for state carry + Triton trsm, inner 16-blocks for the fp32 ratio, would reclaim
   speed toward the (invalid) 87× the A/B first found. See `kernel_ab/POSTMORTEM.md`.
2. **Longer-context heal** — the ≥16k cliff is because the heal trained at seq_len
   512. Train longer-context (now affordable with the fast kernel) to push the
   crossover/cliff right. See `KMD2_STATUS.md`.
3. **The A/B auto-research loop** pattern (`kernel_ab/run_speed_loop.sh`) worked well
   for kernels — reusable for CMS optimization targets, given an honest fitness fn.

## Archives (moved, not deleted)
- `research/archive_autoresearch/` — phase-1 GLM 41-experiment loop (raw configs,
  per-exp outputs, probe scripts, `research_log.md`, `leaderboard.jsonl`). All
  conclusions are folded into the ledger; kept for provenance.
- `research/runs_fable/experiment_data/` — testbed / frontier / mamba-3 / seed /
  rope-mod / trap raw result JSONs+logs. Conclusions in `KMD2_STATUS.md`.

## Notes
- `~/gdn3_fable` is the untouched working original if anything here is missing.
- This copy was pruned of superseded checkpoints (kept only `kmd2_native_heal/final`),
  the phase-0 flattened `code\` export, a compile-debug dump, and Discord tokens.
