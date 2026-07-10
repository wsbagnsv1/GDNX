# research/ — index

Start with the repo-root [`HANDOFF.md`](../HANDOFF.md) for the big picture. This
folder holds the experimental record.

## Read first (the summaries)
- **`EXPERIMENT_LEDGER.md`** — one row per experiment (GLM auto-research → Fable
  KMD-2 → kernel A/B), with results, confidence, and revisit verdicts.
- **`KMD2_STATUS.md`** — the working state: every architecture decision, the heal
  milestone, and the RULER falloff table (512→32768).
- **`kernel_ab/POSTMORTEM.md`** — the scan-kernel A/B race, why the 87× winner was
  invalid (decay-ratio underflow), and the 33× repair.

## Live tools
- `proxy_mqar.py` — frozen-backbone CE MQAR proxy (fast fitness stand-in).
- `testbed_mqar.py` — from-scratch atomic-MQAR testbed (where MIMO+compaction were
  first shown to work).
- `bench_gdn_memory_throughput.py` — weights/state/prefill bench across variants.
- `runs_fable/` — RULER benchmark (`ruler_kmd2.py`), the final plotting scripts, the
  milestone plots (`*.png`), and the final RULER result JSONs (teacher vs native,
  short + long context).
- `kernel_ab/` — the kernel speed A/B: fitness harness (`bench_scan.py` +
  `ref_scan.py`), the two competitors' workspaces (`glm/`, `qwen/`), the repair
  diagnostics (`measure_decay.py`, `diag_real_inputs.py`), and `ab_speedup_race.png`.

## Archives (provenance, not active)
- `archive_autoresearch/` — the phase-1 GLM 41-experiment loop (configs, per-exp
  outputs, probe scripts, `research_log.md`, `leaderboard.jsonl`).
- `runs_fable/experiment_data/` — testbed/frontier/mamba-3/seed/rope-mod/trap raw
  result JSONs + logs (conclusions folded into `KMD2_STATUS.md`).
