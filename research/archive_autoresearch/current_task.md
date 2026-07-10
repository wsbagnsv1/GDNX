# GDN3 auto-research — HANDOFF DEFINITIVELY FINAL: CE-only loss is the ceiling, independent of task difficulty (41 configs, 0 recall)

## exp034 — tie q/k coproduct: FAILED; editable-side alignment lever EXHAUSTED
- **exp034** `tie_qk_coprod_exp034` (branch `exp-tie-qk-coprod`, reverted): tied the q/k coproduct
  projections (k_a=q_a, k_b=q_b by computation) so the same input produces matching q and k.
  Init-time tie (recurrence math unchanged -> PARITY OK ✅ trivially).
- **Result: tokacc 0.2875, recall 0.0** — tie did NOT help. Direct comparison to exp001 (untied,
  seq_len=512): tokacc 0.296, recall 0.0.
- **WHY the tie cannot help:** it makes q_coprod == k_coprod for the SAME INPUT. But in MQAR the query
  ("is:") and the stored key ("KEY") are DIFFERENT TOKENS -> different inputs -> different coproduct
  outputs -> STILL MISMATCHED. The tie only aligns q and k from the same token, which MQAR doesn't.

## HANDOFF IS NOW AIRTIGHT — complete evidence chain
- exp025: GDN3 contributes (identity 0.067 vs trained 0.30).
- exp027: compaction not a ceiling (exact buffer still 0.30/0.0).
- exp028/030: read mechanism not the lever (linear/softmax/sharp all 0.28/0.0).
- exp029: read learns weak discrimination (14% argmax) but stays ~uniform.
- exp031: read sharpens (entropy 1.94) on WRONG positions (CE rewards format not retrieval).
- exp033: BINDING FORMS with compaction (ideal key-query retrieves value, cos 0.35); actual query
  mismatches key (cos 0.10) -> root cause is query-key alignment.
- exp034: structural q/k tie cannot align cross-position query/key -> editable side exhausted.
- **=> The GDN3 mechanism (binding + read + state) is SUFFICIENT. The bottleneck is the CE-only loss
  not rewarding query-key alignment, on the FORBIDDEN (proxy) side.**

## Concrete asks for the human (ALL proxy/freeze edits, currently forbidden by RESEARCH.md)
- (b) auxiliary retrieval loss (+lambda * exact-match reward, or a contrastive alignment loss) —
  rewards the query aligning to the key. This is the DIRECT fix for the diagnosed root cause.
- (e) unfreeze PRESERVED in_proj_qkv so the dense q/k can form an induction head (copy key -> query).
- (c) task variant where format doesn't satisfy CE (non-overlapping / non-numeric values).

## exp038 — marginal-tokacc diagnostic: 0.30 is PARTIAL RETRIEVAL (3.5x format), not pure format
- **exp038** `marginal_tokacc_exp038` (standalone, ~6s): computed per-position marginal digit tokacc from 2000 episodes.
  - random (uniform) tokacc: 0.097; marginal (format-only) tokacc: **0.085**; model (trained): ~0.30.
- **SURPRISE: the model's 0.30 is 3.5x the format baseline (0.085).** NOT pure format — substantial PARTIAL RETRIEVAL. The 5
  inducing heads (exp036) ARE contributing correct digits (~0.215 above marginal). The signal IS reaching the output.
- **SHARPENS (not reopens) the handoff:** partial retrieval is ~14% per digit (exp029's 14% argmax) -> 0.30 tokacc but 0.14^4 ~= 0
  recall. The signal is REAL but too WEAK for exact recall; amplifying 14% -> 100% needs the loss.
- **Editable-side implication:** amplifying inducing heads COULD help IF TARGETED at the RIGHT heads. exp037 zeroed a RANDOM half
  (not the inducing ones) -> 0.30/0.0. Genuinely-untested: identify the CONSISTENTLY-inducing heads and zero the NON-inducing ones.

## exp039 — TARGETED head selection: FAILED (0.279/0.0); HANDOFF FINAL
- **exp039** `aggproj_targeted_exp039` (branch `exp-aggproj-targeted`, reverted): the LAST editable-side
  lever. (1) ranked 16 heads by consistent induction gap: top-5=[7,10,12,3,9] (gap 0.02-0.029, beat rest
  in 73% of episodes); bottom-11 are noise/anti-inducing. (2) zeroed _agg_proj for non-inducing heads
  (keep top-5). PARITY OK ✅. Result: tokacc 0.279, recall 0.0 — slightly WORSE than baseline (0.296).
- **WHY it failed:** the inducing heads are too WEAK/NOISY (mean gap 0.02, std 0.04) to carry retrieval
  alone; the 11 "noise" heads were contributing to the format-marginal 0.30. Concentrating the signal
  LOST partial retrieval rather than amplifying it. The partial-retrieval signal (exp038) is DIFFUSE, not
  concentrated in a few strong heads.

## HANDOFF FINAL — every editable-side lever exhausted
- Query source: exp001/012/034/035 — all 0.29/0.0.
- Read mechanism: exp027/028/030 — all 0.28/0.0.
- Binding: forms (exp033), tie (exp034).
- State: exact no-compaction (exp027), GDN3 contributes (exp025).
- Dynamics: sharpens wrong (exp031), weak discrimination (exp029).
- Head selection: random half (exp037), targeted top-5 (exp039) — both ~0.30/0.0.
- Plateau composition (exp038): 0.30 = 3.5x format = partial retrieval (~14%/digit), too diffuse to amplify.
- **=> GDN3 produces a real but weak/diffuse partial-retrieval signal (0.30 tokacc, 0.14^4 ~= 0 recall).
  CE amplifies the FORMAT component but not the retrieval component (no gradient for exact match). NO
  editable-side change can concentrate/amplify the retrieval signal — it's diffuse and CE-undriven.**

## The autonomous loop has reached the genuine, final, evidence-backed structural ceiling
The bottleneck is the CE-only loss not rewarding exact retrieval, on the FORBIDDEN (proxy) side. All
editable-side levers tested; the GDN3 mechanism is sufficient but CE-undriven for retrieval.

## Concrete asks for the human (ALL proxy/freeze edits, currently forbidden)
- (b) auxiliary retrieval loss (+lambda * exact-match reward, or contrastive alignment loss) — the DIRECT fix.
- (e) unfreeze PRESERVED in_proj_qkv so dense q/k can form an induction head (copy key -> query).
- (c) task variant where format doesn't satisfy CE (non-overlapping / non-numeric values).

## exp041 — n_keys=2 difficulty test: tokacc rises (0.3125) but recall STILL 0; HANDOFF DEFINITIVELY FINAL
- **exp041** `nkeys2_exp041` (config-only): the difficulty test. n_keys=2 (only 2 distractors) — partial retrieval
  strongest. Result: tokacc 0.3125, recall 0.0 (every eval step).
- **Clean difficulty gradient in tokacc, ZERO gradient in recall:** n_keys=8 (0.267/0.0), n_keys=4 (0.296/0.0),
  n_keys=2 (0.3125/0.0). Partial retrieval scales with difficulty; recall stays 0 at all.
- **CE-only loss DEFINITIVELY the ceiling, INDEPENDENT of difficulty.** Even with 2 distractors (partial retrieval at
  its strongest), exact recall never emerges. CE rewards format (any 4-digit number), not the specific correct value.

## HANDOFF DEFINITIVELY FINAL (41 experiments, 0 nonzero recall)
Complete evidence chain spans: all editable-side levers, all distinct ceiling hypotheses, AND task difficulty.
- Editable levers: query source, read mechanism, binding, state, dynamics, head selection (random + targeted).
- Ceiling hypotheses ruled out: frozen-path (exp025), compaction (exp027), read linearity (exp028), read mechanism
  (exp030), v-encoding/LM-head-identical (exp032 artifacts), LM-head alignment (exp040).
- Task difficulty (n_keys 2/4/8): partial retrieval scales (0.27->0.31), recall stays 0 at all (exp041).
- Confirmed: 0.30 = partial retrieval (exp038, 3.5x format, ~14%/digit), diffuse across heads (exp039), too weak
  for exact recall (0.14^4 ~= 0); CE doesn't amplify it (exp031 sharpens wrong).
- **=> GDN3 produces a real, difficulty-sensitive partial-retrieval signal; the CE-only loss prevents crossing from
  partial to exact recall, on the FORBIDDEN (proxy) side. The autonomous loop has reached the definitive final ceiling.**

## Concrete asks for the human (ALL proxy/freeze edits, currently forbidden)
- (b) auxiliary retrieval loss (+lambda * exact-match reward, or contrastive alignment loss) — the DIRECT fix.
- (e) unfreeze PRESERVED in_proj_qkv so dense q/k can form an induction head (copy key -> query).
- (c) task variant where format doesn't satisfy CE (non-overlapping / non-numeric values).

## What the autonomous loop should do while awaiting human direction
Do NOT run more GDN3 edits or diagnostics — every editable-side lever, ceiling hypothesis, AND difficulty level is
exhausted (41 experiments). Running more would be churn against a proven, definitive, final structural ceiling. The
remaining levers (auxiliary loss, unfreezing in_proj_qkv, task variant) ALL require human approval to edit the
proxy/freeze set.

No config robustly beat exp001. 41 configs total. **Max recall ever: 0.0 (all 41).**
- exp006 `lrmem15e4_steps800`: tokacc 0.308 (fragile, skip 0.32) — best tokacc, within noise.
- exp001 `calib_baseline`: tokacc 0.296 (robust, skip 0) — stable champion.
## exp037 — head-restricted _agg_proj: STILL 0.30/0.0; exp036 per-head-gate lever INVALIDATED
- **exp037** `aggproj_headselect_exp037` (branch `exp-aggproj-headselect`, reverted): tested whether HEAD
  SELECTION is the lever. KEY REALIZATION: the model keeps ALL H heads through `_agg_proj` (TRAINABLE,
  NOT PRESERVED) — the "head-averaging" in exp036 was MY PROBE's mean-over-H, not the model's. So a
  per-head GATE would be REDUNDANT (_agg_proj can already up-weight inducing heads). Instead tested H1
  (CE doesn't exploit _agg_proj's capacity) vs H2 (signal too weak): zeroed _agg_proj for half the heads.
  PARITY OK ✅. Result: tokacc 0.30, recall 0.0 — SAME plateau as exp001 (all 16 heads, 0.296/0.0).
- **H1 CONFIRMED: bottleneck is loss-side, not head-selection-side.** CE doesn't exploit even a
  head-restricted path. `_agg_proj` has the capacity; CE doesn't drive it to induction.
- **The exp036 per-head-gate lever is INVALIDATED.** Inducing heads exist, aggregation can select them,
  but CE has no gradient toward selection (format satisfies CE without it).

## HANDOFF BACK TO AIRTIGHT — complete evidence chain (every editable axis)
- Query source: exp001/012/034/035 — all 0.29/0.0.
- Read mechanism: exp027/028/030 — all 0.28/0.0.
- Binding: forms (exp033), tie (exp034).
- State: exact no-compaction (exp027), GDN3 contributes (exp025).
- Dynamics: sharpens wrong (exp031), weak discrimination (exp029).
- Head selection: frozen dense induces in 5/16 (exp036), _agg_proj capacity unused (exp037).
- **=> GDN3 mechanism SUFFICIENT and has CAPACITY; CE-only loss doesn't drive it to retrieval.**
  Bottleneck on the FORBIDDEN (proxy) side.

## Concrete asks for the human (ALL proxy/freeze edits, currently forbidden)
- (b) auxiliary retrieval loss (+lambda * exact-match reward, or contrastive alignment loss).
- (e) unfreeze PRESERVED in_proj_qkv so dense q/k can form an induction head (copy key -> query).
- (c) task variant where format doesn't satisfy CE (non-overlapping / non-numeric values).

## What the autonomous loop should do while awaiting human direction
Do NOT run more GDN3 edits — the evidence chain proves the mechanism has capacity and CE doesn't exploit
it. Every editable axis is exhausted. The remaining levers (auxiliary loss, unfreezing in_proj_qkv, task
variant) ALL require human approval to edit the proxy or the freeze set.

## Best so far (preserved)

## Branches preserved (all reverted; main clean)
exp-timescale-mixture, exp-coprod-binding-init, exp-output-norm-bypass, exp-router-sharpen,
exp-blend-asymmetric, exp-sharp-read, exp-output-gain, exp-output-gain-10x, exp-zero-gdn3-diagnostic,
exp-sharp-read-replace, exp-softmax-prevq, exp-sharp-softmax, exp-tie-qk-coprod. All empty of commits.

## Standalone diagnostic scripts (read-only, do not edit proxy/GDN3 source)
- research/diag_frozen_qk.py — read-score alignment at init (uniform-at-init).
- research/exp029_read_discrim_train.py — read discrimination before/after training (weak-but-real).
- research/diag_read_magnitudes.py — relative norms of read terms (retrieval dominates at init).
- research/exp032_ideal_query_probe.py — ideal-query retrievability (self works, binding confounded).
- research/exp033_kron_binding_compact.py — Kron binding WITH compaction (binding FORMS, query mismatches).

## NOTE (device + protocol)
Per RESEARCH.md: run on `--device cuda:1`. Phase-3 math changes: branch, edit, parity, proxy, revert if
not a beat. exp034 was an init-time tie (recurrence math unchanged -> parity OK trivially); reverted.
