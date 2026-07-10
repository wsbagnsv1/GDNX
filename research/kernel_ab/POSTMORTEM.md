# KMD-2 Scan Kernel A/B — Outcome & Postmortem (2026-07-08)

## Result of the A/B
Two `pi` auto-research loops optimized the KMD-2 recurrence scan for fwd+bwd speed
(hard correctness gate vs frozen `ref_scan.py`). Both plateaued and were stopped.

| competitor | model | best speedup_fb | approach |
|---|---|---|---|
| A | GLM-5.2 | **86.6×** (255k tok/s) | chunk-parallel gated delta rule (C=128) + bf16 gemms + **hand-written Triton fwd/bwd `trsm` kernels** + `max_autotune` compile |
| B | super-qwen-preview | 5.1× (17k tok/s) | per-token Triton fwd kernel + compiled bwd (never reached chunk-parallel) |

GLM independently converged on exactly the chunk-parallel + Triton + compile
"merge" and dominated. qwen's per-token Triton was strictly subsumed.

## THE CATCH — the winner is INVALID on the real model
Wiring GLM's 86× kernel into `kmd2_native.py` (env `GDN3_FAST_SCAN=1`) gave **0.00
RULER recall** (512/16:4) vs the reference scan's ~0.96.

Root cause: GLM's chunk-parallel form uses the within-chunk decay-**ratio** trick
`kUp = k*gcumF`, `kDn = k/gcumF` (so between-token decay becomes a ratio
`Kmat = kUp @ kDn^T`). This is only stable when the per-chunk cumulative decay
`gcumF` stays well above underflow — i.e. decay ≈ 1 across the whole chunk.

The **real trained model's decay** `g` spans the **full (0,1] with mean 0.78 and
reaches ~0** (measured on the native heal ckpt). Over a C=128 chunk `gcumF`
**underflows to exactly 0** → `kDn = k/0` → all precision lost on the decayed
tokens that carry the retrieval signal. Real-input per-layer relMSE = 5.2e-3
(already over the 2e-3 gate); under realistic-decay bench inputs relMSE = 0.61.

## Why the gate missed it (the proxy blind spot)
`bench_scan.py::_mk_inputs` originally drew decay `sigmoid(randn*0.5+3.0) ≈ 0.97`
(benign, `gcumF` never underflowed), so GLM's kernel passed at relMSE 1.4e-5. The
fitness function did not exercise the decay regime the real model actually uses.
This is the "generous single-layer results that fall apart on the real model"
pattern again — the fitness must match reality.

## Fixes applied
- `bench_scan.py` decay now `exp(-softplus(randn*1.2-0.6))` — spans (0,1], mean
  ~0.78, reaches ~0, mirroring the native mechanism. It now correctly DISQUALIFIES
  the ratio-trick kernel (fwd relMSE 0.61) while the reference passes at 1.0×.
- `gdn3/kmd2_fast_scan.py` = GLM's snapshot, kept for reference. **Do NOT enable
  `GDN3_FAST_SCAN` until a decay-stable kernel passes the corrected bench.** Default
  is off; the model uses the reference scan and all prior results stand.

## Path forward
The ratio trick is structurally wrong for decay→0 (fp32 underflows too; not a
precision knob). The correct fast form is the **numerically-stable chunk scan**
(as in fla-org `chunk_gated_delta_rule`): work in **log-cumsum decay**, normalize
intra-chunk *relative to chunk start*, and apply decay via masked weights — never
divide by an underflowing cumulative product. Re-run the A/B against the corrected
bench (which now enforces this), or hand-port the stable chunk form.

## REPAIR (2026-07-08, hand-fixed, GLM offline)
Measured the real decay first: `g` min **3.4e-5** (no exact zeros), mean 0.78,
median 0.999. `1/gcumF` overflows fp32 for **C≥32** (3% @32, 18% @128) but is
SAFE at **C=16** (max 4.1e36, 0% underflow). So the fix, applied to
`gdn3/kmd2_fast_scan.py`:
- **C 128→16** (largest fp32-safe chunk for this decay).
- **decay-ratio gemms (Kmat, A, W) → fp32** (`_bmm_r`); bf16 kept only on the
  bounded S-coupled path (m_inter, term1, term2).
- clamp floor 1e-12→1e-38 (never clamps real gcumF≥2.4e-37).
- compile mode `max-autotune`→**default** (max-autotune compiles >7 min on the
  8×-larger C=16 unroll; default compiles fast).

Result — CORRECT and still fast:
- fwd relMSE vs ref: **8.3e-6** synthetic-realistic-decay, **4.2e-6** on real model
  inputs (was 0.61 / 5.2e-3 broken).
- speed (train B=2,T=512 fwd+bwd): **33.2×** compiled (100.6k tok/s), 9.7× eager
  (vs GLM's invalid 87×; the gap is the 8× smaller chunk needed for fp32 safety).
- RULER 512/16:4 (n=8): 0.88 vs ref-path 0.94 (2/32 argmax-borderline flips; the
  4e-6 relMSE confirms equivalence — within n=8 noise).
- **Caveat**: default compile recompiles per distinct seq_len (fine for the fixed-512
  heal; poor for variable-length inference). 33× turns ~18 s/step into ~0.5–0.6 s/step.

Remaining upside (future): two-level chunking (outer C=64/128 for state carry +
Triton trsm, inner 16-blocks for the fp32 ratio) would reclaim speed toward GLM's
number while staying numerically valid — a real but multi-turn rewrite.
