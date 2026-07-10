#!/usr/bin/env python3
"""A/B kernel-speed race: GLM-5.2 vs super-qwen speedup multiplier over iterations.
Speedup = train_fb_toks / workspace baseline (first correct entry). Log-y so both
arcs read. Per-model regime-change callouts; disqualified attempts marked. Note on
GLM's final kernel (which won on speed but is invalid end-to-end)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/dev/gdn3_fable/research/kernel_ab"
C_GLM, C_QW, C_DQ, INK = "#0072B2", "#E69F00", "#D55E00", "#1a1a1a"


def load(ws):
    rows = [json.loads(l) for l in open(f"{BASE}/{ws}/leaderboard.jsonl")]
    base = next(r["train_fb_toks"] for r in rows if r.get("correct"))
    it, spd, dq = [], [], []
    for i, r in enumerate(rows):
        if r.get("correct") and r.get("train_fb_toks"):
            it.append(i); spd.append(r["train_fb_toks"] / base)
        else:
            dq.append(i)
    return it, spd, dq, base


gi, gs, gdq, gbase = load("glm")
qi, qs, qdq, qbase = load("qwen")

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.plot(gi, gs, "-o", color=C_GLM, ms=4, lw=2, label="GLM-5.2  (GPU0)", zorder=5)
ax.plot(qi, qs, "-o", color=C_QW, ms=3, lw=1.6, label="super-qwen-preview  (GPU1)", zorder=4)

# disqualified attempts
for x in gdq:
    ax.scatter(x, 0.72, marker="x", color=C_DQ, s=40, zorder=6)
for x in qdq:
    ax.scatter(x, 0.72, marker="x", color=C_DQ, s=40, zorder=6)
ax.scatter([], [], marker="x", color=C_DQ, s=40, label="disqualified (wrong output)")

# ---- GLM regime callouts (idx, label, dy) ----
glm_notes = [
    (1, "chunk-parallel\nΔ-rule (C=64)", 20),
    (2, "+ torch.compile", 42),
    (3, "C=128", 90),
    (10, "bf16 gemms", -18),
    (12, "+ Triton fwd/bwd\ntrsm  → peak 87.6×", 14),
]
for idx, lab, dy in glm_notes:
    y = gs[gi.index(idx)]
    ax.annotate(lab, (idx, y), xytext=(idx + (0.4 if dy > 0 else 0.4), y * (1 + dy / 100.0) if dy > 0 else y * 0.6),
                fontsize=8, color=C_GLM, ha="left",
                arrowprops=dict(arrowstyle="-", color=C_GLM, lw=0.8, alpha=0.6))

# ---- qwen regime callouts ----
qw_notes = [
    (3, "custom autograd\nbackward", -0.35),
    (12, "Triton fwd kernel", 0.55),
    (19, "reduced-intermediate\nbwd", 0.62),
    (31, "pre-alloc buffers → 5.3× plateau", 0.30),
]
for idx, lab, dyf in qw_notes:
    y = qs[qi.index(idx)]
    ax.annotate(lab, (idx, y), xytext=(idx, y * (1 + dyf)),
                fontsize=8, color=C_QW, ha="center",
                arrowprops=dict(arrowstyle="-", color=C_QW, lw=0.8, alpha=0.6))
# qwen's long plateau bracket
ax.annotate("", xy=(31, 5.1), xytext=(76, 5.1),
            arrowprops=dict(arrowstyle="<->", color=C_QW, lw=0.8, alpha=0.5))
ax.text(53, 4.35, "45 turns, flat ~5× ceiling", fontsize=7.5, color=C_QW, ha="center", style="italic")

ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×", "50×", "100×"])
ax.set_ylim(0.6, 130)
ax.set_xlim(-1, 78)
ax.set_xlabel("optimization iteration (one benched candidate each)", color=INK)
ax.set_ylabel("fwd+bwd speedup vs reference scan (log)", color=INK)
ax.set_title("KMD-2 scan kernel A/B race — GLM-5.2 vs super-qwen-preview\n"
             "speedup multiplier per benched iteration (train B=2,T=512; hard correctness gate vs frozen reference)",
             fontsize=12, color=INK)
ax.grid(which="both", axis="y", alpha=0.13)
ax.grid(axis="x", alpha=0.06)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="center right", frameon=False, fontsize=9)

note = ("GLM final kernel — chunk-parallel gated Δ-rule (C=128) · bf16 gemms (fp32 accum) · "
        "hand-written Triton fwd+bwd trsm · max_autotune compile.  Peak 87.6× (255k tok/s).\n"
        "⚠ WON on speed but INVALID end-to-end: the intra-chunk decay-ratio kDn=k/gcumF underflows "
        "on the real model's decay (mean 0.78, →0) → 0.00 RULER recall. Bench used benign decay; "
        "now corrected.")
fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8, color=INK,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff6e5", ec=C_DQ, lw=1))

fig.tight_layout(rect=(0, 0.06, 1, 1))
out = f"{BASE}/ab_speedup_race.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print(f"GLM: {len(gi)} correct, peak {max(gs):.1f}x | qwen: {len(qi)} correct, peak {max(qs):.1f}x, {len(qdq)} DQ")
