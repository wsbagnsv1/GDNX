#!/usr/bin/env python3
"""RULER n=16: teacher (native GDN) vs KMD-2 native warm-start heal (r_out=4).
Cells: {256,512,1024,2048} tok x {1,4} queries. Okabe-Ito, 95% CI bars."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_T, C_N = "#0072B2", "#009E73"
INK = "#1a1a1a"
BASE = "/home/dev/gdn3_fable/research/runs_fable"

teacher = json.load(open(f"{BASE}/ruler_teacher.json"))
native = json.load(open(f"{BASE}/ruler_native.json"))
cells = [(256, 1), (256, 4), (512, 1), (512, 4),
         (1024, 1), (1024, 4), (2048, 1), (2048, 4)]


def get(rows, ctx, q):
    for r in rows:
        if r["ctx"] == ctx and r["queries"] == q and r["needles"] == 16:
            return r["recall"], r["n"]
    return None, 0


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if (p is not None and n) else 0.0


series = [("teacher (native GDN)", C_T, teacher),
          ("KMD-2 native heal (r_out=4)", C_N, native)]
x = np.arange(len(cells))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.4))
for i, (lab, col, rows) in enumerate(series):
    vals = [get(rows, c, q) for c, q in cells]
    ps = [p if p is not None else 0 for p, _ in vals]
    errs = [ci95(p, n) for p, n in vals]
    off = (i - 0.5) * w
    ax.bar(x + off, ps, w, yerr=errs, capsize=3, label=lab, color=col, zorder=3)
    for xi, (p, _), e in zip(x, vals, errs):
        if p is not None:
            ax.text(xi + off, p + e + 0.012, f"{p:.2f}", ha="center",
                    fontsize=7.5, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([f"{c}/{q}q" for c, q in cells])
ax.set_xlabel("context length / #queries", color=INK)
ax.set_ylabel("RULER recall (teacher-forced exact value)", color=INK)
ax.set_ylim(0, 1.15)
ax.set_title("Multi-key RULER, 16 needles — teacher vs KMD-2 native warm-start heal\n"
             "single-query matches teacher (1.00); multi-query MIMO (r_out=4) beats teacher "
             "at every context\n95% CI bars (binomial over queried values), n=8 samples/cell",
             fontsize=10.5, color=INK)
ax.grid(axis="y", alpha=0.15)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=9)
fig.tight_layout()
out = f"{BASE}/ruler_native_vs_teacher.png"
fig.savefig(out, dpi=130)
print("saved", out)

tvals = [get(teacher, c, q)[0] for c, q in cells]
nvals = [get(native, c, q)[0] for c, q in cells]
rel = [n / t for n, t in zip(nvals, tvals) if t]
print(f"mean recall/teacher (native heal): {sum(rel)/len(rel):.3f}")
for (c, q), t, n in zip(cells, tvals, nvals):
    print(f"  {c:>5}/{q}q  teacher {t:.2f}  native {n:.2f}")
