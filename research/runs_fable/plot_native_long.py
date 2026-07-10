#!/usr/bin/env python3
"""Long-context RULER falloff: teacher (native GDN) vs KMD-2 native heal.
Teacher from ruler_teacher_long.json; native parsed from ruler_native_long.log
(run killed before writing json; 32768/4q,8q intentionally not measured — the
model has already collapsed to 0.00 at 32768/1q).
3 panels (1/4/8 queries), recall vs context length."""
import json, re, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/dev/gdn3_fable/research/runs_fable"
C_T, C_N = "#0072B2", "#009E73"
INK = "#1a1a1a"
NSAMP = 32

teacher = json.load(open(f"{BASE}/ruler_teacher_long.json"))

# parse native from the log
native = {}
pat = re.compile(r"^\s*(\d+)\s+16\s+(\d+)\s+([\d.]+)\s+([\d.]+)")
for line in open(f"{BASE}/ruler_native_long.log"):
    m = pat.match(line)
    if m:
        ctx, q, rec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        native[(ctx, q)] = rec


def tget(ctx, q):
    for r in teacher:
        if r["ctx"] == ctx and r["queries"] == q and r["needles"] == 16:
            return r["recall"]
    return None


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if p is not None else 0.0


ctxs = [4096, 8192, 16384, 32768]
xpos = np.arange(len(ctxs))
queries = [1, 4, 8]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for ax, q in zip(axes, queries):
    tv = [tget(c, q) for c in ctxs]
    nv = [native.get((c, q)) for c in ctxs]
    te = [ci95(p, NSAMP) for p in tv]
    ne = [ci95(p, NSAMP) for p in nv]
    ax.errorbar(xpos, tv, yerr=te, marker="o", color=C_T, lw=2, ms=6,
                capsize=3, label="teacher (native GDN)")
    nx = [x for x, v in zip(xpos, nv) if v is not None]
    nvv = [v for v in nv if v is not None]
    nee = [e for e, v in zip(ne, nv) if v is not None]
    ax.errorbar(nx, nvv, yerr=nee, marker="s", color=C_N, lw=2, ms=6,
                capsize=3, label="KMD-2 native heal (r_out=4)")
    for x, v in zip(xpos, tv):
        if v is not None:
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=C_T)
    for x, v in zip(xpos, nv):
        if v is not None:
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, -13), ha="center", fontsize=7, color=C_N)
    # mark unmeasured native 32768 multi-query cells
    if q in (4, 8) and native.get((32768, q)) is None:
        ax.annotate("collapsed\n(not measured)", (xpos[-1], 0.02),
                    ha="center", va="bottom", fontsize=7, color=C_N, style="italic")
    ax.axvline(np.interp(512, ctxs, xpos) if 512 >= ctxs[0] else -0.4,
               color="#999999", ls=":", lw=1)
    ax.set_title(f"{q} quer{'y' if q == 1 else 'ies'}", fontsize=11, color=INK)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{c//1024}k" for c in ctxs])
    ax.set_xlabel("context length", color=INK)
    ax.set_ylim(-0.03, 1.1)
    ax.grid(alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("RULER recall (16 needles)", color=INK)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Long-context RULER falloff — teacher vs KMD-2 native heal (trained @ seq_len 512, n=32)\n"
             "MIMO wins at 4k, ties ~8k, then the recurrent-state extrapolation cliff hits at 16k+",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
out = f"{BASE}/ruler_native_long_falloff.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print("\nctx      q   teacher  native")
for c in ctxs:
    for q in queries:
        t, n = tget(c, q), native.get((c, q))
        print(f"{c:>6} {q:>3}   {t if t is None else f'{t:.2f}':>6}   "
              f"{'n/a' if n is None else f'{n:.2f}':>6}")
