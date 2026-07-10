#!/usr/bin/env python3
"""Full-range RULER falloff 512..32768: teacher (native GDN) vs KMD-2 native heal.
Teacher from ruler_teacher_{short,long}.json; native parsed from
ruler_native_{short,long}.log (long run killed before json). n=32, 16 needles.
32768/4q,8q for native intentionally not measured (already 0.00 @ 32768/1q)."""
import json, re, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/dev/gdn3_fable/research/runs_fable"
C_T, C_N = "#0072B2", "#009E73"
INK = "#1a1a1a"
NSAMP = 32

teacher = {}
for f in ("ruler_teacher_short.json", "ruler_teacher_long.json"):
    for r in json.load(open(f"{BASE}/{f}")):
        teacher[(r["ctx"], r["queries"])] = r["recall"]

native = {}
pat = re.compile(r"^\s*(\d+)\s+16\s+(\d+)\s+([\d.]+)\s+([\d.]+)")
for f in ("ruler_native_short.log", "ruler_native_long.log"):
    for line in open(f"{BASE}/{f}"):
        m = pat.match(line)
        if m:
            native[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))


def ci95(p):
    return 1.96 * math.sqrt(p * (1 - p) / NSAMP) if p is not None else 0.0


ctxs = [512, 1024, 2048, 4096, 8192, 16384, 32768]
xpos = np.arange(len(ctxs))
queries = [1, 4, 8]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
for ax, q in zip(axes, queries):
    tv = [teacher.get((c, q)) for c in ctxs]
    nv = [native.get((c, q)) for c in ctxs]
    ax.errorbar(xpos, tv, yerr=[ci95(p) for p in tv], marker="o", color=C_T,
                lw=2, ms=6, capsize=3, label="teacher (native GDN)")
    nx = [x for x, v in zip(xpos, nv) if v is not None]
    ax.errorbar(nx, [v for v in nv if v is not None],
                yerr=[ci95(v) for v in nv if v is not None], marker="s",
                color=C_N, lw=2, ms=6, capsize=3, label="KMD-2 native heal (r_out=4)")
    for x, v in zip(xpos, tv):
        if v is not None:
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6.5, color=C_T)
    for x, v in zip(xpos, nv):
        if v is not None:
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=6.5, color=C_N)
    if q in (4, 8):
        ax.annotate("not measured\n(collapsed)", (xpos[-1], 0.05), ha="center",
                    va="bottom", fontsize=6.5, color=C_N, style="italic")
    # training window seq_len=512 = first x tick
    ax.axvline(0, color="#cc3311", ls=":", lw=1.2)
    ax.annotate("train len 512", (0, 1.06), fontsize=7, color="#cc3311",
                ha="left", rotation=0)
    ax.set_title(f"{q} quer{'y' if q == 1 else 'ies'}", fontsize=11, color=INK)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in ctxs],
                       fontsize=8)
    ax.set_xlabel("context length", color=INK)
    ax.set_ylim(-0.03, 1.12)
    ax.grid(alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("RULER recall (16 needles, n=32)", color=INK)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("RULER recall vs context — KMD-2 native heal vs native-GDN teacher (n=32, 16 needles)\n"
             "heal WINS every multi-query cell 512–4k (peak +0.20 @512/4q); crossover ~8k; "
             "seq_len-512 extrapolation cliff at 16k+",
             fontsize=11.5, color=INK, y=1.03)
fig.tight_layout()
out = f"{BASE}/ruler_native_full_falloff.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

print("\nctx      q   teacher  native   delta")
for c in ctxs:
    for q in queries:
        t, n = teacher.get((c, q)), native.get((c, q))
        d = f"{n - t:+.2f}" if (t is not None and n is not None) else "  - "
        print(f"{c:>6} {q:>3}   {'—' if t is None else f'{t:.2f}':>6}   "
              f"{'n/a' if n is None else f'{n:.2f}':>6}   {d:>6}")
