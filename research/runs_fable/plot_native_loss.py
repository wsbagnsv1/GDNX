#!/usr/bin/env python3
"""Log-log loss-vs-step for the native warm-start heal.
Parses runs/native_heal.log. Total loss prominent; kl/lw/ce as components.
Okabe-Ito palette."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/dev/gdn3_fable/runs/native_heal.log"
OUT = "/home/dev/gdn3_fable/research/runs_fable/native_heal_loss_loglog.png"
INK = "#1a1a1a"
C_LOSS, C_KL, C_CE, C_LW = "#1a1a1a", "#0072B2", "#E69F00", "#009E73"

pat = re.compile(r"step (\d+)/\d+ \| loss ([\d.]+) kl ([\d.]+) ce ([\d.]+) lw ([\d.]+)")
steps, loss, kl, ce, lw = [], [], [], [], []
for line in open(LOG):
    m = pat.search(line)
    if m:
        s, l, k, c, w = m.groups()
        steps.append(int(s)); loss.append(float(l)); kl.append(float(k))
        ce.append(float(c)); lw.append(float(w))

fig, ax = plt.subplots(figsize=(9, 5.4))
ax.loglog(steps, loss, "-o", color=C_LOSS, lw=2.4, ms=6, label="total loss", zorder=5)
ax.loglog(steps, kl, "-s", color=C_KL, lw=1.6, ms=4, label="KL (w=1.0)")
ax.loglog(steps, lw, "-^", color=C_LW, lw=1.6, ms=4, label="layerwise MSE (w=1.0)")
ax.loglog(steps, ce, "-d", color=C_CE, lw=1.6, ms=4, label="CE (w=0.02)")

for x, y in zip(steps, loss):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5, color=INK)

ax.set_xlabel("training step (log)", color=INK)
ax.set_ylabel("loss (log)", color=INK)
ax.set_title("KMD-2 native warm-start heal — loss vs step (log-log)\n"
             "warm start begins near-converged (loss 0.074); KL/layerwise flat-low, "
             "CE stays ~2.28", fontsize=10.5, color=INK)
ax.grid(which="both", alpha=0.15)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="center left")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("saved", OUT)
print("steps:", steps)
print("loss :", loss)
