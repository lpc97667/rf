#!/usr/bin/env python

# Paper figure: E2E speedup roofline
# Speedups are read from <gpu>-timing-comparison.csv written by timing_to_csv.py

# python plot/plot_speedup_roofline.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import paper_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "speedup-roofline")

SYSNAME = "RF"
ACCENT = "#0072B2"   
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"

GPUS = {
    "a100": ("A100 PCIe", 19.5, 1.555),
    "h100": ("H100 PCIe", 51.2, 2.000),
    "l40s": ("L40S", 91.6, 0.864),
}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
})
paper_style.use_times()

fig, ax = plt.subplots(figsize=(3.3, 2.1))

for key, (label, tf, bw) in GPUS.items():
    df = pd.read_csv(os.path.join(ROOT, f"{key}-timing-comparison.csv"))
    sp = df["speedup"]
    ratio = tf * 1e12 / (bw * 1e12)
    ax.scatter([ratio] * len(sp), sp, s=14, color=ACCENT, alpha=0.35, lw=0,
               zorder=3)
    ax.scatter([ratio], [sp.mean()], s=34, color=ACCENT, zorder=4)
    ax.annotate(label, (ratio, sp.mean()),
                xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK)

ax.axhline(2.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(10.5, 2.03, "2$\\times$ GEMM weight-traffic bound", fontsize=7,
        color=MUTED, va="bottom")
ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
ax.text(10.5, 1.02, "LayerCast parity", fontsize=7, color=MUTED, va="bottom")

ax.set_xscale("log")
ax.set_xticks([12.5, 25.6, 106])
ax.set_xticklabels(["12.5", "25.6", "106"])
ax.minorticks_off()
ax.set_xlim(9, 160)
ax.set_ylim(0.9, 3.35)
ax.set_xlabel("FP32 compute / memory bandwidth (FLOP/byte)")
ax.set_ylabel("E2E speedup vs. LayerCast")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=200)
print("wrote", OUT + ".{pdf,png}")
