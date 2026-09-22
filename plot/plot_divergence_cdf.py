#!/usr/bin/env python

# Paper figure: cumulative cross-GPU divergence vs. token position.

# python plot/plot_divergence_cdf.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "divergence-cdf")

SYSNAME = "RF"
C_OURS, C_BASE = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
MODELS = [("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B"),
          ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B")]
TASK_STYLE = {"gsm8k": "-", "math500": (0, (4, 2))}
TASK_LABEL = {"gsm8k": "GSM8K", "math500": "MATH500"}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
})
paper_style.use_times()

firsts = pd.read_csv(os.path.join(ROOT, "divergence_firsts.csv"))
summ = pd.read_csv(os.path.join(ROOT, "divergence_summary.csv"))
denom = {(r.model, r.task, r.method): r.xgpu_prob_pairs
         for r in summ.itertuples()}

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9))

for ax, (model, mlabel) in zip(axes, MODELS):
    for method, color in (("layercast", C_BASE), ("rf", C_OURS)):
        for task in ("gsm8k", "math500"):
            n = denom[(model, task, method)]
            f = firsts.query("model == @model and task == @task and "
                             "method == @method")["first_div"].sort_values()
            x = np.concatenate([[1], f.values, [4096]])
            y = np.concatenate([[0], np.arange(1, len(f) + 1), [len(f)]]) / n * 100
            ax.step(x, y, where="post", color=color, lw=1.4,
                    ls=TASK_STYLE[task], solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlim(1, 4096)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_title(mlabel, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylim(0, 30)
axes[0].set_ylabel("diverged pairs (%)")
axes[1].set_ylim(0, 0.2)
axes[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2])

axes[0].text(420, 27.6, "LayerCast MATH500", fontsize=6.5, color=C_BASE, ha="center")
axes[0].text(1.35, 10.5, "LayerCast\nGSM8K", fontsize=6.5, color=C_BASE)
axes[0].text(2.2, 2.2, f"{SYSNAME}: 0 (both tasks)", fontsize=6.5,
             color=C_OURS)
axes[1].text(11, 0.168, "LayerCast GSM8K", fontsize=6.5, color=C_BASE)
axes[1].text(500, 0.062, f"{SYSNAME}\nGSM8K", fontsize=6.5, color=C_OURS)
axes[1].text(2.2, 0.012, "MATH500: 0 (both)", fontsize=6.5, color=MUTED)

fig.supxlabel("token position $t$ (log)", fontsize=8, y=0.04)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=200)
print("wrote", OUT + ".{pdf,png}")
