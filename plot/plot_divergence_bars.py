#!/usr/bin/env python

# Paper figure: cross-GPU token divergence per method, all models x benchmarks.
# Reads divergence_summary.csv from divergence_analysis.py
# python plot/plot_divergence_bars.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "divergence-bars")
SYSNAME = "RF"

INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
METHODS = [("bf16", "BF16 (unmitigated)", "#9a9a9a"),
           ("layercast", "LayerCast", "#D55E00"),
           ("rf", SYSNAME, "#0072B2")]
MODEL_SHORT = {"meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
               "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
               "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-8B"}
MODEL_ORDER = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
               "meta-llama/Llama-3.2-3B-Instruct",
               "Qwen/Qwen3-4B-Instruct-2507"]
TASK_ORDER = ["gsm8k", "math500", "aime24", "gpqa_diamond"]
TASK_LABEL = {"gsm8k": "GSM8K", "math500": "MATH500",
              "aime24": "AIME24", "gpqa_diamond": "GPQA-D"}


def ordered(present, preferred):
    return [v for v in preferred if v in present] + \
           sorted(v for v in present if v not in preferred)


def main():
    paper_style.use_times()
    plt.rcParams.update({"pdf.fonttype": 42})
    df = pd.read_csv(os.path.join(ROOT, "divergence_summary.csv"))
    df["pct"] = df["xgpu_problem_rate"] * 100.0

    models = ordered(df["model"].unique(), MODEL_ORDER)
    tasks = ordered(df["task"].unique(), TASK_ORDER)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 2.5), squeeze=False,
                             sharey=True)
    axes = axes[0]
    x = np.arange(len(tasks))
    bw = 0.8 / len(METHODS)

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].set_index(["task"])
        for i, (key, label, color) in enumerate(METHODS):
            vals = []
            for t in tasks:
                v = np.nan
                if t in sub.index:
                    row = sub.loc[t]
                    row = row[row["method"] == key]
                    if len(row):
                        v = float(row["pct"].iloc[0])
                vals.append(v)
            offs = (i - (len(METHODS) - 1) / 2) * bw
            bars = ax.bar(x + offs, vals, bw, label=label, color=color,
                          edgecolor="black", linewidth=0.4)
            labels = ["" if np.isnan(v) else (f"{v:.0f}" if v >= 1
                      else ("0" if v == 0 else f"{v:.1f}")) for v in vals]
            ax.bar_label(bars, labels=labels, padding=1.5, fontsize=5.6,
                         rotation=90)
        ax.set_title(MODEL_SHORT.get(model, model.split("/")[-1]), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=7.5,
                           rotation=20, ha="right")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 108)

    axes[0].set_ylabel("cross-GPU divergence\n(\\% of problems)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.92))
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200)
    print("wrote", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
