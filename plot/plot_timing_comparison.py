#!/usr/bin/env python

# Paper figure:  RF, LayerCast, and unmitigated end-to-end times.
# Auto-discovers every <gpu>-timing-comparison.csv in the repo root.

# python plot/plot_timing_comparison.py


import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "timing-comparison.png")
OUT_PDF = os.path.join(ROOT, "timing-comparison.pdf")


RF_LABEL = "RF"

METHODS = [
    ("rf", RF_LABEL, "#0072B2"),
    ("layercast", "LayerCast", "#D55E00"),    
    ("bf16", "Unmitigated", "#9a9a9a"),       
]
GPU_ORDER = ["a100", "l40s", "h100"]
GPU_LABEL = {"a100": "A100", "l40s": "L40S", "h100": "H100"}
TASK_LABEL = {"gsm8k": "GSM8K", "math500": "MATH500",
              "aime24": "AIME24", "gpqa_diamond": "GPQA-Diamond"}
MODEL_ORDER = ["Llama-3.2-3B-Instruct", "Qwen3-4B-Instruct-2507",
               "DeepSeek-R1-Distill-Llama-8B"]
MODEL_LABEL = {"Llama-3.2-3B-Instruct": "Llama-3.2-3B",
               "Qwen3-4B-Instruct-2507": "Qwen3-4B",
               "DeepSeek-R1-Distill-Llama-8B": "DS-R1-Llama-8B"}

Y_SPLIT = {"gsm8k": 1}
FIG_W = 6.95        
GROUP_W = 0.58      
GROUPS_PER_ROW = 2  
PANEL_H = 0.62      
ROW_GAP = 0.47      
PAD_TOP = 0.49      
PAD_BOTTOM = 0.16   
M_LEFT, M_RIGHT = 0.057, 0.955
WSPACE_GROUP = 0.20
WSPACE_MODEL = 0.07
FS_TICK = 7.5
FS_TITLE = 8
FS_GROUP = 9
FS_LABEL = 8.5
FS_LEGEND = 8.5


def short_model(name):
    return name.split("/")[-1]


def load_data():
    csvs = sorted(glob.glob(os.path.join(ROOT, "*-timing-comparison.csv")))
    if not csvs:
        raise SystemExit("No *-timing-comparison.csv files found in repo root.")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csvs)} CSV(s): "
          f"{', '.join(os.path.basename(c) for c in csvs)}")
    return df


def ordered(values, preferred):
    present = [v for v in preferred if v in values]
    extras = sorted(v for v in values if v not in preferred)
    return present + extras


def place_group_headers(fig, groups):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for task, group in groups:
        left = min(ax.get_position().x0 for ax in group)
        right = max(ax.get_position().x1 for ax in group)
        tops = [inv.transform(ax.title.get_window_extent(renderer))[1][1]
                for ax in group]
        fig.text((left + right) / 2, max(tops) + 0.010,
                 TASK_LABEL.get(task, task), ha="center", va="bottom",
                 fontsize=FS_GROUP, fontweight="bold")


def main():
    paper_style.use_times()
    df = load_data()
    df["model_short"] = df["model"].map(short_model)

    models = ordered(df["model_short"].unique(), MODEL_ORDER)
    tasks = ordered(df["task"].unique(), list(TASK_LABEL))
    gpus = ordered(df["gpu"].unique(), GPU_ORDER)

    nrows = -(-len(tasks) // GROUPS_PER_ROW)
    fig_h = (nrows * PANEL_H + (nrows - 1) * ROW_GAP + PAD_TOP + PAD_BOTTOM)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    outer = fig.add_gridspec(
        nrows, GROUPS_PER_ROW,
        wspace=WSPACE_GROUP, hspace=ROW_GAP / PANEL_H,
        left=M_LEFT, right=M_RIGHT,
        top=1 - PAD_TOP / fig_h, bottom=PAD_BOTTOM / fig_h,
    )

    x = np.arange(len(gpus))
    bar_w = GROUP_W / len(METHODS)

    def col(sub, g, name):
        if g not in sub.index or name not in sub.columns:
            return np.nan
        return pd.to_numeric(sub.loc[g, name], errors="coerce")

    groups = [] 

    for t, task in enumerate(tasks):
        r, c = t // GROUPS_PER_ROW, t % GROUPS_PER_ROW
        inner = outer[r, c].subgridspec(1, len(models), wspace=WSPACE_MODEL)

        split = min(Y_SPLIT.get(task, len(models) - 1), len(models))
        parts = [p for p in (list(range(split)),
                             list(range(split, len(models)))) if p]

        group_axes = [None] * len(models)
        for part in parts:
            for j, m in enumerate(part):
                group_axes[m] = fig.add_subplot(
                    inner[0, m],
                    sharey=group_axes[part[0]] if j else None)
        groups.append((task, group_axes))
        tops = [0.0] * len(models)

        for mi, (ax, model) in enumerate(zip(group_axes, models)):
            sub = df[(df["model_short"] == model) & (df["task"] == task)]
            sub = sub.set_index("gpu")

            for i, (key, label, color) in enumerate(METHODS):
                means = [col(sub, g, f"{key}_mean_s") for g in gpus]
                errs = []
                for g in gpus:
                    n = col(sub, g, f"{key}_n")
                    s = col(sub, g, f"{key}_std_s")
                    errs.append(s / np.sqrt(n) if (n and n > 0) else np.nan)
                offset = (i - (len(METHODS) - 1) / 2) * bar_w
                ax.bar(
                    x + offset, means, bar_w, yerr=errs, label=label,
                    color=color, edgecolor="black", linewidth=0.4,
                    capsize=1.8, error_kw=dict(elinewidth=0.6, capthick=0.6),
                )
                bar_tops = [v + (e if np.isfinite(e) else 0.0)
                            for v, e in zip(means, errs) if np.isfinite(v)]
                tops[mi] = max([tops[mi]] + bar_tops)

            ax.set_xticks(x)
            ax.set_xticklabels([GPU_LABEL.get(g, g.upper()) for g in gpus],
                               fontsize=FS_TICK)
            ax.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.5)
            ax.set_title(MODEL_LABEL.get(model, model), fontsize=FS_TITLE)
            ax.grid(axis="y", linestyle=":", color="0.45", alpha=0.85,
                    linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.5, len(gpus) - 0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        for pi, part in enumerate(parts):
            lead = group_axes[part[0]]
            lead.set_ylim(0, max(tops[m] for m in part))
            lead.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            right_side = pi == len(parts) - 1 and len(parts) > 1
            labelled = group_axes[part[-1] if right_side else part[0]]
            for m in part:
                ax = group_axes[m]
                if ax is labelled and right_side:
                    ax.yaxis.set_ticks_position("right")
                    ax.spines["right"].set_visible(True)
                    ax.tick_params(axis="y", left=False, right=True,
                                   labelleft=False, labelright=True,
                                   labelsize=FS_TICK, length=2.5, pad=1.5)
                elif ax is not labelled:
                    ax.tick_params(axis="y", left=False, labelleft=False)
                if m - 1 in part:
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", left=False)

    handles, labels = groups[0][1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               frameon=False, fontsize=FS_LEGEND, handlelength=1.5,
               handleheight=0.8, columnspacing=1.8, borderpad=0.1,
               bbox_to_anchor=(0.5, 1.01))
    fig.supylabel("Mean E2E time (s)", fontsize=FS_LABEL, x=0.004)

    place_group_headers(fig, groups)

    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
