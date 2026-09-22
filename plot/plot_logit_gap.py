#!/usr/bin/env python
# Paper figure: greedy decision margins vs. cross-architecture logit noise.
# Input: logit_hist.npz from logit_analysis.py
# python plot/plot_logit_gap.py

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSNAME = "RF"

MODELS = [("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "#009E73"),
          ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B", "#CC79A7"),
          ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-8B", "#56B4E9")]
METHODS = [("bf16", "Unmitigated", "#8a8a8a"),
           ("layercast", "LayerCast", "#D55E00"),
           ("rf", SYSNAME, "#0072B2")]
PAIRS = [("A100 vs. L40S", ["a100|l40s"], "-"),
         ("Either vs. H100", ["a100|h100", "l40s|h100"], (0, (2.6, 1.3)))]

XLO, XHI = 2e-8, 3e2
XTICKLO = 1e-7
YLO = 3e-7          
FS_TICK, FS_LAB, FS_LEG, FS_ANN = 7.2, 8.0, 8.0, 6.6


def cdf_from_hist(npz, prefix, keys):
    edges = npz["edges"]
    counts = np.zeros(len(edges) - 1)
    n = nz = 0
    for k in keys:
        counts = counts + npz[f"{prefix}::{k}"]
        m = npz[f"{prefix}meta::{k}"]
        n += int(m[0])
        nz += int(m[1])
    if n == 0:
        return None
    cum = (nz + np.cumsum(counts)) / n
    return 10.0 ** edges[1:], cum, n, nz / n


def quant_from_hist(x, cum, q):
    if cum[-1] < q:
        return np.nan
    i = int(np.searchsorted(cum, q))
    return float(x[min(i, len(x) - 1)])


def quant_nonzero(x, cum, fz, q):
    if fz >= 1.0:
        return np.nan
    return quant_from_hist(x, (cum - fz) / (1.0 - fz), q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-models", nargs="*", default=None,
                    help="restrict panel (b) to models matching these substrings")
    ap.add_argument("--out", default="logit-gap")
    args = ap.parse_args()

    paper_style.use_times()
    npz = np.load(os.path.join(ROOT, "logit_hist.npz"), allow_pickle=False)
    keys = list(npz.keys())
    tasks = sorted({k.split("::")[1].split("|")[1]
                    for k in keys if k.startswith("gap::")})

    def noise_keys(method, pairs):
        out = []
        for k in keys:
            if not k.startswith("noise::"):
                continue
            tag = k.split("::")[1]
            model, task, meth, kind = tag.split("|", 3)
            if meth != method or kind not in pairs:
                continue
            if args.noise_models and not any(s in model for s in args.noise_models):
                continue
            out.append(tag)
        return out

    fig = plt.figure(figsize=(3.35, 3.30))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0], hspace=0.10,
                          left=0.205, right=0.985, top=0.985, bottom=0.125)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    rows, guides = [], {}
    for mkey, mlabel, color in METHODS:
        for plabel, pkinds, style in PAIRS:
            tags = noise_keys(mkey, pkinds)
            r = cdf_from_hist(npz, "noise", tags) if tags else None
            rows.append((mkey, mlabel, plabel, style, color, r))
        allr = cdf_from_hist(npz, "noise",
                             noise_keys(mkey, [p for _, ps, _ in PAIRS for p in ps]))
        if allr:
            guides[mkey] = (quant_from_hist(allr[0], allr[1], 0.5), color)

    for mkey, mlabel, color in MODELS:
        r = cdf_from_hist(npz, "gap", [f"{mkey}|{t}" for t in tasks
                                       if f"gap::{mkey}|{t}" in keys])
        if r is None:
            continue
        x, cum, n, _ = r
        ax0.plot(x, np.maximum(cum, YLO / 10), lw=1.35, color=color,
                 label=mlabel, solid_joinstyle="round")

    for mkey, (med, color) in guides.items():
        if med and np.isfinite(med) and med > XLO:
            ax0.axvline(med, color=color, lw=0.8, ls=(0, (1.4, 1.6)), alpha=0.9,
                        zorder=0)

    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_xlim(XLO, XHI)
    ax0.set_ylim(YLO, 1.9)
    ax0.set_ylabel("fraction of positions\nwith margin $\\leq x$",
                   fontsize=FS_LAB, linespacing=1.15)
    ax0.grid(True, which="major", color="#dcdcdc", lw=0.5)
    ax0.set_axisbelow(True)
    ax0.tick_params(labelsize=FS_TICK, length=2.5, pad=1.6, labelbottom=False)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.legend(fontsize=FS_LEG, frameon=False, loc="upper left",
               handlelength=1.4, borderpad=0.1, labelspacing=0.22,
               borderaxespad=0.2)
    ax0.text(0.985, 0.055, "(a)", transform=ax0.transAxes,
             ha="right", va="bottom", fontsize=FS_LAB)

    for mkey, mlabel, plabel, style, color, r in rows:
        if r is None:
            continue
        x, cum, n, fz = r
        ax1.plot(np.concatenate([[XLO], x]), np.concatenate([[fz], cum]),
                 lw=1.3, ls=style, color=color, solid_joinstyle="round")
        ax1.plot([XLO * 1.5], [fz], marker="o", ms=2.9, color=color,
                 mec="black", mew=0.4, zorder=5)

    handles = [plt.Line2D([], [], color=c, lw=1.3, label=l)
               for _, l, c in METHODS]
    handles += [plt.Line2D([], [], color="#444444", lw=1.1, ls=s, label=l)
                for l, _, s in PAIRS]
    ax1.legend(handles=handles, fontsize=FS_LEG - 0.6, frameon=False,
               loc="upper left", handlelength=1.7, borderpad=0.1,
               labelspacing=0.2, borderaxespad=0.25, ncol=2, columnspacing=0.9,
               handletextpad=0.5)

    ax1.set_ylim(-0.04, 1.42)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("fraction of positions\nwith $|\\Delta| \\leq x$",
                   fontsize=FS_LAB, linespacing=1.15)
    ax1.set_xlabel("$x$: decision margin $g$, and its cross-GPU shift "
                   "$|\\Delta|$ (nats)", fontsize=FS_LAB, labelpad=1.5)
    ax1.grid(True, axis="both", which="major", color="#dcdcdc", lw=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1])
    ax1.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.6)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.985, 0.045, "(b)", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=FS_LAB)
    ax1.annotate("Bitwise\nidentical", xy=(XLO * 1.95, 0.945), xytext=(1.5e-7, 0.60),
                 fontsize=FS_ANN, ha="left", va="center", color="#333333",
                 linespacing=1.05,
                 arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.36",
                                 lw=0.55, color="#333333", shrinkA=3, shrinkB=3))

    out = os.path.join(ROOT, args.out)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=400)
    print("wrote", out + ".{pdf,png}")

    print("\nmedian / 99th-pct cross-GPU gap perturbation (nats), bitwise-equal fraction")
    for mkey, mlabel, plabel, style, color, r in rows:
        if r is None:
            continue
        x, cum, n, fz = r
        print(f"  {mlabel:12s} {plabel:16s} n={n:9d} "
              f"med={quant_from_hist(x, cum, 0.5):.3g} "
              f"p99={quant_from_hist(x, cum, 0.99):.3g} bitwise={100 * fz:.2f}%")
    print("\ngap CDF, pooled over benchmarks")
    for mkey, mlabel, color in MODELS:
        r = cdf_from_hist(npz, "gap", [f"{mkey}|{t}" for t in tasks
                                       if f"gap::{mkey}|{t}" in keys])
        if r is None:
            continue
        x, cum, n, _ = r
        row = "  ".join(f"P(g<{t:g})={cum[np.searchsorted(x, t)]:.2e}"
                        for t in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
        print(f"  {mlabel:15s} n={n:9d}  {row}")


if __name__ == "__main__":
    main()
