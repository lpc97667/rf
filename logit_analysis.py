#!/usr/bin/env python

# Logit-level analysis of cross-GPU divergence: near-ties vs. numerical noise.

# Outputs:
# logit_hist.npz          log-spaced histograms of g and of e, per cell
# logit_gap_summary.csv   per (model, task) gap quantiles + exposure fractions
# logit_noise_summary.csv per (model, task, method, pair-kind) noise quantiles
# logit_divpoint.csv      gap at the first divergent token of each flip event

# python logit_analysis.py

import argparse
import collections
import csv
import glob
import itertools
import os
import re
import sys

import numpy as np
import torch

# Run dirs: timing-<gpu>_<task>_timing_<method>-<seed>
RUN_RE = re.compile(
    r"(a100|h100|l40s)_(.+?)(?:_timing)?_(rf|layercast(?:_bs\d+)?|bf16)-(\w+)$")
DEFAULT_ROOTS = ["outputs/vllm_layercast", "outputs/vllm_main"]
DEFAULT_SEEDS = ["1", "42", "777"]
GPU_ORDER = ["a100", "l40s", "h100"]

LO, HI, NBINS = -12.0, 3.0, 300
EDGES = np.linspace(LO, HI, NBINS + 1)
QUANTILES = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def hist(vals):
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.size
    nz = int((vals == 0).sum())
    pos = vals[vals > 0]
    lg = np.clip(np.log10(pos), LO + 1e-9, HI - 1e-9)
    counts, _ = np.histogram(lg, bins=EDGES)
    return n, nz, counts.astype(np.int64)


class Acc:
    CAP = 400_000

    def __init__(self):
        self.n = self.nz = 0
        self.counts = np.zeros(NBINS, dtype=np.int64)
        self.res = []
        self.res_n = 0

    def add(self, vals):
        n, nz, c = hist(vals)
        self.n += n
        self.nz += nz
        self.counts += c
        if self.res_n < self.CAP:
            take = min(len(vals), self.CAP - self.res_n)
            self.res.append(np.asarray(vals[:take], dtype=np.float32))
            self.res_n += take

    def sample(self):
        return np.concatenate(self.res) if self.res else np.zeros(0, np.float32)

    def quantiles(self, qs=QUANTILES):
        s = self.sample()
        if not len(s):
            return {f"q{q}": float("nan") for q in qs}
        return {f"q{q}": float(np.quantile(s, q)) for q in qs}

    def frac_below(self, thr):
        if not self.n:
            return float("nan")
        below = self.nz + int(self.counts[EDGES[1:] <= np.log10(thr)].sum())
        return below / self.n


def discover(roots, seeds):
    runs = collections.defaultdict(dict)
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "timing-*"))):
            base = os.path.basename(d)
            if "unmitigatedfp" in base:
                continue
            m = RUN_RE.search(base)
            if not m:
                continue
            gpu, task, method, seed = m.groups()
            if seed not in seeds:
                continue
            method = "layercast" if method.startswith("layercast") else method
            for org in sorted(os.listdir(d)):
                orgd = os.path.join(d, org)
                if not os.path.isdir(orgd):
                    continue
                for model in sorted(os.listdir(orgd)):
                    md = os.path.join(orgd, model)
                    if os.path.isdir(md) and glob.glob(
                            os.path.join(md, "problem_*logprobs*.pt")):
                        runs[(f"{org}/{model}", task, method)][(gpu, seed)] = md
    return runs


def problem_ids(rundir):
    ids = []
    for f in glob.glob(os.path.join(rundir, "problem_*_logprobs_*.pt")):
        ids.append(int(re.search(r"problem_(\d+)_", os.path.basename(f)).group(1)))
    return sorted(ids)


def _load(rundir, pid, kind):
    f = glob.glob(os.path.join(rundir, f"problem_{pid}_*{kind}*.pt"))
    if not f:
        return None
    try:
        return torch.load(f[0], map_location="cpu", weights_only=True)
    except Exception:
        return None


def load_run(rundir, pid):
    lp = _load(rundir, pid, "logprobs")
    ti = _load(rundir, pid, "token_ids")
    if lp is None or ti is None or lp.ndim != 2 or lp.shape[1] < 2:
        return None
    T = min(len(lp), len(ti))
    if T == 0:
        return None
    gap = (lp[:T, 0] - lp[:T, 1]).double().numpy()
    return gap, ti[:T, 0].numpy()


def agreeing_prefix(a, b):
    n = min(len(a), len(b))
    neq = np.flatnonzero(a[:n] != b[:n])
    return int(neq[0]) if len(neq) else n


def subset(ids, cap):
    if not cap or len(ids) <= cap:
        return ids
    step = len(ids) / cap
    return [ids[int(i * step)] for i in range(cap)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--gap-problems", type=int, default=250,
                    help="problems sampled per cell for the gap distribution")
    ap.add_argument("--noise-problems", type=int, default=80,
                    help="problems sampled per cell for the noise distribution")
    ap.add_argument("--firsts", default="divergence_firsts.csv")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    runs = discover(args.outputs, set(args.seeds))
    models = sorted({k[0] for k in runs})
    tasks = sorted({k[1] for k in runs})
    print(f"discovered {len(runs)} cells: {len(models)} models x {len(tasks)} tasks "
          f"x {len(sorted({k[2] for k in runs}))} methods", flush=True)
    if not runs:
        sys.exit(f"no runs found under {args.outputs}; produce token dumps "
                 f"with sweeps/run_sweep.py first (writing empty summaries "
                 f"would be worse than stopping here).")

    gap_acc = {}
    gap_meta = {}
    noise_acc = {}
    pair_acc = {}
    npz = {}

    for (model, task, method), by_gs in sorted(runs.items()):
        if method != "rf":
            continue
        ref = None
        for gpu in GPU_ORDER:
            for seed in args.seeds:
                if (gpu, seed) in by_gs:
                    ref = (gpu, seed)
                    break
            if ref:
                break
        if ref is None:
            continue
        rundir = by_gs[ref]
        acc = Acc()
        lens = []
        for pid in subset(problem_ids(rundir), args.gap_problems):
            r = load_run(rundir, pid)
            if r is None:
                continue
            gap, _ = r
            gap = gap[np.isfinite(gap)]
            acc.add(gap)
            lens.append(len(gap))
        gap_acc[(model, task)] = acc
        gap_meta[(model, task)] = dict(ref_gpu=ref[0], ref_seed=ref[1],
                                       n_problems=len(lens),
                                       mean_len=float(np.mean(lens)) if lens else float("nan"))
        print(f"  gaps  {model.split('/')[-1]:28s} {task:13s} "
              f"n={acc.n:8d} median={np.median(acc.sample()):.3f} "
              f"P(g<1e-4)={acc.frac_below(1e-4):.2e}", flush=True)


    for (model, task, method), by_gs in sorted(runs.items()):
        xg = Acc()
        xs = Acc()
        pairs = collections.defaultdict(Acc)

        for seed in args.seeds:
            present = [g for g in GPU_ORDER if (g, seed) in by_gs]
            if len(present) < 2:
                continue
            ids = subset(problem_ids(by_gs[(present[0], seed)]),
                         args.noise_problems)
            for pid in ids:
                loaded = {g: load_run(by_gs[(g, seed)], pid) for g in present}
                for ga, gb in itertools.combinations(present, 2):
                    ra, rb = loaded[ga], loaded[gb]
                    if ra is None or rb is None:
                        continue
                    n = agreeing_prefix(ra[1], rb[1])
                    if n == 0:
                        continue
                    e = np.abs(ra[0][:n] - rb[0][:n])
                    e = e[np.isfinite(e)]
                    xg.add(e)
                    pairs[f"{ga}|{gb}"].add(e)

        for gpu in GPU_ORDER:
            present = [s for s in args.seeds if (gpu, s) in by_gs]
            if len(present) < 2:
                continue
            ids = subset(problem_ids(by_gs[(gpu, present[0])]),
                         args.noise_problems)
            for pid in ids:
                loaded = {s: load_run(by_gs[(gpu, s)], pid) for s in present}
                for sa, sb in itertools.combinations(present, 2):
                    ra, rb = loaded[sa], loaded[sb]
                    if ra is None or rb is None:
                        continue
                    n = agreeing_prefix(ra[1], rb[1])
                    if n == 0:
                        continue
                    e = np.abs(ra[0][:n] - rb[0][:n])
                    xs.add(e[np.isfinite(e)])
                    
        noise_acc[(model, task, method, "xgpu")] = xg
        noise_acc[(model, task, method, "xseed")] = xs
        for p, a in pairs.items():
            pair_acc[(model, task, method, p)] = a
        print(f"  noise {model.split('/')[-1]:28s} {task:13s} {method:9s} "
              f"xgpu n={xg.n:8d} exact0={xg.nz / max(xg.n, 1):.4f} "
              f"med={np.median(xg.sample()) if xg.n else float('nan'):.3e} "
              f"| xseed n={xs.n:8d} exact0={xs.nz / max(xs.n, 1):.4f}", flush=True)

    divpts = collections.defaultdict(list)
    if os.path.exists(args.firsts):
        events = collections.defaultdict(set)
        for r in csv.DictReader(open(args.firsts)):
            events[(r["model"], r["task"], r["method"])].add(
                (r["gpu_a"], r["seed"], int(r["problem"]), int(r["first_div"])))
            
        for key, evs in sorted(events.items()):
            by_gs = runs.get(key, {})
            cache_pid, cache_gap = None, None
            for gpu, seed, pid, fd in sorted(evs):
                rundir = by_gs.get((gpu, seed))
                if rundir is None:
                    continue
                if cache_pid != (gpu, seed, pid):
                    r = load_run(rundir, pid)
                    cache_gap = r[0] if r else None
                    cache_pid = (gpu, seed, pid)
                if cache_gap is None or fd >= len(cache_gap):
                    continue
                g = float(cache_gap[fd])
                if np.isfinite(g):
                    divpts[key].append(g)
            if divpts[key]:
                a = np.array(divpts[key])
                print(f"  flips {key[0].split('/')[-1]:28s} {key[1]:13s} {key[2]:9s} "
                      f"n={len(a):6d} median_gap={np.median(a):.3e}", flush=True)

    p = args.prefix
    with open(f"{p}logit_gap_summary.csv", "w", newline="") as f:
        w = None
        for (model, task), acc in sorted(gap_acc.items()):
            row = dict(model=model, task=task, **gap_meta[(model, task)],
                       n_positions=acc.n, **acc.quantiles())
            for t in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                row[f"P_gap_below_{t:g}"] = acc.frac_below(t)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)

    with open(f"{p}logit_noise_summary.csv", "w", newline="") as f:
        w = None
        allacc = {(m, t, me, k): a for (m, t, me, k), a in noise_acc.items()}
        allacc.update(pair_acc)
        for (model, task, method, kind), acc in sorted(allacc.items()):
            row = dict(model=model, task=task, method=method, kind=kind,
                       n_positions=acc.n, n_exact_zero=acc.nz,
                       frac_exact_zero=acc.nz / acc.n if acc.n else float("nan"),
                       **acc.quantiles())
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)

    with open(f"{p}logit_divpoint.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "method", "n", "median_gap", "q90_gap",
                    "frac_below_1e-3", "frac_below_1e-1"])
        for key, vals in sorted(divpts.items()):
            a = np.array(vals)
            if not len(a):
                continue
            w.writerow([*key, len(a), float(np.median(a)),
                        float(np.quantile(a, 0.9)),
                        float((a < 1e-3).mean()), float((a < 1e-1).mean())])

    for (model, task), acc in gap_acc.items():
        npz[f"gap::{model}|{task}"] = acc.counts
        npz[f"gapmeta::{model}|{task}"] = np.array([acc.n, acc.nz])
    for (model, task, method, kind), acc in {**noise_acc, **pair_acc}.items():
        npz[f"noise::{model}|{task}|{method}|{kind}"] = acc.counts
        npz[f"noisemeta::{model}|{task}|{method}|{kind}"] = np.array([acc.n, acc.nz])
    for key, vals in divpts.items():
        if vals:
            npz["divpt::" + "|".join(key)] = np.array(vals, dtype=np.float32)
    npz["edges"] = EDGES
    np.savez_compressed(f"{p}logit_hist.npz", **npz)
    print(f"\nwrote {p}logit_gap_summary.csv, {p}logit_noise_summary.csv, "
          f"{p}logit_divpoint.csv, {p}logit_hist.npz")


if __name__ == "__main__":
    main()
