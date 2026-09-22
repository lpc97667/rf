#!/usr/bin/env python

# Logit-gap analysis: computes distribution of logit gaps and gap at divergence
# points

# Outputs logit_flip_prediction.csv and logit_gap_dist.npz

# python logit_gap_analysis.py [--outputs outputs/vllm_layercast]
#                              [--max-problems 300] [--firsts divergence_firsts.csv]

import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np
import torch

RUN_RE = re.compile(
    r"^(?:timing-)?(?:\d+)?(a100|h100|l40s|a6000|h200|b200)_([a-z0-9_]+?)(?:_timing)?_"
    r"(rf|layercast(?:_bs\d+)?|bf16)(?:-(\w+))?$")

DEFAULT_ROOTS = ["outputs/vllm_layercast", "outputs/vllm_main"]
THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
QUANTILES = [0.0, 0.001, 0.01, 0.05, 0.5]


def discover_runs(roots):
    if isinstance(roots, str):
        roots = [roots]
    runs = collections.defaultdict(dict)
    dirs = [d for root in roots for d in sorted(glob.glob(os.path.join(root, "*")))]
    for d in dirs:
        m = RUN_RE.match(os.path.basename(d))
        if not m:
            continue
        gpu, task, method, seed = m.groups()
        method = "layercast" if method.startswith("layercast") else method
        seed = seed or "noseed"
        for org in sorted(os.listdir(d)):
            orgd = os.path.join(d, org)
            if not os.path.isdir(orgd):
                continue
            for model in sorted(os.listdir(orgd)):
                md = os.path.join(orgd, model)
                if os.path.isdir(md):
                    runs[(f"{org}/{model}", task, method)][(gpu, seed)] = md
                    
    return runs


def logprob_path(rundir, pid):
    hits = glob.glob(os.path.join(rundir, f"problem_{pid}_*logprobs*.pt"))
    return hits[0] if hits else None


def load_gaps(path):
    lp = torch.load(path, map_location="cpu", weights_only=True)
    if lp.ndim != 2 or lp.shape[1] < 2:
        return None
    gap = (lp[:, 0] - lp[:, 1]).to(torch.float32).numpy()
    return gap[np.isfinite(gap)]


def pooled_gaps(rundir, max_problems):
    files = sorted(glob.glob(os.path.join(rundir, "problem_*logprobs*.pt")))
    if max_problems and len(files) > max_problems:
        # deterministic even stride
        step = len(files) / max_problems
        files = [files[int(i * step)] for i in range(max_problems)]
        
    chunks = []
    for f in files:
        try:
            lp = torch.load(f, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if lp.ndim == 2 and lp.shape[1] >= 2:
            g = (lp[:, 0] - lp[:, 1]).to(torch.float32).numpy()
            chunks.append(g[np.isfinite(g)])
            
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def divergence_point_gaps(firsts_csv, runs):
    if not os.path.exists(firsts_csv):
        print(f"  (no {firsts_csv}; skipping divergence-point gaps)")
        return {}
    rows = list(csv.DictReader(open(firsts_csv)))
    events = collections.defaultdict(set) # distinct flip events
    for r in rows:
        key = (r["model"], r["task"], r["method"])
        events[key].add((r["gpu_a"], r.get("seed", "noseed"),
                          int(r["problem"]), int(r["first_div"])))
        
    out = {}
    for key, evs in events.items():
        by_gs = runs.get(key, {})
        cache_key, cache_lp = None, None
        gaps = []
        for gpu, seed, pid, fd in sorted(evs):
            rundir = by_gs.get((gpu, seed))
            if rundir is None:
                continue
            if cache_key != (gpu, seed, pid):
                path = logprob_path(rundir, pid)
                cache_lp = None
                if path is not None:
                    try:
                        cache_lp = torch.load(path, map_location="cpu",
                                              weights_only=True)
                    except Exception:
                        cache_lp = None
                cache_key = (gpu, seed, pid)
            if cache_lp is None or cache_lp.ndim != 2 or fd >= cache_lp.shape[0]:
                continue
            g = float(cache_lp[fd, 0] - cache_lp[fd, 1])
            if np.isfinite(g):
                gaps.append(g)
                
        if gaps:
            out[key] = np.array(gaps, dtype=np.float32)
            
    return out


def frac_below(arr, thr):
    return float((arr < thr).mean()) if len(arr) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--firsts", default="divergence_firsts.csv")
    ap.add_argument("--max-problems", type=int, default=300,
                    help="problems sampled per run for the pooled distribution")
    ap.add_argument("--summary-csv", default="logit_flip_prediction.csv")
    ap.add_argument("--dist-npz", default="logit_gap_dist.npz")
    args = ap.parse_args()

    runs = discover_runs(args.outputs)
    print(f"discovered {len(runs)} (model, task, method) groups")

    print("computing divergence-point gaps ...")
    dp_gaps = divergence_point_gaps(args.firsts, runs)

    rows = []
    npz = {}
    for key in sorted(runs):
        model, task, method = key
        by_gs = runs[key]
        gaps = np.array([], dtype=np.float32)
        gpu, seed = sorted(by_gs)[0]
        for cand in sorted(by_gs):
            g = pooled_gaps(by_gs[cand], args.max_problems)
            if len(g):
                gpu, seed, gaps = cand[0], cand[1], g
                break
            
        dpg = dp_gaps.get(key, np.array([], dtype=np.float32))
        tag = f"{model}|{task}|{method}"
        npz[f"pooled::{tag}"] = gaps.astype(np.float16)
        npz[f"divpt::{tag}"] = dpg.astype(np.float32)

        row = dict(model=model, task=task, method=method,
                   rep_gpu=gpu, rep_seed=seed, n_positions=len(gaps),
                   n_divpts=len(dpg))
        for q in QUANTILES:
            row[f"q{q}"] = float(np.quantile(gaps, q)) if len(gaps) else float("nan")
        for t in THRESHOLDS:
            row[f"frac_below_{t:g}"] = frac_below(gaps, t)
        row["divpt_median"] = float(np.median(dpg)) if len(dpg) else float("nan")
        row["divpt_q90"] = float(np.quantile(dpg, 0.9)) if len(dpg) else float("nan")
        rows.append(row)
        
        print(f"  {model.split('/')[-1]:22s} {task:8s} {method:9s} "
              f"npos={len(gaps):7d} ndiv={len(dpg):5d} "
              f"median_gap={row['q0.5']:.3f} "
              f"frac<1e-3={row['frac_below_0.001']:.2e} "
              f"divpt_median={row['divpt_median']:.2e}", flush=True)

    if not rows:
        sys.exit(f"\nno runs found under {args.outputs}; produce token dumps "
                 f"with sweeps/run_sweep.py first.")

    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(args.dist_npz, **npz)
    print(f"\nwrote {args.summary_csv} ({len(rows)} rows) and {args.dist_npz}")


if __name__ == "__main__":
    main()
