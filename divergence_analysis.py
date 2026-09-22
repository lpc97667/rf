#!/usr/bin/env python

# Token-level cross-GPU / cross-seed divergence analysis: RF vs LayerCast.

# Scans outputs/vllm_layercast/<gpu>_<task>_<method>-<seed>/<org>/<model>/
# problem_*_token_ids_*.pt files (shape [gen_len, 5]; column 0 is the greedy
# token) and reports cross-GPU and cross-seed divergence.

# Writes divergence_summary.csv, divergence_firsts.csv and prints a summary table.

# python divergence_analysis.py [--outputs outputs/vllm_layercast]

import argparse
import collections
import csv
import glob
import itertools
import os
import re
import sys

import torch

# Run dirs are named  timing-<gpu>_<task>_timing_<method>-<seed>
# Methods: rf, layercast, bf16 (unmitigated, from eval_main.py)
RUN_RE = re.compile(
    r"(a100|h100|l40s)_(.+?)(?:_timing)?_(rf|layercast(?:_bs\d+)?|bf16)-(\w+)$")

# vllm_layercast also contains rf results
DEFAULT_ROOTS = ["outputs/vllm_layercast", "outputs/vllm_main"]
# Seeds shared across every method/GPU (uniform cross-seed / cross-GPU analysis).
DEFAULT_SEEDS = ["1", "42", "777"]


def _has_tokens(md):
    return bool(glob.glob(os.path.join(md, "problem_*token_ids*.pt")))


def discover_runs(roots, gpus=None, seeds=None, exclude=()):
    if isinstance(roots, str):
        roots = [roots]

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
            if (gpus and gpu not in gpus) or (seeds and seed not in seeds):
                continue
            method = "layercast" if method.startswith("layercast") else method
            if any(x in base for x in exclude):
                continue
            for org in sorted(os.listdir(d)):
                orgd = os.path.join(d, org)
                if not os.path.isdir(orgd) or any(x in org for x in exclude):
                    continue
                for model in sorted(os.listdir(orgd)):
                    md = os.path.join(orgd, model)
                    if not (os.path.isdir(md) and _has_tokens(md)):
                        continue
                    runs[(f"{org}/{model}", task, method)][(gpu, seed)] = md
    return runs


def load_streams(rundir):
    streams = {}
    for f in glob.glob(os.path.join(rundir, "problem_*_token_ids_*.pt")):
        pid = int(re.search(r"problem_(\d+)_", os.path.basename(f)).group(1))
        t = torch.load(f, map_location="cpu", weights_only=True)
        streams[pid] = t[:, 0].contiguous()
    return streams


def first_div(a, b):
    n = min(len(a), len(b))
    neq = (a[:n] != b[:n]).nonzero()
    if len(neq):
        return int(neq[0])
    return n if len(a) != len(b) else -1  # -1 == identical


def compare_pair(sa, sb):
    common = sorted(set(sa) & set(sb))
    firsts = []
    ndiv = 0
    for p in common:
        fd = first_div(sa[p], sb[p])
        if fd >= 0:
            ndiv += 1
            firsts.append((p, fd, max(len(sa[p]), len(sb[p]))))
    return len(common), ndiv, firsts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", default=DEFAULT_ROOTS,
                    help="output root(s) to scan (default: vllm_layercast + vllm_main)")
    ap.add_argument("--gpus", nargs="+", default=None,
                    help="restrict to these GPUs (default: all present)")
    ap.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS,
                    help="restrict to these seeds (default: 1 42 777)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="skip any run dir or model-org dir whose name "
                         "contains one of these substrings (e.g. --exclude "
                         "broken, to quarantine bad runs without deleting)")
    ap.add_argument("--csv", default="divergence_summary.csv")
    ap.add_argument("--firsts_csv", default="divergence_firsts.csv")
    args = ap.parse_args()

    runs = discover_runs(args.outputs, gpus=args.gpus, seeds=args.seeds,
                         exclude=tuple(args.exclude))
    rows = []
    first_rows = []
    for (model, task, method), by_gs in sorted(runs.items()):
        cache = {}

        def S(gs):
            if gs not in cache:
                cache[gs] = load_streams(by_gs[gs])
            return cache[gs]

        gpus = sorted({g for g, _ in by_gs})
        seeds = sorted({s for _, s in by_gs})

        # cross-GPU, matched seed
        xg_pairs = xg_div = 0
        xg_prob_pairs = xg_prob_div = 0
        xg_firsts = []
        xg_div_problems = set()
        all_problems = set()
        for gs in by_gs:
            all_problems |= set(S(gs))
        for seed in seeds:
            have = [g for g in gpus if (g, seed) in by_gs]
            for ga, gb in itertools.combinations(have, 2):
                nc, nd, fs = compare_pair(S((ga, seed)), S((gb, seed)))
                xg_pairs += 1
                xg_div += nd > 0
                xg_prob_pairs += nc
                xg_prob_div += nd
                xg_firsts += fs
                for prob, fd, glen in fs:
                    xg_div_problems.add(prob)
                    first_rows.append(dict(model=model, task=task,
                                           method=method, gpu_a=ga, gpu_b=gb,
                                           seed=seed, problem=prob,
                                           first_div=fd, gen_len=glen))

        # cross-seed, same GPU
        xs_prob_pairs = xs_prob_div = 0
        for gpu in gpus:
            have = [s for s in seeds if (gpu, s) in by_gs]
            for sa, sb in itertools.combinations(have, 2):
                nc, nd, _ = compare_pair(S((gpu, sa)), S((gpu, sb)))
                xs_prob_pairs += nc
                xs_prob_div += nd

        row = dict(
            model=model, task=task, method=method,
            n_runs=len(by_gs), gpus="+".join(gpus), n_seeds=len(seeds),
            xgpu_run_pairs=xg_pairs,
            xgpu_prob_pairs=xg_prob_pairs, xgpu_prob_div=xg_prob_div,
            xgpu_div_rate=(xg_prob_div / xg_prob_pairs) if xg_prob_pairs else float("nan"),
            xgpu_mean_first_div=(sum(f for _, f, _ in xg_firsts) / len(xg_firsts)) if xg_firsts else float("nan"),
            n_problems=len(all_problems),
            xgpu_div_problems=len(xg_div_problems),
            xgpu_problem_rate=(len(xg_div_problems) / len(all_problems)) if all_problems else float("nan"),
            xseed_prob_pairs=xs_prob_pairs, xseed_prob_div=xs_prob_div,
            xseed_div_rate=(xs_prob_div / xs_prob_pairs) if xs_prob_pairs else float("nan"),
        )
        rows.append(row)
        print(f"{model:38s} {task:8s} {method:10s} "
              f"xGPU {row['xgpu_prob_div']:5d}/{row['xgpu_prob_pairs']:6d} "
              f"({100*row['xgpu_div_rate']:6.2f}%)  "
              f"xSeed {row['xseed_prob_div']:5d}/{row['xseed_prob_pairs']:6d} "
              f"({100*row['xseed_div_rate']:6.2f}%)  "
              f"perProblem {row['xgpu_div_problems']:4d}/{row['n_problems']:4d} "
              f"({100*row['xgpu_problem_rate']:6.2f}%)  "
              f"meanFirstDiv={row['xgpu_mean_first_div']:.0f}", flush=True)

    if not rows:
        sys.exit(f"\nno runs found under {args.outputs} for seeds {args.seeds}. "
                 f"Expected directories named "
                 f"timing-<gpu>_<task>_timing_<method>-<seed>/<org>/<model>/ "
                 f"holding problem_*_token_ids_*.pt; produce them with "
                 f"sweeps/run_sweep.py first.")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv} ({len(rows)} rows)")

    if first_rows:
        with open(args.firsts_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(first_rows[0].keys()))
            w.writeheader()
            w.writerows(first_rows)
        print(f"wrote {args.firsts_csv} ({len(first_rows)} rows)")


if __name__ == "__main__":
    main()
