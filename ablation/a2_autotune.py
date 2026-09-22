#!/usr/bin/env python

# A2: ablation of R2 autotuning.

# R2 fixes the kernel configuration to a compile-time constant chosen by a pure
# function of the problem shape. The alternative is what every Triton GEMM does
# which is benchmark a candidate space on the local device and keep the fastest.
# This script an autotuner to show what autotuning would have changed.

# Per (case, candidate) the results are the median time and the output digest.

# python a2_autotune.py [--repeats 3]


import torch

import ablation_common as C
from ablation_kernels import (CANDIDATE_CFGS, DECODE_M_THRESHOLD, abl_linear,
                              cfg_name, pinned_cfg, split_shape_pure)

SPLITK_AXIS = [1, 2, 4, 8]
PREC_AXIS = ["ieee", "tf32", "tf32x3"]


def candidates(M):
    out = []
    for cfg in CANDIDATE_CFGS:
        for s in (SPLITK_AXIS if M <= DECODE_M_THRESHOLD else [1]):
            out.append((f"{cfg_name(cfg)}_S{s}", cfg, s))
    return out


def pinned_label(M, N, K):
    cfg = pinned_cfg(M)
    return f"{cfg_name(cfg)}_S{split_shape_pure(M, N, K, cfg)}"


def main():
    ap = C.base_argparser("A2: ablation of R2 autotuning")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent repetitions of the whole search")
    ap.add_argument("--rep-ms", type=int, default=50,
                    help="do_bench measurement window per candidate")
    args = ap.parse_args()
    cases = C.select_cases(args.cases, args.quick)

    C.print_env()
    print(f"\nR2 ablation: autotuning over {len(CANDIDATE_CFGS)} tile configs "
          f"x split-K in {SPLITK_AXIS} (decode) -- "
          f"{args.repeats} independent searches.\n")

    out = {"splitk_axis": SPLITK_AXIS, "repeats": args.repeats, "cases": {}}
    for name, M, N, K, wdtype, bucket in cases:
        x, w, b = C.case_inputs((name, M, N, K, wdtype, bucket))
        menu = candidates(M)
        pin = pinned_label(M, N, K)

        digests, failures = {}, {}
        for label, cfg, s in menu:
            try:
                y = abl_linear(x, w, b, cfg=cfg, split_s=s)
                digests[label] = C.digest(y)
                del y
            except Exception as e:
                failures[label] = f"{type(e).__name__}: {e}"
        torch.cuda.empty_cache()

        times = [] # times[repeat][label]
        for r in range(args.repeats):
            row = {}
            for label, cfg, s in menu:
                if label in failures:
                    continue
                row[label] = C.bench_ms(
                    lambda cfg=cfg, s=s: abl_linear(x, w, b, cfg=cfg, split_s=s),
                    rep=args.rep_ms)
            times.append(row)

        winners = [min(row, key=row.get) for row in times]
        best_t = [row[wn] for row, wn in zip(times, winners)]
        pin_t = [row.get(pin, float("nan")) for row in times]

        prec_t = {}
        for p in PREC_AXIS:
            try:
                prec_t[p] = C.bench_ms(
                    lambda p=p: abl_linear(x, w, b, cfg=pinned_cfg(M), prec=p),
                    rep=args.rep_ms)
            except Exception:
                pass
        prec_win = min(prec_t, key=prec_t.get) if prec_t else None

        classes = {}
        for label, d in digests.items():
            classes.setdefault(d, []).append(label)

        out["cases"][name] = {
            "shape": (M, N, K, str(wdtype), bucket),
            "menu": [lab for lab, _, _ in menu],
            "pinned": pin, "digests": digests, "failures": failures,
            "times": times, "winners": winners,
            "prec_times": prec_t, "prec_winner": prec_win,
            "n_digest_classes": len(classes),
        }

        stable = len(set(winners)) == 1
        same_bits = (digests.get(winners[0]) == digests.get(pin))
        print(f"{name:16s} winner={winners[0]:24s} "
              f"{'stable' if stable else 'UNSTABLE ' + str(sorted(set(winners)))}")
        print(f"{'':16s} pinned={pin:24s} "
              f"pinned/best={pin_t[0] / best_t[0]:.3f}x  "
              f"bits(winner)=={'pinned' if same_bits else 'DIFFERENT'}  "
              f"{len(classes)} distinct output(s) over {len(digests)} candidates")
        if prec_win:
            print(f"{'':16s} precision search would pick '{prec_win}' "
                  + "  ".join(f"{p}={t:.3f}ms" for p, t in prec_t.items()))
        if failures:
            print(f"{'':16s} {len(failures)} failed to compile")
        torch.cuda.empty_cache()

    C.save("a2", out)


if __name__ == "__main__":
    main()
