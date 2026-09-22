#!/usr/bin/env python

# A3: ablation of R3 deterministic split-K.

# Checks how the reduction dimension is split without atomics and when the
# split factor is a pure function of shape. Timed.

# python a3_splitk.py [--runs 50]


import torch

import ablation_common as C
from ablation_kernels import (abl_linear, pinned_cfg, split_occupancy_tuned,
                              split_shape_pure)

S_SWEEP = [1, 2, 4, 8, 16]
VARIANTS = ["none", "det", "atomic"]


def main():
    ap = C.base_argparser("A3: ablation of R3 deterministic split-K")
    ap.add_argument("--runs", type=int, default=50,
                    help="repetitions used to detect run-to-run variation")
    args = ap.parse_args()

    cases = [c for c in C.select_cases(args.cases, args.quick) if c[5] == "decode"]
    if not cases:
        raise SystemExit("A3 applies to decode-bucket cases only")

    e = C.print_env()
    print(f"\nR3 ablation: {args.runs} repeated runs per variant on the pinned decode config.\n")
    hdr = (f"{'case':16s} {'variant':7s} {'S':>3s} {'distinct/runs':>14s} "
           f"{'max spread':>11s} {'time (ms)':>10s} {'vs det':>8s}")
    print(hdr)
    print("-" * len(hdr))

    out = {"runs": args.runs, "s_sweep": S_SWEEP, "cases": {}}
    for case in cases:
        name, M, N, K, wdtype, bucket = case
        x, w, b = C.case_inputs(case)
        cfg = pinned_cfg(M)
        S_policy = split_shape_pure(M, N, K, cfg)
        S_occ = split_occupancy_tuned(M, N, K, cfg)

        rec = {"shape": (M, N, K, str(wdtype), bucket),
               "S_policy": S_policy, "S_occupancy": S_occ,
               "sm_count": e["sm_count"], "variants": {}}

        t_det = None
        
        for v in VARIANTS:
            kw = dict(splitk=v, split_s=None if v == "none" else S_policy)
            first = abl_linear(x, w, b, cfg=cfg, **kw)
            fp = C.fingerprint(first)
            seen, spread = {fp["sha256"]: 1}, 0.0
            
            for _ in range(args.runs - 1):
                y = abl_linear(x, w, b, cfg=cfg, **kw)
                d = C.digest(y)
                seen[d] = seen.get(d, 0) + 1
                if d != fp["sha256"]:
                    diff = (C.sample_of(y).double() - fp["sample"].double()).abs()
                    spread = max(spread, float(diff.max()))
                del y
                
            t = C.bench_ms(lambda kw=kw: abl_linear(x, w, b, cfg=cfg, **kw))
            if v == "det":
                t_det = t
            rec["variants"][v] = {
                **fp, "S": 1 if v == "none" else S_policy, "ms": t,
                "n_distinct": len(seen), "counts": seen,
                "max_spread_abs": spread,
                "max_spread_scaled": spread / max(fp["absmax"], 1e-30)}
            ratio = f"{t / t_det:.2f}x" if t_det else "-"
            sp = f"{spread / max(fp['absmax'], 1e-30):.2e}" if spread else "0"
            print(f"{name:16s} {v:7s} {rec['variants'][v]['S']:3d} "
                  f"{len(seen):6d}/{args.runs:<7d} {sp:>11s} "
                  f"{t:10.3f} {ratio:>8s}")


        # split factor function of shape
        rec["s_sweep"] = {}
        for s in S_SWEEP:
            y = abl_linear(x, w, b, cfg=cfg, splitk="det", split_s=s)
            rec["s_sweep"][s] = C.fingerprint(y)
            del y
            
        n_distinct = len({f["sha256"] for f in rec["s_sweep"].values()})
        rec["s_sweep_distinct"] = n_distinct
        
        print(f"{name:16s} split factor: shape-pure rule picks S={S_policy}, "
              f"occupancy rule picks S={S_occ} on {e['sm_count']} SMs; "
              f"S in {S_SWEEP} gives {n_distinct} distinct outputs")

        out["cases"][name] = rec
        torch.cuda.empty_cache()

    C.save("a3", out)


if __name__ == "__main__":
    main()
