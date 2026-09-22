#!/usr/bin/env python

# A4: ablation of R4 batch invariance

# Computes row 0 of the same input under various batch sizes and past the
# bucket boundary and sees whether the bits of row 0 ever move.

# Four policies compared: rf, occupancy, autotuned, cublas. Timed.

# python a4_batch_invariance.py


import torch
import torch.nn.functional as F

import ablation_common as C
from ablation_kernels import (DECODE_M_THRESHOLD, SPLIT_SLICES, abl_linear,
                              cfg_name, pinned_cfg, split_occupancy_tuned,
                              split_shape_pure)

M_SWEEP = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 65, 128]
M_MAX = max(M_SWEEP)

MENU_CFGS = [
    dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2),
]


def main():
    ap = C.base_argparser("A4: ablation of R4 batch invariance")
    ap.add_argument("--rep-ms", type=int, default=20,
                    help="do_bench window used by the per-M autotuner")
    args = ap.parse_args()
    cases = [c for c in C.select_cases(args.cases, args.quick) if c[5] == "decode"]
    if not cases:
        raise SystemExit("A4 applies to decode-bucket cases only")

    e = C.print_env()
    print(f"\nR4 ablation: row 0 of the batch-M result against the M=1 result. "
          f"'=' identical bits, 'X' differs.\nThe shape bucket boundary is at "
          f"M={DECODE_M_THRESHOLD}; R4 claims invariance up to there.\n")

    out = {"m_sweep": M_SWEEP, "bucket_threshold": DECODE_M_THRESHOLD,
           "cases": {}}
    for name, _, N, K, wdtype, bucket in cases:
        x, w, b = C.make_inputs(M_MAX, N, K, wdtype)

        def run(policy, M):
            """-> (output, label describing the chosen configuration)."""
            if policy == "cublas":
                return F.linear(x[:M].contiguous(), w.to(torch.float32), b), "cuBLAS"
            if policy == "autotuned":
                cfg, s = autotune(M)
            else:
                cfg = pinned_cfg(M)
                s = (split_shape_pure if policy == "rf"
                     else split_occupancy_tuned)(M, N, K, cfg)
            return (abl_linear(x[:M], w, b, cfg=cfg, split_s=s),
                    f"{cfg_name(cfg)}_S{s}")

        _tuned = {}

        def autotune(M):
            if M not in _tuned:
                best, best_t = None, float("inf")
                for cfg in MENU_CFGS:
                    for s in SPLIT_SLICES:
                        try:
                            t = C.bench_ms(
                                lambda cfg=cfg, s=s: abl_linear(
                                    x[:M], w, b, cfg=cfg, split_s=s),
                                rep=args.rep_ms)
                        except Exception:
                            continue
                        if t < best_t:
                            best, best_t = (cfg, s), t
                _tuned[M] = best
            return _tuned[M]

        rec = {"shape": (N, K, str(wdtype)), "sm_count": e["sm_count"],
               "policies": {}}
        print(f"{name}  (N={N}, K={K})")
        print(f"  {'policy':10s} " + " ".join(f"{m:>4d}" for m in M_SWEEP)
              + f"   verdict for M<={DECODE_M_THRESHOLD}")

        for policy in ("rf", "occupancy", "autotuned", "cublas"):
            rows, marks, ref = {}, [], None
            for M in M_SWEEP:
                y, label = run(policy, M)
                fp = C.fingerprint(y[0])
                fp["config"] = label
                fp["ms"] = C.bench_ms(lambda M=M: run(policy, M)[0])
                rows[M] = fp
                if ref is None:
                    ref = fp["sha256"]
                marks.append(" =  " if fp["sha256"] == ref else " X  ")
                del y
                
            in_bucket = [M for M in M_SWEEP if M <= DECODE_M_THRESHOLD]
            n_bad = sum(rows[M]["sha256"] != ref for M in in_bucket)
            rec["policies"][policy] = {"rows": rows, "invariant": n_bad == 0,
                                       "n_varying_in_bucket": n_bad,
                                       "n_in_bucket": len(in_bucket)}
            verdict = ("BATCH-INVARIANT" if n_bad == 0
                       else f"VARIES ({n_bad}/{len(in_bucket)} differ from M=1)")
            print(f"  {policy:10s} " + "".join(marks) + f"   {verdict}")
            
            if policy in ("occupancy", "autotuned"):
                chosen = {rows[M]["config"] for M in in_bucket}
                if len(chosen) > 1:
                    print(f"  {'':10s} chose {len(chosen)} different "
                          f"configurations over the batch range")

        rp = rec["policies"]["rf"]["rows"]
        for other in ("occupancy", "autotuned"):
            ratios = {M: rec["policies"][other]["rows"][M]["ms"] / rp[M]["ms"]
                      for M in M_SWEEP}
            rec[f"{other}_over_rf"] = ratios
            vals = [ratios[M] for M in M_SWEEP if M <= DECODE_M_THRESHOLD]
            print(f"  {'cost of R4':10s} vs {other}: "
                  + " ".join(f"{ratios[M]:.2f}" for M in M_SWEEP)
                  + f"   (best case for tuning: {min(vals):.2f}x)")
        print()

        out["cases"][name] = rec
        torch.cuda.empty_cache()

    C.save("a4", out)


if __name__ == "__main__":
    main()
