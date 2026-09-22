#!/usr/bin/env python

# A1: ablation of R1 IEEE FMA

# R1 forces input_precision="ieee" on every tl.dot. This script runs the
# otherwise identical kernel under four settings of that one argument and asks
# what R1 buys and what it costs (ieee, tf32, tf32x3, default).

# Results in bitwise fingerprint for the cross-architecture comparison, error
# against an fp64 reference, and median kernel time.

# python a1_precision.python


import torch

import ablation_common as C
from ablation_kernels import abl_linear

VARIANTS = ["ieee", "tf32", "tf32x3", "default"]


def main():
    ap = C.base_argparser("A1: ablation of R1 IEEE FMA")
    args = ap.parse_args()
    cases = C.select_cases(args.cases, args.quick)

    C.print_env()
    print("\nR1 ablation: input_precision on every tl.dot, everything else "
          "pinned.\n")
    hdr = (f"{'case':16s} {'variant':8s} {'digest':10s} {'max rel err':>12s} "
           f"{'mean rel err':>13s} {'time (ms)':>10s} {'vs ieee':>9s}")
    print(hdr)
    print("-" * len(hdr))

    out = {"variants": VARIANTS, "cases": {}}
    for name, M, N, K, wdtype, bucket in cases:
        x, w, b = C.case_inputs((name, M, N, K, wdtype, bucket))
        ref64 = C.fp64_reference(x, w, b)
        rec, t_ieee = {}, None
        
        for v in VARIANTS:
            try:
                y = abl_linear(x, w, b, prec=v)
            except Exception as e: # optional tf32x3
                rec[v] = {"unsupported": f"{type(e).__name__}: {e}"}
                print(f"{name:16s} {v:8s} unsupported ({type(e).__name__})")
                continue
            t = C.bench_ms(lambda v=v: abl_linear(x, w, b, prec=v))
            r = {**C.fingerprint(y), **C.err_stats(y, ref64), "ms": t}
            rec[v] = r
            if v == "ieee":
                t_ieee = t
            ratio = f"{t / t_ieee:.2f}x" if t_ieee else "-"
            print(f"{name:16s} {v:8s} {r['sha256'][:10]} "
                  f"{r['max_rel_scaled']:12.3e} {r['mean_rel']:13.3e} "
                  f"{t:10.3f} {ratio:>9s}")
            
        out["cases"][name] = {"shape": (M, N, K, str(wdtype), bucket),
                              "variants": rec}
        del ref64
        torch.cuda.empty_cache()


    print("\nDoes torch.backends.cuda.matmul.allow_tf32=False reach Triton?")
    agree_tf32 = agree_ieee = total = 0
    
    for name, rc in out["cases"].items():
        v = rc["variants"]
        if "sha256" not in v.get("default", {}) or "sha256" not in v.get("tf32", {}):
            continue
        total += 1
        agree_tf32 += v["default"]["sha256"] == v["tf32"]["sha256"]
        agree_ieee += v["default"]["sha256"] == v["ieee"]["sha256"]
        
    print(f"  default == tf32 on {agree_tf32}/{total} cases, default == ieee on {agree_ieee}/{total} cases")
    print("No" if agree_tf32 == total and total else "Yes")

    C.save("a1", out)


if __name__ == "__main__":
    main()
