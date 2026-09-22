#!/usr/bin/env python
import sys
import torch
import ablation_common as C
from ablation_kernels import abl_linear
from rf_kernels import rf_linear


def main():
    C.print_env()
    print("\nparity of abl_linear (all rules on) with the shipped "
          "rf_linear:\n")
    ok = True
    for case in C.GEMM_CASES:
        name, M, N, K, wdtype, bucket = case
        x, w, b = C.case_inputs(case)
        for tag, bias in (("with bias", b), ("no bias", None)):
            a = rf_linear(x, w, bias)
            c = abl_linear(x, w, bias)
            same = torch.equal(a, c)
            ok &= same
            print(f"  {name:16s} {tag:10s} "
                  f"{'identical' if same else 'MISMATCH'}")
            del a, c
        torch.cuda.empty_cache()
    print("\nRESULT:", "parity holds" if ok else
          "PARITY BROKEN: ablation_kernels.py has drifted from "
          "rf_kernels.py; fix before running the ablations")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
