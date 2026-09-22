# Cross-architecture bitwise probe for testing 1. IEEE-754 FFMA is bit-specified for given operands on every arch and 2. the reduction order is a pure function of problem shape, not device.

# python probe_cross_arch.py dump            # writes probe_<gpuname>.pt
# python probe_cross_arch.py compare         # compares all probe_*.pt found

import glob
import hashlib
import os
import re
import sys

import torch

from rf_kernels import rf_linear

CASES = [
    ("decode_qkv", 32, 6144, 4096, torch.bfloat16),
    ("decode_down", 32, 4096, 14336, torch.bfloat16),
    ("decode_lmhead", 32, 128256, 4096, torch.float32),
    ("decode_bs1", 1, 4096, 4096, torch.bfloat16),
    ("prefill_qkv", 2048, 6144, 4096, torch.bfloat16),
    ("prefill_gateup", 2048, 28672, 4096, torch.bfloat16),
    ("ragged", 100, 1000, 1000, torch.bfloat16),
]


def make_inputs(M, N, K, wdtype):
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(M, K, generator=g, dtype=torch.float32)
    w = torch.randn(N, K, generator=g, dtype=torch.float32).to(wdtype)
    bias = torch.randn(N, generator=g, dtype=torch.float32)
    return x.cuda(), w.cuda(), bias.cuda()


def dump():
    name = torch.cuda.get_device_name()
    tag = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    out = {"device": name, "torch": torch.__version__,
           "triton": __import__("triton").__version__, "cases": {}}
    for cname, M, N, K, wdtype in CASES:
        x, w, bias = make_inputs(M, N, K, wdtype)
        y = rf_linear(x, w, bias).cpu()
        digest = hashlib.sha256(y.numpy().tobytes()).hexdigest()
        out["cases"][cname] = {"tensor": y, "sha256": digest}
        print(f"  {cname:16s} sha256={digest[:16]}…")
    path = f"probe_{tag}.pt"
    torch.save(out, path)
    print(f"wrote {path}")


def compare():
    paths = sorted(p for p in glob.glob("probe_*.pt")
                   if not os.path.basename(p).startswith("probe_baseline_"))
    if len(paths) < 2:
        sys.exit(f"need >=2 probe files from different GPUs, found {paths}")
    dumps = {p: torch.load(p, weights_only=False) for p in paths}
    ref_path = paths[0]
    ref = dumps[ref_path]
    print(f"reference: {ref_path} ({ref['device']})")
    all_ok = True
    for p in paths[1:]:
        d = dumps[p]
        print(f"vs {p} ({d['device']}):")
        for cname in ref["cases"]:
            a = ref["cases"][cname]["tensor"]
            b = d["cases"][cname]["tensor"]
            bitwise = torch.equal(a, b)
            all_ok &= bitwise
            if bitwise:
                print(f"  {cname:16s} BITWISE IDENTICAL")
            else:
                diff = (a.double() - b.double()).abs()
                nbad = (a != b).sum().item()
                print(f"  {cname:16s} DIFFERS: {nbad}/{a.numel()} elems, "
                      f"max|diff|={diff.max().item():.3e}, "
                      f"scaled={diff.max().item() / a.abs().max().item():.3e}")
                
    print("\nRESULT:", "ALL BITWISE IDENTICAL, cross-arch determinism holds"
          if all_ok else "DIVERGENCE FOUND, see above")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    {"dump": dump, "compare": compare}[mode]()
