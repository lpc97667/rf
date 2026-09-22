# Substantiate the two baseline claims made in the paper about LayerCast/cuBLAS.

# 1. LayerCast's cross-architecture cast-then-cuBLAS path
# (F.linear(x, w.to(fp32), bias)) produces different bits on A100 / L40S / H100,
# at ~1e-7 scaled magnitude.

# 2. for cuBLAS SGEMM, row 0 of F.linear(X[:M]) changes bits as M varies

# python probe_baseline_claims.py dump      # on each GPU
# python probe_baseline_claims.py compare   # once all dumps exist

import glob
import hashlib
import re
import sys

import torch
import torch.nn.functional as F

from rf_kernels import rf_linear

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CASES_A = [
    ("decode_qkv", 32, 6144, 4096, torch.bfloat16),
    ("decode_down", 32, 4096, 14336, torch.bfloat16),
    ("decode_lmhead", 32, 128256, 4096, torch.float32),
    ("decode_bs1", 1, 4096, 4096, torch.bfloat16),
    ("prefill_qkv", 2048, 6144, 4096, torch.bfloat16),
    ("prefill_down", 2048, 4096, 14336, torch.bfloat16),
    ("ragged", 100, 1000, 1000, torch.bfloat16),
]

CASES_B = [
    ("qkv_proj", 6144, 4096, torch.bfloat16),
    ("down_proj", 4096, 14336, torch.bfloat16),
    ("lm_head", 128256, 4096, torch.bfloat16),
]
BATCH_SIZES = [1, 2, 3, 4, 8, 16, 32, 48, 64]


def layercast_linear(x, w, bias):
    return F.linear(x, w.to(torch.float32),
                    bias.to(torch.float32) if bias is not None else None)


def make_inputs(M, N, K, wdtype, seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g, dtype=torch.float32)
    w = torch.randn(N, K, generator=g, dtype=torch.float32).to(wdtype)
    bias = torch.randn(N, generator=g, dtype=torch.float32)
    return x.cuda(), w.cuda(), bias.cuda()


def dump():
    name = torch.cuda.get_device_name()
    tag = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    out = {"device": name, "torch": torch.__version__,
           "triton": __import__("triton").__version__,
           "layercast": {}, "batch_invariance": {}}

    print("=== Claim A dump: LayerCast (cast-then-cuBLAS) outputs ===")
    for cname, M, N, K, wdtype in CASES_A:
        x, w, bias = make_inputs(M, N, K, wdtype)
        y = layercast_linear(x, w, bias).cpu()
        digest = hashlib.sha256(y.numpy().tobytes()).hexdigest()
        out["layercast"][cname] = {"tensor": y, "sha256": digest}
        print(f"  {cname:16s} sha256={digest[:16]}…")

    print("\n=== Claim B (local): row-0 batch invariance, M in "
          f"{BATCH_SIZES} ===")
    print(f"  {'case':10s} {'method':10s}  " +
          " ".join(f"M={m:<3d}" for m in BATCH_SIZES) + "  verdict")
    for cname, N, K, wdtype in CASES_B:
        x, w, bias = make_inputs(max(BATCH_SIZES), N, K, wdtype)
        for mname, fn in (("layercast", layercast_linear),
                          ("rf", rf_linear)):
            ref = fn(x[:1].contiguous(), w, bias)[0].cpu()
            marks, n_id = [], 0
            digests = {}
            for M in BATCH_SIZES:
                row0 = fn(x[:M].contiguous(), w, bias)[0].cpu()
                same = torch.equal(row0, ref)
                n_id += same
                marks.append("  = " if same else "  X ")
                digests[M] = hashlib.sha256(
                    row0.numpy().tobytes()).hexdigest()
            invariant = n_id == len(BATCH_SIZES)
            out["batch_invariance"][(cname, mname)] = {
                "digests": digests, "invariant": invariant}
            verdict = ("BATCH-INVARIANT" if invariant
                       else f"VARIES ({len(BATCH_SIZES) - n_id}/"
                            f"{len(BATCH_SIZES)} differ from M=1)")
            print(f"  {cname:10s} {mname:10s}  " + " ".join(marks)
                  + f"  {verdict}")

    path = f"probe_baseline_{tag}.pt"
    torch.save(out, path)
    print(f"\nwrote {path}")


def compare():
    paths = sorted(glob.glob("probe_baseline_*.pt"))
    if len(paths) < 2:
        sys.exit(f"need >=2 probe files, found {paths}")
    dumps = {p: torch.load(p, weights_only=False) for p in paths}
    ref_path = paths[0]
    ref = dumps[ref_path]

    print("=== Claim A: LayerCast cross-architecture comparison ===")
    print(f"reference: {ref_path} ({ref['device']})")
    any_diff = False
    worst = 0.0
    for p in paths[1:]:
        d = dumps[p]
        print(f"vs {p} ({d['device']}):")
        for cname in ref["layercast"]:
            a = ref["layercast"][cname]["tensor"]
            b = d["layercast"][cname]["tensor"]
            if torch.equal(a, b):
                print(f"  {cname:16s} BITWISE IDENTICAL")
                continue
            any_diff = True
            diff = (a.double() - b.double()).abs()
            scaled = diff.max().item() / a.abs().max().item()
            worst = max(worst, scaled)
            nbad = (a != b).sum().item()
            print(f"  {cname:16s} DIFFERS: {nbad}/{a.numel()} elems "
                  f"({100.0 * nbad / a.numel():.1f}%), "
                  f"max|diff|={diff.max().item():.3e}, scaled={scaled:.3e}")
    print(f"\nClaim A verdict: LayerCast cross-arch outputs "
          f"{'DIFFER (worst scaled diff %.3e)' % worst if any_diff else 'are identical (claim NOT supported)'}")

    print("\n=== Claim B: batch invariance per node ===")
    for p in paths:
        d = dumps[p]
        print(f"{d['device']}:")
        for (cname, mname), r in d["batch_invariance"].items():
            print(f"  {cname:10s} {mname:10s} "
                  f"{'BATCH-INVARIANT' if r['invariant'] else 'VARIES with M'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    {"dump": dump, "compare": compare}[mode]()
