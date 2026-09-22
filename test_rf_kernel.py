# Standalone tests + benchmark for the RF fused-upcast linear kernel.

import argparse
import torch
import torch.nn.functional as F
from rf_kernels import rf_linear

torch.backends.cuda.matmul.allow_tf32 = False


def layercast_linear(x, w, bias=None):
    weight = w.to(torch.float32)
    b = bias.to(torch.float32) if bias is not None else None
    return F.linear(x, weight, b)


LLAMA8B_SHAPES = {
    "qkv_proj": (6144, 4096),
    "o_proj": (4096, 4096),
    "gate_up_proj": (28672, 4096),
    "down_proj": (4096, 14336),
    "lm_head": (128256, 4096),
}


def check_correctness():
    print("=== correctness vs fp64 reference ===")
    torch.manual_seed(0)
    ok = True
    for name, (N, K) in LLAMA8B_SHAPES.items():
        for M in (1, 32, 2048):
            wdtype = torch.float32 if name == "lm_head" else torch.bfloat16
            x = torch.randn(M, K, device="cuda", dtype=torch.float32)
            w = torch.randn(N, K, device="cuda", dtype=torch.float32).to(wdtype)
            bias = torch.randn(N, device="cuda", dtype=torch.float32) \
                if name == "qkv_proj" else None

            ref = F.linear(x.double(), w.double(),
                           bias.double() if bias is not None else None)
            out_rf = rf_linear(x, w, bias).double()
            out_lc = layercast_linear(x, w, bias).double()

            scale = ref.abs().max()
            rel_rf = ((out_rf - ref).abs().max() / scale).item()
            rel_lc = ((out_lc - ref).abs().max() / scale).item()
            line_ok = rel_rf < 1e-5
            ok &= line_ok
            print(f"  {name:13s} M={M:5d} {str(wdtype):15s} "
                  f"scaled-err rf={rel_rf:.3e} layercast={rel_lc:.3e} "
                  f"{'OK' if line_ok else 'FAIL'}")
    print("correctness:", "PASS" if ok else "FAIL")
    return ok


def check_determinism_and_invariance():
    print("=== run-to-run determinism + batch invariance ===")
    torch.manual_seed(1)
    N, K = 4096, 4096
    w = torch.randn(N, K, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    ok = True

    x = torch.randn(32, K, device="cuda", dtype=torch.float32)
    outs = [rf_linear(x, w) for _ in range(5)]
    r2r = all(torch.equal(outs[0], o) for o in outs[1:])
    ok &= r2r
    print(f"  run-to-run bitwise (M=32, 5 runs): {'PASS' if r2r else 'FAIL'}")

    row = x[:1]
    out_m1 = rf_linear(row, w)
    for M in (32, 64):
        xb = torch.cat([row, torch.randn(M - 1, K, device="cuda")], dim=0)
        out_mb = rf_linear(xb, w)[:1]
        same = torch.equal(out_m1, out_mb)
        ok &= same
        print(f"  batch invariance row0: M=1 vs M={M:3d}: "
              f"{'PASS' if same else 'FAIL'}")

    xb = torch.cat([row, torch.randn(127, K, device="cuda")], dim=0)
    out_mb = rf_linear(xb, w)[:1]
    same = torch.equal(out_m1, out_mb)
    diff = (out_m1 - out_mb).abs().max().item()
    print(f"  across buckets M=1 vs M=128: bitwise={same} "
          f"max|diff|={diff:.3e} (informational)")

    print("determinism/invariance:", "PASS" if ok else "FAIL")
    return ok


def bench():
    print("=== latency: rf vs layercast vs native bf16 ===")
    torch.manual_seed(2)

    def time_fn(fn, iters=200, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    for M, tag, iters in ((32, "decode bs=32", 200), (2048, "prefill 2048", 20)):
        print(f"-- {tag} --")
        tot = {"rf": 0.0, "layercast": 0.0, "bf16": 0.0}
        for name, (N, K) in LLAMA8B_SHAPES.items():
            if name == "lm_head" and M > 64:
                continue
            x = torch.randn(M, K, device="cuda", dtype=torch.float32)
            w = torch.randn(N, K, device="cuda", dtype=torch.float32)
            x_bf = x.to(torch.bfloat16)
            w_bf = w.to(torch.bfloat16)
            w_rf = w_bf
            w_lc = w if name == "lm_head" else w_bf

            t_rf = time_fn(lambda: rf_linear(x, w_rf), iters)
            t_lc = time_fn(lambda: layercast_linear(x, w_lc), iters)
            t_bf = time_fn(lambda: F.linear(x_bf, w_bf), iters)
            tot["rf"] += t_rf
            tot["layercast"] += t_lc
            tot["bf16"] += t_bf

            wbytes = w_rf.numel() * w_rf.element_size()
            gbps = wbytes / (t_rf * 1e-3) / 1e9
            print(f"  {name:13s} N={N:6d} K={K:5d}  "
                  f"rf {t_rf:7.3f} ms ({gbps:6.0f} GB/s W-read)  "
                  f"layercast {t_lc:7.3f} ms  bf16 {t_bf:7.3f} ms  "
                  f"speedup vs LC: {t_lc / t_rf:4.2f}x")
        print(f"  {'TOTAL':13s} {'':15s}  rf {tot['rf']:7.3f} ms"
              f"{'':23s}layercast {tot['layercast']:7.3f} ms  "
              f"bf16 {tot['bf16']:7.3f} ms  "
              f"speedup vs LC: {tot['layercast'] / tot['rf']:4.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="benchmark only")
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name()}")
    if not args.bench:
        ok = check_correctness()
        ok &= check_determinism_and_invariance()
        if not ok:
            raise SystemExit("FAILURES — do not integrate")
    bench()
