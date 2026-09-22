# Offline config sweep for tuning hard-coded rf kernel configs

import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import rf_kernels as rf
from rf_kernels import rf_linear

torch.backends.cuda.matmul.allow_tf32 = False


def time_fn(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def sweep_decode(M, N, K, wdtype=torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32).to(wdtype)

    t_lc = time_fn(lambda: F.linear(x, w.to(torch.float32)))
    print(f"\nM={M} N={N} K={K} w={wdtype}  layercast={t_lc:.3f}ms", flush=True)

    results = []
    for bm, bn, bk, s in itertools.product(
            (16, 32), (32, 64, 128, 256), (32, 64), (1, 2, 4, 8)):
        cfg = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                   num_warps=4, num_stages=2)
        old = rf._DECODE_CFG
        rf._DECODE_CFG = cfg
        try:
            t = time_fn(lambda: rf_linear(x, w, split_s=s),
                        iters=60, warmup=10)
        except Exception:
            continue
        finally:
            rf._DECODE_CFG = old
        results.append((t, cfg, s))
        
    results.sort(key=lambda r: r[0])
    flops = 2 * M * N * K
    for t, cfg, s in results[:5]:
        print(f"  {t:7.3f} ms {flops / (t * 1e-3) / 1e12:5.1f} TF "
              f"vs LC {t_lc / t:4.2f}x  S={s} {cfg}", flush=True)


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(), flush=True)
    sweep_decode(32, 6144, 4096)
    sweep_decode(32, 4096, 4096)
    sweep_decode(32, 28672, 4096)
    sweep_decode(32, 4096, 14336)
    sweep_decode(32, 128256, 4096, torch.bfloat16)
    sweep_decode(1, 4096, 4096)
