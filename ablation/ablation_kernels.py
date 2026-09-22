# Parameterized copy of the RF linear kernel used for ablations.

from typing import Optional

import torch
import triton
import triton.language as tl

# R1 knob
PREC_IEEE = 0
PREC_TF32 = 1
PREC_TF32X3 = 2
PREC_DEFAULT = 3
PREC_IDS = {"ieee": PREC_IEEE, "tf32": PREC_TF32, "tf32x3": PREC_TF32X3,
            "default": PREC_DEFAULT}


@triton.jit
def _abl_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_os, stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    SPLIT_S: tl.constexpr,
    ATOMIC: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # out[M, N] = x[M, K] @ w[N, K]^T (+ bias[N])
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pids_per_seg = num_pid_m * num_pid_n
    sid = pid // pids_per_seg
    pid_in = pid % pids_per_seg

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_in // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid_in % num_pid_in_group) % group_size_m)
    pid_n = (pid_in % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_seg = tl.cdiv(num_k_tiles, SPLIT_S)
    kt_start = sid * tiles_per_seg
    kt_end = tl.minimum(num_k_tiles, kt_start + tiles_per_seg)

    x_ptrs = (x_ptr + offs_m[:, None] * stride_xm
              + (kt_start * BLOCK_K + offs_k[None, :]) * stride_xk)
    w_ptrs = (w_ptr + offs_n[None, :] * stride_wn
              + (kt_start * BLOCK_K + offs_k[:, None]) * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(kt_start, kt_end):
        k_remaining = K - kt * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        w_tile = w_tile.to(tl.float32)
        if PREC == 0:
            acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
        elif PREC == 1:
            acc = tl.dot(x_tile, w_tile, acc, input_precision="tf32")
        elif PREC == 2:
            acc = tl.dot(x_tile, w_tile, acc, input_precision="tf32x3")
        else:
            acc = tl.dot(x_tile, w_tile, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS and SPLIT_S == 1:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias.to(tl.float32)[None, :]

    out_ptrs = (out_ptr + sid * stride_os
                + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def _abl_splitk_reduce(
    part_ptr, bias_ptr, out_ptr,
    MN, N, stride_s,
    HAS_BIAS: tl.constexpr,
    SPLIT_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out[i] = sum_s part[s, i] (+ bias[i % N]), s ascending -- fixed order."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_S):
        acc += tl.load(part_ptr + s * stride_s + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs % N, mask=mask, other=0.0)
        acc += bias.to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=mask)


# R2 knob
DECODE_CFG = dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, GROUP_M=8,
                  num_warps=4, num_stages=2)
PREFILL_CFG = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                   num_warps=4, num_stages=2)
DECODE_M_THRESHOLD = 64
SPLITK_MAX = 8
SPLITK_MIN_KTILES_PER_SEG = 4

SMEM_LIMIT_B = 99 * 1024


def smem_bytes(cfg):
    """Worst-case (fp32 weight) pipelined shared-memory footprint of a config."""
    return cfg["num_stages"] * 4 * (
        cfg["BLOCK_M"] * cfg["BLOCK_K"] + cfg["BLOCK_K"] * cfg["BLOCK_N"])


_RAW_CANDIDATES = [
    dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=16, BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=32, BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2),
]
CANDIDATE_CFGS = [c for c in _RAW_CANDIDATES if smem_bytes(c) <= SMEM_LIMIT_B]


def cfg_name(cfg):
    return (f"M{cfg['BLOCK_M']}_N{cfg['BLOCK_N']}_K{cfg['BLOCK_K']}"
            f"_w{cfg['num_warps']}_s{cfg['num_stages']}")


def pinned_cfg(M):
    return dict(DECODE_CFG if M <= DECODE_M_THRESHOLD else PREFILL_CFG)


# R3 and R4 knobs
def split_shape_pure(M, N, K, cfg, device=None):

    if M > DECODE_M_THRESHOLD:
        return 1
    num_k_tiles = triton.cdiv(K, cfg["BLOCK_K"])
    return max(1, min(SPLITK_MAX, num_k_tiles // SPLITK_MIN_KTILES_PER_SEG))


SPLIT_SLICES = [1, 2, 4, 8, 16]


def split_occupancy_tuned(M, N, K, cfg, device=None):
    if M > DECODE_M_THRESHOLD:
        return 1
    props = torch.cuda.get_device_properties(device or torch.cuda.current_device())
    sms = props.multi_processor_count
    tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    num_k_tiles = triton.cdiv(K, cfg["BLOCK_K"])
    best, best_eff = 1, -1.0
    for s in SPLIT_SLICES:
        if s > num_k_tiles:
            break
        programs = tiles * s
        eff = programs / (triton.cdiv(programs, sms) * sms)
        if eff > best_eff + 1e-12: # ties keep the smaller split
            best, best_eff = s, eff
    return best



def abl_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cfg: Optional[dict] = None,
    prec: str = "ieee",
    splitk: str = "det",
    split_s: Optional[int] = None,
    split_policy=split_shape_pure,
    out: Optional[torch.Tensor] = None,
):
    # RF's linear layer with one rule at a time switched off.
    assert x.dtype == torch.float32, f"x must be fp32, got {x.dtype}"
    assert weight.dtype in (torch.bfloat16, torch.float32)

    out_shape = x.shape[:-1] + (weight.shape[0],)
    K = x.shape[-1]
    x2d = x.reshape(-1, K).contiguous()
    weight = weight.contiguous()
    M, N = x2d.shape[0], weight.shape[0]
    cfg = dict(cfg) if cfg is not None else pinned_cfg(M)

    if splitk == "none":
        S = 1
    elif split_s is not None:
        S = split_s
    else:
        S = split_policy(M, N, K, cfg)

    prec_id = PREC_IDS[prec]
    has_bias = bias is not None
    dummy = x2d
    tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])

    if S == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        _abl_gemm_kernel[(tiles,)](
            x2d, weight, bias if has_bias else dummy, out,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0), weight.stride(1),
            0, out.stride(0), out.stride(1),
            HAS_BIAS=has_bias, SPLIT_S=1, ATOMIC=False, PREC=prec_id, **cfg)
        return out.reshape(out_shape)

    if splitk == "atomic":
        # Deterministic starting point then the S partials are merged by
        # fp32 atomicAdd in whatever order the SMs finish
        if has_bias:
            out = bias.to(torch.float32).expand(M, N).contiguous()
        else:
            out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
        _abl_gemm_kernel[(S * tiles,)](
            x2d, weight, dummy, out,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0), weight.stride(1),
            0, out.stride(0), out.stride(1),
            HAS_BIAS=False, SPLIT_S=S, ATOMIC=True, PREC=prec_id, **cfg)
        return out.reshape(out_shape)

    part = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
    _abl_gemm_kernel[(S * tiles,)](
        x2d, weight, dummy, part,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        weight.stride(0), weight.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        HAS_BIAS=False, SPLIT_S=S, ATOMIC=False, PREC=prec_id, **cfg)

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    MN, RBLOCK = M * N, 1024
    _abl_splitk_reduce[(triton.cdiv(MN, RBLOCK),)](
        part, bias if has_bias else dummy, out,
        MN, N, part.stride(0),
        HAS_BIAS=has_bias, SPLIT_S=S, BLOCK=RBLOCK)
    return out.reshape(out_shape)
