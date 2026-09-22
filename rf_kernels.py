# RF fused-upcast linear kernel.

# Loads the bf16 weights from HBM and upcast to fp32 in registers,
# accumulating in fp32 with IEEE FFMA. Weight HBM traffic in bf16.
# Batch size-independent, deterministic split-K, no autotuning.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rf_linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_os, stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    SPLIT_S: tl.constexpr,
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

    # Grouped program ordering for L2 reuse without affecting reduction order
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_in // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid_in % num_pid_in_group) % group_size_m)
    pid_n = (pid_in % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Contiguous k-segment for this sid in whole BLOCK_K tiles
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_seg = tl.cdiv(num_k_tiles, SPLIT_S)
    kt_start = sid * tiles_per_seg
    kt_end = tl.minimum(num_k_tiles, kt_start + tiles_per_seg)

    x_ptrs = (x_ptr + offs_m[:, None] * stride_xm
              + (kt_start * BLOCK_K + offs_k[None, :]) * stride_xk)
    # w.shape is (N, K) so load the (BLOCK_K, BLOCK_N) transposed
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

        w_tile = w_tile.to(tl.float32) # In-register upcast
        acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee") # IEEE FFMA
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS and SPLIT_S == 1:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias.to(tl.float32)[None, :]

    out_ptrs = (out_ptr + sid * stride_os
                + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _rf_splitk_reduce(
    part_ptr, bias_ptr, out_ptr,
    MN, N, stride_s,
    HAS_BIAS: tl.constexpr,
    SPLIT_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # out[i] = sum_s part[s, i] (+ bias[i % N])
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_S):  # sequential, ascending
        acc += tl.load(part_ptr + s * stride_s + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs % N, mask=mask, other=0.0)
        acc += bias.to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=mask)


# Fixed configs. Shared-memory worst case to fit L40S (~99 KiB/block):
# decode : 2 stages * (32*64*4  + 128*64*4) B = 80 KiB (fp32 W worst case)
# prefill: 2 stages * (128*32*4 + 128*32*4) B = 64 KiB (fp32 W worst case)
_DECODE_CFG = dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, GROUP_M=8,
                   num_warps=4, num_stages=2)
_PREFILL_CFG = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                    num_warps=4, num_stages=2)
_DECODE_M_THRESHOLD = 64
_SPLITK_MAX = 8
_SPLITK_MIN_KTILES_PER_SEG = 4


def _pick_split(M: int, N: int, K: int, cfg: dict) -> int:
    # Deterministic split factor
    if M > _DECODE_M_THRESHOLD:
        return 1
    num_k_tiles = triton.cdiv(K, cfg["BLOCK_K"])
    return max(1, min(_SPLITK_MAX,
                      num_k_tiles // _SPLITK_MIN_KTILES_PER_SEG))


def rf_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    split_s: Optional[int] = None, # override for offline tuning
) -> torch.Tensor:
    """Drop-in replacement for ``F.linear(x, weight.to(torch.float32), bias)``
    where ``x`` is fp32 and ``weight`` is bf16 (or already fp32).

    Returns fp32, shape ``x.shape[:-1] + (weight.shape[0],)``.
    """
    assert x.dtype == torch.float32, f"x must be fp32, got {x.dtype}"
    assert weight.dtype in (torch.bfloat16, torch.float32), (
        f"weight must be bf16 or fp32, got {weight.dtype}")

    if not x.is_cuda:  # CPU fallback
        return torch.nn.functional.linear(
            x, weight.to(torch.float32), bias)

    out_shape = x.shape[:-1] + (weight.shape[0],)
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    M, N = x2d.shape[0], weight.shape[0]

    if M == 0:
        return torch.empty(out_shape, device=x.device, dtype=torch.float32)

    cfg = _DECODE_CFG if M <= _DECODE_M_THRESHOLD else _PREFILL_CFG
    S = split_s if split_s is not None else _pick_split(M, N, K, cfg)

    has_bias = bias is not None
    dummy = x2d  # not dereferenced when the corresponding flag is off
    if S == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
        _rf_linear_kernel[grid](
            x2d, weight, bias if has_bias else dummy, out,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0), weight.stride(1),
            0, out.stride(0), out.stride(1),
            HAS_BIAS=has_bias, SPLIT_S=1, **cfg)
        return out.reshape(out_shape)

    part = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
    grid = (S * triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _rf_linear_kernel[grid](
        x2d, weight, dummy, part,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        weight.stride(0), weight.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        HAS_BIAS=False, SPLIT_S=S, **cfg)

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    MN = M * N
    RBLOCK = 1024
    _rf_splitk_reduce[(triton.cdiv(MN, RBLOCK),)](
        part, bias if has_bias else dummy, out,
        MN, N, part.stride(0),
        HAS_BIAS=has_bias, SPLIT_S=S, BLOCK=RBLOCK)
    return out.reshape(out_shape)
