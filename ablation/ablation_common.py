# Shared harness for the R1-R4 ablations.

import hashlib
import os
import re
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SEED = 1234
SAMPLE_N = 65536


GEMM_CASES = [
    ("dec_qkv_8b",     32, 6144, 4096, torch.bfloat16, "decode"),
    ("dec_down_8b",    32, 4096, 14336, torch.bfloat16, "decode"),
    ("dec_qkv_qwen",   32, 6144, 2560, torch.bfloat16, "decode"),
    ("dec_down_qwen",  32, 2560, 9728, torch.bfloat16, "decode"),
    ("dec_oproj_f32w", 32, 4096, 4096, torch.float32, "decode"),
    ("dec_lmhead_3b",  32, 128256, 3072, torch.bfloat16, "decode"),
    ("pre_qkv_8b",     2048, 6144, 4096, torch.bfloat16, "prefill"),
    ("pre_down_8b",    2048, 4096, 14336, torch.bfloat16, "prefill"),
    ("ragged",         100, 1000, 1000, torch.bfloat16, "prefill"),
]
QUICK_CASES = ["dec_qkv_8b", "dec_qkv_qwen", "pre_qkv_8b", "ragged"]
SYNTHETIC_CASES = {"ragged"}


def select_cases(names=None, quick=False):
    if quick and not names:
        names = QUICK_CASES
    if not names:
        return list(GEMM_CASES)
    out = [c for c in GEMM_CASES if c[0] in set(names)]
    missing = set(names) - {c[0] for c in out}
    if missing:
        raise SystemExit(f"unknown case(s): {sorted(missing)}")
    return out


_ARCH_BY_CC = {(8, 0): "a100", (8, 9): "l40s", (9, 0): "h100"}


def arch_key():
    cc = torch.cuda.get_device_capability()
    if cc in _ARCH_BY_CC:
        return _ARCH_BY_CC[cc]
    name = torch.cuda.get_device_name().lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def env_info():
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "arch": arch_key(),
        "device_name": torch.cuda.get_device_name(),
        "capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "sm_count": p.multi_processor_count,
        "smem_per_block_optin": getattr(p, "shared_memory_per_block_optin", None),
        "total_mem_gb": round(p.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def print_env():
    e = env_info()
    print(f"device : {e['device_name']}  (arch key '{e['arch']}', "
          f"sm_{e['capability'].replace('.', '')}, {e['sm_count']} SMs)")
    print(f"stack  : torch {e['torch']} / cuda {e['cuda']} / triton {e['triton']}"
          f"   torch.backends.cuda.matmul.allow_tf32={e['allow_tf32']}")
    return e


# inputs
_input_cache = {}


def make_inputs(M, N, K, wdtype, seed=SEED):
    key = (M, N, K, wdtype, seed)
    if key in _input_cache:
        return _input_cache[key]
    _input_cache.clear() # one case resident at a time
    torch.cuda.empty_cache()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g, dtype=torch.float32)
    w = torch.randn(N, K, generator=g, dtype=torch.float32)
    bias = torch.randn(N, generator=g, dtype=torch.float32)
    got = (x.cuda(), w.to(wdtype).cuda(), bias.cuda())
    _input_cache[key] = got
    return got


def case_inputs(case, m_override=None):
    _, M, N, K, wdtype, _ = case
    x, w, b = make_inputs(m_override if m_override else M, N, K, wdtype)
    return x, w, b


# output fingerprints
def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sample_of(t):
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n <= SAMPLE_N:
        return flat.cpu().clone()
    return flat[::max(1, n // SAMPLE_N)][:SAMPLE_N].cpu().clone()


def fingerprint(t):
    return {"sha256": digest(t), "sample": sample_of(t),
            "numel": t.numel(), "absmax": float(t.abs().max())}


def fp64_reference(x, w, bias, n_chunk=8192):
    M = x.shape[0]
    out = torch.empty((M, w.shape[0]), device=x.device, dtype=torch.float64)
    xd = x.double()
    for i in range(0, w.shape[0], n_chunk):
        wd = w[i:i + n_chunk].double()
        out[:, i:i + n_chunk] = xd @ wd.T
        if bias is not None:
            out[:, i:i + n_chunk] += bias[i:i + n_chunk].double()
    return out


def err_stats(y, ref):
    d = (y.double() - ref).abs()
    scale = ref.abs().max().clamp_min(1e-30)
    denom = ref.abs().clamp_min(1e-30)
    return {"max_abs": float(d.max()),
            "max_rel_scaled": float(d.max() / scale),
            "mean_rel": float((d / denom).mean())}


def cmp_fingerprints(a, b):
    same = a["sha256"] == b["sha256"]
    r = {"bitwise": same}
    if not same:
        sa, sb = a["sample"].double(), b["sample"].double()
        d = (sa - sb).abs()
        scale = max(a["absmax"], b["absmax"], 1e-30)
        r.update(frac_diff=float((a["sample"] != b["sample"]).float().mean()),
                 max_abs=float(d.max()), max_rel_scaled=float(d.max() / scale))
    return r


# timing
def bench_ms(fn, warmup=25, rep=100):
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep,
                                             return_mode="median"))
    except TypeError:                        # older signature: mean only
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


# io
def save(exp, payload):
    os.makedirs(RESULTS, exist_ok=True)
    payload["env"] = env_info()
    payload["exp"] = exp
    path = os.path.join(RESULTS, f"{exp}_{arch_key()}.pt")
    torch.save(payload, path)
    print(f"\nwrote {os.path.relpath(path, HERE)}")
    return path


def load_all(exp):
    out = {}
    if not os.path.isdir(RESULTS):
        return out
    for f in sorted(os.listdir(RESULTS)):
        m = re.fullmatch(rf"{re.escape(exp)}_(.+)\.pt", f)
        if m:
            out[m.group(1)] = torch.load(os.path.join(RESULTS, f),
                                         weights_only=False)
    return out


ARCH_ORDER = ["a100", "l40s", "h100"]


def arch_sorted(keys):
    known = [a for a in ARCH_ORDER if a in keys]
    return known + sorted(k for k in keys if k not in ARCH_ORDER)


def arch_pairs(keys):
    ks = arch_sorted(keys)
    return [(ks[i], ks[j]) for i in range(len(ks)) for j in range(i + 1, len(ks))]


def base_argparser(desc):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset of case names (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="four representative cases, for a fast smoke run")
    return ap
