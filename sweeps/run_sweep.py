#!/usr/bin/env python
#Timed sweeps on the local GPU: end-to-end evaluation, and memory profiling.

# python sweeps/run_sweep.py [--tasks ...] [--methods ...] [--seeds ...] [--mode ...]

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODELS = ["meta-llama/Llama-3.2-3B-Instruct",
          "Qwen/Qwen3-4B-Instruct-2507",
          "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
TASKS = ["gsm8k", "math500", "aime24", "gpqa_diamond"]
SEEDS = ["42", "1", "777"]

MEMORY_EXTRA_MODEL = {"a100": "Qwen/Qwen3-8B",
                      "l40s": "Qwen/Qwen3-8B",
                      "h100": "Qwen/Qwen3-14B"}

METHODS = {
    "rf": ("eval_layercast.py", "float32", "1"),
    "layercast": ("eval_layercast.py", "float32", "0"),
    "bf16": ("eval_main.py", "bfloat16", None),
}
METHOD_ORDER = ["rf", "layercast", "bf16"]

NO_RESUME_METHODS = {"bf16"}
MEMORY_METHODS = ["rf", "layercast"]

ARCH_BY_CC = {(8, 0): "a100", (8, 9): "l40s", (9, 0): "h100"}


def detect_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; pass --gpu to name it explicitly")
    cc = torch.cuda.get_device_capability(0)
    if cc not in ARCH_BY_CC:
        sys.exit(f"unrecognized compute capability sm{cc[0]}{cc[1]} "
                 f"({torch.cuda.get_device_name(0)}); pass --gpu explicitly")
    return ARCH_BY_CC[cc]


def build(cmd_opts, script, fused):
    prefix = f"RF_FUSED={fused} " if fused is not None else ""
    recorded = f"{prefix}python {script} " + " ".join(cmd_opts)
    env = {"RF_FUSED": fused} if fused is not None else {}
    return recorded, [sys.executable, script] + cmd_opts, env


def eval_plan(args, gpu):
    for method in [m for m in METHOD_ORDER if m in args.methods]:
        script, dtype, fused = METHODS[method]
        for seed in args.seeds:
            for model in args.models:
                for task in args.tasks:
                    opts = []
                    if args.no_resume and method in NO_RESUME_METHODS:
                        opts.append("--no_resume")
                    opts += ["--model", model, "--task", task,
                             "--dtype", dtype, "--seed", seed,
                             "--batch_size", str(args.batch_size),
                             "--max_tokens", str(args.max_tokens),
                             "--exp_name",
                             f"timing-{gpu}_{task}_timing_{method}-{seed}"]
                    yield build(opts, script, fused)


def memory_plan(args, gpu, models):
    for method in [m for m in MEMORY_METHODS if m in args.methods]:
        _, _, fused = METHODS[method]
        for model in models:
            yield build(["--model", model, "--exp_name", f"{gpu}-{method}"],
                        "profile_memory.py", fused)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["eval", "memory"], default="eval",
                    help="eval: timed end-to-end benchmark runs (default). "
                         "memory: weights / forward-spike / KV profiling.")
    ap.add_argument("--methods", nargs="+", choices=METHOD_ORDER,
                    default=METHOD_ORDER,
                    help="which methods to run; 'bf16' is the unmitigated "
                         "baseline via eval_main.py, and is skipped in "
                         "--mode memory (default: all three)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: the three paper models, plus one larger "
                         "model in --mode memory")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", default=SEEDS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu", choices=sorted(set(ARCH_BY_CC.values())),
                    help="override GPU autodetection")
    ap.add_argument("--tag", default="",
                    help="extra token in the log filenames, e.g. --tag rerun")
    ap.add_argument("--no-resume", action="store_true",
                    help=f"pass --no_resume (honored only by: "
                         f"{', '.join(sorted(NO_RESUME_METHODS))})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit without running them")
    args = ap.parse_args()

    os.chdir(REPO)
    gpu = args.gpu or detect_gpu()
    stamp = datetime.now().strftime("%m-%d-%H-%M")
    tag = f"-{args.tag}" if args.tag else ""
    kind = "timing" if args.mode == "eval" else "memory"
    log_path = Path(f"{stamp}-{gpu}{tag}-{kind}.log")
    times_path = Path(f"{stamp}-{gpu}{tag}-{kind}-times.log")

    memory_models = args.models or (MODELS + [MEMORY_EXTRA_MODEL[gpu]])
    args.models = args.models or MODELS
    plan = list(eval_plan(args, gpu) if args.mode == "eval"
                else memory_plan(args, gpu, memory_models))
    if not plan:
        sys.exit(f"nothing to run: no selected method applies to --mode {args.mode}")

    if args.no_resume and not (set(args.methods) & NO_RESUME_METHODS):
        print("note: --no-resume has no effect for the selected methods",
              file=sys.stderr)

    if args.dry_run:
        for recorded, _, _ in plan:
            print(recorded)
        print(f"\n{len(plan)} runs on '{gpu}'; logs would be {log_path} "
              f"and {times_path}", file=sys.stderr)
        return

    print(f"{len(plan)} runs on '{gpu}'  ->  {log_path}, {times_path}",
          file=sys.stderr)

    failures = []
    t_sweep = time.time()
    with open(log_path, "w") as log, open(times_path, "w") as times:
        for i, (recorded, argv, env_over) in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] {recorded}", file=sys.stderr, flush=True)
            log.write(f"@@@@ {recorded}\n")
            log.flush()

            start = time.time()
            rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, **env_over})
            elapsed = time.time() - start

            if rc == 0:
                times.write(f"[{elapsed:.2f}s] {recorded}\n")
                times.flush()
            else:
                failures.append((rc, recorded))
                msg = f"!!!! exit {rc} after {elapsed:.2f}s, not recorded"
                print(msg, file=sys.stderr, flush=True)
                log.write(f"{msg}\n")
                log.flush()

    mins = (time.time() - t_sweep) / 60
    print(f"\ndone: {len(plan) - len(failures)}/{len(plan)} runs in {mins:.1f} min",
          file=sys.stderr)
    for rc, recorded in failures:
        print(f"  FAILED (exit {rc}): {recorded}", file=sys.stderr)
    if args.mode == "eval":
        print(f"\nnext:  python timing_to_csv.py {times_path}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
