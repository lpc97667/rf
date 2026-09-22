# Extracts timing from logs into .csv files.

# python timing_to_csv.py <log> [<log> ...]
# python timing_to_csv.py *-timing-times.log # merges per GPU


import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(?P<time>[0-9.]+)s\]\s+"
    r"(?:RF_FUSED=(?P<fused>[01])\s+)?"
    r"python\s+(?P<script>eval_\w+\.py)\b")
OPT_RE = {k: re.compile(rf"--{k}\s+(\S+)")
          for k in ("model", "task", "seed", "batch_size", "max_tokens", "dtype")}
GPU_RE = re.compile(r"(a100|l40s|h100)", re.IGNORECASE)

def resolve_method(fused, dtype):
    if fused == "1":
        return "rf"
    if fused == "0":
        return "layercast"
    if dtype == "bfloat16":
        return "bf16"
    if dtype == "float32":
        return "fp32"
    return None

METHODS = ["bf16", "layercast", "rf"]


def parse_log(path):
    rows = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            if line.startswith("["):
                print(f"  warn: {path}:{lineno} unrecognized timing line, skipped",
                      file=sys.stderr)
            continue
        opts = {k: (rx.search(line).group(1) if rx.search(line) else "")
                for k, rx in OPT_RE.items()}
        method = resolve_method(m["fused"], opts["dtype"])
        if method is None:
            print(f"  warn: {path}:{lineno} cannot resolve method "
                  f"(fused={m['fused']}, dtype={opts['dtype']}), skipped",
                  file=sys.stderr)
            continue
        if not (opts["model"] and opts["task"]):
            raise ValueError(f"{path}:{lineno} missing --model/--task: {line}")
        rows.append({"time": float(m["time"]), "method": method, **opts})
    return rows


def gpu_from_name(path):
    m = GPU_RE.search(Path(path).name)
    return m.group(1).lower() if m else Path(path).stem


def aggregate(rows):
    latest = {}
    for r in rows:
        latest[(r["model"], r["task"], r["method"], r["seed"])] = r["time"]

    groups = defaultdict(list)
    for (model, task, method, seed), t in latest.items():
        groups[(model, task, method)].append((seed, t))
    configs = defaultdict(dict)
    for (model, task, method), pairs in groups.items():
        times = [t for _, t in pairs]
        configs[(model, task)][method] = {
            "n": len(times),
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "seeds": ",".join(s for s, _ in sorted(pairs)),
        }
    return configs


def write_csv(out_path, gpu, configs):
    fields = ["gpu", "model", "task", "seeds", "n_seeds"]
    for meth in METHODS:
        fields += [f"{meth}_mean_s", f"{meth}_std_s", f"{meth}_n"]
    fields += ["speedup", "time_saved_s", "pct_faster",
               "rf_overhead_vs_bf16", "layercast_overhead_vs_bf16"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, task) in sorted(configs):
            methods = configs[(model, task)]
            row = {"gpu": gpu, "model": model, "task": task}
            present = [methods[k] for k in METHODS if k in methods]
            row["seeds"] = max((p["seeds"] for p in present),
                               key=lambda s: len(s.split(",")), default="")
            row["n_seeds"] = max((p["n"] for p in present), default="")
            for meth in METHODS:
                st = methods.get(meth)
                row[f"{meth}_mean_s"] = f"{st['mean']:.2f}" if st else ""
                row[f"{meth}_std_s"] = f"{st['std']:.2f}" if st else ""
                row[f"{meth}_n"] = st["n"] if st else ""
            rf, lc, bf = methods.get("rf"), methods.get("layercast"), methods.get("bf16")
            if rf and lc:
                sp = lc["mean"] / rf["mean"]
                row["speedup"] = f"{sp:.3f}"
                row["time_saved_s"] = f"{lc['mean'] - rf['mean']:.2f}"
                row["pct_faster"] = f"{(sp - 1) * 100:.1f}"
            else:
                row["speedup"] = row["time_saved_s"] = row["pct_faster"] = ""
            row["rf_overhead_vs_bf16"] = f"{rf['mean'] / bf['mean']:.3f}" if (rf and bf) else ""
            row["layercast_overhead_vs_bf16"] = f"{lc['mean'] / bf['mean']:.3f}" if (lc and bf) else ""
            w.writerow(row)
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="*-timing-times.log files")
    args = ap.parse_args()

    by_gpu = defaultdict(list)
    for log in args.logs:
        by_gpu[gpu_from_name(log)].append(log)

    for gpu, logs in sorted(by_gpu.items()):
        rows = []
        for log in logs:
            rows += parse_log(log)
        if not rows:
            print(f"{gpu}: no usable lines from {logs}", file=sys.stderr)
            continue
        configs = aggregate(rows)
        out_path = Path(logs[0]).with_name(f"{gpu}-timing-comparison.csv")
        write_csv(out_path, gpu, configs)

        print(f"\n{gpu}: {len(rows)} runs from {len(logs)} log(s) -> {out_path.name}")
        for (model, task) in sorted(configs):
            m = configs[(model, task)]
            parts = []
            for meth in METHODS:
                if meth in m:
                    parts.append(f"{meth} {m[meth]['mean']:7.1f}s")
            extra = ""
            if "rf" in m and "layercast" in m:
                extra = f"  [{m['layercast']['mean'] / m['rf']['mean']:.2f}x vs LC]"
            print(f"  {model.split('/')[-1]:26s} {task:13s}  "
                  + "  ".join(parts) + extra)


if __name__ == "__main__":
    main()
