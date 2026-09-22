#!/usr/bin/env python

# Cross-architecture analysis of the results from a1 to a4.

# python analyze.py


import argparse
import csv
import os
import statistics

import ablation_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
ARCH_TEX = {"a100": "A100", "l40s": "L40S", "h100": "H100"}


def head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def tex_arch(a):
    return ARCH_TEX.get(a, a.upper())


def write_csv(name, fieldnames, rows):
    os.makedirs(TABLES, exist_ok=True)
    p = os.path.join(TABLES, name)
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    print(f"  wrote {os.path.relpath(p, HERE)}")


def write_tex(name, body):
    os.makedirs(TABLES, exist_ok=True)
    p = os.path.join(TABLES, name)
    with open(p, "w") as f:
        f.write(body if body.endswith("\n") else body + "\n")
    print(f"  wrote {os.path.relpath(p, HERE)}")


def xarch(archs, what):
    if len(archs) > 1:
        return True
    print(f"  {what}: skipped, only one architecture present")
    return False


def require(dumps, exp):
    if not dumps:
        print(f"  no results/{exp}_*.pt found; run {exp} on each GPU first")
        return False
    print(f"  architectures present: {', '.join(C.arch_sorted(dumps))}")
    for a in C.arch_sorted(dumps):
        e = dumps[a]["env"]
        print(f"    {a:5s} {e['device_name']}  sm_{e['capability'].replace('.', '')}"
              f"  {e['sm_count']} SMs  triton {e['triton']}")
    if len(dumps) < 2:
        print("  (only one architecture: cross-architecture claims are skipped)")
    return True



def analyze_a1():
    head("A1 / R1: IEEE FMA only")
    dumps = C.load_all("a1")
    if not require(dumps, "a1"):
        return
    archs = C.arch_sorted(dumps)
    variants = dumps[archs[0]]["variants"]
    cases = list(dumps[archs[0]]["cases"])

    print("\n(1) Cross-architecture bitwise agreement, per precision setting")
    rows, worst = [], {}
    multi = xarch(archs, "bitwise agreement across GPUs")
    for v in variants:
        ident = diff = 0
        wr = 0.0
        for case in cases:
            fps = [dumps[a]["cases"][case]["variants"].get(v) for a in archs]
            if any(f is None or "sha256" not in f for f in fps):
                continue
            if len({f["sha256"] for f in fps}) == 1:
                ident += 1
                continue
            diff += 1
            for a, bnd in C.arch_pairs(archs):
                r = C.cmp_fingerprints(dumps[a]["cases"][case]["variants"][v],
                                       dumps[bnd]["cases"][case]["variants"][v])
                if not r["bitwise"]:
                    wr = max(wr, r["max_rel_scaled"])
        worst[v] = wr
        errs = [dumps[a]["cases"][c]["variants"][v]["max_rel_scaled"]
                for a in archs for c in cases
                if "max_rel_scaled" in dumps[a]["cases"][c]["variants"].get(v, {})]
        ts = [dumps[a]["cases"][c]["variants"][v]["ms"]
              / dumps[a]["cases"][c]["variants"]["ieee"]["ms"]
              for a in archs for c in cases
              if "ms" in dumps[a]["cases"][c]["variants"].get(v, {})]
        rows.append({"variant": v, "cases_bitwise_identical": ident,
                     "cases_divergent": diff,
                     "worst_cross_arch_rel": f"{wr:.3e}" if wr else "0",
                     "max_rel_err_vs_fp64": f"{max(errs):.3e}" if errs else "",
                     "median_time_vs_ieee": f"{statistics.median(ts):.3f}" if ts else ""})
        print(f"  {v:8s} "
              + (f"identical on {ident}/{ident + diff} cases" if multi
                 else f"{ident + diff} cases (single GPU: no comparison)")
              + (f", worst cross-arch scaled diff {wr:.2e}" if diff else "")
              + (f", max err vs fp64 {max(errs):.2e}" if errs else "")
              + (f", {statistics.median(ts):.2f}x the ieee time" if ts else ""))

    print("\n(2) Does the framework TF32 switch reach Triton?"
          "  (allow_tf32=False was set on every node)")
    for a in archs:
        cs = dumps[a]["cases"]
        n = sum(1 for c in cs
                if cs[c]["variants"].get("default", {}).get("sha256")
                == cs[c]["variants"].get("tf32", {}).get("sha256"))
        m = sum(1 for c in cs
                if cs[c]["variants"].get("default", {}).get("sha256")
                == cs[c]["variants"].get("ieee", {}).get("sha256"))
        print(f"  {a:5s} omitting input_precision == tf32 on {n}/{len(cs)} cases, "
              f"== ieee on {m}/{len(cs)} cases")

    print("\n(3) Per-case detail")
    print(f"  {'case':16s} {'variant':8s} " +
          " ".join(f"{a:>10s}" for a in archs) + "   bitwise")
    detail = []
    for case in cases:
        for v in variants:
            fps = [dumps[a]["cases"][case]["variants"].get(v, {}) for a in archs]
            if any("sha256" not in f for f in fps):
                continue
            ok = len({f["sha256"] for f in fps}) == 1
            print(f"  {case:16s} {v:8s} " +
                  " ".join(f"{f['sha256'][:10]:>10s}" for f in fps) +
                  (("   identical" if ok else "   DIFFER") if multi else "   n/a"))
            detail.append({"case": case, "variant": v, "identical": ok,
                           **{a: f["sha256"][:16] for a, f in zip(archs, fps)}})

    write_csv("a1_precision.csv",
              ["variant", "cases_bitwise_identical", "cases_divergent",
               "worst_cross_arch_rel", "max_rel_err_vs_fp64",
               "median_time_vs_ieee"], rows)
    write_csv("a1_detail.csv", ["case", "variant", "identical"] + archs, detail)

    label = {"ieee": r"\texttt{ieee} (R1)", "tf32": r"\texttt{tf32}",
             "tf32x3": r"\texttt{tf32x3}", "default": "argument omitted"}
    body = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            r"\texttt{input\_precision} & Bitwise-identical & Worst cross-GPU "
            r"& Max error vs.\ & Time vs.\ \\",
            r" & across GPUs & difference & fp64 & \texttt{ieee} \\", r"\midrule"]
    for r in rows:
        tot = r["cases_bitwise_identical"] + r["cases_divergent"]
        wr = "--" if r["worst_cross_arch_rel"] == "0" else f"${_sci(r['worst_cross_arch_rel'])}$"
        body.append(f"{label.get(r['variant'], r['variant'])} & "
                    f"{r['cases_bitwise_identical']}/{tot} & {wr} & "
                    f"${_sci(r['max_rel_err_vs_fp64'])}$ & "
                    f"${r['median_time_vs_ieee']}\\times$ \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_tex("a1_precision.tex", "\n".join(body))


def _sci(s):
    """'4.348e-04' -> '4.3\\times10^{-4}' for LaTeX math mode."""
    if not s or s == "0":
        return "0"
    m, e = f"{float(s):.1e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"



def analyze_a2():
    head("A2 / R2: no autotuning")
    dumps = C.load_all("a2")
    if not require(dumps, "a2"):
        return
    archs = C.arch_sorted(dumps)
    cases = [c for c in dumps[archs[0]]["cases"]
             if all(c in dumps[a]["cases"] for a in archs)]

    print("\n(1) What the autotuner selects, per architecture")
    print(f"  {'case':16s} " + " ".join(f"{a:>26s}" for a in archs)
          + "  same choice?")
    rows = []
    n_same = 0
    for case in cases:
        wins = [dumps[a]["cases"][case]["winners"][0] for a in archs]
        same = len(set(wins)) == 1
        n_same += same
        print(f"  {case:16s} " + " ".join(f"{w:>26s}" for w in wins)
              + (("  yes" if same else "  NO") if len(archs) > 1 else "  n/a"))
        rows.append({"case": case, "same_winner_across_archs": same,
                     **{f"winner_{a}": w for a, w in zip(archs, wins)}})
    if xarch(archs, "agreement of the selected configuration"):
        print(f"  -> the same configuration wins on all {len(archs)} "
              f"architectures for {n_same}/{len(cases)} shapes")

    print("\n(2) Is the search even stable on one device?"
          f"  ({dumps[archs[0]]['repeats']} independent searches per node)")
    unstable = 0
    for a in archs:
        u = [c for c in cases if len(set(dumps[a]["cases"][c]["winners"])) > 1]
        unstable += len(u)
        print(f"  {a:5s} winner flips between repeats on {len(u)}/{len(cases)} "
              f"shapes" + (f": {', '.join(u)}" if u else ""))

    print("\n(3) Consequences for the bits")
    print(f"  {'case':16s} {'classes':>8s} {'winner==pinned':>15s} "
          f"{'pinned identical':>17s} {'autotuned identical':>20s}")
    bitrows = []
    for case in cases:
        classes = max(dumps[a]["cases"][case]["n_digest_classes"] for a in archs)
        pin = dumps[archs[0]]["cases"][case]["pinned"]
        pin_d = {dumps[a]["cases"][case]["digests"].get(pin) for a in archs}
        auto_d = {dumps[a]["cases"][case]["digests"].get(
            dumps[a]["cases"][case]["winners"][0]) for a in archs}
        pin_ok, auto_ok = len(pin_d) == 1, len(auto_d) == 1
        local = [dumps[a]["cases"][case]["digests"].get(
            dumps[a]["cases"][case]["winners"][0])
            == dumps[a]["cases"][case]["digests"].get(pin) for a in archs]
        print(f"  {case:16s} {classes:8d} "
              f"{sum(local)}/{len(archs)} GPUs".rjust(16)
              + f"{('yes' if pin_ok else 'NO') if len(archs) > 1 else 'n/a':>17s} "
              f"{('yes' if auto_ok else 'NO') if len(archs) > 1 else 'n/a':>20s}")
        bitrows.append({"case": case, "n_digest_classes": classes,
                        "winner_matches_pinned_on_n_gpus": sum(local),
                        "pinned_bitwise_across_archs": pin_ok,
                        "autotuned_bitwise_across_archs": auto_ok})
    n_local = sum(len(archs) - r["winner_matches_pinned_on_n_gpus"]
                  for r in bitrows)
    print(f"  -> on {n_local}/{len(cases) * len(archs)} (shape, GPU) cells the "
          f"autotuned choice already disagrees bitwise with the pinned one on "
          f"that same GPU")
    n_pin = sum(r["pinned_bitwise_across_archs"] for r in bitrows)
    n_auto = sum(r["autotuned_bitwise_across_archs"] for r in bitrows)
    if xarch(archs, "bitwise agreement of pinned vs autotuned outputs"):
        print(f"  -> pinned: {n_pin}/{len(cases)} shapes bitwise identical "
              f"across architectures; autotuned: {n_auto}/{len(cases)}")
    print("     ('classes' = distinct outputs the candidate space can produce "
          "on one device; >1 means the tuner's choice is a numerical choice)")

    print("\n(4) What R2 costs: pinned time / best found time")
    costrows = []
    for case in cases:
        cells = []
        for a in archs:
            d = dumps[a]["cases"][case]
            t = d["times"][0]
            ratio = t[d["pinned"]] / t[d["winners"][0]]
            cells.append(ratio)
        print(f"  {case:16s} " + "  ".join(f"{a}={r:.3f}x"
                                           for a, r in zip(archs, cells)))
        costrows.append({"case": case,
                         **{f"pinned_over_best_{a}": round(r, 4)
                            for a, r in zip(archs, cells)}})
    allr = [v for r in costrows if r["case"] not in C.SYNTHETIC_CASES
            for k, v in r.items() if k != "case"]
    if allr:
        print(f"  -> over served shapes, pinning costs "
              f"{100 * (statistics.median(allr) - 1):.1f}% at the median, "
              f"{100 * (max(allr) - 1):.1f}% at worst")
    for r in costrows:
        if r["case"] in C.SYNTHETIC_CASES:
            v = [x for k, x in r.items() if k != "case"]
            print(f"  -> {r['case']} (synthetic mask-path probe, not a served "
                  f"shape) is the outlier at {max(v):.1f}x: the pinned prefill "
                  f"tile leaves a shape this small far too little parallelism")

    print("\n(5) If precision were also in the search space")
    for a in archs:
        picks = {}
        for case in cases:
            p = dumps[a]["cases"][case].get("prec_winner")
            if p:
                picks[p] = picks.get(p, 0) + 1
        print(f"  {a:5s} fastest precision per shape: "
              + ", ".join(f"{k} on {v}" for k, v in sorted(picks.items())))

    write_csv("a2_winners.csv",
              ["case", "same_winner_across_archs"] + [f"winner_{a}" for a in archs],
              rows)
    write_csv("a2_bits.csv",
              ["case", "n_digest_classes", "winner_matches_pinned_on_n_gpus",
               "pinned_bitwise_across_archs",
               "autotuned_bitwise_across_archs"], bitrows)
    write_csv("a2_cost.csv",
              ["case"] + [f"pinned_over_best_{a}" for a in archs], costrows)

    body = [r"\begin{tabular}{@{}l" + "c" * len(archs) + r"c@{}}", r"\toprule",
            r" & \multicolumn{%d}{c}{Configuration chosen by autotuning} & "
            r"Bitwise across GPUs \\" % len(archs),
            r"\cmidrule(lr){2-%d}" % (len(archs) + 1),
            r"Shape & " + " & ".join(tex_arch(a) for a in archs)
            + r" & pinned / autotuned \\", r"\midrule"]
    for r, br in zip(rows, bitrows):
        cells = " & ".join(r[f"winner_{a}"].replace("_", r"\_") for a in archs)
        body.append(f"\\texttt{{{r['case'].replace('_', chr(92) + '_')}}} & {cells} & "
                    f"{'yes' if br['pinned_bitwise_across_archs'] else 'no'} / "
                    f"{'yes' if br['autotuned_bitwise_across_archs'] else 'no'} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_tex("a2_autotune.tex", "\n".join(body))



def analyze_a3():
    head("A3 / R3: deterministic split-K")
    dumps = C.load_all("a3")
    if not require(dumps, "a3"):
        return
    archs = C.arch_sorted(dumps)
    cases = [c for c in dumps[archs[0]]["cases"]
             if all(c in dumps[a]["cases"] for a in archs)]
    runs = dumps[archs[0]]["runs"]

    print(f"\n(1) Run-to-run determinism on a fixed device ({runs} runs)")
    print(f"  {'case':16s} {'arch':5s} " +
          " ".join(f"{v:>16s}" for v in ("none", "det", "atomic")))
    rr = []
    for case in cases:
        for a in archs:
            v = dumps[a]["cases"][case]["variants"]
            print(f"  {case:16s} {a:5s} " +
                  " ".join(f"{v[k]['n_distinct']:>4d} distinct"
                           f"{'':>2s}" for k in ("none", "det", "atomic")))
            rr.append({"case": case, "arch": a,
                       **{f"{k}_distinct": v[k]["n_distinct"]
                          for k in ("none", "det", "atomic")},
                       "atomic_max_spread_scaled":
                           f"{v['atomic']['max_spread_scaled']:.3e}"})
    worst = max(float(r["atomic_max_spread_scaled"]) for r in rr)
    n_atomic_bad = sum(r["atomic_distinct"] > 1 for r in rr)
    print(f"  -> atomic split-K produced more than one result on "
          f"{n_atomic_bad}/{len(rr)} (case, GPU) cells; "
          f"worst run-to-run spread {worst:.2e} of the output scale")
    print(f"  -> ordered split-K and no-split produced exactly one result "
          f"everywhere: "
          f"{all(r['det_distinct'] == 1 and r['none_distinct'] == 1 for r in rr)}")

    print("\n(2) Cross-architecture bitwise agreement")
    if xarch(archs, "bitwise agreement across GPUs"):
        for v in ("none", "det", "atomic"):
            ident = sum(len({dumps[a]["cases"][c]["variants"][v]["sha256"]
                             for a in archs}) == 1 for c in cases)
            print(f"  {v:7s} identical on {ident}/{len(cases)} shapes")

    print("\n(3) Must the split factor be a pure function of shape?")
    print(f"  {'case':16s} " + " ".join(f"{a + ' S':>10s}" for a in archs)
          + f"  {'distinct outputs over S sweep':>32s}  occupancy rule bitwise")
    srows = []
    for case in cases:
        socc = [dumps[a]["cases"][case]["S_occupancy"] for a in archs]
        occ_d = {dumps[a]["cases"][case]["s_sweep"][s]["sha256"]
                 for a, s in zip(archs, socc)
                 if s in dumps[a]["cases"][case]["s_sweep"]}
        nd = max(dumps[a]["cases"][case]["s_sweep_distinct"] for a in archs)
        ok = len(occ_d) == 1 and len(occ_d) > 0
        print(f"  {case:16s} " + " ".join(f"{s:>10d}" for s in socc)
              + f"  {nd:>32d}  "
              + (("yes" if ok else "NO") if len(archs) > 1 else "n/a"))
        srows.append({"case": case, "n_distinct_over_S_sweep": nd,
                      "occupancy_bitwise_across_archs": ok,
                      **{f"S_occupancy_{a}": s for a, s in zip(archs, socc)},
                      **{f"S_policy_{a}": dumps[a]["cases"][case]["S_policy"]
                         for a in archs}})
    print("  (the shape-pure rule picks "
          + ", ".join(f"{a}: S={dumps[a]['cases'][cases[0]]['S_policy']}"
                      for a in archs) + " on the first shape, by construction)")

    print("\n(4) Is split-K worth having, and what does ordering it cost?")
    print(f"  {'case':16s} {'arch':5s} {'none (ms)':>10s} {'det (ms)':>9s} "
          f"{'atomic (ms)':>12s} {'det speedup':>12s} {'det vs atomic':>14s}")
    trows = []
    for case in cases:
        for a in archs:
            v = dumps[a]["cases"][case]["variants"]
            sp = v["none"]["ms"] / v["det"]["ms"]
            oa = v["det"]["ms"] / v["atomic"]["ms"]
            print(f"  {case:16s} {a:5s} {v['none']['ms']:10.3f} "
                  f"{v['det']['ms']:9.3f} {v['atomic']['ms']:12.3f} "
                  f"{sp:11.2f}x {oa:13.2f}x")
            trows.append({"case": case, "arch": a,
                          "none_ms": round(v["none"]["ms"], 4),
                          "det_ms": round(v["det"]["ms"], 4),
                          "atomic_ms": round(v["atomic"]["ms"], 4),
                          "det_speedup_over_nosplit": round(sp, 3),
                          "det_over_atomic": round(oa, 3)})
    sps = [r["det_speedup_over_nosplit"] for r in trows]
    oas = [r["det_over_atomic"] for r in trows]
    med = statistics.median(oas)
    verb = (f"is {100 * (1 - med):.1f}% faster than" if med < 1
            else f"costs {100 * (med - 1):.1f}% against")
    print(f"  -> split-K is worth {min(sps):.2f}--{max(sps):.2f}x over no split; "
          f"the ordered reduction {verb} atomics at the median "
          f"(det/atomic {min(oas):.2f}--{max(oas):.2f}x)")

    write_csv("a3_runtorun.csv",
              ["case", "arch", "none_distinct", "det_distinct", "atomic_distinct",
               "atomic_max_spread_scaled"], rr)
    write_csv("a3_splitfactor.csv",
              ["case", "n_distinct_over_S_sweep", "occupancy_bitwise_across_archs"]
              + [f"S_occupancy_{a}" for a in archs]
              + [f"S_policy_{a}" for a in archs], srows)
    write_csv("a3_timing.csv",
              ["case", "arch", "none_ms", "det_ms", "atomic_ms",
               "det_speedup_over_nosplit", "det_over_atomic"], trows)

    body = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
            r"Shape & GPU & \multicolumn{3}{c}{Distinct outputs over %d runs} \\"
            % runs, r"\cmidrule(l){3-5}",
            r" & & no split & ordered (R3) & atomic \\", r"\midrule"]
    for r in rr:
        body.append(f"\\texttt{{{r['case'].replace('_', chr(92) + '_')}}} & "
                    f"{tex_arch(r['arch'])} & {r['none_distinct']} & "
                    f"\\textbf{{{r['det_distinct']}}} & {r['atomic_distinct']} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_tex("a3_splitk.tex", "\n".join(body))



def analyze_a4():
    head("A4 / R4: batch invariance by construction")
    dumps = C.load_all("a4")
    if not require(dumps, "a4"):
        return
    archs = C.arch_sorted(dumps)
    cases = [c for c in dumps[archs[0]]["cases"]
             if all(c in dumps[a]["cases"] for a in archs)]
    msweep = dumps[archs[0]]["m_sweep"]
    thr = dumps[archs[0]]["bucket_threshold"]
    in_bucket = [m for m in msweep if m <= thr]
    policies = list(dumps[archs[0]]["cases"][cases[0]]["policies"])

    print(f"\n(1) Row 0 vs the M=1 result, batch sizes M <= {thr}")
    print(f"  {'case':16s} {'arch':5s} " +
          " ".join(f"{p:>12s}" for p in policies))
    rows = []
    for case in cases:
        for a in archs:
            pol = dumps[a]["cases"][case]["policies"]
            cells = []
            for p in policies:
                n = pol[p]["n_varying_in_bucket"]
                cells.append("invariant" if n == 0 else f"{n}/{len(in_bucket)} differ")
            print(f"  {case:16s} {a:5s} " + " ".join(f"{c:>12s}" for c in cells))
            rows.append({"case": case, "arch": a,
                         **{f"{p}_varying": pol[p]["n_varying_in_bucket"]
                            for p in policies}})
    for p in policies:
        n_ok = sum(r[f"{p}_varying"] == 0 for r in rows)
        print(f"  -> {p:10s} batch-invariant on {n_ok}/{len(rows)} (shape, GPU) cells")

    print(f"\n(2) Where each policy first breaks (M > {thr} crosses the shape "
          f"bucket and is outside R4's claim)")
    for case in cases:
        for a in archs:
            pol = dumps[a]["cases"][case]["policies"]
            for p in policies:
                r = pol[p]["rows"]
                ref = r[msweep[0]]["sha256"]
                first = next((m for m in msweep if r[m]["sha256"] != ref), None)
                if first is not None:
                    print(f"  {case:16s} {a:5s} {p:10s} first differs at M={first}"
                          + (f"  (S: {r[msweep[0]]['S']} -> {r[first]['S']})"
                             if r[first].get("S") is not None else ""))

    print("\n(3) Cross-architecture: is row 0 at M=32 the same on every GPU?")
    xrows = []
    for case in cases if xarch(archs, "row 0 across GPUs") else []:
        cells = {}
        for p in policies:
            ds = {dumps[a]["cases"][case]["policies"][p]["rows"][32]["sha256"]
                  for a in archs}
            cells[p] = len(ds) == 1
        print(f"  {case:16s} " + "  ".join(
            f"{p}={'yes' if v else 'NO'}" for p, v in cells.items()))
        xrows.append({"case": case, **{f"{p}_bitwise_across_archs": v
                                       for p, v in cells.items()}})

    print("\n(4) What R4 costs: batch-tuned time / shape-pure time")
    crows = []
    for case in cases:
        for a in archs:
            row = {"case": case, "arch": a}
            for other in ("occupancy", "autotuned"):
                rt = dumps[a]["cases"][case][f"{other}_over_rf"]
                vals = [rt[m] for m in in_bucket]
                row[f"median_{other}_over_rf"] = round(statistics.median(vals), 4)
                row[f"best_{other}_over_rf"] = round(min(vals), 4)
            print(f"  {case:16s} {a:5s} " + "  ".join(
                f"{o}: median {row[f'median_{o}_over_rf']:.3f}x, "
                f"best {row[f'best_{o}_over_rf']:.3f}x"
                for o in ("occupancy", "autotuned")))
            crows.append(row)
    allv = [r["best_autotuned_over_rf"] for r in crows]
    print(f"  -> re-tuning at every batch size would gain at most "
          f"{100 * (1 - min(allv)):.1f}% on any (shape, GPU) cell, and loses "
          f"time on cells above 1.00x")

    write_csv("a4_invariance.csv",
              ["case", "arch"] + [f"{p}_varying" for p in policies], rows)
    write_csv("a4_crossarch.csv",
              ["case"] + [f"{p}_bitwise_across_archs" for p in policies], xrows)
    write_csv("a4_cost.csv",
              ["case", "arch"] + [f"{p}_{o}_over_rf"
                                  for o in ("occupancy", "autotuned")
                                  for p in ("median", "best")], crows)

    plabel = {"rf": r"\sysname (R4)", "occupancy": "occupancy-sized split",
              "autotuned": "autotuned per batch size",
              "cublas": "cuBLAS (LayerCast)"}
    body = [r"\begin{tabular}{@{}ll" + "c" * len(policies) + r"@{}}", r"\toprule",
            r"Shape & GPU & " + " & ".join(plabel.get(p, p) for p in policies)
            + r" \\", r"\midrule"]
    for r in rows:
        cells = " & ".join(
            (r"\textbf{0}" if r[f"{p}_varying"] == 0 else str(r[f"{p}_varying"]))
            for p in policies)
        body.append(f"\\texttt{{{r['case'].replace('_', chr(92) + '_')}}} & "
                    f"{tex_arch(r['arch'])} & {cells} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_tex("a4_batch_invariance.tex", "\n".join(body))



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["a1", "a2", "a3", "a4"])
    args = ap.parse_args()
    todo = args.only or ["a1", "a2", "a3", "a4"]
    for k in todo:
        {"a1": analyze_a1, "a2": analyze_a2,
         "a3": analyze_a3, "a4": analyze_a4}[k]()


if __name__ == "__main__":
    main()
