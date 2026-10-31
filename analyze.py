#!/usr/bin/env python3
"""Turn raw CSVs into the numbers, tables and figure used in Section 5 of the C&OR manuscript."""
import csv
import glob
import json
import sys
from collections import defaultdict

import numpy as np

FILES = sys.argv[1:] or sorted(glob.glob("main_*_raw.csv"))
rows = []
for f in FILES:
    for r in csv.DictReader(open(f)):
        for c in r:
            if c not in ("size", "regime", "weights", "method"):
                r[c] = float(r[c])
        rows.append(r)

METHODS = ["full", "prune_only", "price_s2", "price_both", "screened", "proposed"]
NAMES = {"full": "Full labelling", "prune_only": "Prune-only", "price_s2": "Stage-2 pricing",
         "price_both": "Two-stage pricing", "screened": "Screened pricing (Prop.~\\ref{prop:screen})",
         "proposed": "OLA (Alg.~\\ref{alg:ola})"}
W = ["high_physical", "balanced", "high_humanistic"]
WN = {"high_physical": "High physical", "balanced": "Balanced", "high_humanistic": "High humanistic"}


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def exact(r):
    return r["s1_bias"] == 0 and abs(r["s2_gap_pct"]) < 1e-6


out = {}
# ---- global exactness / counts
for m in ["price_both", "screened", "proposed"]:
    rs = sel(method=m)
    out[f"{m}_runs"] = len(rs)
    out[f"{m}_all_exact"] = all(exact(r) for r in rs)
p = sel(method="proposed")
out["ola_labels_pct_min"] = min(r["labels_pct"] for r in p)
out["ola_labels_pct_max"] = max(r["labels_pct"] for r in p)
out["ola_ratio_min"] = min(r["labels_per_assign"] for r in p)
out["ola_ratio_max"] = max(r["labels_per_assign"] for r in p)
out["ola_ratio_mean"] = float(np.mean([r["labels_per_assign"] for r in p]))
out["K_mean_nofeas"] = float(np.mean([r["K"] for r in sel(method="full", regime="nofeas")]))
out["K_mean_feas60"] = float(np.mean([r["K"] for r in sel(method="full", regime="feas60")]))
out["seeds"] = sorted({int(r["seed"]) for r in rows})
for reg in ["nofeas", "feas60"]:
    for w in W:
        rs = sel(method="proposed", regime=reg, weights=w)
        if rs:
            out[f"ola_{reg}_{w}_pct"] = float(np.mean([r["labels_pct"] for r in rs]))
            out[f"ola_{reg}_{w}_ratio"] = float(np.mean([r["labels_per_assign"] for r in rs]))
            out[f"ola_{reg}_{w}_acq"] = float(np.mean([r["acq_rounds"] for r in rs]))
            out[f"ola_{reg}_{w}_time"] = float(np.mean([r["time_s"] for r in rs]))
# phantom shortage under stage-2 pricing (feas60)
for T in [12, 18, 24, 30, 45]:
    rs = sel(method="price_s2", regime="feas60", t_prune=T, weights="balanced")
    if rs:
        out[f"phantom_feas60_T{T}_max"] = max(r["s1_bias"] for r in rs)
        out[f"phantom_feas60_T{T}_med"] = float(np.median([r["s1_bias"] for r in rs]))
fs = sel(method="full")
out["full_time_mean"] = float(np.mean([r["time_s"] for r in fs]))
json.dump(out, open("analysis_summary.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(f"{k:32s} {v}")


# ---- Table: main comparison at T = 30, both regimes
def cell(rs, f, fmt):
    return fmt.format(np.mean([f(r) for r in rs])) if rs else "---"


lines = []
for reg, regname in [("nofeas", "Unrestricted"), ("feas60", "60-min feasibility limit")]:
    lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{regname}}}}} \\\\")
    for m in METHODS:
        if m == "full":
            rs_all = sel(method="full", regime=reg)
            tm = cell(rs_all, lambda r: r["time_s"], "{:.1f}")
            q = [cell(sel(method="full", regime=reg, weights=w), lambda r: r["labels_pct"], "{:.1f}") for w in W]
            ratio = cell(rs_all, lambda r: r["labels_per_assign"], "{:.1f}")
            bias, gap = "0", "0"
        else:
            rs_all = sel(method=m, regime=reg, t_prune=30.0)
            if not rs_all:
                continue
            q = [cell(sel(method=m, regime=reg, t_prune=30.0, weights=w), lambda r: r["labels_pct"], "{:.1f}") for w in W]
            ratio = cell(rs_all, lambda r: r["labels_per_assign"], "{:.1f}")
            tm = cell(rs_all, lambda r: r["time_s"], "{:.1f}")
            bias = f"{max(r['s1_bias'] for r in rs_all):.0f}"
            gap = f"{max(abs(r['s2_gap_pct']) for r in rs_all):.2f}"
        lines.append(f"{NAMES[m]} & {' & '.join(q)} & {ratio} & {bias} & {gap} & {tm} \\\\")
    lines.append("\\hline")
open("tab_main.tex", "w").write("\n".join(lines) + "\n")

# ---- Figure data: labels (% of E) vs threshold, unrestricted regime
fig = {}
for w in W:
    for m in ["prune_only", "screened", "proposed"]:
        pts = []
        for T in ([12, 18, 24, 30, 45, 60] if m == "prune_only" else [0, 12, 18, 24, 30, 45, 60]):
            rs = sel(method=m, regime="nofeas", weights=w, t_prune=float(T))
            if rs:
                pts.append((T, float(np.mean([r["labels_pct"] for r in rs]))))
        fig[f"{w}:{m}"] = pts
json.dump(fig, open("fig_threshold.json", "w"), indent=1)
print("\nwrote tab_main.tex, fig_threshold.json, analysis_summary.json")
