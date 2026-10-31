#!/usr/bin/env python3
"""
exp_label_pricing.py
====================
Comparative experiments of Section 5: exact lexicographic assignment when the
ordinal label R_ij in each arc cost must be obtained by a costly query.  Six
solution schemes are compared on the number of labels queried, running time
and exactness.

Setting
-------
Stage 2 minimizes C_ij = a_ij + gamma * U(R_ij), where a_ij is label-free.
Exact reduced-cost pricing of the excluded arcs needs
    cbar_ij = C_ij - u_i - v_j
and hence every label on E, so pruning saves computation but not labels.

Schemes
-------
  full        : label every arc of E and solve both stages on E (ground truth)
  prune_only  : label the pruned graph E' only, solve both stages on E',
                no pricing (heuristic)
  price_s2    : Stage 1 on E' without pricing, Stage 2 priced over E
                (common practice; needs every label; coverage is biased)
  price_both  : two-stage pricing (both stages priced over E); Stage 2
                pricing needs every label on E
  screened    : ablation. Stage 1 solved on E without labels; Stage 2 working
                set E' plus the Stage 1 support, bound-screened pricing
                (an arc with a_ij - u_i - v_j >= 0 is skipped without its label)
  proposed    : optimistic label acquisition (OLA, Algorithm 1). Stage 1 on E
                without labels; Stage 2 by column generation under optimistic
                costs (unknown labels priced at the lower bound), querying only
                unlabelled arcs in the support of the optimistic optimum, until
                the support is fully labelled. Exact.

Usage
-----
  # 1) smoke test on the built-in demo generator (not the paper's instances)
  python exp_label_pricing.py --demo --seeds 2 --thresholds 0 18 60

  # 2) check the regenerated City-X instances against published values
  python adapter_cor.py

  # 3) main experiment (default grid)
  python exp_label_pricing.py

  Output: <out>_raw.csv (one row per run, written incrementally),
          <out>_summary.csv and <out>_table.tex.

Requirements: numpy, scipy (scipy.optimize.linprog with HiGHS).
"""
import argparse
import csv
import math
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

TOL = 1e-9
K_UTIL = 0.2          # utility curvature k (paper: 0.2)
RHO = 0.15            # cross-district settlement penalty (paper reference value)
WEIGHTS = {           # (alpha, beta, gamma)
    "high_physical":   (0.8, 0.1, 0.1),
    "balanced":        (0.4, 0.3, 0.3),
    "high_humanistic": (0.1, 0.1, 0.8),
}


def utility(R, k=K_UTIL):
    return (np.exp(k * R) - 1.0) / (np.exp(10.0 * k) - 1.0)


# ============================================================================
# Instances: adapter_cor.build_instance for the paper; --demo uses the generator below
# ============================================================================
def demo_instance(seed, n_teachers=5236, n_schools=114, total_vac=2273):
    """City-X-like demo instance for smoke tests only; not used for any reported result."""
    rng = np.random.default_rng(seed)
    W, H = 48.0, 52.0
    # schools: clustered
    n_par = max(3, rng.poisson(0.012 * W * H))
    parents = np.column_stack([rng.uniform(0, W, n_par), rng.uniform(0, H, n_par)])
    sch = parents[rng.integers(0, n_par, n_schools)] + rng.normal(0, 3.5, (n_schools, 2))
    sch = np.clip(sch, 0, [W, H])
    # teachers: scattered around randomly chosen schools
    tea = sch[rng.integers(0, n_schools, n_teachers)] + rng.normal(0, 8.0, (n_teachers, 2))
    tea = np.clip(tea, 0, [W, H])
    district = lambda p: np.minimum((p[:, 0] / W * 5).astype(int), 4)
    d_s, d_t = district(sch), district(tea)
    subj_s = rng.integers(0, 8, n_schools)
    subj_t = rng.integers(0, 8, n_teachers)
    # vacancies
    w = rng.gamma(2.0, 1.0, n_schools)
    vac = np.floor(w / w.sum() * total_vac).astype(int)
    vac[: total_vac - vac.sum()] += 1
    # qualified arcs: same or adjacent subject
    ii, jj = np.where(np.abs(subj_t[:, None] - subj_s[None, :]) <= 1)
    dist = np.hypot(*(tea[ii] - sch[jj]).T)
    commute = dist / 25.0 * 60.0
    m_hat = np.where(subj_t[ii] == subj_s[jj], 0.0, 1.0 / 7.0)
    cross = (d_t[ii] != d_s[jj]).astype(float)
    d_hat = commute / commute.max()
    z = 0.030 * commute + 0.55 * cross + rng.normal(0, 0.45, len(ii))
    labels = np.select([z < 0.65, z < 1.35, z < 2.10], [0, 1, 5], 10).astype(float)
    return dict(n_teachers=n_teachers, n_schools=n_schools, vacancies=vac,
                arcs=np.column_stack([ii, jj]), commute=commute, d_hat=d_hat,
                m_hat=m_hat, cross=cross, labels=labels)


# ============================================================================
# Label oracle: a label is known only after it has been requested; requests are counted
# ============================================================================
class LabelOracle:
    def __init__(self, labels):
        self._labels = labels
        self.known = np.zeros(len(labels), dtype=bool)

    def request(self, idx):
        self.known[idx] = True

    @property
    def n_requested(self):
        return int(self.known.sum())


# ============================================================================
# LP construction and solution
# ============================================================================
class Network:
    """Caches the constraint matrices of the full graph E; subgraphs are column slices."""

    def __init__(self, n_t, n_s, vac, arcs):
        self.n_t, self.n_s = n_t, n_s
        self.vac = np.asarray(vac, dtype=float)
        self.arcs = arcs
        K = len(arcs)
        self.K = K
        self.T_full = csr_matrix((np.ones(K), (arcs[:, 0], np.arange(K))), shape=(n_t, K)).tocsc()
        self.S_full = csr_matrix((np.ones(K), (arcs[:, 1], np.arange(K))), shape=(n_s, K)).tocsc()

    def solve(self, cols, arc_cost, dummy_cost, vstar=None):
        """
        min  arc_cost' x + dummy_cost' y
        s.t. T x <= 1                    (teachers, duals u <= 0)
             S x + y = V                 (schools, duals v free)
             1' y = vstar  (Stage 2 only) (dual w)
             0 <= x <= 1,  y >= 0
        Returns obj, x (in the order of cols), y, u, v
        """
        n_s = self.n_s
        k = len(cols)
        T = self.T_full[:, cols]
        S = self.S_full[:, cols]
        from scipy.sparse import hstack, identity
        A_ub = hstack([T, csr_matrix((self.n_t, n_s))]).tocsr()
        A_eq = hstack([S, identity(n_s, format="csr")]).tocsr()
        b_eq = self.vac.copy()
        if vstar is not None:
            row = csr_matrix((np.ones(n_s), (np.zeros(n_s, dtype=int), k + np.arange(n_s))),
                             shape=(1, k + n_s))
            A_eq = vstack([A_eq, row]).tocsr()
            b_eq = np.append(b_eq, vstar)
        c = np.concatenate([arc_cost, dummy_cost])
        bounds = [(0, 1)] * k + [(0, None)] * n_s
        res = linprog(c, A_ub=A_ub, b_ub=np.ones(self.n_t), A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs-ds")
        if res.status != 0:
            raise RuntimeError(f"LP solve failed: {res.message}")
        u = res.ineqlin.marginals            # <= 0
        v = res.eqlin.marginals[:n_s]
        return res.fun, res.x[:k], res.x[k:], u, v


def max_frac(x):
    return float(np.max(np.abs(x - np.round(x)))) if len(x) else 0.0


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def stage1(net, cols, price_over=None):
    """
    Minimize unmet demand. cols: initial working set. If price_over (column
    indices of the candidate set) is given, price over it until dual feasible;
    Stage 1 has zero arc costs, so pricing needs no labels.
    Returns V*, final working set, support (arcs with x = 1, global indices),
    pricing rounds, max fractionality.
    """
    work = np.array(sorted(set(cols.tolist())), dtype=int)
    rounds = 0
    while True:
        rounds += 1
        obj, x, y, u, v = net.solve(work, np.zeros(len(work)), np.ones(net.n_s))
        if price_over is None:
            break
        in_work = np.zeros(net.K, dtype=bool)
        in_work[work] = True
        cand = price_over[~in_work[price_over]]
        rc = 0.0 - u[net.arcs[cand, 0]] - v[net.arcs[cand, 1]]
        add = cand[rc < -TOL]
        if len(add) == 0:
            break
        work = np.union1d(work, add)
    support = work[x > 0.5]
    return int(round(obj)), work, support, rounds, max_frac(x)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def stage2(net, cols, vstar, a, C, oracle, price_over=None, screen=False):
    """
    Minimize the composite cost subject to sum y = vstar.
    a: label-free cost (lower bound); C: full cost (usable only once labelled).
    price_over: column indices of the candidate set; None means no pricing.
    screen=False: standard pricing, exact reduced costs of all candidates
                  (needs every label).
    screen=True : bound screening, labels requested only for arcs whose
                  lower-bound reduced cost is negative.
    """
    work = np.array(sorted(set(cols.tolist())), dtype=int)
    oracle.request(work)
    rounds = 0
    while True:
        rounds += 1
        obj, x, y, u, v = net.solve(work, C[work], np.zeros(net.n_s), vstar=vstar)
        if price_over is None:
            break
        in_work = np.zeros(net.K, dtype=bool)
        in_work[work] = True
        cand = price_over[~in_work[price_over]]
        uu = u[net.arcs[cand, 0]]
        vv = v[net.arcs[cand, 1]]
        if screen:
            lb = a[cand] - uu - vv
            need = lb < -TOL
            oracle.request(cand[need])
            rc = np.full(len(cand), np.inf)
            rc[need] = C[cand[need]] - uu[need] - vv[need]
        else:
            oracle.request(cand)
            rc = C[cand] - uu - vv
        add = cand[rc < -TOL]
        if len(add) == 0:
            break
        work = np.union1d(work, add)
    return obj, work, rounds, max_frac(x)


def stage2_optimistic(net, cols, vstar, a, C, oracle, price_over):
    """
    Optimistic label acquisition (OLA, Algorithm 1).
    Unlabelled arcs are priced at the lower bound a_ij (U(R) >= 0, so
    a_ij <= C_ij); labelled arcs at their true cost.
      inner loop: column generation under optimistic costs, solving the LP to
                  optimality over E (pricing uses optimistic costs only, no labels);
      outer loop: query the unlabelled arcs in the support of that optimum,
                  update the costs and repeat.
    At termination the support is fully labelled, so
    C(x) = c~(x) = min c~ <= min C and x is exactly optimal over E.
    """
    work = np.array(sorted(set(cols.tolist())), dtype=int)
    known = oracle.known
    acq = 0
    solves = 0
    while True:
        while True:
            ctil = np.where(known[work], C[work], a[work])
            obj, x, y, u, v = net.solve(work, ctil, np.zeros(net.n_s), vstar=vstar)
            solves += 1
            in_work = np.zeros(net.K, dtype=bool)
            in_work[work] = True
            cand = price_over[~in_work[price_over]]
            cc = np.where(known[cand], C[cand], a[cand])
            rc = cc - u[net.arcs[cand, 0]] - v[net.arcs[cand, 1]]
            add = cand[rc < -TOL]
            if len(add) == 0:
                break
            work = np.union1d(work, add)
        supp = work[x > 0.5]
        new = supp[~known[supp]]
        if len(new) == 0:
            break
        oracle.request(new)
        acq += 1
    return obj, work, acq, solves, max_frac(x)


# ============================================================================
# The six schemes
# ============================================================================
def run_methods(inst, weights, t_prune, t_feas=None, truth=None, methods=None):
    n_t, n_s = inst["n_teachers"], inst["n_schools"]
    arcs_all = np.asarray(inst["arcs"] if "arcs" in inst else inst["qual_arcs"], dtype=int)
    commute_all = np.asarray(inst["commute"], dtype=float)
    keep = np.ones(len(arcs_all), dtype=bool) if t_feas is None else commute_all <= t_feas
    arcs = arcs_all[keep]
    commute = commute_all[keep]
    d_hat = np.asarray(inst["d_hat"], dtype=float)[keep]
    m_hat = np.asarray(inst["m_hat"], dtype=float)[keep]
    cross = np.asarray(inst["cross"], dtype=float)[keep]
    labels = np.asarray(inst["labels"], dtype=float)[keep]

    alpha, beta, gamma = weights
    a = alpha * d_hat + beta * m_hat + RHO * cross      # label-free part (lower bound)
    C = a + gamma * utility(labels)                      # full cost
    net = Network(n_t, n_s, inst["vacancies"], arcs)
    K = net.K
    E = np.arange(K)
    Ep = np.where(commute <= t_prune)[0]
    out = {}
    methods = methods or ["full", "prune_only", "price_s2", "price_both", "screened", "proposed"]

    def rec(name, t0, oracle, v1, obj2, r1, r2, work, mf):
        out[name] = dict(time_s=time.perf_counter() - t0, labels=oracle.n_requested,
                         labels_pct=100.0 * oracle.n_requested / K,
                         s1_unmet=v1, s2_obj=obj2, s1_rounds=r1, s2_rounds=r2,
                         work_pct=100.0 * len(work) / K, max_frac=mf)

    if "full" in methods and truth is None:
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, _, r1, mf1 = stage1(net, E)
        obj2, w2, r2, mf2 = stage2(net, E, v1, a, C, o)
        rec("full", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
        truth = (v1, obj2)
    elif truth is not None:
        out["full"] = None
    if len(Ep) > 0 and "prune_only" in methods:
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, _, r1, mf1 = stage1(net, Ep)
        obj2, w2, r2, mf2 = stage2(net, Ep, v1, a, C, o)
        rec("prune_only", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
    if len(Ep) > 0 and "price_s2" in methods:
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, w1, _, r1, mf1 = stage1(net, Ep)
        obj2, w2, r2, mf2 = stage2(net, w1, v1, a, C, o, price_over=E, screen=False)
        rec("price_s2", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
    if len(Ep) > 0 and "price_both" in methods:
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, w1, _, r1, mf1 = stage1(net, Ep, price_over=E)
        obj2, w2, r2, mf2 = stage2(net, w1, v1, a, C, o, price_over=E, screen=False)
        rec("price_both", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
    if "screened" in methods:                                  # ablation: bound-screened pricing
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, supp, r1, mf1 = stage1(net, E)                 # label-free, full graph
        init = np.union1d(Ep, supp)                           # keeps Stage 2 feasible
        obj2, w2, r2, mf2 = stage2(net, init, v1, a, C, o, price_over=E, screen=True)
        rec("screened", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
    if "proposed" in methods:                                  # OLA
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, supp, r1, mf1 = stage1(net, E)                 # label-free, full graph
        init = np.union1d(Ep, supp)
        obj2, w2, acq, solves, mf2 = stage2_optimistic(net, init, v1, a, C, o, price_over=E)
        rec("proposed", t0, o, v1, obj2, r1, solves, w2, max(mf1, mf2))
        out["proposed"]["acq_rounds"] = acq
    nu = int(round(net.vac.sum())) - (truth[0] if truth else 0)
    for r in out.values():
        if r is not None:
            r.setdefault("acq_rounds", 0)
            r["labels_per_assign"] = r["labels"] / max(nu, 1)
    return out, truth, K, len(Ep)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use the demo generator (smoke test only)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="explicit list of seeds (overrides --seeds), e.g. for parallel batches")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0, 12, 18, 24, 30, 45, 60],
                    help="pruning thresholds T_prune (min); 0 means the initial working set is the Stage 1 support only")
    ap.add_argument("--weights", nargs="+", default=list(WEIGHTS))
    ap.add_argument("--regimes", nargs="+", default=["nofeas", "feas60"],
                    help="nofeas: E = full qualified graph (main regime); feas60: E = arcs with commute <= 60 min")
    ap.add_argument("--sizes", type=str, default=None,
                    help="scaling experiment, e.g. '2000x50,5236x114,12000x250'")
    ap.add_argument("--methods", nargs="+",
                    default=["full", "prune_only", "price_s2", "price_both", "screened", "proposed"])
    ap.add_argument("--out", default="label_pricing")
    args = ap.parse_args()

    if args.demo:
        build = demo_instance
    else:
        from adapter_cor import build_instance as build
    import platform, scipy
    print(f"# platform={platform.platform()} python={platform.python_version()} "
          f"numpy={np.__version__} scipy={scipy.__version__}", flush=True)

    rows = []
    done = {}
    raw_path = f"{args.out}_raw.csv"
    import os
    if os.path.exists(raw_path):
        with open(raw_path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["size"], int(r["seed"]), r["regime"], r["weights"], float(r["t_prune"]))
                done.setdefault(key, set()).add(r["method"])
                done[("truth",) + key[:4]] = (int(float(r["s1_true"])), float(r["s2_true"]))
                for c in r:
                    if c in ("size", "regime", "weights", "method"):
                        continue
                    try:
                        r[c] = int(r[c]) if c in ("seed", "K", "labels", "s1_unmet", "s1_true", "s1_bias",
                                                    "s1_rounds", "s2_rounds", "acq_rounds") else float(r[c])
                    except ValueError:
                        r[c] = float(r[c])
                rows.append(r)
        print(f"# resume: {len(rows)} rows read", flush=True)
    sizes = [None]
    if args.sizes:
        sizes = [tuple(int(z) for z in s.split("x")) for s in args.sizes.split(",")]

    for size in sizes:
        for seed in (args.seed_list if args.seed_list is not None else range(args.seeds)):
            inst = build(seed) if size is None else build(seed, n_teachers=size[0], n_schools=size[1])
            for regime in args.regimes:
                t_feas = None if regime == "nofeas" else 60.0
                for wname in args.weights:
                    szkey = "" if size is None else f"{size[0]}x{size[1]}"
                    truth = done.get(("truth", szkey, seed, regime, wname))
                    for T in args.thresholds:
                        todo = [m for m in args.methods
                                if m not in done.get((szkey, seed, regime, wname, float(T)), set())]
                        if truth is not None and "full" in todo:
                            todo.remove("full")
                        if truth is not None and not todo:
                            continue
                        res, truth, K, kp = run_methods(inst, WEIGHTS[wname], T, t_feas, truth,
                                                        methods=todo)
                        for m, r in res.items():
                            if r is None:
                                continue
                            row = dict(size="" if size is None else f"{size[0]}x{size[1]}",
                                       seed=seed, regime=regime, weights=wname, t_prune=T,
                                       method=m, K=K, Ep_pct=100.0 * kp / K,
                                       s1_true=truth[0], s2_true=truth[1],
                                       s1_bias=r["s1_unmet"] - truth[0],
                                       s2_gap_pct=100.0 * (r["s2_obj"] - truth[1]) / max(abs(truth[1]), 1e-12),
                                       **r)
                            rows.append(row)
                            _append_csv(raw_path, row, first=not os.path.exists(raw_path))
                            ok = "" if m not in ("price_both", "screened", "proposed") else \
                                ("  OK" if row["s1_bias"] == 0 and abs(row["s2_gap_pct"]) < 1e-6 else "  !!! NOT EXACT")
                            print(f"seed={seed} {regime:7s} {wname:15s} T={T:>4g} {m:11s} "
                                  f"labels={r['labels_pct']:6.2f}%  V*={r['s1_unmet']:5d} (true {truth[0]})  "
                                  f"gap={row['s2_gap_pct']:+.4f}%  t={r['time_s']:.2f}s{ok}", flush=True)

    print(f"\nwritten: {args.out}_raw.csv ({len(rows)} rows, written incrementally)")
    summarize(rows, args.out)


_FIELDS = None


def _append_csv(path, row, first):
    global _FIELDS
    if _FIELDS is None:
        if first:
            _FIELDS = list(row.keys())
        else:
            with open(path, newline="") as fh:
                _FIELDS = next(csv.reader(fh))
    with open(path, "w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        if first:
            w.writeheader()
        w.writerow(row)


def summarize(rows, out):
    """Summarize by (size, regime, weights, threshold, method): label share, coverage bias, cost gap, time."""
    g = defaultdict(list)
    for r in rows:
        g[(r["size"], r["regime"], r["weights"], r["t_prune"], r["method"])].append(r)
    summ = []
    for key, rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        f = lambda c: np.array([x[c] for x in rs], dtype=float)
        summ.append(dict(size=key[0], regime=key[1], weights=key[2], t_prune=key[3], method=key[4],
                         n=len(rs), labels_pct=f("labels_pct").mean(),
                         labels_pct_max=f("labels_pct").max(),
                         s1_bias_med=float(np.median(f("s1_bias"))), s1_bias_max=f("s1_bias").max(),
                         s2_gap_pct_max=np.abs(f("s2_gap_pct")).max(),
                         s2_rounds=f("s2_rounds").mean(), time_s=f("time_s").mean(),
                         work_pct=f("work_pct").mean(),
                         labels_per_assign=f("labels_per_assign").mean(),
                         acq_rounds=f("acq_rounds").mean(),
                         exact_all=int(all(x["s1_bias"] == 0 and abs(x["s2_gap_pct"]) < 1e-6 for x in rs))))
    path = f"{out}_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print(f"written: {path}")

    # console summary per regime x weights: label shares of OLA and two-stage pricing
    print("\n=== summary: share of labels queried (%) and exactness ===")
    print(f"{'regime':8s} {'weights':16s} {'T':>4s} | {'proposed':>9s} {'(x nu)':>6s} {'screened':>8s} {'price_both':>10s} "
          f"{'prune_only gap%':>15s} {'price_s2 V* bias(max)':>21s} | {'t_prop':>7s} {'t_both':>7s} exact")
    idx = {(s["size"], s["regime"], s["weights"], s["t_prune"], s["method"]): s for s in summ}
    for s in summ:
        if s["method"] != "proposed":
            continue
        k = lambda m: idx.get((s["size"], s["regime"], s["weights"], s["t_prune"], m))
        pb, po, p2, sc = k("price_both"), k("prune_only"), k("price_s2"), k("screened")
        print(f"{s['regime']:8s} {s['weights']:16s} {s['t_prune']:>4g} | {s['labels_pct']:9.2f} "
              f"{s['labels_per_assign']:6.2f} {(sc['labels_pct'] if sc else float('nan')):8.2f} "
              f"{(pb['labels_pct'] if pb else float('nan')):10.2f} "
              f"{(po['s2_gap_pct_max'] if po else float('nan')):15.3f} "
              f"{(p2['s1_bias_max'] if p2 else float('nan')):21.0f} | "
              f"{s['time_s']:7.2f} {(pb['time_s'] if pb else float('nan')):7.2f} "
              f"{'Y' if s['exact_all'] and (pb is None or pb['exact_all']) and (sc is None or sc['exact_all']) else 'N'}")

    # LaTeX table (all regimes and weights; details in the CSV)
    tex = f"{out}_table.tex"
    with open(tex, "w") as fh:
        fh.write("% auto-generated by exp_label_pricing.py\n")
        fh.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\hline\n")
        fh.write("Regime / weights & $T_{\\text{prune}}$ & Labels, proposed (\\%) & Labels, two-stage pricing (\\%) "
                 "& Prune-only gap (\\%) & Stage-2-only bias (max) & Time ratio \\\\\n\\hline\n")
        for s in summ:
            if s["method"] != "proposed":
                continue
            k = lambda m: idx.get((s["size"], s["regime"], s["weights"], s["t_prune"], m))
            pb, po, p2 = k("price_both"), k("prune_only"), k("price_s2")
            fmt = lambda v, f: ("---" if v is None else f.format(v))
            fh.write(f"{s['regime']} / {s['weights'].replace('_', ' ')} & {s['t_prune']:g} & "
                     f"{s['labels_pct']:.1f} & {fmt(pb and pb['labels_pct'], '{:.1f}')} & "
                     f"{fmt(po and po['s2_gap_pct_max'], '{:.2f}')} & {fmt(p2 and p2['s1_bias_max'], '{:.0f}')} & "
                     f"{fmt(pb and s['time_s'] / pb['time_s'], '{:.2f}')} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    print(f"written: {tex}")


if __name__ == "__main__":
    main()
