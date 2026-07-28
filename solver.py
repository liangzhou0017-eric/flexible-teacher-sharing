"""Preemptive lexicographic teacher assignment with exact edge augmentation.

Both stages apply a pricing loop over spatially pruned arcs. Stage 1 pricing is
required for exactness: Stage 1 determines V*_unmet, which constraint (6) then
fixes for Stage 2, so pruning that removes augmenting paths would otherwise
inflate the reported shortfall irrecoverably.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import linprog

import citax as cx

BALANCED = (0.4, 0.3, 0.3)
SCENARIOS = cx.SCENARIOS


def capture_environment():
    env = {
        "os": platform.system() + " " + platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": "SciPy linprog / HiGHS (scipy %s)" % scipy.__version__,
        "cpu_model": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
    }
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal"):
                    env["ram_gb"] = round(int(ln.split()[1]) / 1048576, 1)
                    break
    except OSError:
        pass
    return env


def _matrices(rows, cols, n, m, V):
    n_arcs = len(rows)
    A_ub = sp.csr_matrix((np.ones(n_arcs), (rows, np.arange(n_arcs))),
                         shape=(n, n_arcs + m))
    data = np.concatenate([np.ones(n_arcs), np.ones(m)])
    ri = np.concatenate([cols, np.arange(m)])
    ci = np.concatenate([np.arange(n_arcs), n_arcs + np.arange(m)])
    A_eq = sp.csr_matrix((data, (ri, ci)), shape=(m, n_arcs + m))
    return A_ub, A_eq


def stage1(rows, cols, pruned, n, m, V, use_pricing=True, tol=1e-9):
    """Minimise unmet demand. Arc objective coefficients are zero, so the
    reduced cost of a pruned arc is -u_i - v_j."""
    loops = iters = 0
    while True:
        loops += 1
        A_ub, A_eq = _matrices(rows, cols, n, m, V)
        bd = [(0, 1)] * len(rows) + [(0, float(v)) for v in V]
        c = np.concatenate([np.zeros(len(rows)), np.ones(m)])
        r = linprog(c, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq, b_eq=V,
                    bounds=bd, method="highs")
        if not r.success:
            raise RuntimeError("Stage 1 failed: " + r.message)
        iters += int(getattr(r, "nit", 0) or 0)
        if not use_pricing or not pruned.any():
            break
        u = np.asarray(r.ineqlin.marginals)
        v = np.asarray(r.eqlin.marginals)[:m]
        pr, pc = np.nonzero(pruned)
        bad = (-u[pr] - v[pc]) < -tol
        if not bad.any():
            break
        rows = np.concatenate([rows, pr[bad]])
        cols = np.concatenate([cols, pc[bad]])
        pruned = pruned.copy()
        pruned[pr[bad], pc[bad]] = False
    return float(r.fun), rows, cols, loops, iters


def stage2(rows, cols, pruned, C, n, m, V, V_unmet, tol=1e-9):
    """Minimise composite cost subject to total dummy flow = V_unmet."""
    loops = iters = 0
    elapsed = 0.0
    while True:
        loops += 1
        A_ub, A_eq = _matrices(rows, cols, n, m, V)
        bd = [(0, 1)] * len(rows) + [(0, float(v)) for v in V]
        lock = sp.csr_matrix(
            (np.ones(m), (np.zeros(m, int), len(rows) + np.arange(m))),
            shape=(1, len(rows) + m))
        A2 = sp.vstack([A_eq, lock], format="csr")
        b2 = np.concatenate([V, [V_unmet]])
        c = np.concatenate([C[rows, cols], np.zeros(m)])
        t0 = time.perf_counter()
        r = linprog(c, A_ub=A_ub, b_ub=np.ones(n), A_eq=A2, b_eq=b2,
                    bounds=bd, method="highs")
        elapsed += time.perf_counter() - t0
        if not r.success:
            raise RuntimeError("Stage 2 failed: " + r.message)
        iters += int(getattr(r, "nit", 0) or 0)
        if not pruned.any():
            break
        u = np.asarray(r.ineqlin.marginals)
        v = np.asarray(r.eqlin.marginals)[:m]
        pr, pc = np.nonzero(pruned)
        bad = (C[pr, pc] - u[pr] - v[pc]) < -tol
        if not bad.any():
            break
        rows = np.concatenate([rows, pr[bad]])
        cols = np.concatenate([cols, pc[bad]])
        pruned = pruned.copy()
        pruned[pr[bad], pc[bad]] = False
    return (float(r.fun), r.x[:len(rows)], r.x[len(rows):],
            rows, cols, loops, iters, elapsed)


def metrics(inst, rows, cols, x, dummy, dist, R, I, V):
    """mismatch_pct is the share of realised assignments taught out of subject."""
    sel = x > 0.5
    rr, cc = rows[sel], cols[sel]
    if not len(rr):
        return dict(mismatch_pct=0.0, commute_km=0.0, resistance=0.0,
                    cross_pct=0.0, n_assigned=0)
    gap = np.abs(inst.teacher_subject[rr] - inst.school_subject[cc])
    return dict(
        mismatch_pct=100.0 * float((gap > 0).mean()),
        commute_km=float(dist[rr, cc].mean()),
        resistance=float(R[rr, cc].mean()),
        cross_pct=100.0 * float(I[rr, cc].mean()),
        n_assigned=int(sel.sum()),
    )


def build(seed, threshold, k, n_t=5236, n_s=114):
    inst = cx.make_instance(seed, n_t, n_s)
    rng = np.random.default_rng(seed + 10_000)
    dist, com, Q, M, R, I = cx.build_cost_components(inst, rng)
    D = np.clip(com / 60.0, 0, 1)
    U = cx.utility(R, k)
    keep = Q & (com <= threshold)
    return inst, dist, com, Q, M, R, I, D, U, keep, Q & ~keep


def solve_scenario(inst, keep, pruned, C, V, use_s1=True):
    n, m = inst.n_teachers, inst.n_schools
    rows, cols = np.nonzero(keep)
    t0 = time.perf_counter()
    vu, rows, cols, l1, i1 = stage1(rows, cols, pruned.copy(), n, m, V, use_s1)
    obj, x, dummy, rows, cols, l2, i2, _ = stage2(
        rows, cols, pruned.copy(), C, n, m, V, vu)
    return dict(V_unmet=vu, obj=obj, x=x, dummy=dummy, rows=rows, cols=cols,
                s1_loops=l1, s2_loops=l2, simplex_iter=i1 + i2,
                cpu_s=time.perf_counter() - t0,
                max_frac=float(np.abs(x - np.round(x)).max()) if len(x) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--no-stage1-pricing", action="store_true",
                    help="disable Stage 1 pricing to reproduce the shortfall bias")
    ap.add_argument("--out", default="results_main.json")
    args = ap.parse_args()
    use_s1 = not args.no_stage1_pricing

    env = capture_environment()
    print("%s | %s | %s cores" % (env["platform"], env["solver"], env["cpu_count"]))
    print("seeds=%d threshold=%.0f rho=%.2f k=%.2f stage1_pricing=%s\n"
          % (args.seeds, args.threshold, args.rho, args.k, use_s1))

    per = {nm: [] for nm, *_ in SCENARIOS}
    per["TSA (static)"] = []
    t_start = time.perf_counter()

    for seed in range(args.seeds):
        inst, dist, com, Q, M, R, I, D, U, keep, pruned = build(
            seed, args.threshold, args.k)
        V = inst.vacancies.astype(float)

        C = np.where(Q & (I == 0), 0.4 * D + 0.3 * M + 0.3 * U, np.inf)
        res = solve_scenario(inst, keep & (I == 0), pruned & (I == 0), C, V, use_s1)
        mt = metrics(inst, res["rows"], res["cols"], res["x"], res["dummy"],
                     dist, R, I, V)
        per["TSA (static)"].append({**mt, **{k_: res[k_] for k_ in (
            "V_unmet", "cpu_s", "simplex_iter", "s1_loops", "s2_loops", "max_frac")}})

        for nm, a, b, g in SCENARIOS:
            C = np.where(Q, a * D + b * M + g * U + args.rho * I, np.inf)
            res = solve_scenario(inst, keep, pruned, C, V, use_s1)
            mt = metrics(inst, res["rows"], res["cols"], res["x"], res["dummy"],
                         dist, R, I, V)
            per[nm].append({**mt, **{k_: res[k_] for k_ in (
                "V_unmet", "cpu_s", "simplex_iter", "s1_loops", "s2_loops",
                "max_frac")}})
        if (seed + 1) % 5 == 0 or seed == 0:
            print("  [%d/%d] %.0fs" % (seed + 1, args.seeds,
                                       time.perf_counter() - t_start))

    order = ["TSA (static)"] + [nm for nm, *_ in SCENARIOS]
    print("\n%-22s%15s%14s%13s%13s%13s" % ("scenario", "mismatch%", "commute",
                                           "resistance", "cross%", "cpu s"))
    for nm in order:
        cells = []
        for key in ("mismatch_pct", "commute_km", "resistance", "cross_pct", "cpu_s"):
            v = np.array([r[key] for r in per[nm]], float)
            cells.append("%.2f+-%.2f" % (v.mean(), v.std(ddof=1)))
        print("%-22s%15s%14s%13s%13s%13s" % (nm, *cells))

    mf = max(r["max_frac"] for nm in order for r in per[nm])
    print("\nmax |x - round(x)| = %.1e" % mf)
    print("Stage 1 pricing rounds (mean) = %.2f"
          % np.mean([r["s1_loops"] for nm in order for r in per[nm]]))

    json.dump(dict(environment=env, settings=vars(args), scenarios=per),
              open(args.out, "w"), indent=2, default=float)
    print("written to " + args.out)


if __name__ == "__main__":
    main()
