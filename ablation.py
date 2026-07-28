"""Ablations: Big-M penalty, explicit integrality, linear utility, rho sweep."""
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

import citax as cx
import solver as sv

BALANCED = (0.4, 0.3, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--out", default="results_ablation.json")
    args = ap.parse_args()

    inst, dist, com, Q, M, R, I, D, U, keep, pruned = sv.build(
        args.seed, args.threshold, args.k)
    V = inst.vacancies.astype(float)
    n, m = inst.n_teachers, inst.n_schools
    rows, cols = np.nonzero(keep)
    n_arcs = len(rows)
    A_ub, A_eq = sv._matrices(rows, cols, n, m, V)
    bounds = [(0, 1)] * n_arcs + [(0, float(v)) for v in V]
    out = {}

    c1 = np.concatenate([np.zeros(n_arcs), np.ones(m)])
    Vu = float(linprog(c1, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq, b_eq=V,
                       bounds=bounds, method="highs").fun)

    def cost(a, b, g, rho, k):
        return np.where(Q, a * D + b * M + g * cx.utility(R, k) + rho * I, np.inf)

    def lex(C, integral=False):
        costs = C[rows, cols]
        lock = sp.csr_matrix(
            (np.ones(m), (np.zeros(m, int), n_arcs + np.arange(m))),
            shape=(1, n_arcs + m))
        A2 = sp.vstack([A_eq, lock], format="csr")
        b2 = np.concatenate([V, [Vu]])
        c = np.concatenate([costs, np.zeros(m)])
        t0 = time.perf_counter()
        if integral:
            cons = [LinearConstraint(A_ub, -np.inf, np.ones(n)),
                    LinearConstraint(A2, b2, b2)]
            ub = np.concatenate([np.ones(n_arcs), V])
            res = milp(c, constraints=cons,
                       bounds=Bounds(np.zeros(n_arcs + m), ub),
                       integrality=np.ones(n_arcs + m))
            return (res.fun, time.perf_counter() - t0,
                    int(getattr(res, "mip_node_count", -1) or -1), res.x)
        res = linprog(c, A_ub=A_ub, b_ub=np.ones(n), A_eq=A2, b_eq=b2,
                      bounds=bounds, method="highs")
        return float(res.fun), time.perf_counter() - t0, 0, res.x

    C_bal = cost(*BALANCED, args.rho, args.k)
    obj, cpu, _, x = lex(C_bal)
    frac = float(np.abs(x[:n_arcs] - np.round(x[:n_arcs])).max())
    out["proposed"] = dict(obj=obj, cpu=cpu, nodes=0, frac=frac)
    print("proposed: obj=%.4f cpu=%.2fs max_frac=%.1e" % (obj, cpu, frac))

    out["bigm"] = []
    costs = C_bal[rows, cols]
    for Mbig in [1e3, 1e6, 1e9, 1e12]:
        c = np.concatenate([costs, np.full(m, Mbig)])
        t0 = time.perf_counter()
        r = linprog(c, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq, b_eq=V,
                    bounds=bounds, method="highs")
        el = time.perf_counter() - t0
        rec = dict(M=Mbig, cpu=el, nit=int(r.nit or 0),
                   status="ok" if r.success else r.message[:32])
        if r.success:
            rec["unmet"] = float(r.x[n_arcs:].sum())
            rec["err"] = abs(float(costs @ r.x[:n_arcs]) - obj)
        out["bigm"].append(rec)
        print("  big-M %.0e: cpu=%.2fs nit=%d err=%.1e"
              % (Mbig, el, rec["nit"], rec.get("err", float("nan"))))

    obj_b, cpu_b, nodes, _ = lex(C_bal, integral=True)
    out["bnb"] = dict(obj=obj_b, cpu=cpu_b, nodes=nodes,
                      gap=abs(obj_b - obj), speedup=cpu_b / cpu)
    print("branch-and-bound: cpu=%.2fs nodes=%d slowdown=%.2fx"
          % (cpu_b, nodes, cpu_b / cpu))

    C_lin = cost(*BALANCED, args.rho, 0.0)
    obj_l, cpu_l, _, x_l = lex(C_lin)
    def summarise(xv):
        s = xv[:n_arcs] > 0.5
        rr, cc = rows[s], cols[s]
        gap = np.abs(inst.teacher_subject[rr] - inst.school_subject[cc])
        return float(R[rr, cc].mean()), 100.0 * float((gap > 0).mean())
    re_, mi_ = summarise(x)
    rl_, ml_ = summarise(x_l)
    out["linear"] = dict(obj=obj_l, cpu=cpu_l, resist_exp=re_, resist_lin=rl_,
                         mism_exp=mi_, mism_lin=ml_)
    print("linear utility: resistance %.3f vs %.3f (exp); mismatch %.2f%% vs %.2f%%"
          % (rl_, re_, ml_, mi_))

    out["rho"] = []
    for rho in [0.0, 0.05, 0.15, 0.30, 0.60]:
        C = cost(*BALANCED, rho, args.k)
        o, _, _, xv = lex(C)
        s = xv[:n_arcs] > 0.5
        rec = dict(rho=rho, cross_pct=100.0 * float(I[rows[s], cols[s]].mean()),
                   resist=float(R[rows[s], cols[s]].mean()), obj=o)
        out["rho"].append(rec)
        print("  rho=%.2f: cross=%.2f%% resistance=%.3f obj=%.2f"
              % (rho, rec["cross_pct"], rec["resist"], o))

    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print("written to " + args.out)


if __name__ == "__main__":
    main()
