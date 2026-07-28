"""Type II support-set tolerances and stability under ordinal-label jitter."""
import argparse
import json

import numpy as np

import citax as cx
import solver as sv

ANCHORS = np.array([0.0, 1.0, 5.0, 10.0])


def solve(inst, keep, pruned, C, V, forbid=None):
    k = keep.copy()
    if forbid is not None:
        k[forbid] = False
    return sv.solve_scenario(inst, k, pruned.copy(), C, V, use_s1=True)


def type_two(inst, keep, pruned, C, V, n_sample, rng):
    """delta+ per active arc, obtained by forbidding it and re-solving.
    Active arcs have delta- = -inf: lowering the cost of an arc already in the
    optimum cannot displace it."""
    base = solve(inst, keep, pruned, C, V)
    obj0 = base["obj"]
    act = base["x"] > 0.5
    ai, aj = base["rows"][act], base["cols"][act]
    idx = rng.permutation(int(act.sum()))[:min(n_sample, int(act.sum()))]
    deltas = []
    for c, q in enumerate(idx, 1):
        d = solve(inst, keep, pruned, C, V, forbid=(ai[q], aj[q]))["obj"] - obj0
        if np.isfinite(d) and d >= -1e-9:
            deltas.append(max(d, 0.0))
        if c % 20 == 0:
            print("  [%d/%d]" % (c, len(idx)))
    q = np.percentile(np.array(deltas), [0, 25, 50, 75, 95, 100])
    print("delta+ min %.4f med %.4f max %.4f" % (q[0], q[2], q[5]))
    return dict(n_active=int(act.sum()), n_sampled=len(deltas),
                dp_min=q[0], dp_q1=q[1], dp_med=q[2],
                dp_q3=q[3], dp_p95=q[4], dp_max=q[5])


def weighted_kappa(x, y, K=4):
    W = np.array([[1 - abs(i - j) / (K - 1) for j in range(K)] for i in range(K)])
    O = np.zeros((K, K))
    for a, b in zip(x, y):
        O[a, b] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    return float(((W * O).sum() - (W * E).sum()) / (1 - (W * E).sum()))


def jitter(inst, keep, pruned, C, V, dist, R, I, D, M, Q, k, rho, n_runs, flip, rng):
    """Perturb labels to an adjacent level with probability `flip`, re-solve, and
    compare the optimal support set and the aggregate metrics."""
    a, b, g = sv.BALANCED
    base = solve(inst, keep, pruned, C, V)
    sel = base["x"] > 0.5
    E0 = set(zip(base["rows"][sel].tolist(), base["cols"][sel].tolist()))
    m0 = sv.metrics(inst, base["rows"], base["cols"], base["x"],
                    base["dummy"], dist, R, I, V)

    idx = np.nonzero(keep)
    lab0 = np.searchsorted(ANCHORS, R[idx])

    reps = []
    for _ in range(30):
        l = lab0.copy()
        f = rng.random(len(l)) < flip
        l[f] = np.clip(l[f] + rng.choice([-1, 1], f.sum()), 0, 3)
        reps.append(ANCHORS[l])
    reps = np.array(reps)
    stable = float((reps == reps[0]).all(axis=0).mean() * 100)
    sd = cx.utility(reps, k).std(axis=0, ddof=1)

    l1 = lab0.copy()
    f = rng.random(len(l1)) < flip
    l1[f] = np.clip(l1[f] + rng.choice([-1, 1], f.sum()), 0, 3)
    kap = weighted_kappa(lab0, l1)

    jac, mets = [], []
    for t in range(n_runs):
        Rp = R.copy()
        l = lab0.copy()
        f = rng.random(len(l)) < flip
        l[f] = np.clip(l[f] + rng.choice([-1, 1], f.sum()), 0, 3)
        Rp[idx] = ANCHORS[l]
        Cp = np.where(Q, a * D + b * M + g * cx.utility(Rp, k) + rho * I, np.inf)
        r = solve(inst, keep, pruned, Cp, V)
        s = r["x"] > 0.5
        E = set(zip(r["rows"][s].tolist(), r["cols"][s].tolist()))
        jac.append(len(E0 & E) / len(E0 | E))
        mm = sv.metrics(inst, r["rows"], r["cols"], r["x"], r["dummy"],
                        dist, R, I, V)
        mets.append([mm["mismatch_pct"], mm["resistance"], mm["commute_km"]])
    jac = np.array(jac)
    mets = np.array(mets)
    shift = max(abs(mets[:, i].mean() - v) / max(abs(v), 1e-9) * 100
                for i, v in enumerate([m0["mismatch_pct"], m0["resistance"],
                                       m0["commute_km"]]))
    print("Jaccard %.3f+-%.3f; %.1f%% of pairings change; aggregates shift <= %.1f%%"
          % (jac.mean(), jac.std(ddof=1), (1 - jac.mean()) * 100, shift))
    return dict(flip_p=flip, weighted_kappa=round(kap, 3), n_runs=n_runs,
                stable_label_pct=round(stable, 1),
                sigma_med=float(np.median(sd)), sigma_max=float(sd.max()), lam=3,
                jaccard_mean=float(jac.mean()), jaccard_sd=float(jac.std(ddof=1)),
                arc_change_pct=float((1 - jac.mean()) * 100),
                agg_mism=[float(mets[:, 0].mean()), float(mets[:, 0].std(ddof=1))],
                agg_resist=[float(mets[:, 1].mean()), float(mets[:, 1].std(ddof=1))],
                agg_comm=[float(mets[:, 2].mean()), float(mets[:, 2].std(ddof=1))],
                agg_max_shift_pct=round(float(shift), 1),
                baseline=dict(mismatch=m0["mismatch_pct"],
                              resistance=m0["resistance"],
                              commute=m0["commute_km"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--nt", type=int, default=1200)
    ap.add_argument("--ns", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--arcs", type=int, default=120)
    ap.add_argument("--jitter-runs", type=int, default=20)
    ap.add_argument("--flip", type=float, default=0.06)
    ap.add_argument("--out", default="results_sensitivity.json")
    args = ap.parse_args()
    rng = np.random.default_rng(20260727)

    inst, dist, com, Q, M, R, I, D, U, keep, pruned = sv.build(
        args.seed, args.threshold, args.k, args.nt, args.ns)
    V = inst.vacancies.astype(float)
    a, b, g = sv.BALANCED
    C = np.where(Q, a * D + b * M + g * U + args.rho * I, np.inf)

    print("--- Type II tolerances ---")
    E = type_two(inst, keep, pruned, C, V, args.arcs, rng)
    print("--- label jitter ---")
    F = jitter(inst, keep, pruned, C, V, dist, R, I, D, M, Q,
               args.k, args.rho, args.jitter_runs, args.flip, rng)

    json.dump(dict(environment=sv.capture_environment(), settings=vars(args),
                   type_two=E, jitter=F),
              open(args.out, "w"), indent=2, default=float)
    print("written to " + args.out)


if __name__ == "__main__":
    main()
