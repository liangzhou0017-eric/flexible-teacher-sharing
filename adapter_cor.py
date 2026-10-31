#!/usr/bin/env python3
"""
adapter_cor.py -- builds the City-X instances of the paper from citax.py.

  instance        citax.make_instance(seed, n_t, n_s)
  cost components citax.build_cost_components(inst, default_rng(seed + 10000))
  commute         D^ = clip(commute / 60, 0, 1)
  subject gap     M^ = |subject difference| / 7 (0 or 1/7 on qualified arcs)
  labels          R from the seeded ordered-probit generator, defined for every
                  teacher-school pair (including arcs beyond 60 minutes)

Running `python adapter_cor.py` checks the regenerated instances against
previously published values.
"""
import sys
import numpy as np
import citax as cx


def build_instance(seed, n_teachers=5236, n_schools=114):
    inst = cx.make_instance(seed, n_teachers, n_schools)
    rng = np.random.default_rng(seed + 10_000)
    dist, commute, Q, M_hat, R, I = cx.build_cost_components(inst, rng)
    ii, jj = np.nonzero(Q)
    return dict(
        n_teachers=inst.n_teachers,
        n_schools=inst.n_schools,
        vacancies=inst.vacancies.astype(int),
        arcs=np.column_stack([ii, jj]),
        commute=commute[ii, jj],
        d_hat=np.clip(commute[ii, jj] / 60.0, 0, 1),
        m_hat=M_hat[ii, jj],
        cross=I[ii, jj].astype(float),
        labels=R[ii, jj].astype(float),
    )


def check():
    """Check instance size, label shares, optimal objective and the Stage 1 pricing table."""
    import exp_label_pricing as X
    ok = True

    def rep(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name:40s} {detail}", flush=True)

    print("=== check (seed 0) ===")
    inst = build_instance(0)
    comm, R, c = inst["commute"], inst["labels"], inst["cross"]
    at60 = comm <= 60
    rep("candidate arcs within 60 min = 93,381", at60.sum() == 93381, f"{at60.sum()}")
    sh = {k: 100 * np.mean(R[at60] == k) for k in (0, 1, 5, 10)}
    ref = {0: 17.4, 1: 28.5, 5: 34.8, 10: 19.3}
    rep("label shares as published", max(abs(sh[k] - ref[k]) for k in ref) <= 0.05,
        " / ".join(f"{k}:{sh[k]:.1f}%" for k in sh))
    rep("cross-district share of R=10 arcs = 94.1%", abs(100 * np.mean(c[at60 & (R == 10)]) - 94.1) <= 0.05)
    _, truth, _, _ = X.run_methods(inst, X.WEIGHTS["balanced"], 60.0, None, None, methods=["full"])
    rep("balanced optimal objective = 261.0947", abs(truth[1] - 261.0947) < 1e-3, f"{truth[1]:.4f}")
    keep = at60
    net = X.Network(inst["n_teachers"], inst["n_schools"], inst["vacancies"], inst["arcs"][keep])
    v12, *_ = X.stage1(net, np.where(comm[keep] <= 12)[0])
    rep("Stage 1 table: phantom shortage at 12 min = 673", v12 == 673, f"{v12}")
    print("all checks passed." if ok else "some checks FAILED; see above.", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
