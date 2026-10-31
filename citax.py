#!/usr/bin/env python3
"""
City-X 参考实现 (reference implementation)
==========================================

用途：独立于作者原实现，重跑 main.tex 中 Table `tab:results` 的全部指标，
      用于交叉核对。**输出的是本机真实测量值，不是构造出来的"示例数据"。**

严格按 main.tex §4.1 的模型实现：
  Stage 1:  min  sum_j x_Dj
            s.t. x_ij <= Q_ij
                 sum_j x_ij <= 1        (每位教师至多一个岗位)
                 sum_i x_ij + x_Dj = V_j
  Stage 2:  min  sum_ij (a*Dhat + b*Mhat + g*U(R) + rho*I) x_ij
            s.t. 同上 + sum_j x_Dj = V*_unmet        <-- 式 (6)

关键诊断：同时报告两种口径的 "Dummy"
  - unmet_total  = sum_j x_Dj          （按 Proposition 1 应为常量）
  - unmet_school = #{j : x_Dj > 0}     （可随权重变化）
这一项直接回答 main.tex 中 Remark 2 的 P0 问题。

用法:
    python3 citax.py --scale 1.0 --seeds 3
    python3 citax.py --scale 0.25 --seeds 5     # 快速试跑
"""

import argparse, json, platform, time
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

# ----------------------------------------------------------------------
# 实例生成
# ----------------------------------------------------------------------

GRID_W, GRID_H = 48.0, 52.0        # km, 见 main.tex §5.1
N_DISTRICTS    = 5
N_SUBJECTS     = 8

# 2 个衰退老城区 + 3 个新兴郊区，幅度取自 main.tex §5.1
DISTRICT_GROWTH = np.array([-0.146, -0.146, +0.372, +0.372, +0.372])


@dataclass
class Instance:
    n_teachers: int
    n_schools: int
    school_xy: np.ndarray          # (m,2)
    school_district: np.ndarray    # (m,)
    school_subject: np.ndarray     # (m,) 该校缺额学科
    vacancies: np.ndarray          # (m,) V_j
    teacher_xy: np.ndarray         # (n,2)
    teacher_district: np.ndarray   # (n,)
    teacher_subject: np.ndarray    # (n,)
    seed: int

    @property
    def total_vacancies(self):
        return int(self.vacancies.sum())


def neyman_scott(rng, n_target, kappa=0.012, sigma=3.5):
    """Neyman-Scott 聚集点过程。kappa=母点强度(/km^2), sigma=子点扩散(km)。

    先按强度生成母点，再在每个母点周围撒各向同性高斯子点，
    最后裁剪到恰好 n_target 个（论文需要固定的学校数）。
    """
    area = GRID_W * GRID_H
    n_parents = max(3, rng.poisson(kappa * area))
    parents = np.column_stack([
        rng.uniform(0, GRID_W, n_parents),
        rng.uniform(0, GRID_H, n_parents),
    ])
    per_parent = int(np.ceil(n_target / n_parents)) + 3
    pts = []
    for px, py in parents:
        k = rng.poisson(per_parent)
        if k == 0:
            continue
        pts.append(np.column_stack([
            rng.normal(px, sigma, k),
            rng.normal(py, sigma, k),
        ]))
    pts = np.vstack(pts)
    pts[:, 0] = np.clip(pts[:, 0], 0, GRID_W)
    pts[:, 1] = np.clip(pts[:, 1], 0, GRID_H)
    idx = rng.permutation(len(pts))[:n_target]
    if len(idx) < n_target:                       # 极少数种子下补足
        extra = rng.uniform([0, 0], [GRID_W, GRID_H], (n_target - len(idx), 2))
        return np.vstack([pts[idx], extra])
    return pts[idx]


def assign_district(xy):
    """把平面按 5 个区中心做最近邻划分（Voronoi 近似真实行政区划）。"""
    centers = np.array([
        [GRID_W * 0.35, GRID_H * 0.45],   # 老城 1
        [GRID_W * 0.50, GRID_H * 0.55],   # 老城 2
        [GRID_W * 0.20, GRID_H * 0.82],   # 新区 1
        [GRID_W * 0.80, GRID_H * 0.25],   # 新区 2
        [GRID_W * 0.82, GRID_H * 0.78],   # 新区 3
    ])
    d = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
    return d.argmin(axis=1)


def make_instance(seed, n_teachers=5236, n_schools=114) -> Instance:
    rng = np.random.default_rng(seed)

    school_xy = neyman_scott(rng, n_schools)
    school_district = assign_district(school_xy)
    school_subject = rng.integers(0, N_SUBJECTS, n_schools)

    # 缺额规模由该区人口增减驱动：新区缺口大，老城缺口小
    g = DISTRICT_GROWTH[school_district]
    lam = np.clip(9.0 + 60.0 * g, 1.0, None)       # 期望缺额数
    vacancies = rng.poisson(lam) + 1               # 每校至少 1 个缺额

    # 教师住在城里，同样聚集；学科均匀分布
    teacher_xy = neyman_scott(rng, n_teachers, kappa=0.02, sigma=5.0)
    teacher_district = assign_district(teacher_xy)
    teacher_subject = rng.integers(0, N_SUBJECTS, n_teachers)

    return Instance(n_teachers, n_schools, school_xy, school_district,
                    school_subject, vacancies, teacher_xy,
                    teacher_district, teacher_subject, seed)


# ----------------------------------------------------------------------
# 成本要素
# ----------------------------------------------------------------------

def build_cost_components(inst: Instance, rng, commute_kmh=25.0):
    """返回 (dist_km, commute_min, Q, M_hat, R, I) 全部为 (n,m) 稠密阵。"""
    d = np.linalg.norm(
        inst.teacher_xy[:, None, :] - inst.school_xy[None, :, :], axis=2)

    commute_min = d / commute_kmh * 60.0

    # 法定任职资格 Q：可任教即可（本学科或相邻学科），是硬约束
    subj_gap = np.abs(inst.teacher_subject[:, None] - inst.school_subject[None, :])
    Q = subj_gap <= 1

    # 学科匹配度 M：Q=1 内部仍有优劣，非本学科任教记为 mismatch，是软成本。
    # 把 Q(硬) 与 M(软) 分开，"mismatch 率"才可能随权重 beta 变化；
    # 若 Q 直接要求学科完全相符，则 mismatch 恒为 0。
    M_hat = subj_gap / (N_SUBJECTS - 1)

    # 跨区指示
    I = (inst.teacher_district[:, None] != inst.school_district[None, :]).astype(np.float64)

    # 序数抗拒 R ∈ {0,1,5,10}
    # 无 LLM 可用，故按可复现的生成过程模拟：通勤越长、跨区，抗拒越高
    z = 0.030 * commute_min + 0.55 * I + rng.normal(0, 0.45, d.shape)
    R = np.select(
        [z < 0.65, z < 1.35, z < 2.10],
        [0.0, 1.0, 5.0],
        default=10.0,
    )
    return d, commute_min, Q, M_hat, R, I


def utility(R, k=0.2):
    """main.tex 式 (7)。k->0 时退化为线性归一化 R/10。"""
    if k < 1e-9:
        return R / 10.0
    return (np.exp(k * R) - 1.0) / (np.exp(10.0 * k) - 1.0)


# ----------------------------------------------------------------------
# 两阶段字典序求解
# ----------------------------------------------------------------------

def _build_matrices(rows, cols, n, m, V, n_arcs):
    """构造 A_ub (教师容量) 与 A_eq (学校平衡)。"""
    # A_ub: 每位教师一行，其所有出弧系数为 1
    A_ub = sp.csr_matrix(
        (np.ones(n_arcs), (rows, np.arange(n_arcs))),
        shape=(n, n_arcs + m))

    # A_eq: 每所学校一行，入弧系数 1，加上自己的 dummy 变量
    data = np.concatenate([np.ones(n_arcs), np.ones(m)])
    ri = np.concatenate([cols, np.arange(m)])
    ci = np.concatenate([np.arange(n_arcs), n_arcs + np.arange(m)])
    A_eq = sp.csr_matrix((data, (ri, ci)), shape=(m, n_arcs + m))
    return A_ub, A_eq


def solve_lexicographic(inst, C_full, Q, commute_min, threshold=60.0,
                        do_pricing=True, verbose=False):
    """返回解与全部诊断指标。"""
    n, m = inst.n_teachers, inst.n_schools
    V = inst.vacancies.astype(float)

    feasible = Q & np.isfinite(C_full)
    n_candidate = int(feasible.sum())              # 满足资格的弧
    keep = feasible & (commute_min <= threshold)   # 再施加空间剪枝
    n_kept = int(keep.sum())
    pruned_mask = feasible & ~keep

    rows, cols = np.nonzero(keep)
    n_arcs = len(rows)
    costs = C_full[rows, cols]

    A_ub, A_eq = _build_matrices(rows, cols, n, m, V, n_arcs)
    bounds = [(0, 1)] * n_arcs + [(0, float(v)) for v in V]

    # ---------- Stage 1: 最大化匹配量（目标中不含任何成本） ----------
    c1 = np.concatenate([np.zeros(n_arcs), np.ones(m)])
    t0 = time.perf_counter()
    r1 = linprog(c1, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq, b_eq=V,
                 bounds=bounds, method="highs")
    t_stage1 = time.perf_counter() - t0
    if not r1.success:
        raise RuntimeError(f"Stage 1 failed: {r1.message}")
    V_unmet = float(r1.fun)
    it1 = int(getattr(r1, "nit", 0) or 0)

    # ---------- Stage 2: 最小化复合成本，锁定未匹配总量 ----------
    loops, it2, t_stage2 = 0, 0, 0.0
    while True:
        loops += 1
        n_arcs = len(rows)
        costs = C_full[rows, cols]
        A_ub, A_eq = _build_matrices(rows, cols, n, m, V, n_arcs)
        bounds = [(0, 1)] * n_arcs + [(0, float(v)) for v in V]

        # 追加式 (6)：sum_j x_Dj = V*_unmet
        lock = sp.csr_matrix(
            (np.ones(m), (np.zeros(m, dtype=int), n_arcs + np.arange(m))),
            shape=(1, n_arcs + m))
        A_eq2 = sp.vstack([A_eq, lock], format="csr")
        b_eq2 = np.concatenate([V, [V_unmet]])

        c2 = np.concatenate([costs, np.zeros(m)])
        t0 = time.perf_counter()
        r2 = linprog(c2, A_ub=A_ub, b_ub=np.ones(n), A_eq=A_eq2, b_eq=b_eq2,
                     bounds=bounds, method="highs")
        t_stage2 += time.perf_counter() - t0
        if not r2.success:
            raise RuntimeError(f"Stage 2 failed: {r2.message}")
        it2 += int(getattr(r2, "nit", 0) or 0)

        if not do_pricing or pruned_mask.sum() == 0:
            break

        # ---------- 精确定价：扫描被剪掉的弧 ----------
        u = np.asarray(r2.ineqlin.marginals)          # <= 约束，min 问题下 <= 0
        v = np.asarray(r2.eqlin.marginals)[:m]
        pr, pc = np.nonzero(pruned_mask)
        rc = C_full[pr, pc] - u[pr] - v[pc]
        viol = rc < -1e-9
        if not viol.any():
            break
        add_r, add_c = pr[viol], pc[viol]
        rows = np.concatenate([rows, add_r])
        cols = np.concatenate([cols, add_c])
        pruned_mask[add_r, add_c] = False
        if verbose:
            print(f"      定价轮 {loops}: 加回 {viol.sum()} 条弧")

    x = r2.x[:len(rows)]
    x_dummy = r2.x[len(rows):]

    # 整数性检验（TUM 的经验确认）
    frac = float(np.abs(x - np.round(x)).max()) if len(x) else 0.0

    return dict(
        rows=rows, cols=cols, x=x, x_dummy=x_dummy,
        V_unmet=V_unmet,
        unmet_total=float(x_dummy.sum()),
        unmet_schools=int((x_dummy > 1e-7).sum()),
        n_candidate=n_candidate, n_kept_initial=n_kept,
        n_arcs_final=len(rows),
        pricing_loops=loops,
        simplex_iter=it1 + it2,
        cpu_s=t_stage1 + t_stage2,
        max_fractional=frac,
    )


# ----------------------------------------------------------------------
# 场景
# ----------------------------------------------------------------------

SCENARIOS = [
    ("DOEM (pure distance)", 1.0, 0.0, 0.0),
    ("High physical cost",   0.8, 0.1, 0.1),
    ("Balanced",             0.4, 0.3, 0.3),
    ("High humanistic",      0.1, 0.1, 0.8),
    ("Pure humanistic",      0.0, 0.0, 1.0),
]


def run_seed(seed, scale=1.0, rho=0.15, k=0.2, threshold=60.0, verbose=False):
    n_t = max(200, int(5236 * scale))
    n_s = max(20, int(114 * scale))
    inst = make_instance(seed, n_t, n_s)
    rng = np.random.default_rng(seed + 10_000)
    dist, commute, Q, M_hat, R, I = build_cost_components(inst, rng)

    total_arcs = inst.n_teachers * inst.n_schools
    D_hat = np.clip(commute / threshold, 0, 1)      # 归一化通勤
    U = utility(R, k)

    # TSA 基线：禁止跨区流动
    C_tsa = np.where(I > 0, np.inf, 0.4 * D_hat + 0.3 * M_hat + 0.3 * U)
    out = []

    t0 = time.perf_counter()
    res_tsa = solve_lexicographic(inst, C_tsa, Q, commute,
                                  threshold=1e9, do_pricing=False)
    tsa_time = time.perf_counter() - t0
    sel = res_tsa["x"] > 0.5
    gap_tsa = np.abs(inst.teacher_subject[res_tsa["rows"][sel]]
                     - inst.school_subject[res_tsa["cols"][sel]]) if sel.any() else np.array([0])
    out.append(dict(
        scenario="TSA (static baseline)", weights=None,
        mismatch_pct=100.0 * float((gap_tsa > 0).mean()),
        unfilled_pct=100.0 * res_tsa["unmet_total"] / inst.total_vacancies,
        commute_km=float(dist[res_tsa["rows"][sel], res_tsa["cols"][sel]].mean()) if sel.any() else 0.0,
        resistance=float(R[res_tsa["rows"][sel], res_tsa["cols"][sel]].mean()) if sel.any() else 0.0,
        unmet_total=res_tsa["unmet_total"],
        unmet_schools=res_tsa["unmet_schools"],
        cpu_s=round(tsa_time, 3),
        simplex_iter=res_tsa["simplex_iter"],
        pruning_pct=0.0, pricing_loops=0,
        max_fractional=res_tsa["max_fractional"],
    ))

    for name, a, b, g in SCENARIOS:
        C = a * D_hat + b * M_hat + g * U + rho * I
        C = np.where(Q, C, np.inf)
        r = solve_lexicographic(inst, C, Q, commute, threshold, verbose=verbose)
        sel = r["x"] > 0.5
        rr, cc = r["rows"][sel], r["cols"][sel]
        gap = np.abs(inst.teacher_subject[rr] - inst.school_subject[cc]) if sel.any() else np.array([0])
        out.append(dict(
            scenario=name, weights=(a, b, g),
            mismatch_pct=100.0 * float((gap > 0).mean()),
            unfilled_pct=100.0 * r["unmet_total"] / inst.total_vacancies,
            commute_km=float(dist[rr, cc].mean()) if sel.any() else 0.0,
            resistance=float(R[rr, cc].mean()) if sel.any() else 0.0,
            unmet_total=r["unmet_total"],
            unmet_schools=r["unmet_schools"],
            cpu_s=round(r["cpu_s"], 3),
            simplex_iter=r["simplex_iter"],
            pruning_pct=100.0 * (1 - r["n_kept_initial"] / max(1, r["n_candidate"])),
            pricing_loops=r["pricing_loops"],
            max_fractional=r["max_fractional"],
        ))

    meta = dict(seed=seed, n_teachers=inst.n_teachers, n_schools=inst.n_schools,
                total_arcs=total_arcs, total_vacancies=inst.total_vacancies,
                qualified_arcs=int(Q.sum()), rho=rho, k=k, threshold=threshold)
    return meta, out


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--out", default="citax_results.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("City-X 参考实现  —  真实测量输出")
    print("=" * 78)
    print(f"平台   : {platform.platform()}")
    print(f"Python : {platform.python_version()}   NumPy {np.__version__}")
    import scipy
    print(f"求解器 : SciPy {scipy.__version__} / HiGHS (linprog method='highs')")
    print(f"设置   : scale={args.scale}  seeds={args.seeds}  rho={args.rho}  k={args.k}")
    print()

    all_runs = []
    for s in range(args.seeds):
        t0 = time.perf_counter()
        meta, rows = run_seed(s, args.scale, args.rho, args.k, verbose=args.verbose)
        print(f"[seed {s}] |T|={meta['n_teachers']}  |S|={meta['n_schools']}  "
              f"sum V_j={meta['total_vacancies']}  合格弧={meta['qualified_arcs']:,}"
              f"  ({time.perf_counter()-t0:.1f}s)")
        all_runs.append(dict(meta=meta, rows=rows))

    with open(args.out, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\n结果已写入 {args.out}")
    return all_runs


if __name__ == "__main__":
    main()
