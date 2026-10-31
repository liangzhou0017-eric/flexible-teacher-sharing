#!/usr/bin/env python3
"""
exp_label_pricing.py
====================
C&OR 改投版的核心实验：在"标签获取有成本"的前提下，比较六种求解方案
要了多少标签、花了多少时间、结果是否精确。

背景（为什么要这组实验）
------------------------
原稿说剪枝是为了"少要标签"，但第二阶段定价要算被剪掉弧的约化成本
    cbar_ij = C_ij - u_i - v_j,   C_ij = a_ij + gamma * U(R_ij)
而 C_ij 里含标签 R_ij —— 所以原稿的两阶段定价其实需要 E 上全部标签，
剪枝并没有省标签。新方法用两点修掉这个漏洞：

  (1) 第一阶段（最大覆盖）不含任何成本系数，直接在完整图 E 上求解，
      完全不需要标签，也从源头上消除"幻影缺编"。
  (2) 第二阶段用下界筛选定价：U(R) >= 0，所以
          a_ij - u_i - v_j  <=  cbar_ij
      若下界 >= 0，该弧可证明不会改进目标，无需标签；只有下界 < 0 的弧
      才去要标签，再算精确约化成本。终止时所有弧满足对偶可行，解在 E 上
      精确最优。

五种方案
--------
  full        : 给 E 上全部弧打标签，两阶段都在 E 上直接求解（真值）
  prune_only  : 只给剪枝后的 E' 打标签，两阶段都在 E' 上求解，不定价（启发式）
  price_s2    : 第一阶段在 E' 上求解（不定价），第二阶段在 E 上定价
                —— 常见做法；需要 E 全部标签；覆盖量有偏
  price_both  : 原稿方法。两阶段都定价；第二阶段定价需要 E 全部标签
  screened    : 消融。第一阶段在 E 上免标签求解；第二阶段工作集
                = E' ∪ 第一阶段最优匹配的支撑，下界筛选定价
  proposed    : 新方法。第一阶段在 E 上免标签求解；第二阶段用"乐观成本"
                （未知标签按下界 0 计）做列生成，只对最优解实际用到的弧要标签，
                反复直到最优解用到的弧全部已知 —— 精确，且标签数接近"实际
                分配人数"这一下限

用法
----
  # 1) 先用自带的演示生成器自测流程（不代表论文结果）
  python3 exp_label_pricing.py --demo --seeds 2 --thresholds 0 18 60

  # 2) 在 adapter_cor.py 里接上你的生成器，跑校验
  python3 adapter_cor.py

  # 3) 正式实验（默认网格，约 1-2 小时，视机器而定）
  python3 exp_label_pricing.py

  产出：label_pricing_raw.csv（逐次记录）、label_pricing_summary.csv、
        label_pricing_table.tex。把这三个文件和终端最后的汇总输出发回即可。

依赖：numpy, scipy（与论文一致：SciPy linprog + HiGHS）
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
K_UTIL = 0.2          # 效用变换系数 k，论文取 0.2
RHO = 0.15            # 跨区清算惩罚，论文参考值
WEIGHTS = {           # (alpha, beta, gamma)
    "high_physical":   (0.8, 0.1, 0.1),
    "balanced":        (0.4, 0.3, 0.3),
    "high_humanistic": (0.1, 0.1, 0.8),
}


def utility(R, k=K_UTIL):
    return (np.exp(k * R) - 1.0) / (np.exp(10.0 * k) - 1.0)


# ============================================================================
# 实例来源：正式实验用 adapter_cor.build_instance；--demo 用下面的演示生成器
# ============================================================================
def demo_instance(seed, n_teachers=5236, n_schools=114, total_vac=2273):
    """仿 City-X 的演示实例，只用于检查流程，不用于论文结果。"""
    rng = np.random.default_rng(seed)
    W, H = 48.0, 52.0
    # 学校：簇状分布
    n_par = max(3, rng.poisson(0.012 * W * H))
    parents = np.column_stack([rng.uniform(0, W, n_par), rng.uniform(0, H, n_par)])
    sch = parents[rng.integers(0, n_par, n_schools)] + rng.normal(0, 3.5, (n_schools, 2))
    sch = np.clip(sch, 0, [W, H])
    # 教师：住在学校附近（围绕随机学校扩散）
    tea = sch[rng.integers(0, n_schools, n_teachers)] + rng.normal(0, 8.0, (n_teachers, 2))
    tea = np.clip(tea, 0, [W, H])
    district = lambda p: np.minimum((p[:, 0] / W * 5).astype(int), 4)
    d_s, d_t = district(sch), district(tea)
    subj_s = rng.integers(0, 8, n_schools)
    subj_t = rng.integers(0, 8, n_teachers)
    # 空缺
    w = rng.gamma(2.0, 1.0, n_schools)
    vac = np.floor(w / w.sum() * total_vac).astype(int)
    vac[: total_vac - vac.sum()] += 1
    # 合格弧：学科相邻
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
# 标签预言机：只有被请求过的弧才"知道"标签，并计数
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
# LP 构造与求解
# ============================================================================
class Network:
    """缓存完整图 E 的约束矩阵；子图通过列索引切片得到。"""

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
        s.t. T x <= 1                    (教师，对偶 u <= 0)
             S x + y = V                 (学校，对偶 v 自由)
             1' y = vstar  (仅第二阶段)  (对偶 w)
             0 <= x <= 1,  y >= 0
        返回 obj, x(按 cols 顺序), y, u, v
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
            raise RuntimeError(f"LP 求解失败: {res.message}")
        u = res.ineqlin.marginals            # <= 0
        v = res.eqlin.marginals[:n_s]
        return res.fun, res.x[:k], res.x[k:], u, v


def max_frac(x):
    return float(np.max(np.abs(x - np.round(x)))) if len(x) else 0.0


# ---------------------------------------------------------------------------
# 第一阶段
# ---------------------------------------------------------------------------
def stage1(net, cols, price_over=None):
    """
    最小化未满足需求。cols: 初始工作集。若给出 price_over（候选全集的列索引），
    则在其上做约化成本定价直至对偶可行（第一阶段成本为 0，定价不需要标签）。
    返回 V*, 最终工作集, 支撑弧（x=1 的弧，全局索引）, 定价轮数
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
# 第二阶段
# ---------------------------------------------------------------------------
def stage2(net, cols, vstar, a, C, oracle, price_over=None, screen=False):
    """
    最小化综合成本，保持 sum y = vstar。
    a: 不含标签项的成本（下界），C: 完整成本（只在标签已知时可用）
    price_over: 候选全集列索引；None 表示不定价
    screen=False：传统定价，对所有候选弧算精确约化成本（需要全部标签）
    screen=True ：下界筛选，只对下界约化成本 < 0 的弧请求标签
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
    新方法：乐观标签获取（optimistic label acquisition）。
    未知标签的弧按下界 a_ij 计价（U(R)>=0，故 a_ij <= C_ij），已知标签的弧按真成本计价。
      内层：在乐观成本下用列生成把 LP 解到 E 上最优（定价只用乐观成本，不需要标签）；
      外层：最优解支撑里若有未知标签的弧，就只对这些弧请求标签，更新成本，重复。
    终止时支撑全部已知标签：C(x) = c~(x) = min c~ <= min C，故 x 在 E 上精确最优。
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
# 六种方案
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
    a = alpha * d_hat + beta * m_hat + RHO * cross      # 免标签部分（下界）
    C = a + gamma * utility(labels)                      # 完整成本
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
    if "screened" in methods:                                  # 消融：下界筛选定价
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, supp, r1, mf1 = stage1(net, E)                 # 免标签，完整图
        init = np.union1d(Ep, supp)                           # 保证第二阶段可行
        obj2, w2, r2, mf2 = stage2(net, init, v1, a, C, o, price_over=E, screen=True)
        rec("screened", t0, o, v1, obj2, r1, r2, w2, max(mf1, mf2))
    if "proposed" in methods:                                  # 新方法：乐观标签获取
        t0 = time.perf_counter(); o = LabelOracle(labels)
        v1, _, supp, r1, mf1 = stage1(net, E)                 # 免标签，完整图
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
# 主程序
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用演示生成器（只测流程）")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="直接指定 seed 列表（覆盖 --seeds），便于并行分批")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0, 12, 18, 24, 30, 45, 60],
                    help="剪枝阈值 T_prune（分钟）；0 表示初始工作集只含第一阶段最优匹配")
    ap.add_argument("--weights", nargs="+", default=list(WEIGHTS))
    ap.add_argument("--regimes", nargs="+", default=["nofeas", "feas60"],
                    help="nofeas: E=完整合格图（主实验口径）；feas60: E=通勤<=60分钟")
    ap.add_argument("--sizes", type=str, default=None,
                    help="规模实验，如 '2000x50,5236x114,12000x250'（需 adapter 支持）")
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
        print(f"# 续跑：已读入 {len(rows)} 行", flush=True)
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
                                ("  OK" if row["s1_bias"] == 0 and abs(row["s2_gap_pct"]) < 1e-6 else "  !!! 不精确")
                            print(f"seed={seed} {regime:7s} {wname:15s} T={T:>4g} {m:11s} "
                                  f"labels={r['labels_pct']:6.2f}%  V*={r['s1_unmet']:5d} (true {truth[0]})  "
                                  f"gap={row['s2_gap_pct']:+.4f}%  t={r['time_s']:.2f}s{ok}", flush=True)

    print(f"\n已写入 {args.out}_raw.csv（{len(rows)} 行，逐行写入，中途中断也不会丢）")
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
    """按 (规模, 口径, 权重, 阈值, 方法) 汇总：标签比例、覆盖偏差、成本差距、时间。"""
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
    print(f"已写入 {path}")

    # 终端汇总：每个口径 × 权重，展示 proposed 与 price_both 的标签比例
    print("\n=== 汇总：标签请求比例（%）与精确性 ===")
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

    # LaTeX 表（主口径 nofeas，balanced 权重；其余口径见 CSV）
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
    print(f"已写入 {tex}")


if __name__ == "__main__":
    main()
