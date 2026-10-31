#!/usr/bin/env python3
"""
adapter_cor.py —— 已接好你的 City-X 生成器（citax.py），不需要修改。

与原稿主实验完全一致：
  实例        citax.make_instance(seed, n_t, n_s)
  成本要素    citax.build_cost_components(inst, default_rng(seed + 10000))
  归一化通勤  D^ = clip(commute / 60, 0, 1)
  学科错配    M^ = |学科差| / 7（合格弧内取 0 或 1/7）
  标签        R 由有序 probit 生成器给出，覆盖全部教师-学校对（含 60 分钟外长弧）

单独运行 `python adapter_cor.py` 会用原稿已发表的数字做校验。
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
    """用原稿已发表的数字校验：实例、标签、目标值、第一阶段定价表。"""
    import exp_label_pricing as X
    ok = True

    def rep(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name:40s} {detail}", flush=True)

    print("=== 校验（seed 0）===")
    inst = build_instance(0)
    comm, R, c = inst["commute"], inst["labels"], inst["cross"]
    at60 = comm <= 60
    rep("60 分钟内候选弧数 = 93,381", at60.sum() == 93381, f"{at60.sum()}")
    sh = {k: 100 * np.mean(R[at60] == k) for k in (0, 1, 5, 10)}
    ref = {0: 17.4, 1: 28.5, 5: 34.8, 10: 19.3}
    rep("标签分布与原稿一致", max(abs(sh[k] - ref[k]) for k in ref) <= 0.05,
        " / ".join(f"{k}:{sh[k]:.1f}%" for k in sh))
    rep("严重阻力弧中跨区比例 = 94.1%", abs(100 * np.mean(c[at60 & (R == 10)]) - 94.1) <= 0.05)
    _, truth, _, _ = X.run_methods(inst, X.WEIGHTS["balanced"], 60.0, None, None, methods=["full"])
    rep("balanced 最优目标值 = 261.0947", abs(truth[1] - 261.0947) < 1e-3, f"{truth[1]:.4f}")
    keep = at60
    net = X.Network(inst["n_teachers"], inst["n_schools"], inst["vacancies"], inst["arcs"][keep])
    v12, *_ = X.stage1(net, np.where(comm[keep] <= 12)[0])
    rep("第一阶段定价表：阈值 12 分钟幻影缺编 = 673", v12 == 673, f"{v12}")
    print("全部通过。" if ok else "有 FAIL，请把上面的输出发给我。", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
