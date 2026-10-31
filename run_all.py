#!/usr/bin/env python3
"""
run_all.py —— 一键运行 C&OR 改投所需的全部新实验。

用法（在本文件所在文件夹里）：
    python run_all.py            # 校验 + 主实验 + 规模实验，约 2-4 小时
    python run_all.py --quick    # 只跑校验和一个小样（约 5 分钟），确认环境没问题

运行结束后，本文件夹里会生成 cor_results.zip，把它发给我即可。
中途关掉也没关系：已完成的结果是逐行写入 CSV 的。
"""
import argparse
import os
import platform
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable


def run(args, log):
    print(f"\n>>> {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([PY, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            fh.write(line)
            if ("OK" in line and "seed=" in line) or "不精确" in line or "PASS" in line or "FAIL" in line \
                    or "已写入" in line or "汇总" in line or "Error" in line or "Traceback" in line:
                print("   " + line.rstrip(), flush=True)
        p.wait()
    print(f"    ({(time.time() - t0) / 60:.1f} 分钟，日志 {log})", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print(f"Python {platform.python_version()}  {platform.platform()}")
    print(f"CPU: {platform.processor()}  核数: {os.cpu_count()}")

    # 1. 校验
    if run(["adapter_cor.py"], "log_check.txt") != 0:
        print("\n校验未通过，已停止。请把 log_check.txt 发给我。")
        sys.exit(1)

    if a.quick:
        run(["exp_label_pricing.py", "--seed-list", "0", "--thresholds", "18", "60",
             "--weights", "balanced", "--out", "quick"], "log_quick.txt")
        print("\n小样跑通。正式运行请去掉 --quick。")
        return

    # 2. 主实验：8 个 seed × 2 种口径 × 3 组权重 × 7 个阈值 × 6 种方案
    run(["exp_label_pricing.py", "--out", "label_pricing"], "log_main.txt")

    # 3. 规模实验：标签数随实例规模的变化
    run(["exp_label_pricing.py", "--sizes", "2000x50,3500x80,5236x114,8000x170,12000x250",
         "--seeds", "3", "--regimes", "nofeas", "--weights", "balanced", "--thresholds", "60",
         "--methods", "full", "price_both", "proposed", "--out", "scaling"], "log_scaling.txt")

    # 4. 打包
    files = [f for f in os.listdir(HERE)
             if f.startswith(("label_pricing_", "scaling_", "log_")) and f.endswith((".csv", ".tex", ".txt"))]
    with zipfile.ZipFile("cor_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)
    print(f"\n全部完成。请把 {os.path.join(HERE, 'cor_results.zip')} 发给我。")


if __name__ == "__main__":
    main()
