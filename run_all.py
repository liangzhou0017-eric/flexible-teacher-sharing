#!/usr/bin/env python3
"""
run_all.py -- runs all comparative experiments of Section 5 in one go.

Usage (from this folder):
    python run_all.py            # check + main experiment + scaling, a few hours
    python run_all.py --quick    # check + one small run (minutes), to test the setup

Results are written to CSV row by row, so an interrupted run resumes where it
stopped. At the end all results and logs are collected in cor_results.zip.
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
            if ("OK" in line and "seed=" in line) or "NOT EXACT" in line or "PASS" in line or "FAIL" in line \
                    or "written:" in line or "summary" in line or "Error" in line or "Traceback" in line:
                print("   " + line.rstrip(), flush=True)
        p.wait()
    print(f"    ({(time.time() - t0) / 60:.1f} min, log {log})", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print(f"Python {platform.python_version()}  {platform.platform()}")
    print(f"CPU: {platform.processor()}  cores: {os.cpu_count()}")

    # 1. check
    if run(["adapter_cor.py"], "log_check.txt") != 0:
        print("\nCheck failed; stopped. See log_check.txt.")
        sys.exit(1)

    if a.quick:
        run(["exp_label_pricing.py", "--seed-list", "0", "--thresholds", "18", "60",
             "--weights", "balanced", "--out", "quick"], "log_quick.txt")
        print("\nQuick run finished. Run without --quick for the full experiments.")
        return

    # 2. main experiment: 8 seeds x 2 regimes x 3 weight settings x 7 thresholds x 6 schemes
    run(["exp_label_pricing.py", "--out", "label_pricing"], "log_main.txt")

    # 3. scaling experiment: labels queried against instance size
    run(["exp_label_pricing.py", "--sizes", "2000x50,3500x80,5236x114,8000x170,12000x250",
         "--seeds", "3", "--regimes", "nofeas", "--weights", "balanced", "--thresholds", "60",
         "--methods", "full", "price_both", "proposed", "--out", "scaling"], "log_scaling.txt")

    # 4. collect results
    files = [f for f in os.listdir(HERE)
             if f.startswith(("label_pricing_", "scaling_", "log_")) and f.endswith((".csv", ".tex", ".txt"))]
    with zipfile.ZipFile("cor_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)
    print(f"\nDone. Results collected in {os.path.join(HERE, 'cor_results.zip')}")


if __name__ == "__main__":
    main()
