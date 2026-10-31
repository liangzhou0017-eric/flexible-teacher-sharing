# Comparative experiments: exact lexicographic assignment with costly labels

Files added for the C&OR version.

- `citax.py` — City-X instance generator (unchanged from v1.0.0).
- `adapter_cor.py` — builds the full qualified graph and the cost components (normalized commute, subject mismatch, cross-district indicator, ordinal labels) exactly as in the paper; `python adapter_cor.py` checks the regenerated instances against previously published values.
- `exp_label_pricing.py` — the six solution schemes (full labelling, prune-only, Stage-2 pricing, two-stage pricing, bound-screened pricing, optimistic label acquisition) and the experiment driver.
- `run_all.py` — runs the check, the main experiment and the scaling experiment, and zips the results.
- `analyze.py`, `make_figure.py` — produce the tables and Figure 1 of Section 5.
- `main_a_raw.csv`, `main_b_raw.csv` — raw results of the main experiment (seeds 0–3 and 4–7).
- `scaling_raw.csv` — raw results of the scaling experiment.

Reproduce:

    python run_all.py           # check + main + scaling
    python analyze.py main_a_raw.csv main_b_raw.csv
    python make_figure.py

Requirements: Python ≥ 3.10, NumPy, SciPy (HiGHS via scipy.optimize.linprog).
