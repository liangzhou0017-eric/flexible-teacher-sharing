# Flexible Teacher Sharing: Reproduction Code

Code and data accompanying

> Zhou, L., Yu, J. *Exact Lexicographic Assignment with Costly Cost Coefficients:
> Optimistic Label Acquisition for Cross-Boundary Teacher Sharing.*
> Submitted to *Computers & Operations Research*.

Release v2.0.0 adds the comparative experiments of that paper (Section 5) to
release v1.0.0, which accompanied an earlier version of the work. The scripts
and data of v1.0.0 are retained unchanged.

## Requirements

Python 3.9+, `numpy`, `scipy` (>= 1.9, for `linprog(method="highs")` and `milp`),
`pandas` (case study only).

```
pip install -r requirements.txt
```

No solver licence is required; the HiGHS backend ships with SciPy.

## Contents

| File | Purpose |
|---|---|
| `citax.py` | City-X synthetic benchmark: instance generation and cost components |
| `adapter_cor.py` | Builds the full qualified graph and the cost components of the paper; `python adapter_cor.py` checks the regenerated instances against previously published values |
| `exp_label_pricing.py` | The six solution schemes compared in Section 5 and the experiment driver |
| `run_all.py` | Runs the check, the main experiment and the scaling experiment |
| `analyze.py`, `make_figure.py` | Produce the numbers, tables and Figure 1 of Section 5 from the raw CSVs |
| `main_a_raw.csv`, `main_b_raw.csv` | Raw results of the main experiment (seeds 0–3 and 4–7), one row per run |
| `scaling_raw.csv` | Raw results of the scaling experiment |
| `*_summary.csv`, `Figure_1.tex` | Summaries and figure source generated from the raw results |
| `solver.py` | Two-stage lexicographic assignment with exact edge augmentation (v1.0.0) |
| `ablation.py` | Big-M, branch-and-bound and linear-utility ablations; rho sweep (v1.0.0) |
| `sensitivity.py` | Type II support-set tolerances; stability under label jitter (v1.0.0) |
| `casestudy.py` | Haishu District accessible-supply analysis |
| `data_haishu_schools.csv` | De-identified school-level case study data |

## Solution schemes (Section 5)

| Scheme | Description |
|---|---|
| `full` | Label every arc of the candidate graph E and solve both stages on E (ground truth) |
| `prune_only` | Label and solve on the pruned graph E' only, no pricing (heuristic) |
| `price_s2` | Stage 1 on E' without pricing, Stage 2 priced over E |
| `price_both` | Two-stage pricing: both stages priced over E (needs every label) |
| `screened` | Ablation: label-free Stage 1 on E, bound-screened pricing in Stage 2 |
| `proposed` | Optimistic label acquisition (OLA, Algorithm 1) |

## Reproducing the results of the paper

Comparative experiments (Section 5):

```bash
python adapter_cor.py                            # check the regenerated instances
python run_all.py --quick                        # minutes: test the setup
python run_all.py                                # check + main + scaling experiments
python analyze.py main_a_raw.csv main_b_raw.csv  # tables and numbers of Section 5
python make_figure.py                            # Figure 1
```

`run_all.py` writes `label_pricing_raw.csv` and `scaling_raw.csv`; pass them to
`analyze.py` to analyse a fresh run instead of the archived results. Results are
written row by row, and an interrupted run resumes where it stopped. Label counts
and exactness are deterministic given the seeds; running times depend on the
machine (those reported in the paper were measured on a 2-core Intel Xeon at
2.10 GHz, see Section 5.1).

Case study (Section 6):

```bash
python casestudy.py                              # binary partition
python casestudy.py --partition sub_district     # finer partition
```

Scripts of release v1.0.0:

```bash
python solver.py --seeds 30                      # scenario table
python ablation.py                               # ablations and rho sweep
python sensitivity.py                            # Type II and jitter analysis
python solver.py --seeds 6 --threshold 18 --no-stage1-pricing   # Stage 1 bias
```

## Case study data

`data_haishu_schools.csv` contains one row per school:

| Column | Description |
|---|---|
| `school_id` | Anonymised identifier |
| `level` | `primary`, `junior` or `nine-year` |
| `zone` | `core` (pre-2016 district) or `transferred` (merged in 2016) |
| `sub_district` | Township or sub-district, where determined from policy documents |
| `lon_gcj02`, `lat_gcj02` | Coordinates in the GCJ-02 datum |
| `coord_source` | `geocoded`, or `township-centroid` for four schools whose addresses resolved only to a district centroid |
| `teachers_in_roster` | Teachers matched to the school in the education bureau's in-service training roster |

Notes and limitations, discussed further in the paper:

- The training roster is a sample, not a census; coverage is roughly 57% of the
  establishment headcount implied by the national staffing standard.
- No teacher-level records are distributed. `teachers_in_roster` is an aggregate
  count; names, identifiers and contact details are excluded.
- School names are omitted. The underlying directory is published by the Ningbo
  Haishu District Education Bureau.
- Four coordinates are township centroids; excluding them leaves the reported
  magnitudes unchanged.

## Citation

If you use this code, please cite the accompanying paper. Machine-readable
metadata is provided in `CITATION.cff`.

## License

Code released under the MIT License (`LICENSE`). The case study data is derived
from publicly published administrative sources and is provided for verification
of the reported analysis.
