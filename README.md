# Flexible Teacher Sharing: Reproduction Code

Code and data accompanying *Optimizing Educational Resources: A Lexicographic
Assignment Framework for Flexible Teacher Sharing with AI-Assisted Ordinal
Cost Estimation*.

## Requirements

Python 3.9+, `numpy`, `scipy` (>= 1.9 for `linprog(method="highs")` and `milp`),
`pandas` (case study only).

```
pip install -r requirements.txt
```

No solver licence is required; the HiGHS backend ships with SciPy.

## Contents

| File | Purpose |
|---|---|
| `citax.py` | City-X synthetic benchmark: instance generation and cost components |
| `solver.py` | Two-stage lexicographic assignment with exact edge augmentation |
| `ablation.py` | Big-M, branch-and-bound and linear-utility ablations; rho sweep |
| `sensitivity.py` | Type II support-set tolerances; stability under label jitter |
| `casestudy.py` | Haishu District accessible-supply analysis |
| `data_haishu_schools.csv` | De-identified school-level case study data |

## Reproducing the reported results

```bash
python solver.py --seeds 30                    # main scenario table
python ablation.py                             # ablations and rho sweep
python sensitivity.py                          # Type II and jitter analysis
python casestudy.py                            # case study, binary partition
python casestudy.py --partition sub_district   # case study, finer partition
```

Each script writes a JSON file recording the run environment, the settings and
the results. Runtimes on a 12-core laptop: roughly 20 minutes for `solver.py`
at 30 seeds, 1 minute for `ablation.py`, 25 minutes for `sensitivity.py`
(dominated by per-arc re-solves), and seconds for `casestudy.py`.

## Stage 1 pricing

Stage 1 is solved on the spatially pruned graph and determines `V*_unmet`, which
constraint (6) then fixes for Stage 2. Pricing applied only in Stage 2 improves
cost but cannot recover assignment volume lost to pruning, so the reported
shortfall is biased upward. Both stages therefore price excluded arcs. The bias
can be reproduced with:

```bash
python solver.py --seeds 6 --threshold 18 --no-stage1-pricing
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
