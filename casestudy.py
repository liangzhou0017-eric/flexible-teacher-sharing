"""Haishu District case study: accessible qualified teacher supply.

Measures, for each school, the number of same-level teachers reachable within a
commute limit, comparing supply within the school's own administrative unit
against the district as a whole. This is the empirical analogue of the Stage 1
structural effect and requires no per-school vacancy data.
"""
import argparse
import json

import numpy as np
import pandas as pd

SPEED_KMH = 25.0
THRESHOLDS = [15, 20, 30, 45, 60]


def haversine(lat, lon):
    R = 6371.0
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dla, dlo = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    h = (np.sin(dla / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(h))


def accessible(D, same_level, cross, supply, minutes):
    reach = (D <= SPEED_KMH * minutes / 60) & same_level
    within = (reach & ~cross) @ supply - supply
    total = reach @ supply - supply
    return within.sum(), total.sum(), within, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_haishu_schools.csv")
    ap.add_argument("--partition", default="zone", choices=["zone", "sub_district"])
    ap.add_argument("--drop-centroid", action="store_true",
                    help="exclude schools located by township centroid")
    ap.add_argument("--out", default="results_casestudy.json")
    args = ap.parse_args()

    g = pd.read_csv(args.data)
    if args.drop_centroid:
        g = g[g["coord_source"] == "geocoded"]
    g = g[(g["teachers_in_roster"] > 0) & g[args.partition].notna()]
    g = g[g[args.partition].astype(str) != ""].reset_index(drop=True)
    print("%d schools, %d teachers, partition=%s (%d units)"
          % (len(g), g["teachers_in_roster"].sum(), args.partition,
             g[args.partition].nunique()))

    D = haversine(g["lat_gcj02"], g["lon_gcj02"])
    lv = g["level"].to_numpy(str)
    same = lv[:, None] == lv[None, :]
    z = g[args.partition].to_numpy(str)
    cross = z[:, None] != z[None, :]
    supply = g["teachers_in_roster"].to_numpy(float)

    rows = []
    print("\n%9s%9s%14s%13s%9s" % ("limit", "radius", "within", "district", "gain"))
    for mins in THRESHOLDS:
        w, t, _, _ = accessible(D, same, cross, supply, mins)
        gain = 100 * (t - w) / max(w, 1)
        print("%7d m%9.1f%14s%13s%8.1f%%"
              % (mins, SPEED_KMH * mins / 60, format(int(w), ","),
                 format(int(t), ","), gain))
        rows.append(dict(minutes=mins, within=int(w), district=int(t),
                         gain_pct=round(gain, 1)))

    _, _, w30, t30 = accessible(D, same, cross, supply, 30)
    g = g.assign(within=w30, district=t30)
    g["gain_pct"] = 100 * (g["district"] - g["within"]) / g["within"].clip(lower=1)

    by_unit = []
    print("\nby %s (30 min):" % args.partition)
    for u, sub in g.groupby(args.partition):
        gain = 100 * (sub["district"].sum() - sub["within"].sum()) / max(sub["within"].sum(), 1)
        print("  %-14s n=%2d  within %7s  district %7s  gain %7.1f%%"
              % (u, len(sub), format(int(sub["within"].sum()), ","),
                 format(int(sub["district"].sum()), ","), gain))
        by_unit.append(dict(unit=u, n=len(sub), within=int(sub["within"].sum()),
                            district=int(sub["district"].sum()),
                            gain_pct=round(gain, 1)))

    print("\nmost affected schools:")
    for _, r in g.nlargest(5, "gain_pct").iterrows():
        print("  %-8s %-10s within %4d -> district %4d  (+%.0f%%)"
              % (r["school_id"], r["level"], r["within"], r["district"],
                 r["gain_pct"]))

    json.dump(dict(settings=vars(args), thresholds=rows, by_unit=by_unit),
              open(args.out, "w"), indent=2, default=float)
    print("written to " + args.out)


if __name__ == "__main__":
    main()
