"""Panama D141 near the trace: the screen recipes on the main band as cut, and with its local cycles repaired.

usage: .../envs/remote_sensing/bin/python -u run_d141_integers.py

The cells and tied main band of `run_nisar.py` (Section 5's cut, the south block on the side band's tie). The same pair
was also unwrapped without the cut (v5: a phase model from the range offsets taken out before SNAPHU and put back), and
the two unwrappings' integers differ, cell by cell, beyond each block's modal difference: in patches near the trace,
mostly north of it. Here each such cell takes v5's integer (``repaired``), on the per-cell form, and the screens are
remade with the production filter (a 10 km plane fill, ISCE3's weighted mean) and with the alternatives of Section 7
(narrower fills, the ladders, the first-order filter), the 2 km fault zone left out. v5's integers are not known to be
right either; the comparison asks how much each recipe moves when the integers do.

Writes ``../results/d141_integers.csv`` (each recipe's median minus the production screen's, by signed distance from the
trace, as cut and repaired) and ``../results/d141_repair.csv`` (the share of cells repaired, by distance and side).
"""

import functools
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402,F401  (the filter's nanmedian, sped up in this process)
import run_nisar as RN  # noqa: E402
from nisar_tools import split_spectrum as ss  # noqa: E402

RESULTS = RN.RESULTS
WS = os.path.expanduser("~/Documents/data/D141_panama/workspace_reg")
RECIPES = {"production": dict(fill="plane", fill_m=10000.0, fill_reach_m=12000.0),
           "production, first order": dict(fill="plane", fill_m=10000.0, fill_reach_m=12000.0, order=1),
           "plane 2.6": dict(fill="plane", fill_m=2600.0, fill_reach_m=12000.0),
           "ladder 1.5": dict(fill="plane", fill_m=(1500.0, 3000.0, 6000.0, 12000.0), fill_reach_m=12000.0),
           "ladder 1.5, first order": dict(fill="plane", fill_m=(1500.0, 3000.0, 6000.0, 12000.0), fill_reach_m=12000.0,
                                           order=1),
           "ladder 2.6, first order": dict(fill="plane", fill_m=(2600.0, 5000.0, 10000.0), fill_reach_m=12000.0, order=1)}
EDGES = np.r_[-10, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 10]
LOS_MM = 299792458.0 / 1239e6 / (4 * np.pi) * 1e3


def main():
    I = RN.inputs_d141()
    ca, cb, pa, valid = I["ca"], I["cb"], I["phase_a"], I["valid"]
    X, Y = np.meshgrid(ca.x.values, ca.y.values)
    dist = RN.signed_distance(X, Y, *I["trace"])
    v6 = xr.open_dataset(f"{WS}/iono_split_spectrum_v6_s5.nc").sel(x=ca.x, y=ca.y, method="nearest")
    v5 = xr.open_dataset(f"{WS}/iono_split_spectrum_v5_s5.nc").sel(x=ca.x, y=ca.y, method="nearest")
    k = np.round((v6["phase_a"].values - v5["phase_a"].values) / (2 * np.pi))
    local = np.zeros(k.shape)
    for side in (dist > 0, dist < 0):
        m = side & np.isfinite(k) & (np.abs(dist) > 10e3)
        mode = np.bincount((k[m] - k[m].min()).astype(int)).argmax() + k[m].min()
        local = np.where(side & np.isfinite(k), k - mode, local)
    rep = []
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 20)):
        for side, sgn in (("N", 1), ("S", -1)):
            m = valid & np.isfinite(k) & (sgn * dist >= lo * 1e3) & (sgn * dist < hi * 1e3)
            rep.append(dict(side=side, lo_km=lo, hi_km=hi, cells=int(m.sum()), repaired=float(np.mean(local[m] != 0)),
                            plus1=float(np.mean(local[m] == 1)), minus1=float(np.mean(local[m] == -1)),
                            more=float(np.mean(np.abs(local[m]) > 1))))
            print(f"{side} {lo}-{hi} km: {rep[-1]['repaired']:6.1%} of {m.sum()} cells repaired "
                  f"(+1 {rep[-1]['plus1']:5.1%}, -1 {rep[-1]['minus1']:5.1%}, more {rep[-1]['more']:5.1%})")
    pd.DataFrame(rep).to_csv(os.path.join(RESULTS, "d141_repair.csv"), index=False)
    near = ss.fault_distance(ca, *I["trace"]) < 2000.0
    import panama_pipeline as PP                    # loaded by inputs_d141: the drawn trace's branches stay out too
    for bx, by in PP.drawn_branches():
        near |= ss.fault_distance(ca, bx, by) < 2000.0
    screens = {}
    for integ, ph in (("as cut", pa), ("repaired", pa - 2 * np.pi * local)):
        for name, kw in RECIPES.items():
            screens[(integ, name)] = ss.split_spectrum_screen(ca, cb, ph, valid=valid, exclude=near, **kw).screen.values
            print(f"  {integ:9s} {name}")
    ref = screens[("as cut", "production")]
    far = valid & (np.abs(dist) > 5e3) & (np.abs(dist) < 20e3)
    rows = []
    for (integ, name), s in screens.items():
        g = s - np.median((s - ref)[far])
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = valid & (dist >= lo * 1e3) & (dist < hi * 1e3)
            rows.append(dict(integers=integ, recipe=name, lo_km=lo, hi_km=hi,
                             mm=float(np.median((g - ref)[m]) * LOS_MM)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "d141_integers.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(df.pivot_table(index=["integers", "recipe"], columns="lo_km", values="mm", sort=False).round(1))


if __name__ == "__main__":
    main()
