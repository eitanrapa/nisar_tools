"""Which fill suits which gap: synthetic ionospheres plus white noise on the cells, gaps cut out, refilled.

usage: .../envs/remote_sensing/bin/python -u run_fill_synthetic.py [seeds=8] [workers=3]

The ionospheres of `run_tie_iono_only.py` (default, rough: 8 rad at 8 km, travelling waves of 3 rad at 25 km) on the
synthetic scene's 240 m cells, 0.7 rad of noise per cell. Five gaps: the through-going fault zone at 1, 2 and 4 km
half-width (data on both sides), a 15 km square bay (data all around) and a 12 km strip at the grid's edge (data on one
side only, as NISAR's partially focused near range). Each is refilled by the filter's local plane at one width (1-15 km)
or by the ladders 1.5-3-6-12 km and 2.6-5-10 km (`split_spectrum.ionosphere_filter`'s lever band), all with reach
12 km. The error
is the screen minus the cell-averaged truth, levelled by its median over the data beyond 20 km of the gap, binned by
distance into the gap (and 0-2 / 2-5 km outside it). Writes ``../results/fill_synthetic.csv``.
"""

import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402,F401  (the filter's nanmedian, sped up in this process)
import synth  # noqa: E402
from nisar_tools import split_spectrum as ss  # noqa: E402

VARIANTS = {"default": {}, "rough": dict(gradient=0.6, rf_rms=8.0, rf_km=8.0),
            "tid": dict(gradient=0.6, rf_rms=4.0, rf_km=15.0, tid_amp=3.0, tid_km=25.0)}
NOISE = 0.7
LADDER = (1500.0, 3000.0, 6000.0, 12000.0)
LADDER26 = (2600.0, 5000.0, 10000.0)
FILLS = {**{f"plane {w / 1e3:g}": w for w in (1000.0, 1500.0, 2000.0, 2600.0, 4000.0, 6000.0, 10000.0, 15000.0)},
         "ladder": LADDER, "ladder 2.6": LADDER26}
IN_BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 12))


def gaps(Xc, Yc, dist):
    return {"fault 1": np.abs(dist) < 1000.0, "fault 2": np.abs(dist) < 2000.0, "fault 4": np.abs(dist) < 4000.0,
            "bay 15": (np.abs(Xc + 10000) < 7500) & (np.abs(Yc + 15000) < 7500),
            "edge 12": Xc > Xc.max() - 12000.0}


def one(job):
    variant, seed = job
    x, y = synth.grid()
    k = int(round(synth.CELL / synth.PX))
    X, Y = np.meshgrid(x, y)
    Xc, Yc = np.meshgrid(synth._cells(X, k)[0] / (k * k), synth._cells(Y, k)[:, 0] / (k * k))
    dist = synth.signed_distance(Xc, Yc, synth.LENGTH_THROUGH)
    io = synth.ionosphere(X, Y, np.random.default_rng(seed), **VARIANTS[variant])
    truth = synth._cells(io, k) / (k * k)
    d = truth + NOISE * np.random.default_rng(1000 + seed).standard_normal(truth.shape)
    sig = np.full(d.shape, NOISE)
    rows = []
    for gname, gap in gaps(Xc, Yc, dist).items():
        mask = ~gap
        D_in = ndimage.distance_transform_edt(gap) * synth.CELL
        D_out = ndimage.distance_transform_edt(mask) * synth.CELL
        for fname, fm in FILLS.items():
            scr, _ = ss.ionosphere_filter(d, sig, mask, synth.CELL, fill="plane", fill_m=fm, fill_reach_m=12000.0)
            e = scr - truth
            e = e - np.median(e[mask & (D_out > 20000.0)])
            row = dict(variant=variant, seed=seed, gap=gname, fill=fname)
            for lo, hi in IN_BINS:
                m = gap & (D_in > lo * 1e3) & (D_in <= hi * 1e3)
                row[f"in_{lo}_{hi}"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() > 20 else np.nan
            for lo, hi in ((0, 2), (2, 5)):
                m = mask & (D_out > lo * 1e3) & (D_out <= hi * 1e3)
                row[f"out_{lo}_{hi}"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() > 20 else np.nan
            rows.append(row)
    print(variant, seed, flush=True)
    return rows


def main():
    opt = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    seeds, workers = int(opt.get("seeds", 8)), int(opt.get("workers", 3))
    jobs = [(v, s) for v in VARIANTS for s in range(seeds)]
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(HERE, "..", "results", "fill_synthetic.csv"), index=False)
    mm = 0.24196 / (4 * np.pi) * 1e3
    cols = [c for c in df.columns if c.startswith("in_") or c.startswith("out_")]
    g = df.groupby(["gap", "variant", "fill"], sort=False)[cols].apply(lambda t: np.sqrt((t ** 2).mean())) * mm
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(g.round(1))


if __name__ == "__main__":
    main()
