"""The main band taken out pixel by pixel, and the fault zone it lets back in (`synth.py`, the truth known).

usage: .../envs/remote_sensing/bin/python -u run_pixel.py [workers=3] [seeds=5] [quick=1] [example=1]

The through-going rupture of `run_synthetic.py` (D141-like damage zone, far-field coherence 0.6, the default
ionosphere), at 0 and 4 m of slip, with and without the rough near-field deformation of `synth.NEARFIELD_D141` (the
Okada field changes by ~3 rad/km within 2 km of the trace, D141's main band by a median 13-24 rad/km north of its
trace). Three ways to form the cells, all from the same pixel draws:

- ``cell``: each band summed over the 240 m cells, the scaled main band taken out of the side band's cell phase
  (the production screens, `split_spectrum_screen` with ``phase_a``);
- ``pixel``: the main band taken out of each 40 m pixel of both bands before the sums (`gslc_cells(phase_model=)`),
  the model being the A pixels' own phase after a one-pixel Gaussian on the true integers;
- ``oracle``: the same with the true A phase as the model.

Each runs with the cells within 0, 0.5, 1 and 2 km of the trace left out, through three filters: v6's (a 10 km
plane fill, reach 12 km, ISCE3's weighted Gaussian mean), ``matched`` (the fill matched to the gap,
`split_spectrum.ionosphere_filter`'s ``fill_m`` ladder 1.5-3-6-12 km) and ``matched1`` (that fill, and a weighted
local plane through the final Gaussian, ``order=1``).
The library's own `split_spectrum_screen` makes every screen. Scored against the true ionosphere (levelled beyond 20
km): rms by distance from the trace, the 2-4 km step, and the leak (`metrics.synthetic`); and the raw estimate on the
cells used, by distance.

Writes `../results/pixel.csv` and `../results/pixel_example.npz` (4 m, near field, seed 0; ``example=1`` makes that
scene alone, for the figure).
"""

import functools
import os
import sys
import time

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402,F401  (the filter's nanmedian, sped up in this process)
import metrics as M  # noqa: E402
import synth  # noqa: E402
from nisar_tools import split_spectrum as ss  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
EXCLUDE = (0.0, 500.0, 1000.0, 2000.0)
LADDER = (1500.0, 3000.0, 6000.0, 12000.0)
FILTERS = {"v6": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=10000.0, fill_reach_m=12000.0),
           "matched": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=LADDER, fill_reach_m=12000.0),
           "matched1": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=LADDER, fill_reach_m=12000.0, order=1)}
BINS = ((0, .5), (.5, 1), (1, 2), (2, 5), (5, 10), (10, 20))
ARMS = ("cell", "pixel", "oracle")


def raw_scores(out, truth):
    """The raw estimate on the cells it used, against the true ionosphere (levelled by the median beyond 20 km)."""
    d, use, dist = out.dispersive.values, out.mask.values, truth["dist"]
    e = d - truth["iono"]
    far = use & (np.abs(dist) > 20e3)
    e = e - np.median(e[far])
    land = truth["land"]
    row = {}
    for a, b in BINS:
        band = land & (np.abs(dist) >= a * 1e3) & (np.abs(dist) < b * 1e3)
        m = band & use
        row[f"used_{a}_{b}"] = float(m.sum() / max(band.sum(), 1))
        row[f"raw_rms_{a}_{b}"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() > 10 else np.nan
        row[f"raw_med_{a}_{b}"] = float(np.median(e[m])) if m.sum() > 10 else np.nan
        row[f"cohb_{a}_{b}"] = float(np.nanmedian(out.coherence_b.values[band])) if band.any() else np.nan
    return row


def screen_scores(screen, truth):
    row, err = M.synthetic(screen, truth, synth.CELL)
    land, dist = truth["land"], truth["dist"]
    for a, b in BINS:
        m = land & np.isfinite(err) & (np.abs(dist) >= a * 1e3) & (np.abs(dist) < b * 1e3)
        row[f"rms_{a}_{b}"] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else np.nan
    return row, err


def run_scene(job):
    slip, nf, seed, keep = job
    t0 = time.time()
    ca, cb, tr = synth.scene(slip=slip, seed=seed, length=synth.LENGTH_THROUGH,
                             nearfield=synth.NEARFIELD_D141 if nf else None, pixel_models=("multilook", "oracle"))
    cells = {"cell": (ca, cb, tr["phase_a"]), "pixel": (*tr["pixel_cells"]["multilook"], None),
             "oracle": (*tr["pixel_cells"]["oracle"], None)}
    rows, keepers = [], {}
    for arm in ARMS:
        a, b, ph = cells[arm]
        for excl in EXCLUDE:
            exclude = np.abs(tr["dist"]) < excl if excl > 0 else None
            for fname, fkw in FILTERS.items():
                out = ss.split_spectrum_screen(a, b, ph, valid=tr["land"], exclude=exclude, **fkw)
                srow, err = screen_scores(out.screen.values, tr)
                rows.append(dict(slip=slip, nearfield=int(nf), seed=seed, arm=arm, exclude_m=excl, filter=fname,
                                 **srow, **raw_scores(out, tr)))
                if keep and ((fname in ("matched", "matched1") and excl in (0.0, 500.0, 2000.0))
                             or (fname == "v6" and excl == 2000.0)):
                    keepers[f"{arm}_{int(excl)}_{fname}"] = dict(screen=out.screen.values.astype(np.float32),
                                                         raw=out.dispersive.values.astype(np.float32),
                                                         coh_b=out.coherence_b.values.astype(np.float32))
    r = pd.DataFrame(rows)
    sel = r[r["filter"] == "matched1"]
    mm = 0.24196 / (4 * np.pi) * 1e3
    print(f"slip {slip} nf {int(nf)} seed {seed} [{time.time() - t0:.0f}s]: "
          + "; ".join(f"{a} excl {e / 1e3:.1f}: {sel[(sel.arm == a) & (sel.exclude_m == e)].rms_0_2.iloc[0] * mm:.1f}/"
                      f"{sel[(sel.arm == a) & (sel.exclude_m == e)].rms_2_5.iloc[0] * mm:.1f} mm"
                      for a in ARMS for e in (0.0, 2000.0)))
    extra = None
    if keep:
        extra = dict(truth_iono=tr["iono"].astype(np.float32), truth_psi=tr["psi"].astype(np.float32),
                     dist=tr["dist"].astype(np.float32), x=tr["x"], y=tr["y"],
                     **{f"{k}_{v}": a for k, d in keepers.items() for v, a in d.items()})
    return rows, extra


def main():
    opt = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    workers, seeds = int(opt.get("workers", 3)), int(opt.get("seeds", 5))
    quick, example = int(opt.get("quick", 0)), int(opt.get("example", 0))
    synth.unit_slip_los(synth.LENGTH_THROUGH)
    if quick or example:
        jobs = [(4.0, True, 0, True)]
    else:
        jobs = [(s, nf, k, (s == 4.0 and nf and k == 0)) for s in (0.0, 4.0) for nf in (False, True) for k in range(seeds)]
    rows, extra = [], None
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for r, ex in pool.imap_unordered(run_scene, jobs):
                rows += r
                extra = ex if ex is not None else extra
    else:
        for j in jobs:
            r, ex = run_scene(j)
            rows += r
            extra = ex if ex is not None else extra
    df = pd.DataFrame(rows)
    name = "pixel_quick" if quick else "pixel"
    if not example:                          # the example alone (its screens for the figure) leaves pixel.csv alone
        df.to_csv(os.path.join(RESULTS, f"{name}.csv"), index=False)
    if extra is not None:
        np.savez_compressed(os.path.join(RESULTS, f"{name}_example.npz"), **extra)
    mm = 0.24196 / (4 * np.pi) * 1e3
    cols = [f"rms_{a}_{b}" for a, b in BINS] + ["step"]
    g = df.groupby(["nearfield", "slip", "filter", "arm", "exclude_m"])[cols].mean() * mm
    g["leak"] = df.groupby(["nearfield", "slip", "filter", "arm", "exclude_m"])["leak"].mean()
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(g.round(2))
    print("wrote", os.path.join(RESULTS, f"{name}_example.npz" if example else f"{name}.csv"))


if __name__ == "__main__":
    main()
