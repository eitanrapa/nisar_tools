"""The side-band tie against the ionosphere alone: many synthetic ionospheres (as `synth.ionosphere` draws them) plus
white noise on the cells, split by the through-going trace with its 2 km gap, nothing planted. Every estimate should be
0; its spread over seeds is the estimator's error from the ionosphere's structure, with more seeds than the full
synthetic scenes can afford (no deformation, no unwrapping: those are exact shifts, `run_tie_synthetic.py`).

    .../envs/remote_sensing/bin/python -u run_tie_iono_only.py [seeds=20] [kriging=1] [workers=3]
        -> ../results/tie_iono_only.csv
"""
import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T  # noqa: E402
import synth  # noqa: E402

VARIANTS = {"default": {}, "rough": dict(gradient=0.6, rf_rms=8.0, rf_km=8.0),
            "tid": dict(gradient=0.6, rf_rms=4.0, rf_km=15.0, tid_amp=3.0, tid_km=25.0)}
NOISE = 0.7                      # rad per cell, as the screens' cells at coherence ~0.5-0.6


def geometry():
    x, y = synth.grid()
    k = int(round(synth.CELL / synth.PX))
    X, Y = np.meshgrid(x, y)
    xc = synth._cells(X, k)[0] / (k * k)
    yc = synth._cells(Y, k)[:, 0] / (k * k)
    Xc, Yc = np.meshgrid(xc, yc)
    dist = synth.signed_distance(Xc, Yc, synth.LENGTH_THROUGH)
    s = np.array([np.sin(np.deg2rad(synth.STRIKE_DEG)), np.cos(np.deg2rad(synth.STRIKE_DEG))])
    return X, Y, k, xc, yc, dist, Xc * s[0] + Yc * s[1]


def one(job):
    variant, seed, kriging = job
    X, Y, k, xc, yc, dist, along = geometry()
    io = synth.ionosphere(X, Y, np.random.default_rng(seed), **VARIANTS[variant])
    d = synth._cells(io, k) / (k * k) + NOISE * np.random.default_rng(1000 + seed).standard_normal(dist.shape)
    use = np.abs(dist) > 2000.0
    sig = np.full(d.shape, NOISE)
    bn, bs = dist > 0, dist < 0
    p = T.block_tie(d, use, bn, bs, xc, yc, sigma=sig, reach_m=12e3, window_m=20e3, along=along, deg=2, min_cells=60)
    row = dict(variant=variant, seed=seed, polynomial=p["e"], poly_spread=p["spread"], poly_windows=len(p["windows"]),
               poly_worst=max((abs(w["e"]) for w in p["windows"]), default=np.nan))
    if kriging:
        g = T.gp_tie(d, use, bn, bs, xc, yc, sigma=sig, reach_m=12e3, window_m=20e3, along=along)
        row.update(kriging=g["e"], kriging_wmean=g["e_wmean"], kriging_se=g["se_wmean"])
    print(f"{variant} seed {seed}: polynomial {p['e']:+.2f}" + (f", kriging {row['kriging']:+.2f}" if kriging else ""),
          flush=True)
    return row


def main():
    opt = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    seeds, kriging, workers = int(opt.get("seeds", 20)), int(opt.get("kriging", 1)), int(opt.get("workers", 3))
    jobs = [(v, s, kriging) for v in VARIANTS for s in range(seeds)]
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(one, jobs, chunksize=1)
    else:
        rows = [one(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "..", "results", "tie_iono_only.csv"), index=False)
    cols = ["polynomial"] + (["kriging"] if kriging else [])
    for c in cols:
        g = df.groupby("variant")[c].agg(rms=lambda v: float(np.sqrt(np.mean(v ** 2))), worst=lambda v: float(np.abs(v).max()),
                                         wrong=lambda v: int((np.abs(v) > 0.5).sum()), n="count")
        print(c)
        print(g.round(3))


if __name__ == "__main__":
    main()
