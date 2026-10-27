"""The filter's fill on REAL ionosphere: fake gaps cut out of the production screens' raw estimates, refilled.

usage: .../envs/remote_sensing/bin/python -u run_fill_null.py [workers=3] [windows=6] [fills=1|2]

In each of the eight scenes of `run_tie_null.py` (the Panama north blocks, the Venezuela main regions), windows of
100 x 100 km inside the block, at least 15 km from the real trace and at least 60% used, are cut out of the raw
dispersive estimate. In each window:

- ``fault``: a 40 km segment through the centre, at a random orientation, has the cells within ``w`` = 1, 2 or 4 km of
  it removed (a fake fault zone, data on both sides);
- ``edge``: everything beyond a line through the centre is removed, and the error read 0-12 km beyond it (a one-sided
  strip, as NISAR's partially focused near range).

Each gap is refilled by the filter with each fill (planes of 1.5, 2.6 and 10 km; the ladder 1.5-3-6-12 km, also
with the first-order final step; ISCE3's guide) and compared with the same filter run on all the window's data: the
difference inside the gap is the fill's error against the ionosphere the data saw, at the filter's resolution. Writes
``../results/fill_null.csv``.
"""

import functools
import os
import sys
import time

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T  # noqa: E402
import estimators as E  # noqa: E402,F401  (the filter's nanmedian, sped up in this process)
from nisar_tools import split_spectrum as ss  # noqa: E402
from run_nisar import signed_distance  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
WINDOW_M, SEGMENT_M, AWAY_M = 100e3, 40e3, 15e3
WIDTHS = (1000.0, 2000.0, 4000.0)
LADDER = (1500.0, 3000.0, 6000.0, 12000.0)
LADDER26 = (2600.0, 5000.0, 10000.0)
FILLS = {"plane 1.5": dict(fill="plane", fill_m=1500.0), "plane 2.6": dict(fill="plane", fill_m=2600.0),
         "plane 10": dict(fill="plane", fill_m=10000.0), "ladder": dict(fill="plane", fill_m=LADDER),
         "ladder, order 1": dict(fill="plane", fill_m=LADDER, order=1), "guide": dict(fill="guide")}
#: the second pass (fills=2): a ladder from the guide's own width, and wider planes
FILLS2 = {"plane 4": dict(fill="plane", fill_m=4000.0), "plane 6": dict(fill="plane", fill_m=6000.0),
          "ladder 2.6": dict(fill="plane", fill_m=LADDER26),
          "ladder 2.6, order 1": dict(fill="plane", fill_m=LADDER26, order=1)}
BANDS_IN = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 12))
SCENES = (("D141", "v8"), ("D040", "v8"), ("A076", "v8"), ("A004", "v6"), ("A061", None), ("D126", None),
          ("A162", None), ("D054", None))


def load(track, screen):
    if screen is not None:
        import panama_scenes as P
        S = P.load(track, screen, roff=False)
        return S, S["block_n"]
    import venezuela_scenes as V
    S = V.load(track)
    return S, S["region"]


def windows(S, region, n, rng):
    """Up to ``n`` window centres (cell indices) whose 100 km square lies in the block, 60% used, away from the trace."""
    X, Y = np.meshgrid(S["x"], S["y"])
    far = np.abs(signed_distance(X, Y, *S["trace"])) > AWAY_M
    ok = S["use"] & region & far
    half = int(round(WINDOW_M / 2 / S["cell"]))
    frac = ndimage.uniform_filter(ok.astype(float), 2 * half + 1, mode="constant")
    cand = np.argwhere(frac > 0.6)
    cand = cand[(cand[:, 0] >= half) & (cand[:, 0] < ok.shape[0] - half) & (cand[:, 1] >= half) & (cand[:, 1] < ok.shape[1] - half)]
    out = []
    for i in rng.permutation(len(cand)):
        c = cand[i]
        if all(np.hypot(*(c - o)) * S["cell"] > 0.6 * WINDOW_M for o in out):
            out.append(c)
        if len(out) == n:
            break
    return out, half, ok


def job(args):
    track, screen, nwin, fills = args
    FILLS = globals()["FILLS"] if fills == 1 else FILLS2
    t0 = time.time()
    S, region = load(track, screen)
    rng = np.random.default_rng(sum(map(ord, track)))
    centres, half, ok = windows(S, region, nwin, rng)
    sig_all = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
    cell = S["cell"]
    rows = []
    for wi, (ci, cj) in enumerate(centres):
        sl = (slice(ci - half, ci + half + 1), slice(cj - half, cj + half + 1))
        d, use, sig = S["d"][sl], S["use"][sl] & np.isfinite(S["d"][sl]), sig_all[sl]
        yy, xx = np.mgrid[sl[0], sl[1]]
        px, py = (xx - cj) * cell, (ci - yy) * cell
        th = rng.uniform(0, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        along, across = px * u[0] + py * u[1], -px * u[1] + py * u[0]
        refs = {f: ss.ionosphere_filter(d, sig, use, cell, fill_reach_m=12000.0, **kw)[0] for f, kw in FILLS.items()}
        gaps = []
        for w in WIDTHS:
            seg = np.hypot(np.clip(np.abs(along) - SEGMENT_M / 2, 0, None), across)
            gaps.append((f"fault {w / 1e3:g}", seg < w, seg))
        gaps.append(("edge", across > 0, across))
        for gname, gap, _ in gaps:
            test_use = use & ~gap
            D = ndimage.distance_transform_edt(~test_use) * cell            # distance to the data left
            D_gap = ndimage.distance_transform_edt(~gap) * cell              # distance to the gap
            for f, kw in FILLS.items():
                scr = ss.ionosphere_filter(d, sig, test_use, cell, fill_reach_m=12000.0, **kw)[0]
                e = scr - refs[f]
                row = dict(scene=f"{track} {screen or 'v6'}", window=wi, orient_deg=float(np.rad2deg(th)), gap=gname,
                           fill=f, gap_cells=int((gap & use).sum()))
                inner = gap & use                  # where the withheld data were
                for a, b in BANDS_IN:
                    m = inner & (D > a * 1e3) & (D <= b * 1e3)
                    row[f"in_{a}_{b}"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() > 20 else np.nan
                m = inner & np.isfinite(e)
                row["in_all"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() > 20 else np.nan
                for a, b in ((0, 2), (2, 5)):
                    mb = test_use & (D_gap > a * 1e3) & (D_gap <= b * 1e3)
                    row[f"out_{a}_{b}"] = float(np.sqrt(np.mean(e[mb] ** 2))) if mb.sum() > 20 else np.nan
                rows.append(row)
    print(f"{track} {screen or 'v6'}: {len(centres)} windows [{time.time() - t0:.0f}s]")
    return rows


def main():
    opt = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    workers, nwin, fills = int(opt.get("workers", 3)), int(opt.get("windows", 6)), int(opt.get("fills", 1))
    jobs = [(t, s, nwin, fills) for t, s in SCENES]
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    path = os.path.join(RESULTS, "fill_null.csv")
    if fills != 1 and os.path.exists(path):          # the second pass adds its fills to the first's rows
        old = pd.read_csv(path)
        df = pd.concat([old[~old["fill"].isin(df["fill"].unique())], df], ignore_index=True)
    df.to_csv(path, index=False)
    mm = 0.24196 / (4 * np.pi) * 1e3
    cols = ["in_0_1", "in_1_2", "in_2_4", "in_4_8", "in_8_12", "in_all", "out_0_2", "out_2_5"]
    g = df.groupby(["gap", "fill"])[cols].apply(lambda t: np.sqrt((t ** 2).mean())) * mm
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(g.round(1))
    print("->", os.path.join(RESULTS, "fill_null.csv"))


if __name__ == "__main__":
    main()
