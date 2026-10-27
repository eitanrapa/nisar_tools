"""The range-offset tie against the side band's, on the same cells and in the same geometry.

The Panama pipelines (`panama_pipeline._south_block_tie`) tie the south block by m = 4 pi / lambda * range offset - phi_A:
a plane a + b s + c n per side on cells 6-12 km from the trace (s along its chord), the two planes compared at the
trace every 2 km of s, the median. Here that geometry runs on the cells, on m and on the raw dispersive estimate,
and each field's departure from its own far-field plane (6-20 km) is binned by distance from the trace.

Both fields read the block's integer (m at -2 pi per cycle added, the raw estimate at +3.21 rad) and the ionosphere
(m twice, at -2, the raw estimate once), so their difference in one geometry is the step of m + 2 x screen across
the trace: what the offsets and the phase disagree on, beyond the integer and the ionosphere.

    .../envs/remote_sensing/bin/python -u run_tie_offsets.py        -> ../results/tie_offsets.csv, tie_departures.csv
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
import panama_scenes as P  # noqa: E402
from run_nisar import signed_distance  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
BINS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8))


def chord_frame(X, Y, tx, ty):
    tx, ty = np.asarray(tx, float), np.asarray(ty, float)
    if tx[-1] < tx[0]:
        tx, ty = tx[::-1], ty[::-1]
    u = np.array([tx[-1] - tx[0], ty[-1] - ty[0]])
    L = float(np.hypot(*u))
    u /= L
    return signed_distance(X, Y, tx, ty), (X - tx[0]) * u[0] + (Y - ty[0]) * u[1], L


def pipeline_tie(f, ok, n, s, L, bn, bs, per_cycle, s_hi=None):
    """Cycles to add to the south block the pipeline's way. ``per_cycle``: f's change on the south block per cycle
    added to its main band (+3.21 rad for the raw estimate, -2 pi for m)."""
    s_hi = L if s_hi is None else s_hi
    fits = {}
    for nm, blk, sgn in (("N", bn, 1), ("S", bs, -1)):
        b = ok & blk & (sgn * n > 6e3) & (sgn * n < 12e3) & (s > 0) & (s < s_hi)
        if b.sum() < 30:
            return np.nan, np.nan
        A = np.c_[np.ones(b.sum()), s[b] / 1e4, np.abs(n[b]) / 1e4]
        fits[nm] = np.linalg.lstsq(A, f[b], rcond=None)[0]
    s_tr = np.arange(2e3, s_hi - 1e3, 2e3)
    jump = (fits["S"][0] + fits["S"][1] * s_tr / 1e4) - (fits["N"][0] + fits["N"][1] * s_tr / 1e4)
    cyc = -jump / per_cycle
    return float(np.median(cyc)), float(np.ptp(cyc))


def main():
    rows, deps = [], []
    for track, screen in (("D141", "v6"), ("D141", "v8"), ("D040", "v8"), ("A076", "v8"), ("A004", "v6")):
        S = P.load(track, screen)
        X, Y = np.meshgrid(S["x"], S["y"])
        n, s, L = chord_frame(X, Y, *S["trace"])
        s_hi = 16e3 if track == "A004" else None
        okd = S["use"] & np.isfinite(S["d"])
        okm = np.isfinite(S["m"])
        pd_, pd_r = pipeline_tie(S["d"], okd, n, s, L, S["block_n"], S["block_s"], T.STEP, s_hi)
        pm_, pm_r = pipeline_tie(S["m"], okm, n, s, L, S["block_n"], S["block_s"], -2 * np.pi, s_hi)
        k = int(S["ties"]["south_block_cycles"])
        rows.append(dict(scene=f"{track} {screen}", offsets_k=k, offsets_measured=k + float(S["ties"]["south_residual_cycles"]),
                         pipe_side_band=pd_, pipe_side_band_range=pd_r, pipe_m=pm_, pipe_m_range=pm_r,
                         disagreement=pd_ - pm_))
        print(f"{track} {screen}: the pipeline's geometry on the cells -- side band {pd_:+.2f} (along the trace "
              f"{pd_r:.2f}), m {pm_:+.2f} ({pm_r:.2f}; the pipeline measured {rows[-1]['offsets_measured'] - k:+.2f}); "
              f"they disagree by {pd_ - pm_:+.2f} cycles")
        for name, f, per, ok in (("side band", S["d"], T.STEP, okd), ("m", S["m"], -2 * np.pi, okm)):
            for blk, sgn, side in ((S["block_n"], 1, "N"), (S["block_s"], -1, "S")):
                b = ok & blk & (sgn * n > 6e3) & (sgn * n < 20e3)
                if b.sum() < 30:
                    continue
                A = np.c_[np.ones(b.sum()), s[b] / 1e4, n[b] / 1e4]
                cf = np.linalg.lstsq(A, f[b], rcond=None)[0]
                res = -(f - (cf[0] + cf[1] * s / 1e4 + cf[2] * n / 1e4)) / per
                for lo, hi in BINS:
                    m = ok & blk & (sgn * n >= lo * 1e3) & (sgn * n < hi * 1e3)
                    deps.append(dict(scene=f"{track} {screen}", field=name, side=side, lo_km=lo, hi_km=hi,
                                     cycles=float(np.nanmedian(res[m])) if m.sum() > 15 else np.nan, cells=int(m.sum())))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tie_offsets.csv"), index=False)
    dep = pd.DataFrame(deps)
    dep.to_csv(os.path.join(RESULTS, "tie_departures.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(dep.pivot_table(index=["scene", "field", "side"], columns="lo_km", values="cycles").round(2))
    print("->", os.path.join(RESULTS, "tie_offsets.csv"))


if __name__ == "__main__":
    main()
