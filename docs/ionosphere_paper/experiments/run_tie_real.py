"""The side-band tie on the real scenes.

- Panama D141 / D040 / A076 (v8, the production screens) and A004 (v6): the south block's tie, by the robust polynomial
  (`dispersive_tie.block_tie`, reach 12 km, 20 km windows along the trace, degree 2) and by kriging (`gp_tie`), against
  the range-offset tie the pipeline applied; and the same estimators on m = 4 pi / lambda * offset - phi_A.
- Controls: Venezuela A162 and D054 (main band unwrapped through the rupture's west end: the answer is 0), and D141's
  two unwrappings v5 / v5defo (no cut) against v6 (cut, tied): their integer difference against the estimates'.
- What one cycle does to each production screen near the trace (the filter re-run on the raw estimate).
- Cross-fault profiles of the raw estimate and of m, for the figure.

    .../envs/remote_sensing/bin/python -u run_tie_real.py      -> ../results/tie_real*.csv, tie_profiles.npz
"""
import functools
import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "3")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T  # noqa: E402
import panama_scenes as P  # noqa: E402
import venezuela_scenes as V  # noqa: E402
from run_nisar import signed_distance  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
MM = 299792458.0 / 1239e6 / (4 * np.pi) * 1e3          # mm of LOS per rad at f_A
POLY = dict(reach_m=12e3, window_m=20e3, deg=2, min_cells=60)
GP = dict(reach_m=12e3, window_m=20e3)


def trace_frame(X, Y, tx, ty):
    """Signed distance (+ north) and arc length along the trace of each cell's nearest trace point (m)."""
    tx, ty = np.asarray(tx, float), np.asarray(ty, float)
    if tx[-1] < tx[0]:
        tx, ty = tx[::-1], ty[::-1]
    n = signed_distance(X, Y, tx, ty)
    s_tr = np.r_[0.0, np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))]
    ss = np.linspace(0.0, s_tr[-1], int(s_tr[-1] / 50.0) + 2)
    _, idx = cKDTree(np.c_[np.interp(ss, s_tr, tx), np.interp(ss, s_tr, ty)]).query(np.c_[X.ravel(), Y.ravel()])
    return n, ss[idx].reshape(X.shape), float(s_tr[-1])


def windows_rows(scene, field, estimator, r):
    return [dict(scene=scene, field=field, estimator=estimator, **{k: v for k, v in w.items()}) for w in r["windows"]]


def profiles(d, n, s, ok, L, seg_m=10e3, edges=np.arange(-20, 20.01, 0.5)):
    """Median of d in 0.5 km bins of n, per segment of the trace."""
    out = []
    for a in np.arange(0, L, seg_m):
        m = ok & (s >= a) & (s < a + seg_m)
        out.append([np.nanmedian(d[m & (n / 1e3 >= lo) & (n / 1e3 < lo + 0.5)])
                    if np.sum(m & (n / 1e3 >= lo) & (n / 1e3 < lo + 0.5)) > 8 else np.nan for lo in edges[:-1]])
    return np.array(out)


def screen_effect(S, n):
    """The corrected LOS change, beyond each block's constant, for one more cycle on the south block (mm by band)."""
    from nisar_tools import split_spectrum as ss
    sig = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
    kw = dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=10000.0, fill_reach_m=12000.0)
    s0 = ss.ionosphere_filter(S["d"], sig, S["use"], S["cell"], **kw)[0]
    s1 = ss.ionosphere_filter(np.where(S["block_s"] & S["use"], S["d"] + T.STEP, S["d"]), sig, S["use"], S["cell"], **kw)[0]
    land = np.isfinite(S["phase_a"]) & (S["block_n"] | S["block_s"])
    dl = np.where(S["block_s"], 2 * np.pi, 0.0) - (s1 - s0)
    far_n = np.nanmedian(dl[land & S["block_n"] & (n > 15e3) & (n < 30e3)])
    far_s = np.nanmedian(dl[land & S["block_s"] & (n < -15e3) & (n > -30e3)])
    rel = (dl - np.where(S["block_s"], far_s, far_n)) * MM
    rows = []
    for lo, hi in ((-8, -6), (-6, -4), (-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8)):
        m = land & (n / 1e3 >= lo) & (n / 1e3 < hi)
        rows.append(dict(lo_km=lo, hi_km=hi, mm=float(np.nanmedian(rel[m])) if m.sum() > 10 else np.nan))
    return rows, float(far_s * MM), float(np.sqrt(np.nanmean((s0 - S["screen_f"]) ** 2)))


def main():
    wins, summ, eff, prof = [], [], [], {}
    for track, screen in (("D141", "v8"), ("D040", "v8"), ("A076", "v8"), ("A004", "v6")):
        S = P.load(track, screen)
        X, Y = np.meshgrid(S["x"], S["y"])
        n, s, L = trace_frame(X, Y, *S["trace"])
        smax = 16e3 if track == "A004" else L
        al = np.where(s < smax, s, np.nan)
        sig = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
        name = f"{track} {screen}"
        p = T.block_tie(S["d"], S["use"], S["block_n"], S["block_s"], S["x"], S["y"], sigma=sig, along=al, **POLY)
        g = T.gp_tie(S["d"], S["use"], S["block_n"], S["block_s"], S["x"], S["y"], sigma=sig, along=al, **GP)
        okm = np.isfinite(S["m"])
        pm = T.block_tie(-S["m"], okm, S["block_n"], S["block_s"], S["x"], S["y"], along=al, step=2 * np.pi, **POLY)
        pm6 = T.block_tie(-S["m"], okm, S["block_n"], S["block_s"], S["x"], S["y"], along=al, step=2 * np.pi,
                          min_gap_m=6e3, **POLY)
        p6 = T.block_tie(S["d"], S["use"], S["block_n"], S["block_s"], S["x"], S["y"], sigma=sig, along=al,
                         min_gap_m=6e3, **POLY)
        wins += windows_rows(name, "dispersive", "polynomial", p) + windows_rows(name, "dispersive", "kriging", g) \
            + windows_rows(name, "m", "polynomial", pm)
        k = int(S["ties"]["south_block_cycles"])
        summ.append(dict(scene=name, offsets_k=k, offsets_measured=k + float(S["ties"]["south_residual_cycles"]),
                         poly=p["e"], poly_wmean=p["e_wmean"], poly_spread=p["spread"], poly_windows=len(p["windows"]),
                         poly_6_12=p6["e"], kriging=g["e"], kriging_wmean=g["e_wmean"], kriging_se=g["se_wmean"],
                         kriging_windows=len(g["windows"]), m_poly=pm["e"], m_poly_6_12=pm6["e"],
                         side_band_k=(k + int(np.round(p["e"]))) if np.isfinite(p["e"]) else None))
        print(f"{name}: offsets {k} (measured {summ[-1]['offsets_measured']:+.2f}); side band polynomial {p['e']:+.2f} "
              f"({len(p['windows'])} windows: " + " ".join(f"{w['e']:+.2f}" for w in p["windows"])
              + f"), 6-12 km {p6['e']:+.2f}; kriging {g['e']:+.2f} (wmean {g['e_wmean']:+.2f} +- {g['se_wmean']:.2f}: "
              + " ".join(f"{w['e']:+.2f}" for w in g["windows"]) + f"); m {pm['e']:+.2f}, m 6-12 km {pm6['e']:+.2f}")
        if track != "A004":
            rows, const, emu = screen_effect(S, n)
            eff += [dict(scene=name, **r) for r in rows]
            print(f"   one more cycle: the south block's corrected LOS +{const:.1f} mm (its own offset); near the trace "
                  + " ".join(f"{r['lo_km']:+d}..{r['hi_km']:+d} {r['mm']:+.1f}" for r in rows) + f" mm (emulation rms {emu:.3f} rad)")
            ok = S["use"] & np.isfinite(S["d"]) & (((n > 0) & S["block_n"]) | ((n < 0) & S["block_s"]))
            prof[f"{track}_d"] = profiles(S["d"], n, s, ok, L)
            prof[f"{track}_m"] = profiles(S["m"], n, s, okm & (((n > 0) & S["block_n"]) | ((n < 0) & S["block_s"])), L)
            prof[f"{track}_tie"] = np.array([k, p["e"]])
    # controls: a rupture with the main band unwrapped through it
    for track in ("A162", "D054"):
        S = V.load(track)
        X, Y = np.meshgrid(S["x"], S["y"])
        n, s, L = trace_frame(X, Y, *S["trace"])
        interior = (s > 1e3) & (s < L - 1e3)
        bn, bs = S["region"] & (n > 0) & interior, S["region"] & (n < 0) & interior
        sig = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
        al = np.where(interior, s, np.nan)
        p = T.block_tie(S["d"], S["use"], bn, bs, S["x"], S["y"], sigma=sig, along=al, **dict(POLY, min_cells=40))
        g = T.gp_tie(S["d"], S["use"], bn, bs, S["x"], S["y"], sigma=sig, along=al, **GP)
        name = f"{track} v6 (control)"
        wins += windows_rows(name, "dispersive", "polynomial", p) + windows_rows(name, "dispersive", "kriging", g)
        summ.append(dict(scene=name, offsets_k=None, poly=p["e"], poly_wmean=p["e_wmean"], poly_spread=p["spread"],
                         poly_windows=len(p["windows"]), kriging=g["e"], kriging_wmean=g["e_wmean"],
                         kriging_se=g["se_wmean"], kriging_windows=len(g["windows"])))
        print(f"{name}: polynomial {p['e']:+.2f} (" + " ".join(f"{w['e']:+.2f}" for w in p["windows"])
              + f"); kriging {g['e']:+.2f} (" + " ".join(f"{w['e']:+.2f}" for w in g["windows"]) + ")")
    # D141: two unwrappings differing by whole cycles on the south block
    S6 = P.load("D141", "v6", roff=False)
    X, Y = np.meshgrid(S6["x"], S6["y"])
    n, s, L = trace_frame(X, Y, *S6["trace"])
    al = np.where((s > 1e3) & (s < L - 1e3), s, np.nan)      # v5 has no cut: its sides join beyond the trace's ends
    W = os.path.join(P.DATA, P.SCENES["D141"]["ws"])
    sig = T.approx_sigma(S6["coh_a"], S6["coh_b"], S6["f_a"], S6["f_b"], S6["cell"])
    p6 = T.block_tie(S6["d"], S6["use"], S6["block_n"], S6["block_s"], S6["x"], S6["y"], sigma=sig, along=al, **POLY)
    summ.append(dict(scene="D141 v6", offsets_k=int(S6["ties"]["south_block_cycles"]),
                     offsets_measured=int(S6["ties"]["south_block_cycles"]) + float(S6["ties"]["south_residual_cycles"]),
                     poly=p6["e"], poly_windows=len(p6["windows"])))
    for name in ("v5", "v5defo"):
        z = xr.open_dataset(f"{W}/iono_split_spectrum_{name}_s5.nc").load()
        region = np.isfinite(z["phase_a"].values)
        both = region & np.isfinite(S6["phase_a"])
        dc = np.round((z["phase_a"].values - S6["phase_a"]) / (2 * np.pi))
        m = both & S6["block_s"] & (np.abs(n) > 3e3) & (np.abs(n) < 30e3)
        mn = both & S6["block_n"] & (np.abs(n) > 3e3) & (np.abs(n) < 30e3)
        diff = float(np.median(dc[m]) - np.median(dc[mn]))
        sg = T.approx_sigma(z["coherence_a"].values, z["coherence_b"].values, S6["f_a"], S6["f_b"], S6["cell"])
        p = T.block_tie(z["dispersive"].values, z["mask"].values.astype(bool), region & (n > 0), region & (n < 0),
                        S6["x"], S6["y"], sigma=sg, along=al, **POLY)
        summ.append(dict(scene=f"D141 {name} (no cut)", poly=p["e"], poly_windows=len(p["windows"]),
                         cycles_vs_v6=diff, share_vs_v6=float(np.mean(dc[m] == np.median(dc[m])))))
        print(f"D141 {name} vs v6: the south block {diff:+.0f} cycles ({summ[-1]['share_vs_v6']:.1%} of its cells); "
              f"side band {p['e']:+.2f} vs v6's {p6['e']:+.2f}: difference {p['e'] - p6['e']:+.2f}")
    pd.DataFrame(wins).to_csv(os.path.join(RESULTS, "tie_real_windows.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(RESULTS, "tie_real.csv"), index=False)
    pd.DataFrame(eff).to_csv(os.path.join(RESULTS, "tie_screen_effect.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, "tie_profiles.npz"), edges=np.arange(-20, 20.01, 0.5), step=T.STEP, **prof)
    print("->", os.path.join(RESULTS, "tie_real.csv"))


if __name__ == "__main__":
    main()
