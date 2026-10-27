"""Two checks of the Panama ties that the Antarctic pairs prompted (Section 8; 2026-10-26).

User, after Section 8: "What do these say about the Panama ties?", then "sure" to the two checks below.

1. The ties read with the GUNW's own range offsets. In Antarctica the GUNW's offsets (ISCE3's dense offsets, culled,
   refilled and filtered for its rubbersheet) tie a cut-off component as the side band does, across a 74-188 rad step.
   In Panama our registration offsets tie the south blocks one cycle short of the side band (Section 6). The GUNW's
   offsets there say whether that is our offsets' processing or something about the ruptures. D141 (frames 084, 085),
   D040 (084, 085) and A076 (005), 40 + 5 MHz products. On the v8 cells,

       m_G = 4 pi / lambda x the GUNW's slant-range offset - phi_A

   is read as our offsets' m is in run_tie_real.py / run_tie_offsets.py: by the side band's estimator (block_tie: reach
   12 km, 20 km windows, degree 2; and with the cells within 6 km of the gap left out) and the pipelines' way (a plane
   per side 6-12 km from the trace, compared at it). Each number is the cycles to add to the offsets' tie.
1b. The GUNW's offsets turn out not to follow the coseismic offsets near the ruptures: they are one quadratic per frame
   (ISCE3's polynomial rubbersheet, found 2026-10-27; the Antarctic GUNWs keep the dense field), so the same reading
   with our own correlator's offsets
   before the pipelines' smoothing: the band / PR ROFF lattices as geocoded into offsets/offsets_{T}_{frame}_range.zarr
   (los = -slantRangeOffset, bilinear, no filling), with the Antarctic run's gross-outlier cull only (> 1 m from the
   5 x 5 median on the 80 m grid), binned onto the cells (>= 3 of 9 pixels).
2. D141's tie on the per-pixel cells (Section 7.3, results/nisar_pixel_D141.npz) against the per-cell ones, both on
   Section 5's cut with the south block on the side band's tie (34): the per-cell sums are biased where fringes are
   steep, and the Antarctic check used per-pixel cells. 0 says 34 stands on both; -1 would mean the per-cell sums had
   moved the tie by a cycle.

usage: .../envs/remote_sensing/bin/python -u run_tie_panama_checks.py   -> ../results/tie_panama_checks.csv
"""
import functools
import glob
import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "2")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T  # noqa: E402
import panama_scenes as P  # noqa: E402
from run_tie_offsets import chord_frame, pipeline_tie  # noqa: E402
from run_tie_real import POLY, trace_frame  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
GUNW = {"D141": "D141_panama/GUNW/NISAR_L2_PR_GUNW_031_141_D_08[45]_032_4000_*.h5",
        "D040": "D040_panama/GUNW/NISAR_L2_PR_GUNW_032_040_D_08[45]_033_4000_*.h5",
        "A076": "A076_panama/GUNW/NISAR_L2_PR_GUNW_032_076_A_005_033_4000_*.h5"}
OFFSETS = "science/LSAR/GUNW/grids/frequencyA/pixelOffsets"
RAW = {"D141": "D141_panama/offsets/offsets_D141_085_range.zarr", "D040": "D040_panama/offsets/offsets_D040_085_range.zarr",
       "A076": "A076_panama/offsets/offsets_A076_005_range.zarr"}


def raw_offsets_on_cells(path, x, y, cell, cull_m=1.0):
    """Our ROFF's slant-range offset (m, + away from the sensor) from an offset store, culled (> cull_m from the 5 x 5
    median) and averaged onto the cells; NaN where fewer than 3 of a cell's 9 pixels survive."""
    import xarray as xr
    from scipy.ndimage import generic_filter
    ds = xr.open_zarr(path)
    rho = -np.asarray(ds["los"].isel(pair=0).values, float)
    gx, gy = ds.x.values, ds.y.values
    med = generic_filter(rho, np.nanmedian, size=5, mode="constant", cval=np.nan)
    n_in = int(np.isfinite(rho).sum())
    rho = np.where(np.abs(rho - med) <= cull_m, rho, np.nan)
    culled = 1.0 - np.isfinite(rho).sum() / max(n_in, 1)
    shape = (y.size, x.size)
    GX, GY = np.meshgrid(gx, gy)
    ok = np.isfinite(rho)
    ix = np.floor((GX[ok] - (x[0] - cell / 2)) / cell).astype(int)
    iy = np.floor(((y[0] + cell / 2) - GY[ok]) / cell).astype(int)
    inb = (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
    num, den = np.zeros(shape), np.zeros(shape)
    np.add.at(num, (iy[inb], ix[inb]), rho[ok][inb])
    np.add.at(den, (iy[inb], ix[inb]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= 3, num / den, np.nan), float(culled)


def gunw_offsets_on_cells(files, x, y, cell, epsg=32617):
    """The GUNWs' slant-range offsets (m) averaged onto the cells; later frames levelled on the first by their median
    difference where both have data (returned, with the counts)."""
    import h5py
    shape = (y.size, x.size)
    mosaic = np.full(shape, np.nan)
    shifts = []
    for f in files:
        with h5py.File(f, "r") as h:
            g = h[OFFSETS]
            val = g["HH/slantRangeOffset"][()].astype(float)
            gx, gy = g["xCoordinates"][()], g["yCoordinates"][()]
            if int(g["projection"][()]) != epsg:
                raise SystemExit(f"{os.path.basename(f)}: EPSG {int(g['projection'][()])}, the cells are {epsg}")
        ok = np.isfinite(val) & (val != 0)
        GX, GY = np.meshgrid(gx, gy)
        ix = np.floor((GX[ok] - (x[0] - cell / 2)) / cell).astype(int)
        iy = np.floor(((y[0] + cell / 2) - GY[ok]) / cell).astype(int)
        inb = (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
        num, den = np.zeros(shape), np.zeros(shape)
        np.add.at(num, (iy[inb], ix[inb]), val[ok][inb])
        np.add.at(den, (iy[inb], ix[inb]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frame = np.where(den > 0, num / den, np.nan)
        both = np.isfinite(mosaic) & np.isfinite(frame)
        if both.sum() > 100:
            shift = float(np.median(frame[both] - mosaic[both]))
            shifts.append((os.path.basename(f)[:46], shift, int(both.sum())))
            frame = frame - shift
        mosaic = np.where(np.isfinite(mosaic), np.where(np.isfinite(frame), 0.5 * (mosaic + frame), mosaic), frame)
    return mosaic, shifts


def gunw_check():
    rows = []
    for track in ("D141", "D040", "A076"):
        S = P.load(track, "v8")
        x, y = S["x"], S["y"]
        X, Y = np.meshgrid(x, y)
        files = sorted(glob.glob(os.path.join(P.DATA, GUNW[track])))
        rho_g, shifts = gunw_offsets_on_cells(files, x, y, S["cell"])
        lam = 299792458.0 / S["f_a"]
        m_g = 4 * np.pi / lam * rho_g - S["phase_a"]
        rho_r, culled = raw_offsets_on_cells(os.path.join(P.DATA, RAW[track]), x, y, S["cell"])
        m_r = 4 * np.pi / lam * rho_r - S["phase_a"]
        rho_ours = (S["m"] + S["phase_a"]) * lam / (4 * np.pi)          # our registration offsets on the same cells
        both = np.isfinite(rho_g) & np.isfinite(rho_ours) & (S["block_n"] | S["block_s"])
        slope = float(np.polyfit(rho_ours[both], rho_g[both], 1)[0])
        diff = rho_g[both] - rho_ours[both]
        n, s, L = trace_frame(X, Y, *S["trace"])
        al = np.where(s < L, s, np.nan)
        nc, sc, Lc = chord_frame(X, Y, *S["trace"])
        sig = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
        k = int(S["ties"]["south_block_cycles"])
        row = dict(track=track, offsets_tie=k, frames=len(files), frame_shift_m=shifts[0][1] if shifts else np.nan,
                   gunw_vs_ours_slope=slope, gunw_minus_ours_median_m=float(np.median(diff)),
                   gunw_minus_ours_rsd_m=float(1.4826 * np.median(np.abs(diff - np.median(diff)))))
        okd = S["use"] & np.isfinite(S["d"])
        for name, f, ok, per in (("side band", S["d"], okd, None), ("our offsets", -S["m"], np.isfinite(S["m"]), 2 * np.pi),
                                 ("our offsets, raw", -m_r, np.isfinite(m_r), 2 * np.pi),
                                 ("GUNW offsets", -m_g, np.isfinite(m_g), 2 * np.pi)):
            kw = dict(POLY) if per is None else dict(POLY, step=per)
            extra = dict(sigma=sig) if per is None else {}
            p = T.block_tie(f, ok, S["block_n"], S["block_s"], x, y, along=al, **kw, **extra)
            p6 = T.block_tie(f, ok, S["block_n"], S["block_s"], x, y, along=al, min_gap_m=6e3, **kw, **extra)
            pp, pr = pipeline_tie(-f if per is not None else f, ok, nc, sc, Lc, S["block_n"], S["block_s"],
                                  -2 * np.pi if per is not None else T.STEP)
            row[f"{name}: 2-12 km"] = p["e"]
            row[f"{name}: 6-12 km"] = p6["e"]
            row[f"{name}: pipeline"] = pp
            row[f"{name}: windows"] = " ".join(f"{w['e']:+.2f}" for w in p["windows"])
        rows.append(row)
        print(f"{track} (offsets' tie {k}; {len(files)} GUNW frame(s)"
              + (f", the second levelled by {shifts[0][1] * 100:+.1f} cm on {shifts[0][2]} cells" if shifts else "")
              + f"): GUNW vs our offsets slope {slope:.3f}, GUNW - ours {np.median(diff) * 100:+.1f} cm "
                f"(robust sd {row['gunw_minus_ours_rsd_m'] * 100:.1f} cm)")
        bothr = np.isfinite(rho_r) & np.isfinite(rho_ours) & (S["block_n"] | S["block_s"])
        dr_ = rho_r[bothr] - rho_ours[bothr]
        row["raw_vs_smoothed_rsd_m"] = float(1.4826 * np.median(np.abs(dr_ - np.median(dr_))))
        row["raw_cells"] = int(np.isfinite(rho_r).sum())
        row["raw_culled"] = culled
        print(f"   our raw offsets on {row['raw_cells']} cells ({culled:.1%} of the pixels culled); raw - smoothed {np.median(dr_) * 100:+.1f} cm (robust sd "
              f"{row['raw_vs_smoothed_rsd_m'] * 100:.1f} cm)")
        for name in ("side band", "our offsets", "our offsets, raw", "GUNW offsets"):
            print(f"   {name:13s}: 2-12 km {row[name + ': 2-12 km']:+.2f}, 6-12 km {row[name + ': 6-12 km']:+.2f}, the "
                  f"pipelines' planes {row[name + ': pipeline']:+.2f}; windows {row[name + ': windows']}")
    return rows


def pixel_check():
    z = np.load(os.path.join(RESULTS, "nisar_pixel_D141.npz"), allow_pickle=True)
    S = P.load("D141", "v6", roff=False)
    if not (np.allclose(z["x"], S["x"]) and np.allclose(z["y"], S["y"])):
        raise SystemExit("nisar_pixel_D141.npz is not on the v6 cells")
    X, Y = np.meshgrid(S["x"], S["y"])
    n, s, L = trace_frame(X, Y, *S["trace"])
    al = np.where(s < L, s, np.nan)
    rows = []
    for form in ("cell", "pixel"):
        d = z[f"raw_{form}_2000_v6"].astype(float)
        sig = T.approx_sigma(z[f"coh_a_{form}"].astype(float), z[f"coh_b_{form}"].astype(float), S["f_a"], S["f_b"],
                             S["cell"])
        ok = np.isfinite(d)
        p = T.block_tie(d, ok, S["block_n"], S["block_s"], S["x"], S["y"], sigma=sig, along=al, **POLY)
        p6 = T.block_tie(d, ok, S["block_n"], S["block_s"], S["x"], S["y"], sigma=sig, along=al, min_gap_m=6e3, **POLY)
        near = ok & (np.abs(n) < 12e3) & (S["block_n"] | S["block_s"])
        rows.append(dict(track="D141 (Section 5's cut, south at 34)", form=form, e=p["e"], e_6_12=p6["e"],
                         windows=" ".join(f"{w['e']:+.2f}" for w in p["windows"]), cells_within_12km=int(near.sum())))
        print(f"D141, {form:5s} cells: {p['e']:+.2f} from 2 km ({p6['e']:+.2f} from 6 km), windows "
              + " ".join(f"{w['e']:+.2f}" for w in p["windows"]) + f"; {near.sum()} cells within 12 km")
    return rows


if __name__ == "__main__":
    rows = gunw_check()
    prow = pixel_check()
    out = pd.concat([pd.DataFrame(rows).assign(check="GUNW offsets"), pd.DataFrame(prow).assign(check="per-pixel cells")])
    out.to_csv(os.path.join(RESULTS, "tie_panama_checks.csv"), index=False)
    print("->", os.path.join(RESULTS, "tie_panama_checks.csv"))
