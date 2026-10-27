"""The estimator comparison on NISAR data: Panama D141 and Venezuela's four tracks.

usage: .../envs/remote_sensing/bin/python -u run_nisar.py SCENE [SCENE ...]      SCENE in D141 A061 D126 A162 D054
       .../envs/remote_sensing/bin/python -u run_nisar.py --gunw-phase SCENE ...  (after the first: the GUNW's own
                                                                                    main band in place of ours)
       .../envs/remote_sensing/bin/python -u run_nisar.py --footprint SCENE ...   (after the first: the side band's
                                                                                    range footprint widened as the GUNW's)

Each scene's inputs are rebuilt exactly as its v6 screen was (2026-10-14), from the pipelines' own cached cells and
unwrapped stores:

- D141: `~/Documents/data/D141_panama/scripts/panama_pipeline.py` with SIGMA=5 and V_CUT=v3 (`unw_reg_tied_v3_s5`: the
  drawn-trace cut, the south block tied). Frames 084 + 085. phi_A from the north and south blocks only. The south
  block takes ``D141_TIE_EXTRA`` more cycles (default 1) than the range offsets gave it (33): the side band's tie
  (`run_tie_real.py`, Section 6 of the paper). The reproduction check then runs ``ours`` once more on the offsets' tie.
- Venezuela: `~/Documents/data/Venezuela_reprocess/scripts/vz_pipeline.py` (IONO_SOURCE: v3 with the airport cut;
  D054 its reg store). phi_A from the main region only.
- Both: 240 m cells, band A's partially focused near-range strip left out, cells within 2 km of the trace excluded,
  coherence > 0.2 in both bands, v6's filter (plane fill, 10 km, reach 12 km).

Every estimator of `estimators.py` runs on those cells, plus ``diff_isce3_noexcl`` (the GUNW recipe with no
fault-zone exclusion, as ISCE3 runs it). The PR GUNW's own `ionospherePhaseScreen` is binned onto the same cells for
reference; its frames are levelled on their overlap. The ``ours`` arm must reproduce the saved v6 screen: checked, and
printed. Writes `../results/nisar_<scene>.npz` (screens, masks, distances) and `../results/nisar_<scene>.csv`.
"""

import functools
import glob
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.expanduser("~/Documents/data")
CELL = 240.0
EXCLUDE_M = 2000.0
GUNW_GRID = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
GUNW_FILES = {
    "D141": sorted(glob.glob(f"{DATA}/D141_panama/GUNW/NISAR_L2_PR_GUNW_031_141_D_08[45]_032_4000_*.h5")),
    "A061": sorted(glob.glob(f"{DATA}/NISAR_L2_PR_GUNW_023_061_A_00[67]_024_4000_*.h5")),
    "D126": sorted(glob.glob(f"{DATA}/NISAR_L2_PR_GUNW_023_126_D_08[34]_024_4000_*.h5")),
    "A162": sorted(glob.glob(f"{DATA}/NISAR_L2_PR_GUNW_022_162_A_00[67]_023_4000_*.h5")),
    "D054": sorted(glob.glob(f"{DATA}/NISAR_L2_PR_GUNW_023_054_D_08[34]_024_4000_*.h5")),
}


def signed_distance(X, Y, tx, ty, chunk=200000):
    """Distance (m) from (X, Y) to the polyline (tx, ty), + to the left of its west-to-east direction (north)."""
    tx, ty = np.asarray(tx, float), np.asarray(ty, float)
    if tx[-1] < tx[0]:
        tx, ty = tx[::-1], ty[::-1]
    P = np.stack([np.ravel(X), np.ravel(Y)], 1)
    out = np.empty(P.shape[0])
    a = np.stack([tx[:-1], ty[:-1]], 1)
    ab = np.stack([np.diff(tx), np.diff(ty)], 1)
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    for i in range(0, P.shape[0], chunk):
        p = P[i:i + chunk, None, :] - a[None]
        t = np.clip((p * ab[None]).sum(-1) / L2[None], 0.0, 1.0)
        q = p - t[..., None] * ab[None]
        dd = np.hypot(q[..., 0], q[..., 1])
        k = np.argmin(dd, 1)
        idx = np.arange(k.size)
        cross = ab[k, 0] * p[idx, k, 1] - ab[k, 1] * p[idx, k, 0]
        out[i:i + chunk] = dd[idx, k] * np.where(cross >= 0, 1.0, -1.0)
    return out.reshape(np.shape(X))


def inputs_d141():
    os.environ.update(TRACK="D141", SIGMA="5", V_CUT="v3", V_SCREEN="v6")
    sys.path.insert(0, f"{DATA}/D141_panama/scripts")
    import panama_pipeline as PP
    from nisar_tools import UnwrappedStack
    from nisar_tools import split_spectrum as ss
    from pyproj import Transformer
    from scipy.ndimage import label

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    lo, la = np.linspace(PP.BBOX[0], PP.BBOX[1], 30), np.linspace(PP.BBOX[2], PP.BBOX[3], 30)
    LO, LA = np.meshgrid(lo, la)
    bx, by = tr.transform(LO.ravel(), LA.ravel())
    bounds = (min(bx), min(by), max(bx), max(by))
    cells = {}
    for band, pr in (("A", PP.pair), ("B", PP.b_pair)):
        per = [PP._cells_cached(*pr(fr), band, fr, bounds, "near_range", CELL, PP.NEAR_RANGE_M_V6) for fr in PP.frames()]
        cells[band] = ss.merge_cells(*per)
    xs = np.intersect1d(cells["A"].x.values, cells["B"].x.values)
    ys = np.intersect1d(cells["A"].y.values, cells["B"].y.values)[::-1]
    ca, cb = cells["A"].sel(x=xs, y=ys), cells["B"].sel(x=xs, y=ys)
    unw = UnwrappedStack.from_zarr(PP.WS.path("unw_reg_tied_v3" + PP.SFX))
    ph = unw.ds["unw"].values[0].astype(float)
    lab, _ = label(np.isfinite(ph))
    ties = unw.ds.attrs.get("component_ties")
    ph_cells = np.where(np.isin(lab, [ties["north_label"], ties["south_label"]]), ph, np.nan)
    phase_saved, wsum = ss.cell_average(ph_cells, unw.ds.x.values, unw.ds.y.values, ca)
    extra = int(os.environ.get("D141_TIE_EXTRA", "1"))
    ph_cells = np.where(lab == ties["south_label"], ph_cells + 2 * np.pi * extra, ph_cells)
    phase_a, _ = ss.cell_average(ph_cells, unw.ds.x.values, unw.ds.y.values, ca)
    frac = wsum / (CELL / abs(float(unw.ds.x[1] - unw.ds.x[0]))) ** 2
    tx, ty = PP.drawn_trace()
    near = ss.fault_distance(ca, tx, ty) < EXCLUDE_M
    for bx_, by_ in PP.drawn_branches():
        near |= ss.fault_distance(ca, bx_, by_) < EXCLUDE_M
    saved = f"{PP.WS_DIR}/iono_split_spectrum_v6_s5.nc"
    return dict(ca=ca, cb=cb, phase_a=phase_a, valid=np.isfinite(phase_a) & (frac > 0.5), near=near, trace=(tx, ty),
                epsg=32617, saved=saved, label="Panama D141 (2026-09-27 to 10-09)",
                phase_saved=phase_saved if extra else None, tie=int(ties["south_block_cycles"]) + extra,
                tie_saved=int(ties["south_block_cycles"]))


def inputs_venezuela(track):
    sys.path.insert(0, f"{DATA}/Venezuela_reprocess/scripts")
    import vz_pipeline as VZ
    from nisar_tools import UnwrappedStack
    from nisar_tools import split_spectrum as ss
    from pyproj import Transformer
    from scipy.ndimage import label

    frames, bbox = VZ.TRACKS[track]
    unw = UnwrappedStack.from_zarr(VZ.WS.path(VZ.IONO_SOURCE[track]))
    epsg = int(unw.ds.attrs["epsg"])
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    lo, la = np.linspace(bbox[0], bbox[1], 30), np.linspace(bbox[2], bbox[3], 30)
    LO, LA = np.meshgrid(lo, la)
    bx, by = tr.transform(LO.ravel(), LA.ravel())
    bounds = (min(bx), min(by), max(bx), max(by))
    cells = {}
    for band, pr in (("A", VZ.pair), ("B", VZ.b_pair)):
        per = [VZ._cells_cached(*pr(track, f), band, f, bounds, "near_range", CELL, VZ.NEAR_RANGE_M_V6.get(track, 11000.0))
               for f in frames]
        cells[band] = ss.merge_cells(*per)
    xs = np.intersect1d(cells["A"].x.values, cells["B"].x.values)
    ys = np.intersect1d(cells["A"].y.values, cells["B"].y.values)[::-1]
    ca, cb = cells["A"].sel(x=xs, y=ys), cells["B"].sel(x=xs, y=ys)
    ph = unw.ds["unw"].values[0].astype(float)
    lab, _ = label(np.isfinite(ph))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    ph_main = np.where(lab == int(np.argmax(sizes)), ph, np.nan)
    phase_a, wsum = ss.cell_average(ph_main, unw.ds.x.values, unw.ds.y.values, ca)
    frac = wsum / (CELL / abs(float(unw.ds.x[1] - unw.ds.x[0]))) ** 2
    tx, ty = VZ.trace2_xy(epsg)
    near = ss.fault_distance(ca, tx, ty) < EXCLUDE_M
    saved = os.path.join(VZ.V, "workspace", f"iono_split_spectrum_{track}_v6.nc")
    return dict(ca=ca, cb=cb, phase_a=phase_a, valid=np.isfinite(phase_a) & (frac > 0.5), near=near, trace=(tx, ty),
                epsg=epsg, saved=saved, label=f"Venezuela {track}")


def gunw_on_cells(files, x, y, epsg, layer="ionospherePhaseScreen", cycles=False):
    """A PR GUNW layer (``ionospherePhaseScreen``, rad at f_A, or ``unwrappedPhase``) averaged onto the cells; later
    frames levelled on the first by their median difference where both have data (rounded to whole cycles when
    ``cycles``: each frame's unwrapping has its own integer)."""
    import h5py
    from pyproj import Transformer

    shape = (y.size, x.size)
    mosaic = np.full(shape, np.nan)
    for f in files:
        with h5py.File(f, "r") as h:
            g = h[GUNW_GRID]
            scr = g[f"HH/{layer}"][()].astype(float)
            gx, gy = g["xCoordinates"][()], g["yCoordinates"][()]
            g_epsg = int(g["projection"].attrs["epsg_code"])
        ok = np.isfinite(scr) & (scr != 0)
        GX, GY = np.meshgrid(gx, gy)
        px, py, val = GX[ok], GY[ok], scr[ok]
        if g_epsg != epsg:
            px, py = Transformer.from_crs(g_epsg, epsg, always_xy=True).transform(px, py)
        ix = np.floor((px - (x[0] - CELL / 2)) / CELL).astype(int)
        iy = np.floor(((y[0] + CELL / 2) - py) / CELL).astype(int)
        inb = (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
        num, den = np.zeros(shape), np.zeros(shape)
        np.add.at(num, (iy[inb], ix[inb]), val[inb])
        np.add.at(den, (iy[inb], ix[inb]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frame = np.where(den > 0, num / den, np.nan)
        both = np.isfinite(mosaic) & np.isfinite(frame)
        if both.sum() > 100:
            shift = np.median(frame[both] - mosaic[both])
            frame = frame - (2 * np.pi * np.round(shift / (2 * np.pi)) if cycles else shift)
        mosaic = np.where(np.isfinite(mosaic), np.where(np.isfinite(frame), 0.5 * (mosaic + frame), mosaic), frame)
    return mosaic


def run(scene):
    import xarray as xr

    t0 = time.time()
    I = inputs_d141() if scene == "D141" else inputs_venezuela(scene)
    ca, cb = I["ca"], I["cb"]
    f_a, f_b = float(ca.attrs["center_frequency"]), float(cb.attrs["center_frequency"])
    x, y = ca.x.values, ca.y.values
    X, Y = np.meshgrid(x, y)
    psi_a, psi_b = np.angle(ca.ifg.values), np.angle(cb.ifg.values)
    ga, gb = ca.coherence.values, cb.coherence.values
    na, nb = ca.n.values / float(ca.attrs["pixels_per_cell"]), cb.n.values / float(cb.attrs["pixels_per_cell"])
    base = (I["valid"] & np.isfinite(I["phase_a"]) & (ga > 0.2) & (gb > 0.2) & (na >= 0.5) & (nb >= 0.5))
    use = base & ~I["near"]
    sig = E.raw_sigma(ga, gb, na, nb, f_a, f_b, CELL)
    dist = signed_distance(X, Y, *I["trace"])
    print(f"{scene}: {ca.sizes['y']} x {ca.sizes['x']} cells, {use.sum()} used ({base.sum() - use.sum()} in the fault "
          f"zone); f_A {f_a / 1e6:.1f} MHz, f_B {f_b / 1e6:.1f} MHz [{time.time() - t0:.0f}s]")
    screens, rows = {}, []
    arms = [(m, use) for m in E.METHODS] + [("diff_isce3_noexcl", base)]
    for name, u in arms:
        t1 = time.time()
        method = name.replace("_noexcl", "")
        out = E.estimate(method, I["phase_a"], psi_a, psi_b, ga, gb, u, sig, f_a, f_b, CELL, x=x, y=y, data=I["valid"])
        screens[name] = out["screen"].astype(np.float32)
        rows.append(dict(scene=scene, method=name, cycles_a=out["cycles_a"], cycles_b=out["cycles_b"],
                         seconds=time.time() - t1))
        print(f"  {name:18s} [{time.time() - t1:5.0f}s] cycles moved A {out['cycles_a']:.4f} B {out['cycles_b']:.4f}")
    # the harness reproduces the saved v6 screen (on the tie it was made with)
    saved = xr.open_dataset(I["saved"])
    sv = saved["screen"].sel(x=x, y=y, method="nearest").values
    if I.get("phase_saved") is not None:
        same = E.estimate("ours", I["phase_saved"], psi_a, psi_b, ga, gb, use, sig, f_a, f_b, CELL, x=x, y=y,
                          data=I["valid"])["screen"]
        dv = same - sv
        print(f"  (the south block tied at {I['tie']}; the saved screen at {I['tie_saved']}: checked on that)")
    else:
        dv = screens["ours"] - sv
    print(f"  ours vs the saved v6 screen: max |diff| {np.nanmax(np.abs(dv)):.2e} rad, "
          f"{np.mean(np.isfinite(sv) == np.isfinite(screens['ours'])):.4%} same coverage")
    gunw = gunw_on_cells(GUNW_FILES[scene], x, y, I["epsg"]) if GUNW_FILES.get(scene) else np.full(X.shape, np.nan)
    screens["gunw"] = gunw.astype(np.float32)
    rows.append(dict(scene=scene, method="gunw"))
    land = I["valid"]
    phi_a = I["phase_a"] + E.ss.wrap(psi_a - I["phase_a"])
    phi_nd = phi_a - screens["ours"]
    for r in rows:
        s = screens[r["method"]].astype(float)
        r.update(M.nisar(s, screens["ours"].astype(float), phi_nd, dist, land & np.isfinite(s), X, Y, CELL))
    df = pd.DataFrame(rows)
    df["saved_v6_maxdiff"] = float(np.nanmax(np.abs(dv)))
    df["tie"] = I.get("tie")
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f"nisar_{scene}.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, f"nisar_{scene}.npz"), x=x, y=y, dist=dist.astype(np.float32),
                        land=land, use=use, base=base, phi_nd=phi_nd.astype(np.float32),
                        coh_a=ga.astype(np.float32), coh_b=gb.astype(np.float32),
                        trace_x=np.asarray(I["trace"][0]), trace_y=np.asarray(I["trace"][1]),
                        **{f"screen_{k}": v for k, v in screens.items()}, label=I["label"], epsg=I["epsg"])
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(df.drop(columns=["scene"]))
    print(f"{scene} done [{time.time() - t0:.0f}s]")


def run_gunw_phase(scene):
    """The reordered estimator and the GUNW recipe, both on the PR GUNW's OWN unwrapped main band (its
    ``unwrappedPhase`` averaged onto the cells) instead of ours: how much of the product's leak its main-band integers
    carry (3.21 rad of raw estimate per cycle in either form). Adds rows ``ours_gunwphase`` and
    ``diff_isce3_noexcl_gunwphase`` to results/nisar_<scene>.csv and their screens to a separate npz."""
    t0 = time.time()
    I = inputs_d141() if scene == "D141" else inputs_venezuela(scene)
    ca, cb = I["ca"], I["cb"]
    f_a, f_b = float(ca.attrs["center_frequency"]), float(cb.attrs["center_frequency"])
    x, y = ca.x.values, ca.y.values
    X, Y = np.meshgrid(x, y)
    psi_a, psi_b = np.angle(ca.ifg.values), np.angle(cb.ifg.values)
    ga, gb = ca.coherence.values, cb.coherence.values
    na, nb = ca.n.values / float(ca.attrs["pixels_per_cell"]), cb.n.values / float(cb.attrs["pixels_per_cell"])
    sig = E.raw_sigma(ga, gb, na, nb, f_a, f_b, CELL)
    gphi = gunw_on_cells(GUNW_FILES[scene], x, y, I["epsg"], layer="unwrappedPhase", cycles=True)
    gok = np.isfinite(gphi) & (gphi != 0)
    base = (I["valid"] & gok & (ga > 0.2) & (gb > 0.2) & (na >= 0.5) & (nb >= 0.5))
    use = base & ~I["near"]
    dist = signed_distance(X, Y, *I["trace"])
    # where both main-band unwrappings exist, how many cycles apart are they (cells of our use)?
    ours_phi = I["phase_a"] + E.ss.wrap(psi_a - I["phase_a"])
    g_phi = gphi + E.ss.wrap(psi_a - gphi)
    dcyc = np.round((g_phi - ours_phi) / (2 * np.pi))
    both = use & np.isfinite(dcyc)
    mode = np.bincount((dcyc[both] - dcyc[both].min()).astype(int)).argmax() + dcyc[both].min()
    off = both & (dcyc != mode)
    print(f"{scene}: GUNW unwrapped phase on {gok.mean():.1%} of the cells; where both exist it differs from ours by "
          f"a non-modal whole cycle at {off.mean() / max(both.mean(), 1e-12):.2%} of them "
          f"({np.mean(off[np.abs(dist) < 10e3]) / max(np.mean(both[np.abs(dist) < 10e3]), 1e-12):.2%} within 10 km)")
    z = np.load(os.path.join(RESULTS, f"nisar_{scene}.npz"), allow_pickle=True)
    ours_screen = z["screen_ours"].astype(float)
    phi_nd = z["phi_nd"].astype(float)
    land = z["land"]
    rows, screens = [], {}
    for name, method, u in (("ours_gunwphase", "ours", use), ("diff_isce3_noexcl_gunwphase", "diff_isce3", base)):
        t1 = time.time()
        out = E.estimate(method, gphi, psi_a, psi_b, ga, gb, u, sig, f_a, f_b, CELL, x=x, y=y, data=I["valid"] & gok)
        screens[name] = out["screen"].astype(np.float32)
        row = dict(scene=scene, method=name, cycles_a=out["cycles_a"], cycles_b=out["cycles_b"], seconds=time.time() - t1)
        row.update(M.nisar(out["screen"], ours_screen, phi_nd, dist, land & np.isfinite(out["screen"]), X, Y, CELL))
        rows.append(row)
        print(f"  {name:28s} [{time.time() - t1:5.0f}s] leak {row['leak']:+.3f} (r {row['leak_r']:+.2f}), diff from ours "
              f"near {row['diff_near']:.2f} rad")
    path = os.path.join(RESULTS, f"nisar_{scene}.csv")
    df = pd.read_csv(path)
    df = pd.concat([df[~df.method.isin([r["method"] for r in rows])], pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path, index=False)
    np.savez_compressed(os.path.join(RESULTS, f"nisar_{scene}_gunwphase.npz"), x=x, y=y,
                        gunw_cycles_vs_ours=np.where(both, dcyc - mode, np.nan).astype(np.float32),
                        **{f"screen_{k}": v for k, v in screens.items()})
    print(f"{scene} gunw-phase done [{time.time() - t0:.0f}s]")


#: The PR GUNW's side band is multilooked over 13 of its range pixels (325 m slant, ~540 m ground at 37 deg) and its
#: main band decimated from a 40 m (slant) look: a range footprint mismatch whose second moment, (0.54^2 - 0.067^2)/12
#: = 0.0239 km^2, a [w, 1, w] range kernel over our 240 m side-band cells reproduces at w = 0.365 (0.0243 km^2).
FOOTPRINT_W = 0.365


def _range_kernel(cells, w):
    """The side band's cell sums widened in range (x, within 13 deg of range for these tracks) by [w, 1, w]."""
    import xarray as xr

    out = {}
    for v in ("ifg", "p1", "p2", "n"):
        a = cells[v].values.astype(np.complex128 if v == "ifg" else float)
        b = a.copy()
        b[:, 1:] += w * a[:, :-1]
        b[:, :-1] += w * a[:, 1:]
        out[v] = b
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(out["ifg"]) / np.sqrt(out["p1"] * out["p2"])
    ds = xr.Dataset({"ifg": (("y", "x"), out["ifg"]), "p1": (("y", "x"), out["p1"]), "p2": (("y", "x"), out["p2"]),
                     "n": (("y", "x"), out["n"] / (1 + 2 * w)), "coherence": (("y", "x"), coh)},
                    coords=cells.coords, attrs=dict(cells.attrs))
    return ds


def run_footprint(scene, w=FOOTPRINT_W):
    """The reordered estimator and the GUNW recipe with the side band's range footprint widened as the PR GUNW's is
    (`FOOTPRINT_W`): how much of the product's leak the bands' footprint mismatch carries. Adds rows
    ``ours_footprint`` and ``diff_isce3_noexcl_footprint`` to results/nisar_<scene>.csv."""
    t0 = time.time()
    I = inputs_d141() if scene == "D141" else inputs_venezuela(scene)
    ca, cb = I["ca"], _range_kernel(I["cb"], w)
    f_a, f_b = float(ca.attrs["center_frequency"]), float(cb.attrs["center_frequency"])
    x, y = ca.x.values, ca.y.values
    X, Y = np.meshgrid(x, y)
    psi_a, psi_b = np.angle(ca.ifg.values), np.angle(cb.ifg.values)
    ga, gb = ca.coherence.values, cb.coherence.values
    na, nb = ca.n.values / float(ca.attrs["pixels_per_cell"]), cb.n.values / float(cb.attrs["pixels_per_cell"])
    base = (I["valid"] & np.isfinite(I["phase_a"]) & (ga > 0.2) & (gb > 0.2) & (na >= 0.5) & (nb >= 0.5))
    use = base & ~I["near"]
    sig = E.raw_sigma(ga, gb, na, nb, f_a, f_b, CELL)
    dist = signed_distance(X, Y, *I["trace"])
    z = np.load(os.path.join(RESULTS, f"nisar_{scene}.npz"), allow_pickle=True)
    ours_screen, phi_nd, land = z["screen_ours"].astype(float), z["phi_nd"].astype(float), z["land"]
    rows, screens = [], {}
    for name, method, u in (("ours_footprint", "ours", use), ("diff_isce3_noexcl_footprint", "diff_isce3", base)):
        t1 = time.time()
        out = E.estimate(method, I["phase_a"], psi_a, psi_b, ga, gb, u, sig, f_a, f_b, CELL, x=x, y=y, data=I["valid"])
        screens[name] = out["screen"].astype(np.float32)
        row = dict(scene=scene, method=name, cycles_a=out["cycles_a"], cycles_b=out["cycles_b"], seconds=time.time() - t1)
        row.update(M.nisar(out["screen"], ours_screen, phi_nd, dist, land & np.isfinite(out["screen"]), X, Y, CELL))
        rows.append(row)
        print(f"  {name:28s} [{time.time() - t1:5.0f}s] leak {row['leak']:+.3f} (r {row['leak_r']:+.2f}), diff from ours "
              f"near {row['diff_near']:.2f} rad")
    path = os.path.join(RESULTS, f"nisar_{scene}.csv")
    df = pd.read_csv(path)
    df = pd.concat([df[~df.method.isin([r["method"] for r in rows])], pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path, index=False)
    np.savez_compressed(os.path.join(RESULTS, f"nisar_{scene}_footprint.npz"), x=x, y=y,
                        **{f"screen_{k}": v for k, v in screens.items()})
    print(f"{scene} footprint done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--gunw-phase":
        for s in args[1:]:
            run_gunw_phase(s)
    elif args and args[0] == "--footprint":
        for s in args[1:]:
            run_footprint(s)
    else:
        for s in args:
            run(s)
