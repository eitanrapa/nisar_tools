"""The main band taken out pixel by pixel on NISAR data, against the per-cell way, with the fault zone narrowed.

usage: .../envs/remote_sensing/bin/python -u run_nisar_pixel.py SCENE [SCENE ...]      SCENE in D141 A061 D126 A162 D054

Both bands' cells are made again from the GSLC pairs with `split_spectrum.gslc_cells(phase_model=)`, the model being
the tied main band `run_nisar.py` averages onto its cells, on its own 50 m grid (D141: the north and south blocks, the
south one on the side band's tie; Venezuela: the main region). Same lattice, bounds and near-range strip as the
pipelines' cached cells. Cached in ``../results/cells_pixel_<scene>_<band>_<frame>.npz`` (git-ignored): the GSLCs are
read once more, ~1 h a scene.

Screens, all by the library's `split_spectrum_screen`, for ``cell`` (the production's per-cell form, `run_nisar`'s
inputs: reproduces the v6 screen) and ``pixel`` (the new cells), with the cells within 2, 1, 0.5 and 0 km of the trace
(and branches) left out, through v6's filter (a 10 km plane fill, ISCE3's weighted Gaussian mean), the fill matched
to the gap (the 1.5-3-6-12 km ladder), and that fill with a weighted local plane as the final step (``order=1``).
Scored (no truth on real data):

- the cells used and their coherence, by distance from the trace;
- the raw estimate's scatter about the screen (1.4826 MAD), by distance, against its Cramer-Rao sigma;
- the leak and the 2-4 km step (`metrics.nisar`, against the non-dispersive phase of the production arm);
- each screen against the production one (cell, 2 km, v6), near the trace and far.

Writes ``../results/nisar_pixel_<scene>.csv`` and ``.npz`` (the screens, raw estimates and masks).
"""

import functools
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402,F401  (the filter's nanmedian, sped up in this process)
import metrics as M  # noqa: E402
import run_nisar as RN  # noqa: E402
from nisar_tools import split_spectrum as ss  # noqa: E402

RESULTS = RN.RESULTS
CELL = RN.CELL
EXCLUDE = (2000.0, 1000.0, 500.0, 0.0)
LADDER = (1500.0, 3000.0, 6000.0, 12000.0)
FILTERS = {"v6": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=10000.0, fill_reach_m=12000.0),
           "matched": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=LADDER, fill_reach_m=12000.0),
           "matched1": dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=LADDER, fill_reach_m=12000.0, order=1)}
BINS = ((0, .5), (.5, 1), (1, 2), (2, 5), (5, 10), (10, 20))


def model_d141():
    """The tied main band of `run_nisar.inputs_d141` on its 50 m grid, and how the pipeline made its cells."""
    os.environ.update(TRACK="D141", SIGMA="5", V_CUT="v3", V_SCREEN="v6")
    sys.path.insert(0, f"{RN.DATA}/D141_panama/scripts")
    import panama_pipeline as PP
    from nisar_tools import UnwrappedStack
    from scipy.ndimage import label

    unw = UnwrappedStack.from_zarr(PP.WS.path("unw_reg_tied_v3" + PP.SFX))
    ph = unw.ds["unw"].values[0].astype(float)
    lab, _ = label(np.isfinite(ph))
    ties = unw.ds.attrs.get("component_ties")
    ph = np.where(np.isin(lab, [ties["north_label"], ties["south_label"]]), ph, np.nan)
    extra = int(os.environ.get("D141_TIE_EXTRA", "1"))
    ph = np.where(lab == ties["south_label"], ph + 2 * np.pi * extra, ph)
    model = xr.DataArray(ph.astype(np.float32), coords={"y": unw.ds.y.values, "x": unw.ds.x.values}, dims=("y", "x"),
                         attrs={"description": f"D141 unw_reg_tied_v3_s5, north + south blocks, south +{extra} cycle"})
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    lo, la = np.linspace(PP.BBOX[0], PP.BBOX[1], 30), np.linspace(PP.BBOX[2], PP.BBOX[3], 30)
    LO, LA = np.meshgrid(lo, la)
    bx, by = tr.transform(LO.ravel(), LA.ravel())
    pairs = {(band, fr): pr(fr) for band, pr in (("A", PP.pair), ("B", PP.b_pair)) for fr in PP.frames()}
    return model, pairs, (min(bx), min(by), max(bx), max(by)), PP.NEAR_RANGE_M_V6


def model_venezuela(track):
    sys.path.insert(0, f"{RN.DATA}/Venezuela_reprocess/scripts")
    import vz_pipeline as VZ
    from nisar_tools import UnwrappedStack
    from pyproj import Transformer
    from scipy.ndimage import label

    frames, bbox = VZ.TRACKS[track]
    unw = UnwrappedStack.from_zarr(VZ.WS.path(VZ.IONO_SOURCE[track]))
    epsg = int(unw.ds.attrs["epsg"])
    ph = unw.ds["unw"].values[0].astype(float)
    lab, _ = label(np.isfinite(ph))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    ph = np.where(lab == int(np.argmax(sizes)), ph, np.nan)
    model = xr.DataArray(ph.astype(np.float32), coords={"y": unw.ds.y.values, "x": unw.ds.x.values}, dims=("y", "x"),
                         attrs={"description": f"{track} {VZ.IONO_SOURCE[track]}, main region"})
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    lo, la = np.linspace(bbox[0], bbox[1], 30), np.linspace(bbox[2], bbox[3], 30)
    LO, LA = np.meshgrid(lo, la)
    bx, by = tr.transform(LO.ravel(), LA.ravel())
    pairs = {(band, fr): pr(track, fr) for band, pr in (("A", VZ.pair), ("B", VZ.b_pair)) for fr in frames}
    return model, pairs, (min(bx), min(by), max(bx), max(by)), VZ.NEAR_RANGE_M_V6.get(track, 11000.0)


def pixel_cells(scene, model, pairs, bounds, near_range_m, f_a):
    """Both bands' cells with the model taken out per pixel, per frame, cached; merged per band."""
    out = {}
    for band in ("A", "B"):
        per = []
        for (b, fr), (ref, sec) in sorted(pairs.items()):
            if b != band:
                continue
            path = os.path.join(RESULTS, f"cells_pixel_{scene}_{band}_{fr}.npz")
            key = json.dumps({"files": [[str(f), os.path.getsize(f), os.path.getmtime(f)] for f in (ref, sec)],
                              "band": band, "cell": CELL, "bounds": [float(v) for v in bounds],
                              "near_range_m": float(near_range_m), "model": ss._model_reader(model)[1]}, sort_keys=True)
            if os.path.exists(path):
                with np.load(path) as z:
                    if str(z["cache_key"]) == key:
                        print(f"  cells {band} {fr}: from {os.path.basename(path)}")
                        per.append(xr.Dataset({v: (("y", "x"), z[v]) for v in ("ifg", "p1", "p2", "n", "coherence", "model")},
                                              coords={"y": z["y"], "x": z["x"]}, attrs=json.loads(str(z["attrs"]))))
                        continue
            t0 = time.time()
            c = ss.gslc_cells(ref, sec, band, cell=CELL, bounds=bounds, subswath_mask="near_range",
                              near_range_m=near_range_m, phase_model=model, model_frequency=f_a)
            tmp = path[:-4] + ".partial.npz"
            np.savez(tmp, **{v: c[v].values for v in ("ifg", "p1", "p2", "n", "coherence", "model")}, y=c.y.values,
                     x=c.x.values, cache_key=np.array(key),
                     attrs=np.array(json.dumps(c.attrs, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
            os.replace(tmp, path)
            print(f"  cells {band} {fr}: {c.sizes['y']} x {c.sizes['x']}, made in {time.time() - t0:.0f}s")
            per.append(c)
        out[band] = ss.merge_cells(*per)
    return out["A"], out["B"]


def by_distance(out, dist, land):
    """Cells used, coherence, and the raw estimate's scatter about the screen, by distance from the trace and side
    (N: dist > 0)."""
    use = out.mask.values
    dev = out.dispersive.values - out.screen.values
    row = {}
    for a, b in BINS:
        for side, sgn in (("N", 1), ("S", -1)):
            band = land & (sgn * dist >= a * 1e3) & (sgn * dist < b * 1e3)
            m = band & use
            row[f"used_{side}_{a}_{b}"] = float(m.sum() / max(band.sum(), 1))
            row[f"coh_a_{side}_{a}_{b}"] = float(np.nanmedian(out.coherence_a.values[band])) if band.any() else np.nan
            row[f"coh_b_{side}_{a}_{b}"] = float(np.nanmedian(out.coherence_b.values[band])) if band.any() else np.nan
            row[f"scatter_{side}_{a}_{b}"] = (float(1.4826 * np.median(np.abs(dev[m] - np.median(dev[m]))))
                                              if m.sum() > 20 else np.nan)
    return row


def held_out(screen, raw, dist, land):
    """A screen against raw estimates it did not use: the median of ``raw - screen`` by distance and side, levelled by
    its median at 5-20 km (both sides). Inside a screen's excluded zone the raw cells of the arm without exclusion are
    independent of it: a screen that is right there leaves medians near 0 on both sides."""
    dev = raw - screen
    ok = land & np.isfinite(dev)
    lev = np.median(dev[ok & (np.abs(dist) >= 5e3) & (np.abs(dist) < 20e3)])
    row = {}
    for a, b in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 5)):
        for side, sgn in (("N", 1), ("S", -1)):
            m = ok & (sgn * dist >= a * 1e3) & (sgn * dist < b * 1e3)
            row[f"held_{side}_{a}_{b}"] = float(np.median(dev[m]) - lev) if m.sum() > 20 else np.nan
            row[f"held_n_{side}_{a}_{b}"] = int(m.sum())
    return row


def run(scene):
    t0 = time.time()
    I = RN.inputs_d141() if scene == "D141" else RN.inputs_venezuela(scene)
    ca, cb = I["ca"], I["cb"]
    f_a = float(ca.attrs["center_frequency"])
    model, pairs, bounds, near_range_m = model_d141() if scene == "D141" else model_venezuela(scene)
    am, bm = pixel_cells(scene, model, pairs, bounds, near_range_m, f_a)
    am, bm = am.sel(x=ca.x.values, y=ca.y.values), bm.sel(x=ca.x.values, y=ca.y.values)
    x, y = ca.x.values, ca.y.values
    X, Y = np.meshgrid(x, y)
    dist = RN.signed_distance(X, Y, *I["trace"])
    land = I["valid"]
    tx, ty = I["trace"]
    near = {e: (ss.fault_distance(ca, tx, ty) < e) if e > 0 else np.zeros(X.shape, bool) for e in EXCLUDE}
    if scene == "D141":                       # the drawn trace's branches are left out as the trace is
        import panama_pipeline as PP
        for e in EXCLUDE:
            if e > 0:
                for bx, by in PP.drawn_branches():
                    near[e] |= ss.fault_distance(ca, bx, by) < e
    # the pixel arm's own validity: cells holding enough model pixels (its cells count only those)
    valid_pix = np.isfinite(am.model.values)
    print(f"{scene}: cells ready [{time.time() - t0:.0f}s]; land {land.sum()} cells, pixel-arm model on {valid_pix.sum()}")
    outs, rows = {}, []
    for arm, (a, b, ph, val) in {"cell": (ca, cb, I["phase_a"], land),
                                 "pixel": (am, bm, None, land & valid_pix)}.items():
        for excl in EXCLUDE:
            for fname, fkw in FILTERS.items():
                t1 = time.time()
                out = ss.split_spectrum_screen(a, b, ph, valid=val, exclude=near[excl], **fkw)
                outs[(arm, excl, fname)] = out
                rows.append(dict(scene=scene, arm=arm, exclude_m=excl, filter=fname, seconds=time.time() - t1,
                                 **by_distance(out, dist, land)))
    ref = outs[("cell", 2000.0, "v6")]
    if os.path.exists(I["saved"]):
        sv = xr.open_dataset(I["saved"])["screen"].sel(x=x, y=y, method="nearest").values
        if I.get("phase_saved") is None:
            print(f"  cell arm (2 km, v6) vs the saved v6 screen: max |diff| "
                  f"{np.nanmax(np.abs(ref.screen.values - sv)):.2e} rad")
    phi_nd = ref.phase_a.values - ref.screen.values
    held_raw = {arm: outs[(arm, 0.0, "matched1")].dispersive.values for arm in ("cell", "pixel")}
    for r in rows:
        s = outs[(r["arm"], r["exclude_m"], r["filter"])].screen.values
        r.update(M.nisar(s, ref.screen.values, phi_nd, dist, land & np.isfinite(s), X, Y, CELL))
        # the screen's own step across the trace, closer in
        r["step_1_2"] = M.step(s, dist, land, 1.0, 2.0)
        r["step_0_1"] = M.step(s, dist, land, 0.0, 1.0)
        # against the per-pixel raw estimate of the arm without exclusion (held out where this screen excluded it)
        r.update(held_out(s, held_raw["pixel"], dist, land))
    df = pd.DataFrame(rows)
    df["tie"] = I.get("tie")
    df.to_csv(os.path.join(RESULTS, f"nisar_pixel_{scene}.csv"), index=False)
    keep = {}
    for (arm, excl, fname), out in outs.items():
        if fname in ("matched", "matched1") or (excl == 2000.0 and fname == "v6"):
            tag = f"{arm}_{int(excl)}_{fname}"
            keep[f"screen_{tag}"] = out.screen.values.astype(np.float32)
            keep[f"raw_{tag}"] = out.dispersive.values.astype(np.float32)
    np.savez_compressed(os.path.join(RESULTS, f"nisar_pixel_{scene}.npz"), x=x, y=y, dist=dist.astype(np.float32),
                        land=land, coh_a_cell=ca.coherence.values.astype(np.float32),
                        coh_b_cell=cb.coherence.values.astype(np.float32),
                        coh_a_pixel=am.coherence.values.astype(np.float32),
                        coh_b_pixel=bm.coherence.values.astype(np.float32), phi_nd=phi_nd.astype(np.float32),
                        trace_x=np.asarray(tx), trace_y=np.asarray(ty), **keep, label=I["label"], epsg=I["epsg"])
    mm = 0.24196 / (4 * np.pi) * 1e3
    show = ["arm", "exclude_m", "filter", "leak", "leak_r", "step", "step_1_2", "diff_near", "diff_far"]
    held = [c for c in df.columns if c.startswith("held_") and not c.startswith("held_n_")]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(df[show])
        print("held-out medians (rad; raw - screen, per side and distance):")
        print(df[["arm", "exclude_m", "filter"] + held])
        print(df[df["filter"] == "v6"][["arm", "exclude_m"] + [c for c in df.columns if c.startswith("used_")]])
    print(f"{scene} done [{time.time() - t0:.0f}s] (1 rad = {mm:.1f} mm of LOS)")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
