"""The Panama screens as the tie experiments read them: the raw dispersive estimate on the cells, the two tied blocks on
the cell grid, the main trace, and the ROFF slant-range offset on the cells (for m = 4 pi / lambda * range - phi_A).

    load(track, screen)  -> dict      track D141 | A004 | D040 | A076; screen v6 | v7 | v8 (the pipelines' names)

Read only. The pipelines are ~/Documents/data/D141_panama/scripts/panama_pipeline.py (TRACK=...).
"""
import json
import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

DATA = os.path.expanduser("~/Documents/data")
SCENES = {
    "D141": dict(ws="D141_panama/workspace_reg", tie_frame="085", roff="D141_panama/ancillary/registration_085_roff.npz"),
    "A004": dict(ws="A004_panama/workspace_reg_005_006", tie_frame="005", roff="A004_panama/ancillary/registration_005_roff.npz"),
    "D040": dict(ws="D040_panama/workspace_reg_084_085", tie_frame="085", roff="D040_panama/ancillary/registration_085_roff.npz"),
    "A076": dict(ws="A076_panama/workspace_reg_005_006", tie_frame="005", roff="A076_panama/ancillary/registration_005_roff.npz"),
}
#: screen version -> the cut / unwrapping / tie version it read, and the trace file of that cut
CUT = {"v6": "v3", "v7": "v7", "v8": "v8"}
TRACES = {"v7": f"{DATA}/A076_panama/rupture_trace_20261019.json", "v8": f"{DATA}/A076_panama/rupture_trace_20261020.json"}
TRACE_V3 = {"D141": f"{DATA}/D141_panama/rupture_trace_D141_drawn.json", "A004": f"{DATA}/D141_panama/rupture_trace_D141_drawn.json",
            "D040": f"{DATA}/D040_panama/rupture_trace_D040_drawn.json", "A076": f"{DATA}/D040_panama/rupture_trace_D040_drawn.json"}


def _cells_like(nc):
    return xr.Dataset(coords={"y": nc.y, "x": nc.x}, attrs={"cell": float(nc.attrs["cell"])})


def load(track, screen="v8", roff=True):
    sc = SCENES[track]
    ws = f"{DATA}/{sc['ws']}"
    nc = xr.open_dataset(f"{ws}/iono_split_spectrum_{screen}_s5.nc").load()
    cut = CUT[screen]
    unw = xr.open_zarr(f"{ws}/unw_reg_tied_{cut}_s5.zarr")
    ties = unw.attrs["component_ties"]
    ties = json.loads(ties) if isinstance(ties, str) else ties
    ph = unw["unw"].isel(pair=0).values.astype(np.float32)
    x50, y50 = unw.x.values, unw.y.values
    lab, _ = label(np.isfinite(ph))
    cells = _cells_like(nc)
    from nisar_tools import split_spectrum as ss
    ppc = (float(nc.attrs["cell"]) / abs(float(x50[1] - x50[0]))) ** 2
    frac = {}
    for name, L in (("north", ties["north_label"]), ("south", ties["south_label"])):
        frac[name] = ss.cell_average(np.where(lab == L, 1.0, np.nan), x50, y50, cells)[1] / ppc
    anyb = ss.cell_average(np.where(lab > 0, 1.0, np.nan), x50, y50, cells)[1] / ppc
    del lab
    block_n = (frac["north"] > 0.3) & (frac["north"] >= frac["south"])
    block_s = (frac["south"] > 0.3) & (frac["south"] > frac["north"])
    tr = json.load(open(TRACES.get(screen) or TRACE_V3[track]))
    out = dict(track=track, screen=screen, cut=cut, ties=ties, x=nc.x.values, y=nc.y.values, d=nc["dispersive"].values,
               use=nc["mask"].values.astype(bool), phase_a=nc["phase_a"].values, screen_f=nc["screen"].values,
               coh_a=nc["coherence_a"].values, coh_b=nc["coherence_b"].values, f_a=float(nc.attrs["f_a"]),
               f_b=float(nc.attrs["f_b"]), cell=float(nc.attrs["cell"]), block_n=block_n, block_s=block_s,
               other_block=(anyb > 0.3) & ~block_n & ~block_s,
               trace=(np.asarray(tr["main"]["x"], float), np.asarray(tr["main"]["y"], float)),
               branches=[(np.asarray(tr[k]["x"], float), np.asarray(tr[k]["y"], float)) for k in tr.get("main_branches", [])])
    if roff:
        z = np.load(f"{DATA}/{sc['roff']}")
        yy, xx = z["y"], z["x"]
        flip = yy[1] < yy[0]
        f_r = RegularGridInterpolator((yy[::-1] if flip else yy, xx), z["range_m"][::-1] if flip else z["range_m"],
                                      bounds_error=False)
        f_w = RegularGridInterpolator((yy[::-1] if flip else yy, xx), z["weight"][::-1] if flip else z["weight"],
                                      bounds_error=False, fill_value=0.0)
        X, Y = np.meshgrid(out["x"], out["y"])
        q = np.c_[Y.ravel(), X.ravel()]
        rng = f_r(q).reshape(X.shape)
        wr = f_w(q).reshape(X.shape)
        lam = 299792458.0 / out["f_a"]
        out["m"] = np.where(wr > 0.99, 4 * np.pi / lam * rng - out["phase_a"], np.nan)
    return out
