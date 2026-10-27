"""The Venezuela v6 screens as the tie experiments read them (read only). There the main band was unwrapped WITHOUT a
cut where the north side has data, so north and south of the trace are one region: the side band's step across the
fault must be zero cycles there. That makes the rupture a control for the tie.

    load(track) -> dict     track A061 | D126 | A162 | D054
"""
import os
import sys

import numpy as np
import xarray as xr

DATA = os.path.expanduser("~/Documents/data")
EPSG = {"A061": 32620, "D126": 32619, "A162": 32619, "D054": 32619}


def load(track):
    sys.path.insert(0, f"{DATA}/Venezuela_reprocess/scripts")
    import vz_pipeline as VZ
    nc = xr.open_dataset(os.path.join(VZ.V, "workspace", f"iono_split_spectrum_{track}_v6.nc")).load()
    unw = xr.open_zarr(VZ.WS.path(VZ.IONO_SOURCE[track]))
    epsg = int(unw.attrs.get("epsg", EPSG[track]))
    tx, ty = VZ.trace2_xy(epsg)
    region = np.isfinite(nc["phase_a"].values)       # the main region's cells (the screen read phi_A from it alone)
    return dict(track=track, x=nc.x.values, y=nc.y.values, d=nc["dispersive"].values, use=nc["mask"].values.astype(bool),
                phase_a=nc["phase_a"].values, screen_f=nc["screen"].values, coh_a=nc["coherence_a"].values,
                coh_b=nc["coherence_b"].values, f_a=float(nc.attrs["f_a"]), f_b=float(nc.attrs["f_b"]),
                cell=float(nc.attrs["cell"]), region=region, trace=(np.asarray(tx, float), np.asarray(ty, float)), epsg=epsg)
