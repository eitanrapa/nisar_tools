"""Validation against a real NISAR GUNW granule.

Skipped unless ``NISAR_TEST_GUNW`` points at a complete GUNW ``.h5``. Confirms
the real product layout matches what the synthetic fixture assumes, and that a
crop -> self-contained ``to_los`` runs on real-sized arrays.
"""

import os

import numpy as np
import pytest

from nisar_tools import LOSStack, UnwrappedStack

PATH = os.environ.get("NISAR_TEST_GUNW")

pytestmark = pytest.mark.skipif(not PATH, reason="set NISAR_TEST_GUNW to run")


def test_real_gunw_schema():
    g = UnwrappedStack.from_gunw_file(PATH)
    assert g.sizes["pair"] == 1
    assert g.epsg > 0
    assert g.direction in ("Ascending", "Descending")
    assert set(g.ds.data_vars) >= {"unw", "coherence", "conncomp"}
    assert g.ds["conncomp"].dtype == np.uint32
    assert 0.15 < g.ds.attrs["wavelength"] < 0.30  # L-band, ~0.24 m
    assert g.ds.attrs["source"] == "gunw"
    assert g.ds.attrs["source_files"]
    assert g.ds["ref_time"].values[0] < g.ds["sec_time"].values[0]

    unw = g.ds["unw"].isel(pair=0).values
    assert np.isfinite(unw).any() and (~np.isfinite(unw)).any()  # a real swath edge


def test_real_gunw_to_los():
    g = UnwrappedStack.from_gunw_file(PATH)

    # crop a modest box near the frame centre via its native coords
    from nisar_tools import geo

    cx, cy = float(np.median(g.x)), float(np.median(g.y))
    half = 400 * abs(float(g.x[1] - g.x[0]))
    bbox = geo.native_bbox_to_lonlat(cx - half, cx + half, cy - half, cy + half, g.epsg)
    gc = g.crop(*bbox)
    assert 0 < gc.sizes["y"] < g.sizes["y"]

    los = gc.to_los()  # no GSLC: geometry from the GUNW's own cube
    assert isinstance(los, LOSStack)

    unw = gc.ds["unw"].isel(pair=0).values
    d = los.ds["los"].isel(pair=0).values
    valid = np.isfinite(unw)
    if valid.any():
        wl = los.ds.attrs["wavelength"]
        assert np.allclose(d[valid], wl / (4 * np.pi) * unw[valid], atol=1e-5)

    inc = los.ds["incidence_angle"].values
    up = los.ds["los_up"].values
    fin = np.isfinite(inc)
    assert np.allclose(up[fin], np.cos(np.deg2rad(inc[fin])), atol=1e-4)
    look = los.ds["look_angle"].values
    assert np.nanmax(look - inc) < 0  # look angle smaller than incidence
    assert 10.0 < np.nanmin(inc) and np.nanmax(inc) < 70.0
