"""Validation against a real NISAR GSLC granule.

Skipped unless ``NISAR_TEST_GSLC`` points at a complete granule. Validates that
the real HDF5 layout, attributes, dtypes, and coordinate ordering match what
the synthetic fixtures assume, and that the full lazy pipeline runs on
real-sized arrays without materialising the whole image.

Set ``NISAR_TEST_GSLC2`` to a second same-frame granule (different date) to
also validate a real two-date interferogram.
"""

import os

import numpy as np
import pytest

from nisar_tools import GSLC, GSLCStack, Workspace

PATH = os.environ.get("NISAR_TEST_GSLC")
PATH2 = os.environ.get("NISAR_TEST_GSLC2")

pytestmark = pytest.mark.skipif(not PATH, reason="set NISAR_TEST_GSLC to run")


def test_real_metadata_and_lazy_read():
    g = GSLC(PATH)
    try:
        assert g.epsg > 0
        assert g.direction in ("Ascending", "Descending")
        assert g.datetime is not None
        ny, nx = g.shape
        assert ny > 0 and nx > 0

        data = g.data
        assert data.dims == ("y", "x")
        assert data.dtype == np.complex64
        # Read only a tiny corner — proves lazy, partial reads work.
        corner = data.isel(y=slice(0, 64), x=slice(0, 64)).compute()
        assert corner.shape == (64, 64)
    finally:
        g.close()


def test_real_crop_and_self_interferogram(tmp_path):
    g = GSLC(PATH)
    try:
        # Crop a small box near the granule centre via its native coords.
        from nisar_tools import geo

        cx = float(np.median(g.x_coords))
        cy = float(np.median(g.y_coords))
        half = 50 * abs(g.x_spacing)  # ~100-pixel box
        lon0, lon1, lat0, lat1 = geo.native_bbox_to_lonlat(
            cx - half, cx + half, cy - half, cy + half, g.epsg
        )
        cropped = g.crop(lon0, lon1, lat0, lat1)
        assert 0 < cropped.sizes["y"] < g.shape[0]
        assert 0 < cropped.sizes["x"] < g.shape[1]

        # Self-interferogram: coherence must be ~1 everywhere with signal.
        ws = Workspace(tmp_path / "ws")
        stack = GSLCStack.from_gslcs([g, g], bbox=(lon0, lon1, lat0, lat1))
        stack = stack.persist(ws, "slc_stack")
    finally:
        g.close()

    igrams = stack.form_interferograms(pairs=[(0, 1)], looks=5, downsample=True)
    coh = igrams.ds["coherence"].isel(pair=0).compute().values
    valid = coh[coh > 0]
    assert valid.size > 0
    assert np.nanmedian(valid) > 0.99  # identical inputs => coherent


@pytest.mark.skipif(not PATH2, reason="set NISAR_TEST_GSLC2 for a real pair")
def test_real_two_date_pipeline(tmp_path):
    g1, g2 = GSLC(PATH), GSLC(PATH2)
    try:
        assert g1.epsg == g2.epsg
        ws = Workspace(tmp_path / "ws")
        stack = GSLCStack.from_gslcs([g1, g2]).persist(ws, "slc_stack")
    finally:
        g1.close()
        g2.close()

    igrams = stack.form_interferograms(looks=5, downsample=True).persist(ws, "igrams")
    assert igrams.sizes["pair"] == 1
    unw = igrams.unwrap(ws, nproc=4)
    assert unw.sizes["pair"] == 1
    assert np.isfinite(unw.ds["unw"].isel(pair=0).compute().values).any()
