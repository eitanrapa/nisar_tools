"""Making a GSLC from a real NISAR RSLC.

Skipped unless ``NISAR_TEST_RSLC`` (an RSLC granule) and ``NISAR_TEST_DEM`` (a
DEM covering it, ideally NISAR DEM v1.2) are set. Geocodes a 2 km box at the
swath centre and checks the result reads as a GSLC. ``NISAR_TEST_TEC`` adds the
ionospheric correction; ``NISAR_TEST_GSLC_REF`` (any ASF GSLC) adds a format
comparison, which must find nothing structural to report.

When ``NISAR_TEST_GSLC_REF`` is the GSLC ASF made from ``NISAR_TEST_RSLC`` with
``NISAR_TEST_TEC`` (its `l1SlcGranules` and `tecFiles` name them) and the DEM is
NISAR DEM v1.2, the pixels are also compared with ASF's on ASF's own grid: the
default settings must stay bit-identical to ISCE3 v0.25.16 on almost every pixel.
"""

import importlib.util
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from nisar_tools import GSLC, RSLC, geocode_rslc

RSLC_PATH = os.environ.get("NISAR_TEST_RSLC")
DEM_PATH = os.environ.get("NISAR_TEST_DEM")
TEC_PATH = os.environ.get("NISAR_TEST_TEC")
REF_PATH = os.environ.get("NISAR_TEST_GSLC_REF")

pytestmark = pytest.mark.skipif(not (RSLC_PATH and DEM_PATH),
                                reason="set NISAR_TEST_RSLC and NISAR_TEST_DEM to run")


def _format_diff():
    path = Path(__file__).resolve().parent / "gslc_format_diff.py"
    spec = importlib.util.spec_from_file_location("gslc_format_diff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_gslc(tmp_path_factory):
    with RSLC(RSLC_PATH) as r:
        grid = r.default_geogrid("A", snap=240.0)
    cx = grid.start_x + grid.spacing_x * grid.width / 2
    cy = grid.start_y + grid.spacing_y * grid.length / 2
    cx, cy = 240.0 * np.floor(cx / 240.0), 240.0 * np.floor(cy / 240.0)
    bounds = (cx - 960.0, cy + 960.0, cx + 960.0, cy - 960.0)
    out = tmp_path_factory.mktemp("gslc") / "small_gslc.h5"
    geocode_rslc(RSLC_PATH, out, DEM_PATH, bounds=bounds, epsg=grid.epsg,
                 tec_file=TEC_PATH, workers=4, log=lambda *a: None)
    return out


def test_real_rslc_geocodes_to_a_readable_gslc(small_gslc):
    g = GSLC(small_gslc)
    try:
        assert g.shape == (384, 384)
        hh = g.data.values
        assert hh.dtype == np.complex64
        assert np.isfinite(hh).mean() > 0.99          # the box sits mid-swath
        assert np.median(np.abs(hh[np.isfinite(hh)])) > 0
    finally:
        g.close()


@pytest.mark.skipif(not REF_PATH, reason="set NISAR_TEST_GSLC_REF to compare formats")
def test_real_gslc_format_matches_asf(small_gslc):
    problems = _format_diff().compare(REF_PATH, str(small_gslc))
    assert problems == []


GRIDS = "science/LSAR/GSLC/grids"


def _made_from_inputs(ref):
    """Whether ASF made `ref` from NISAR_TEST_RSLC and NISAR_TEST_TEC."""
    if not TEC_PATH:
        return False
    with h5py.File(ref, "r") as f:
        inputs = f["science/LSAR/GSLC/metadata/processingInformation/inputs"]
        names = lambda key: [Path(v.decode()).name for v in np.atleast_1d(inputs[key][()])]  # noqa: E731
        return Path(RSLC_PATH).name in names("l1SlcGranules") and Path(TEC_PATH).name in names("tecFiles")


@pytest.mark.skipif(not (REF_PATH and _made_from_inputs(REF_PATH)),
                    reason="set NISAR_TEST_GSLC_REF to ASF's GSLC of NISAR_TEST_RSLC, and NISAR_TEST_TEC to its TEC")
def test_real_gslc_pixels_match_asf(tmp_path):
    with h5py.File(REF_PATH, "r") as f:
        xa, ya = f[f"{GRIDS}/frequencyA/xCoordinates"][()], f[f"{GRIDS}/frequencyA/yCoordinates"][()]
        dxb = float(f[f"{GRIDS}/frequencyB/xCoordinates"][1] - f[f"{GRIDS}/frequencyB/xCoordinates"][0]) \
            if f"{GRIDS}/frequencyB" in f else 5.0
        epsg = int(np.asarray(f[f"{GRIDS}/frequencyA/projection"][()]))
    dx, dy = float(xa[1] - xa[0]), float(ya[1] - ya[0])
    left, top = xa[0] - dx / 2, ya[0] - dy / 2
    # a 1920 m box mid-swath on edges common to both bands' grids
    c0 = left + dxb * np.floor((xa[xa.size // 2] - left) / dxb)
    r0 = top + dy * np.floor((ya[ya.size // 2] - top) / dy)
    bounds = (c0, r0, c0 + 1920.0, r0 + 1920.0 * np.sign(dy))
    out = tmp_path / "gslc_on_asf_grid.h5"
    full = tmp_path / "gslc_isce3_compatible.h5"
    for path, compatible in ((out, "pixels"), (full, True)):
        geocode_rslc(RSLC_PATH, path, DEM_PATH, bounds=bounds, epsg=epsg, polarizations=["HH"],
                     tec_file=TEC_PATH, isce3_compatible=compatible, workers=4, log=lambda *a: None)
    with h5py.File(out, "r") as mine, h5py.File(full, "r") as isce3, h5py.File(REF_PATH, "r") as ref:
        for freq in [k[-1] for k in mine[GRIDS] if k.startswith("frequency")]:
            g, r = mine[f"{GRIDS}/frequency{freq}"], ref[f"{GRIDS}/frequency{freq}"]
            x, y = g["xCoordinates"][()], g["yCoordinates"][()]
            xr_, yr_ = r["xCoordinates"][()], r["yCoordinates"][()]
            j0 = int(np.argmin(np.abs(xr_ - x[0])))
            i0 = int(np.argmin(np.abs(yr_ - y[0])))
            assert np.allclose(xr_[j0:j0 + x.size], x) and np.allclose(yr_[i0:i0 + y.size], y)
            a = g["HH"][()]
            b = r["HH"][i0:i0 + y.size, j0:j0 + x.size]
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{freq}: NaN pattern differs"
            # the default fixes only masks and metadata; isce3_compatible=True reproduces those too
            assert np.array_equal(a.view(np.uint64), isce3[f"{GRIDS}/frequency{freq}/HH"][()].view(np.uint64))
            assert np.array_equal(isce3[f"{GRIDS}/frequency{freq}/mask"][()],
                                  r["mask"][i0:i0 + y.size, j0:j0 + x.size]), f"{freq}: mask differs"
            ok = np.isfinite(a)
            same = (a.view(np.uint64) == b.view(np.uint64))[ok]
            # the rest are a few steps of the 10-bit mantissa truncation apart (measured: <= 1.6e-3 of |z|)
            a_, b_ = a[ok][~same], b[ok][~same]
            assert np.all(np.abs(a_ - b_) <= 2.0 ** -8 * np.maximum(np.abs(a_), np.abs(b_))), \
                f"{freq}: differences beyond a few truncation steps"
            assert same.mean() >= 0.995, f"{freq}: only {100 * same.mean():.2f}% bit-identical"
