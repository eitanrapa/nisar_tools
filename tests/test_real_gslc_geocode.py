"""Making a GSLC from a real NISAR RSLC.

Skipped unless ``NISAR_TEST_RSLC`` (an RSLC granule) and ``NISAR_TEST_DEM`` (a
DEM covering it, ideally NISAR DEM v1.2) are set. Geocodes a 2 km box at the
swath centre and checks the result reads as a GSLC. ``NISAR_TEST_TEC`` adds the
ionospheric correction; ``NISAR_TEST_GSLC_REF`` (any ASF GSLC) adds a format
comparison, which must find nothing structural to report.
"""

import importlib.util
import os
from pathlib import Path

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
    path = Path(__file__).resolve().parents[1] / "scripts" / "gslc_format_diff.py"
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
