"""Tests for the download module.

The real backends (earthaccess, sardem) need network access and Earthdata
credentials, so they are faked here via ``sys.modules`` injection -- the same
spirit as the credentialed, env-gated real-data test. These cover the pure logic
(bbox/URL/argument handling) and the orchestration (which backend is called with
which arguments), not live downloads.
"""

import os
import sys
import types

import pytest

from nisar_tools import download as d
from nisar_tools.download import GSLC_SHORT_NAME


# -- fakes -------------------------------------------------------------------
def _fake_earthaccess(record):
    m = types.ModuleType("earthaccess")

    def login(**kwargs):
        record["login"] = kwargs
        return "auth-obj"

    def search_data(**kwargs):
        record.setdefault("searches", []).append(kwargs)
        # A distinct sentinel per call so download() can echo them back.
        return [("granule", kwargs.get("granule_name", kwargs.get("short_name")))]

    def download(results, local_path):
        record["download"] = {"results": list(results), "path": local_path}
        return [f"{local_path}/file{i}.h5" for i in range(len(results))]

    m.login, m.search_data, m.download = login, search_data, download
    return m


@pytest.fixture
def fake_ea(monkeypatch):
    record = {}
    monkeypatch.setitem(sys.modules, "earthaccess", _fake_earthaccess(record))
    return record


@pytest.fixture
def fake_sardem(monkeypatch):
    record = {}
    dem = types.ModuleType("sardem.dem")

    def main(**kwargs):
        record["main"] = kwargs

    dem.main = main
    monkeypatch.setitem(sys.modules, "sardem.dem", dem)
    return record


class _FakeBulk:
    """Stand-in for the ASF bulk downloader; records URLs, hits no network."""

    def __init__(self, download_dir, files):
        self.download_dir = download_dir
        self.files = files
        self.downloaded = False

    def download_files(self):
        self.downloaded = True

    def print_summary(self):
        pass


@pytest.fixture
def fake_bulk(monkeypatch):
    from nisar_tools import _asf_bulk

    made = {}

    def factory(download_dir, files):
        made["obj"] = _FakeBulk(download_dir, files)
        return made["obj"]

    monkeypatch.setattr(_asf_bulk, "bulk_downloader", factory)
    # Don't actually install a SIGINT handler during the test.
    monkeypatch.setattr(d.signal, "signal", lambda *a, **k: None)
    return made


# -- pure helpers ------------------------------------------------------------
def test_bbox_conversion_orders_corners():
    assert d._bbox_wsen((-120.5, -119.5, 34.0, 35.0)) == (-120.5, 34.0, -119.5, 35.0)
    # Corners given in either order still yield (west, south, east, north).
    assert d._bbox_wsen((-119.5, -120.5, 35.0, 34.0)) == (-120.5, 34.0, -119.5, 35.0)


def test_gslc_url_builds_and_strips_extension():
    assert d.gslc_url("G1") == (
        "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/"
        f"{GSLC_SHORT_NAME}/G1/G1.h5"
    )
    assert d.gslc_url("G1.h5").endswith("/G1/G1.h5")


def test_as_list_normalises():
    assert d._as_list(None) == []
    assert d._as_list("a") == ["a"]
    assert d._as_list(["a", "b"]) == ["a", "b"]


# -- earthaccess search ------------------------------------------------------
def test_search_passes_bbox_and_temporal(fake_ea):
    d.search(GSLC_SHORT_NAME, bbox=(-120.5, -119.5, 34.0, 35.0),
             temporal=("2025-11", "2025-12"))
    (call,) = fake_ea["searches"]
    assert call["short_name"] == GSLC_SHORT_NAME
    assert call["bounding_box"] == (-120.5, 34.0, -119.5, 35.0)
    assert call["temporal"] == ("2025-11", "2025-12")
    assert "granule_name" not in call


def test_search_by_granules_is_one_query_per_name(fake_ea):
    d.search(GSLC_SHORT_NAME, granules=["a", "b"])
    names = [c["granule_name"] for c in fake_ea["searches"]]
    assert names == ["a", "b"]


# -- GSLCs -------------------------------------------------------------------
def test_download_gslcs_earthaccess(fake_ea, tmp_path):
    out = d.download_gslcs(tmp_path / "g", bbox=(-120.5, -119.5, 34.0, 35.0))
    assert (tmp_path / "g").is_dir()
    assert fake_ea["searches"][0]["short_name"] == GSLC_SHORT_NAME
    assert fake_ea["download"]["path"] == str(tmp_path / "g")
    assert all(str(p).endswith(".h5") for p in out)


def test_download_gslcs_asf_builds_urls(fake_bulk, tmp_path):
    out = d.download_gslcs(tmp_path / "g", granules=["G1", "G2"], method="asf")
    urls = fake_bulk["obj"].files
    assert urls == [d.gslc_url("G1"), d.gslc_url("G2")]
    assert fake_bulk["obj"].downloaded
    assert [p.name for p in out] == ["G1.h5", "G2.h5"]


def test_download_gslcs_asf_requires_granules(tmp_path):
    with pytest.raises(ValueError, match="requires explicit granule names"):
        d.download_gslcs(tmp_path, method="asf")


def test_download_gslcs_bad_method(tmp_path):
    with pytest.raises(ValueError, match="method must be"):
        d.download_gslcs(tmp_path, method="ftp")


# -- DEM ---------------------------------------------------------------------
def test_download_dem_converts_bbox_and_returns_dest(fake_sardem, tmp_path):
    dest = tmp_path / "dem" / "out.tif"
    out = d.download_dem(dest, (-120.5, -119.5, 34.0, 35.0), data_source="NISAR")
    call = fake_sardem["main"]
    assert call["output_name"] == str(dest)
    assert call["bbox"] == [-120.5, 34.0, -119.5, 35.0]  # [left, bottom, right, top]
    assert call["data_source"] == "NISAR"
    assert out == dest
    assert dest.parent.is_dir()


# -- auth + misc -------------------------------------------------------------
def test_login_passes_strategy(fake_ea):
    assert d.login(strategy="netrc") == "auth-obj"
    assert fake_ea["login"]["strategy"] == "netrc"


def test_download_files_backcompat_uses_asf(fake_bulk, tmp_path):
    out = d.download_files(str(tmp_path / "g"), ["G1"])
    assert fake_bulk["obj"].files == [d.gslc_url("G1")]
    assert [p.name for p in out] == ["G1.h5"]


def test_require_missing_module_is_helpful():
    with pytest.raises(ImportError, match="not installed"):
        d._require("nisar_tools_no_such_module_xyz")


# -- opt-in live download (no Earthdata login needed for Copernicus) ----------
@pytest.mark.skipif(
    not os.environ.get("NISAR_TEST_DOWNLOAD"),
    reason="set NISAR_TEST_DOWNLOAD=1 for a live Copernicus DEM download",
)
def test_download_dem_cop_live(tmp_path):
    # Exercises the real sardem integration end-to-end; a tiny bbox over land on
    # Hawai'i keeps it to a single Copernicus tile.
    import rioxarray  # noqa: F401

    dest = d.download_dem(
        tmp_path / "cop.tif", (-155.60, -155.55, 19.00, 19.05), data_source="COP"
    )
    assert dest.exists() and dest.stat().st_size > 0
    da = rioxarray.open_rasterio(dest)
    assert da.rio.crs is not None and da.size > 0
