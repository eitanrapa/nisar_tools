"""Ionospheric timing corrections from NISAR TEC files, and finding those files.

The LUTs are checked against the physics they encode on a synthetic orbit: a
constant sub-orbital TEC delays range by ``40.31 TEC / f^2 / cos(inc)`` and
shifts nothing in azimuth; a TEC changing along track shifts azimuth by
``40.31 dTEC/dt r / (v^2 f^2)``. The CMR search and the download are faked --
live Earthdata access belongs in a gated test, not the unit suite.
"""

import io
import json
import sys
import types
import urllib.request

import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import download, radar
from nisar_tools.corrections import K_IONO, TECU, tec_luts

FREQ = radar.SPEED_OF_LIGHT / 0.242


def _tec_file(tmp_path, sub_tec, start=82_300.0, n=41, spacing=10.0):
    """An IMAGEN-style TEC JSON with ``tot - top == sub_tec(sample_index)``."""
    t = start + spacing * np.arange(n)
    utc = [str(rs.EPOCH + np.timedelta64(int(s), "s")) for s in t]
    sub = np.array([sub_tec(k) for k in range(n)], float)
    top = np.full(n, 3.0)
    doc = {"utc": utc, "totTecNr": (sub + top).tolist(), "topTecNr": top.tolist(),
           "totTecFr": (sub + top + 0.5).tolist(), "topTecFr": (top + 0.5).tolist()}
    path = tmp_path / "NISAR_ANC_TEC_test.json"
    path.write_text(json.dumps(doc))
    return path


def _incidence(t, r, grid, orbit):
    llh, ok = radar.rdr2geo([t], [r], orbit, height=0.0, wavelength=grid.wavelength,
                            look_side=grid.look_side)
    lon, lat = llh[0, 0], llh[0, 1]
    target = radar.lonlat_to_xyz(lon, lat, 0.0)
    sat, _ = orbit.interpolate(np.array([t]))
    normal = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    los = sat[0] - target
    return np.arccos(abs(normal @ los) / np.linalg.norm(los))


def test_constant_tec_delays_range_and_leaves_azimuth(tmp_path):
    orbit, grid = rs.orbit(), rs.radar_grid()
    az, srg = tec_luts(_tec_file(tmp_path, lambda k: 10.0), FREQ, orbit, grid)
    times = srg.y_start + srg.dy * np.arange(srg.length)
    assert times[0] < grid.sensing_start and grid.sensing_stop < times[-1]
    for i in (0, srg.length - 1):
        for j, r in enumerate((grid.starting_range, grid.end_range)):
            expected = K_IONO * 10.0 * TECU / FREQ ** 2 / np.cos(_incidence(times[i], r, grid, orbit))
            assert srg.data[i, j] == pytest.approx(expected, rel=1e-9)
    assert np.all(az.data == 0.0)
    assert 2.0 < srg.data.mean() < 5.0            # metres: ASF's products carry 2.6-3.2


def test_along_track_tec_gradient_shifts_azimuth(tmp_path):
    orbit, grid = rs.orbit(), rs.radar_grid()
    per_sample = 0.3                                # TECU per 10 s sample
    az, _ = tec_luts(_tec_file(tmp_path, lambda k: 15.0 + per_sample * k), FREQ, orbit, grid)
    times = az.y_start + az.dy * np.arange(az.length)
    _, vel = orbit.interpolate(times)
    speed2 = np.sum(vel ** 2, axis=1)
    for j, r in enumerate((grid.starting_range, grid.end_range)):
        expected = K_IONO * (per_sample / 10.0) * TECU * r / (speed2 * FREQ ** 2)
        assert np.allclose(az.data[:, j], expected, rtol=1e-9)
    assert (az.data > 0).all()


def test_tec_file_that_misses_the_acquisition_is_refused(tmp_path):
    orbit, grid = rs.orbit(), rs.radar_grid()
    early = _tec_file(tmp_path, lambda k: 10.0, start=81_000.0, n=20)
    with pytest.raises(ValueError, match="does not"):
        tec_luts(early, FREQ, orbit, grid)


# -- finding and fetching the file --------------------------------------------------
def _entry(name, start, end, private_too=True):
    base = "https://nisar.asf.earthdatacloud.nasa.gov"
    links = [{"href": f"{base}/NISAR/TEC/{name}/{name}.json"}]
    if private_too:
        links.insert(0, {"href": f"{base}/NISAR-JPL-PRIVATE-DATA/TEC/{name}/{name}.json"})
        links.append({"href": f"{base}/NISAR-JPL-PRIVATE-DATA/TEC/{name}/{name}.context.json"})
    return {"title": name, "time_start": start, "time_end": end, "links": links}


@pytest.fixture
def fake_cmr(monkeypatch):
    calls = []
    feed = {"feed": {"entry": [
        _entry("NISAR_ANC_TEC_20260623T212027_20260621T230002_20260622T235952_s015",
               "2026-06-21T23:00:02.000Z", "2026-06-22T23:59:52.000Z"),
        _entry("NISAR_ANC_TEC_20260624T211956_20260622T230002_20260623T235952_s015",
               "2026-06-22T23:00:02.000Z", "2026-06-23T23:59:52.000Z"),
    ]}}

    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(json.dumps(feed).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_search_tec_keeps_files_covering_the_whole_acquisition(fake_cmr):
    hits = download.search_tec("2026-06-22T22:54:16", "2026-06-22T22:54:51")
    assert [h["name"][14:29] for h in hits] == ["20260623T212027"]
    assert hits[0]["url"].endswith(".json") and "PRIVATE" not in hits[0]["url"]
    assert "C3622219145-ASF" in fake_cmr[0] and "temporal" in fake_cmr[0]
    # After 23:00 both days' files cover it: the newest production comes first.
    late = download.search_tec(np.datetime64("2026-06-22T23:30:00"))
    assert [h["produced"] for h in late] == ["20260624T211956", "20260623T212027"]


def test_download_tec_writes_the_json(tmp_path, monkeypatch, fake_cmr):
    got = {}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, timeout=None):
            got["url"] = url
            return types.SimpleNamespace(content=b'{"utc": []}', raise_for_status=lambda: None)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(Session=Session))
    path = download.download_tec(tmp_path, "2026-06-22T22:54:16", "2026-06-22T22:54:51")
    assert path.name == "NISAR_ANC_TEC_20260623T212027_20260621T230002_20260622T235952_s015.json"
    assert path.read_bytes() == b'{"utc": []}' and got["url"].endswith(path.name)
    assert download.download_tec(tmp_path, "2026-06-22T22:54:16") == path    # cached
    with pytest.raises(ValueError):
        download.download_tec(tmp_path, "2026-06-22T22:54:16", rslc="x.h5")


def test_download_tec_refuses_a_login_page(tmp_path, monkeypatch, fake_cmr):
    class Session:
        __enter__ = lambda self: self                        # noqa: E731
        __exit__ = lambda self, *exc: False                  # noqa: E731

        def get(self, url, timeout=None):
            return types.SimpleNamespace(content=b"<html>login</html>", raise_for_status=lambda: None)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(Session=Session))
    with pytest.raises(IOError, match="did not return JSON"):
        download.download_tec(tmp_path / "tec.json", "2026-06-22T22:54:16")
