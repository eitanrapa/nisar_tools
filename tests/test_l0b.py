"""L0B pulse timing: the dither-table fit, the classification of bad tags, and
their place in an RSLC's zero-Doppler time.

Synthetic L0Bs carry a random 173-interval dither table, a clock drift and the
two defects found in delivered NISAR L0Bs: runs of echoes tagged with the
previous (or next) pulse's transmit time, and a squeezed span of tags.
"""

import os

import h5py
import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import l0b, radar

UNITS = radar.epoch_units(rs.EPOCH)


def _true_times(n, t0, table, phase=37, drift=0.12e-6):
    k = np.arange(n)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, n // table.size + 3))])
    return t0 + (cum[k + phase] - cum[phase]) * (1.0 + drift)


def _write_l0b(path, utc, table, first_line=5000, name="NISAR_L0_PR_RRSD_TEST"):
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/granuleId"] = np.bytes_(name)
        tx = f.create_group("science/LSAR/RRSD/swaths/frequencyA/txH")
        tx["UTCtime"] = utc
        tx["UTCtime"].attrs["units"] = np.bytes_(UNITS)
        tx["rangeLineIndex"] = (first_line + np.arange(utc.size)).astype(np.uint32)
        tx["PRISequence"] = table
        tx["calType"] = np.zeros(utc.size, np.uint8)
    return path


@pytest.fixture
def table():
    return np.random.default_rng(1).uniform(0.423e-3, 0.621e-3, 173)


@pytest.fixture
def defective(tmp_path, table):
    true = _true_times(30000, 1000.0, table)
    utc = true.copy()
    utc[:3000] = true[3000] - 0.6 * (true[3000] - true[:3000])     # squeezed, as in 003D 2026-07-08
    utc[12000:13000] = true[11999:12999]                           # tagged with the previous pulse's time
    utc[20000:20400] = true[20001:20401]                           # tagged with the next pulse's time
    return _write_l0b(tmp_path / "l0b.h5", utc, table), true


def test_pulse_timing_finds_each_defect_and_restores_the_times(defective, table):
    path, true = defective
    pt = l0b.pulse_timing(path)
    segs = pt.segments()
    assert [(s["code"], s["first"], s["count"]) for s in segs] == [
        (l0b.INCONSISTENT, 0, 3000), (l0b.EARLY, 12000, 1000), (l0b.LATE, 20000, 400)]
    assert np.max(np.abs(pt.corrected - true)) < 1e-9
    assert abs(pt.drift - 0.12e-6) < 1e-9
    assert pt.prf_commanded == pytest.approx(table.size / table.sum(), rel=1e-15)
    assert pt.prf_from_model == pytest.approx((true.size - 1) / (true[-1] - true[0]), rel=1e-12)
    assert pt.prf_from_tags / pt.prf_from_model - 1 > 0.02          # what ISCE3 would have focused with
    assert not pt.ok and "next pulse" in pt.summary()


def test_a_clean_l0b_is_consistent_with_any_table_phase(tmp_path, table):
    true = _true_times(5000, 50.0, table, phase=101)
    pt = l0b.pulse_timing(_write_l0b(tmp_path / "clean.h5", true, table))
    assert pt.ok and pt.segments() == [] and pt.table_phase == 101
    assert pt.prf_from_tags == pytest.approx(pt.prf_from_model, rel=1e-12)


def test_a_fixed_prf_l0b_without_a_table(tmp_path):
    t = 10.0 + np.arange(2000) / 1600.0
    t[700:900] -= 1 / 1600.0
    path = tmp_path / "fixed.h5"
    with h5py.File(path, "w") as f:
        tx = f.create_group("science/LSAR/RRSD/swaths/frequencyA/txH")
        tx["UTCtime"] = t
        tx["UTCtime"].attrs["units"] = np.bytes_(UNITS)
        tx["rangeLineIndex"] = np.arange(t.size, dtype=np.uint32)
        tx["nominalAcquisitionPRF"] = 1600.0
    segs = l0b.pulse_timing(path).segments()
    assert [(s["code"], s["first"], s["count"]) for s in segs] == [(l0b.EARLY, 700, 200)]


# -- the report on an RSLC -------------------------------------------------------------
def _report(problems, span=(90.0, 120.0), prf=1909.6, expected=1909.6):
    return l0b.TimingReport(granule="x", timings=[], problems=problems, prf_product=prf, prf_expected=expected,
                            prf_commanded=1909.6, span=span, lead=(2.0, 2.3), aperture=2.8)


def _problem(code=l0b.EARLY):
    return dict(kind=l0b.KIND_NAMES[code], code=code, count=100, start=100.0, end=110.0,
                full=(103.7, 110.6), transitions=[(98.6, 103.7), (110.6, 113.7)])


def test_report_verdicts_shift_and_transitions():
    rep = _report([_problem()])
    assert rep.verdict == "refocus"                                     # a transition inside the frame
    assert _report([_problem()], span=(104.0, 110.0)).verdict == "correctable"
    assert _report([_problem()], span=(200.0, 300.0)).verdict == "ok"
    assert _report([], prf=1909.6 * (1 + 2e-4)).verdict == "refocus"     # focused with a wrong PRF
    assert _report([_problem(l0b.INCONSISTENT)], span=(104.0, 110.0)).verdict == "refocus"
    shift = rep.azimuth_shift([95.0, 105.0, 115.0])
    assert shift[0] == 0 and shift[2] == 0 and shift[1] == pytest.approx(-1 / 1909.6)
    assert _report([_problem(l0b.LATE)]).azimuth_shift([105.0])[0] == pytest.approx(1 / 1909.6)
    assert list(rep.transition([100.0, 105.0, 111.0])) == [True, False, True]


def _write_rslc(path, grid, orbit, prf, l0b_name):
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"] = np.bytes_("Left")
        ident["orbitPassDirection"] = np.bytes_("Descending")
        ident["granuleId"] = np.bytes_("NISAR_L1_PR_RSLC_TEST")
        ident["listOfFrequencies"] = np.array([b"A"])
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(UNITS)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        fa = sw.create_group("frequencyA")
        fa["listOfPolarizations"] = np.array([b"HH"])
        fa["slantRange"] = grid.starting_range + np.arange(grid.width) * grid.range_pixel_spacing
        fa["processedCenterFrequency"] = radar.SPEED_OF_LIGHT / grid.wavelength
        fa["processedAzimuthBandwidth"] = 1262.0
        fa["nominalAcquisitionPRF"] = prf
        o = root.create_group("metadata/orbit")
        o["time"] = orbit.time
        o["time"].attrs["units"] = np.bytes_(UNITS)
        o["position"], o["velocity"] = orbit.position, orbit.velocity
        o["interpMethod"], o["orbitType"] = np.bytes_("Hermite"), np.bytes_("MOE")
        par = root.create_group("metadata/processingInformation/parameters")
        par["zeroDopplerTime"] = np.linspace(grid.sensing_start - 10, grid.sensing_stop + 10, 6)
        par["slantRange"] = np.linspace(grid.starting_range - 1000, grid.end_range + 1000, 5)
        par["frequencyA/dopplerCentroid"] = np.full((6, 5), 950.0)
        root["metadata/processingInformation/inputs/l0bGranules"] = np.array([np.bytes_(l0b_name + ".h5")])
    return path


def test_timing_report_places_a_slip_in_the_rslcs_time(tmp_path, table):
    orbit = rs.orbit()
    grid = rs.radar_grid(length=20 * 1520, width=400, sensing_start=rs.T0 + 60.0)
    true = _true_times(int(40 * 1909.6), rs.T0 + 50.0, table)
    utc = true.copy()
    a = int(np.searchsorted(true, rs.T0 + 66.0))
    utc[a:] = true[a - 1:-1]                                        # previous pulse's time to the end
    l0b_path = _write_l0b(tmp_path / "l0b.h5", utc, table)
    pt = l0b.pulse_timing(l0b_path)
    rslc = _write_rslc(tmp_path / "rslc.h5", grid, orbit, pt.prf_from_tags, "NISAR_L0_PR_RRSD_TEST")
    rep = l0b.timing_report(rslc, l0b=[l0b_path])
    assert 1.0 < rep.lead[0] < rep.lead[1] < 3.0 and 2.0 < rep.aperture < 4.0
    (p,) = rep.problems
    assert p["code"] == l0b.EARLY and p["start"] == pytest.approx(true[a], abs=1e-9)
    start_edge, _ = p["transitions"]
    assert start_edge[0] < true[a] + np.mean(rep.lead) < start_edge[1]
    # the last tag is a pulse early, so ISCE3's PRF is ~1 PRI / 40 s off: small, but not zero
    assert 0 < rep.prf_error < 1e-4 and rep.verdict == "refocus"
    inside = np.mean(p["full"]) if p["full"] else None
    if inside is not None:
        assert rep.azimuth_shift([inside])[0] == pytest.approx(-1 / pt.prf_commanded)
    assert rep.azimuth_shift([grid.sensing_start + 1.0])[0] == 0.0


@pytest.mark.skipif(not os.environ.get("NISAR_TEST_L0B_RSLC"),
                    reason="set NISAR_TEST_L0B_RSLC to an RSLC whose L0B (on ASF) has a slip")
def test_real_rslc_report_reads_its_l0b_remotely():
    rep = l0b.timing_report(os.environ["NISAR_TEST_L0B_RSLC"])
    assert rep.timings and rep.verdict in ("correctable", "refocus")
    assert any(p["code"] in (l0b.EARLY, l0b.LATE) for p in rep.problems)
