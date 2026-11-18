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


def _add_windows(path, changes, empty, samples=1000):
    """Sub-swath limits (the fill value on pulses with no echo) and data-window settings that
    update at the pulses in `changes`."""
    with h5py.File(path, "r+") as f:
        tx = f["science/LSAR/RRSD/swaths/frequencyA/txH"]
        n = tx["UTCtime"].shape[0]
        limits = np.tile(np.array([[10, samples - 10]], np.uint32), (n, 1))
        limits[empty] = np.iinfo(np.uint32).max
        tx["numberOfSubSwaths"] = 1
        tx["validSamplesSubSwath1"] = limits
        tx["slantRange"] = 880e3 + 3.12 * np.arange(samples)
        wd = np.full((n, 12), 4720, np.uint32)
        for c in changes:
            wd[c:] -= 5
        tx["rxH/RD"] = np.full((n, 12), 1403230, np.uint32)
        tx["rxH/WD"] = wd
        tx["rxH/WL"] = np.full((n, 12), 42650, np.uint32)
    return path


@pytest.fixture
def table():
    return np.random.default_rng(1).uniform(0.423e-3, 0.621e-3, 173)


@pytest.fixture
def defective(tmp_path, table):
    true = _true_times(30000, 1000.0, table)
    utc = true.copy()
    utc[:3000] = true[3000] - 0.6 * (true[3000] - true[:3000])     # squeezed
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


def test_a_tag_microseconds_off_does_not_pull_the_fitted_sequence(tmp_path, table):
    # as delivered: the echo before a slip tagged 4.5 us early, inside the tolerance, so it counts as consistent
    true = _true_times(6000, 1000.0, table)
    utc = true.copy()
    utc[4000:] = true[3999:-1]
    utc[3999] -= 4.5e-6
    pt = l0b.pulse_timing(_write_l0b(tmp_path / "l0b.h5", utc, table))
    assert pt.kind[3999] == l0b.CONSISTENT and pt.kind[4000] == l0b.EARLY
    assert np.max(np.abs(pt.model - true)) < 1e-9
    assert abs(pt.drift - 0.12e-6) < 1e-11


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


def test_masked_segments_are_those_whose_first_pulse_has_no_echo(tmp_path, table):
    utc = _true_times(20000, 1000.0, table)
    path = _add_windows(_write_l0b(tmp_path / "l0b.h5", utc, table), [6000, 12000], np.r_[0:500, 6003, 12000])
    first, later = l0b.masked_segments(path)
    assert first.leading and first.empty == 500 and (first.start, first.end) == (utc[0], utc[6000])
    assert first.first_valid == utc[500] and first.lost == pytest.approx(utc[6000] - utc[500])
    assert not later.leading and later.empty == 1 and later.first_valid == utc[12001] and later.end == utc[-1]
    clean = _add_windows(_write_l0b(tmp_path / "clean.h5", utc, table), [6000], np.array([], int))
    assert l0b.masked_segments(clean) == []


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


def test_the_report_restores_a_masked_datatake_start(tmp_path, table):
    orbit = rs.orbit()
    grid = rs.radar_grid(length=20 * 1520, width=400, sensing_start=rs.T0 + 60.0)
    utc = _true_times(int(40 * 1909.6), rs.T0 + 50.0, table)
    changes = [int(np.searchsorted(utc, rs.T0 + 50.0 + 10.0 * k)) for k in (1, 2, 3)]
    path = _add_windows(_write_l0b(tmp_path / "l0b.h5", utc, table), changes, np.arange(500))
    rslc = _write_rslc(tmp_path / "rslc.h5", grid, orbit, l0b.pulse_timing(path).prf_from_tags,
                       "NISAR_L0_PR_RRSD_TEST")
    rep = l0b.timing_report(rslc, l0b=[path])
    (m,) = rep.masked
    assert m["leading"] and m["reaches"] and m["empty"] == 500
    assert rep.masked_start == pytest.approx(utc[changes[0]] - utc[500])
    assert "repair_mask restores it" in rep.summary()
    # ISCE3's mask: full focus half an aperture after the second segment starts, later at far range
    t_full = utc[changes[0]] + rep.lead[0] + 0.5 * rep.aperture * np.linspace(0.85, 1.0, grid.width)
    first = np.ceil((t_full - grid.sensing_start) * grid.prf).astype(int)
    masked = np.zeros((grid.length, 2), np.int64)
    for i in range(grid.length):
        ok = np.flatnonzero(first <= i)
        if ok.size:
            masked[i] = (ok[0], ok[-1] + 1)
    (fixed,) = rep.repair_mask([masked])
    shift = int(round(rep.masked_start * grid.prf))
    fixed_first = np.array([np.flatnonzero((fixed[:, 0] <= j) & (j < fixed[:, 1]))[0] for j in range(grid.width)])
    np.testing.assert_array_equal(fixed_first, np.maximum(first - shift, 0))
    late = masked.copy()
    late[:first.max() + 3 * 1520] = 0                       # a mask that starts somewhere else
    with pytest.warns(UserWarning, match="left as it is"):
        np.testing.assert_array_equal(rep.repair_mask([late])[0], late)
    after = l0b.TimingReport(granule="x", timings=[], problems=[], prf_product=1909.6, prf_expected=1909.6,
                             prf_commanded=1909.6, span=(0.0, 10.0), lead=(2.0, 2.3), aperture=2.8,
                             masked=[dict(m, reaches=False)])
    assert after.masked_start == 0.0
    np.testing.assert_array_equal(after.repair_mask([masked])[0], masked)


@pytest.mark.skipif(not os.environ.get("NISAR_TEST_L0B_RSLC"),
                    reason="set NISAR_TEST_L0B_RSLC to an RSLC whose L0B (on ASF) has a slip")
def test_real_rslc_report_reads_its_l0b_remotely():
    rep = l0b.timing_report(os.environ["NISAR_TEST_L0B_RSLC"])
    assert rep.timings and rep.verdict in ("correctable", "refocus")
    assert any(p["code"] in (l0b.EARLY, l0b.LATE) for p in rep.problems)


def test_tags_that_never_follow_the_table_are_reported(tmp_path, table):
    true = _true_times(20000, 1000.0, table)
    rng = np.random.default_rng(4)
    steps = np.diff(true) * rng.uniform(0.0, 1.2, true.size - 1)        # every step wrong
    utc = np.concatenate([[true[0]], true[0] + np.cumsum(steps)])
    path = _write_l0b(tmp_path / "corrupt.h5", utc, table)
    with pytest.raises(l0b.UnanchoredTags, match="no run of the tags follows"):
        l0b.pulse_timing(path)


def test_transmit_gaps_give_the_table_phase_the_tags_lost(tmp_path, table):
    """A corrupt-tag L0B whose sub-swaths still record where later transmits blanked each window."""
    true = _true_times(4000, 1000.0, table, phase=108, drift=0.0)
    rng = np.random.default_rng(4)
    utc = np.concatenate([[true[0]], true[0] + np.cumsum(np.diff(true) * rng.uniform(0.0, 1.2, true.size - 1))])
    path = _write_l0b(tmp_path / "corrupt.h5", utc, table, first_line=4334)
    r0, dr, n_samples = 888836.24, 3.1228381041437387, 53067
    c = 299792458.0
    fs, tau0 = c / (2 * dr), 2 * r0 / c
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, 40))])
    sub = np.full((5, true.size, 2), np.iinfo(np.uint32).max, np.uint32)
    for kk in range(true.size):
        g = np.round((cum[kk + 108 + np.arange(1, 20)] - cum[kk + 108] - tau0) * fs - 14.5)
        g = g[(g > 600) & (g < n_samples - 2000)]
        edges = [509] + [v for e in g for v in (e, e + 1849)] + [n_samples]
        for i in range(len(edges) // 2):
            sub[i, kk] = edges[2 * i], edges[2 * i + 1]
    with h5py.File(path, "r+") as f:
        tx = f["science/LSAR/RRSD/swaths/frequencyA/txH"]
        tx["numberOfSubSwaths"] = 5
        for i in range(5):
            tx[f"validSamplesSubSwath{i + 1}"] = sub[i]
        tx["slantRange"] = r0 + dr * np.arange(n_samples)
    gp = l0b.tx_gap_phase(path)
    assert gp.phase == 108 and gp.match == 1.0 and gp.first_line == 4334
    pt = l0b.pulse_timing(path)                      # no run of tags anchors: the gaps' sequence instead
    off = pt.model - true                            # on the latest pair of tags: one constant, within a PRI
    assert not pt.anchored and pt.table_phase == 108 and np.ptp(off) < 1e-9 and abs(off[0]) < table.max()
    assert "anchored on the latest tags" in pt.summary()


def test_tags_behind_and_off_their_transmit_gaps_are_re_timed(tmp_path, table):
    """Tags right at first, then behind by seconds and 4 pulses off the transmits."""
    true = _true_times(4004, 1000.0, table, phase=0, drift=0.0)
    tags, true = true[:-4].copy(), true[4:]           # echo k carries pulse k - 4's time ...
    tags[:1500] = true[:1500]                        # ... except at first, where the tags are right,
    tags[1500:] -= 7.6                               # and everything after is seconds behind
    path = _write_l0b(tmp_path / "relabelled.h5", tags, table, first_line=4097)
    r0, dr, n_samples = 888836.24, 3.1228381041437387, 53067
    fs, tau0 = 299792458.0 / (2 * dr), 2 * r0 / 299792458.0
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, 40))])
    sub = np.full((5, tags.size, 2), np.iinfo(np.uint32).max, np.uint32)
    for kk in range(tags.size):
        g = np.round((cum[kk + 4 + np.arange(1, 20)] - cum[kk + 4] - tau0) * fs - 14.5)   # echo k is pulse k + 4
        g = g[(g > 600) & (g < n_samples - 2000)]
        edges = [509] + [v for e in g for v in (e, e + 1849)] + [n_samples]
        for i in range(len(edges) // 2):
            sub[i, kk] = edges[2 * i], edges[2 * i + 1]
    with h5py.File(path, "r+") as f:
        tx = f["science/LSAR/RRSD/swaths/frequencyA/txH"]
        tx["numberOfSubSwaths"] = 5
        for i in range(5):
            tx[f"validSamplesSubSwath{i + 1}"] = sub[i]
        tx["slantRange"] = r0 + dr * np.arange(n_samples)
    pt = l0b.pulse_timing(path)
    assert pt.table_phase == 4 and pt.tag_shift == 4 and np.max(np.abs(pt.corrected - true)) < 1e-9
    # the echo whose step is the jump itself fits nothing; the run after it follows the table, 4 pulses off
    assert [(s["code"], s["first"]) for s in pt.segments()] == [(l0b.INCONSISTENT, 1500), (l0b.RELABELLED, 1501)]
    assert "4 pulses off" in pt.summary() and " 7.60" in pt.summary()     # 7.6 s plus the 4 PRIs


def test_the_echoes_transmit_leakage_overrules_the_sub_swath_limits(tmp_path, table):
    """Sub-swath limits that follow the table at one phase, and the echoes' leakage at another."""
    true = _true_times(1200, 1000.0, table, phase=64, drift=0.0)
    path = _write_l0b(tmp_path / "leakage.h5", true, table, first_line=4334)
    r0, dr, n_samples = 888836.24, 3.1228381041437387, 20000
    fs, tau0 = 299792458.0 / (2 * dr), 2 * r0 / 299792458.0
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, 20))])
    rng = np.random.default_rng(5)
    echoes = (rng.normal(size=(true.size, n_samples)) + 1j * rng.normal(size=(true.size, n_samples))).astype(np.complex64)
    sub = np.full((5, true.size, 2), np.iinfo(np.uint32).max, np.uint32)
    for kk in range(true.size):
        leak = np.round((cum[kk + 64 + np.arange(1, 20)] - cum[kk + 64] - tau0) * fs - 250).astype(int)
        for s0 in leak[(leak > 0) & (leak < n_samples - 1849)]:
            echoes[kk, s0:s0 + 1849] *= 7.0
        g = np.round((cum[kk + 108 + np.arange(1, 20)] - cum[kk + 108] - tau0) * fs - 14.5)   # bookkeeping
        g = g[(g > 600) & (g < n_samples - 2000)]
        edges = [509] + [v for e in g for v in (e, e + 1849)] + [n_samples]
        for i in range(min(5, len(edges) // 2)):
            sub[i, kk] = edges[2 * i], edges[2 * i + 1]
    with h5py.File(path, "r+") as f:
        tx = f["science/LSAR/RRSD/swaths/frequencyA/txH"]
        tx["numberOfSubSwaths"] = 5
        for i in range(5):
            tx[f"validSamplesSubSwath{i + 1}"] = sub[i]
        tx["slantRange"] = r0 + dr * np.arange(n_samples)
        tx["rxH/HH"] = echoes
    assert l0b.tx_gap_phase(path).phase == 108
    lp = l0b.tx_leakage_phase(path)
    assert lp.phase == 64 and lp.clear and abs(lp.offset + 250) <= 40
    pt = l0b.pulse_timing(path)
    assert pt.phase_source == "leakage" and pt.table_phase == 64 and np.max(np.abs(pt.corrected - true)) < 1e-9


def _leakage_l0b(path, table, tx="txH", rx="rxH", leak_phase=64, gap_phase=108, empty=False):
    """An L0B whose echoes leak at `leak_phase` (or are empty) and whose sub-swath limits follow `gap_phase`, with
    the range-line tables a real receive group holds beside its echoes."""
    true = _true_times(1200, 1000.0, table, phase=leak_phase, drift=0.0)
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/granuleId"] = np.bytes_("NISAR_L0_PR_RRSD_TEST")
        g = f.create_group(f"science/LSAR/RRSD/swaths/frequencyA/{tx}")
        g["UTCtime"] = true
        g["UTCtime"].attrs["units"] = np.bytes_(UNITS)
        g["rangeLineIndex"] = (4334 + np.arange(true.size)).astype(np.uint32)
        g["PRISequence"] = table
        g["calType"] = np.zeros(true.size, np.uint8)
    r0, dr, n_samples = 888836.24, 3.1228381041437387, 20000
    fs, tau0 = 299792458.0 / (2 * dr), 2 * r0 / 299792458.0
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, 20))])
    rng = np.random.default_rng(5)
    echoes = (rng.normal(size=(true.size, n_samples)) + 1j * rng.normal(size=(true.size, n_samples))).astype(np.complex64)
    sub = np.full((5, true.size, 2), np.iinfo(np.uint32).max, np.uint32)
    for kk in range(true.size):
        leak = np.round((cum[kk + leak_phase + np.arange(1, 20)] - cum[kk + leak_phase] - tau0) * fs - 250).astype(int)
        for s0 in leak[(leak > 0) & (leak < n_samples - 1849)]:
            echoes[kk, s0:s0 + 1849] *= 7.0
        gaps = np.round((cum[kk + gap_phase + np.arange(1, 20)] - cum[kk + gap_phase] - tau0) * fs - 14.5)
        gaps = gaps[(gaps > 600) & (gaps < n_samples - 2000)]
        edges = [509] + [v for e in gaps for v in (e, e + 1849)] + [n_samples]
        for i in range(min(5, len(edges) // 2)):
            sub[i, kk] = edges[2 * i], edges[2 * i + 1]
    with h5py.File(path, "r+") as f:
        g = f[f"science/LSAR/RRSD/swaths/frequencyA/{tx}"]
        g["numberOfSubSwaths"] = 5
        for i in range(5):
            g[f"validSamplesSubSwath{i + 1}"] = sub[i]
        g["slantRange"] = r0 + dr * np.arange(n_samples)
        g[f"{rx}/{tx[2:]}{rx[2:]}"] = np.zeros_like(echoes) if empty else echoes
        for table_name in ("RD", "WD", "WL"):                # these sort before VH, VV and RH
            g[f"{rx}/{table_name}"] = np.full((true.size, 12), 1403230, np.uint32)
    return path, true


@pytest.mark.parametrize("tx, rx", [("txH", "rxH"), ("txV", "rxH"), ("txV", "rxV"), ("txR", "rxH")])
def test_the_leakage_is_read_from_the_transmits_own_echoes(tmp_path, table, tx, rx):
    path, true = _leakage_l0b(tmp_path / "leak.h5", table, tx=tx, rx=rx)
    lp = l0b.tx_leakage_phase(path)
    assert lp.phase == 64 and lp.clear
    pt = l0b.pulse_timing(path)
    assert pt.table_phase == 64 and np.max(np.abs(pt.corrected - true)) < 1e-9


def test_empty_echoes_leave_the_phase_to_the_sub_swath_limits(tmp_path, table):
    """No leakage to read is no evidence: the limits, which follow the table, decide."""
    path, true = _leakage_l0b(tmp_path / "empty.h5", table, leak_phase=64, gap_phase=64, empty=True)
    assert not l0b.tx_leakage_phase(path).clear
    pt = l0b.pulse_timing(path)
    assert pt.phase_source == "sub-swaths" and pt.table_phase == 64 and np.max(np.abs(pt.corrected - true)) < 1e-9


def test_a_slip_through_either_end_of_the_l0b_is_still_a_slip(tmp_path, table):
    """An L0B starts and ends mid-datatake, so a run of mislabelled tags can begin at its first echo or end at its last."""
    true = _true_times(30000, 1000.0, table)
    utc = true.copy()
    utc[1:3000] = true[:2999]                                    # from the start: the previous pulse's time ...
    utc[0] = true[0] - table[(37 - 1) % table.size] * (1 + 0.12e-6)   # ... the first echo's too
    utc[29600:-1] = true[29601:]                                  # to the end: the next pulse's time ...
    utc[-1] = true[-1] + table[(37 + true.size - 1) % table.size] * (1 + 0.12e-6)
    pt = l0b.pulse_timing(_write_l0b(tmp_path / "ends.h5", utc, table), echoes=False)
    assert [(s["code"], s["first"], s["count"]) for s in pt.segments()] == [(l0b.EARLY, 0, 3000), (l0b.LATE, 29600, 400)]
    assert np.max(np.abs(pt.corrected - true)) < 1e-9


def test_the_aperture_goes_with_the_platform_and_ground_speeds(tmp_path):
    """B lambda R / (2 V_s V_g), the azimuth FM rate's, with V_g the ground track's: not V_s squared."""
    from nisar_tools import roff
    from nisar_tools.rslc import RSLC

    class _Flat:
        def heights(self, lon, lat):
            return np.zeros(np.shape(lon))

    orbit = rs.orbit()
    grid = rs.radar_grid(length=2000, width=400, sensing_start=rs.T0 + 60.0)
    path = _write_rslc(tmp_path / "rslc.h5", grid, orbit, 1909.6, "NISAR_L0_PR_RRSD_TEST")
    with RSLC(path) as r:
        _, aperture = l0b._beam_geometry(r, "A")
        g = r.radar_grid("A")
        far = g.starting_range + 0.98 * (g.width - 1) * g.range_pixel_spacing
        _, vel = r.orbit.interpolate(np.array([g.sensing_mid]))
        vs = float(np.linalg.norm(vel))
        vg = float(roff.ground_track_velocity(r, _Flat(), [g.sensing_mid], [far])[0])
    assert vg < 0.95 * vs
    assert aperture == pytest.approx(1262.0 * g.wavelength * far / (2 * vs * vg), rel=1e-3)


def _gap_limits(path, n_echo, table, phase):
    """Sub-swath limits whose transmit gaps follow the table at `phase`."""
    r0, dr, n_samples = 888836.24, 3.1228381041437387, 53067
    fs, tau0 = 299792458.0 / (2 * dr), 2 * r0 / 299792458.0
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, n_echo // table.size + 40))])
    sub = np.full((5, n_echo, 2), np.iinfo(np.uint32).max, np.uint32)
    for kk in range(n_echo):
        g = np.round((cum[kk + phase + np.arange(1, 20)] - cum[kk + phase] - tau0) * fs - 14.5)
        g = g[(g > 600) & (g < n_samples - 2000)]
        edges = [509] + [v for e in g for v in (e, e + 1849)] + [n_samples]
        for i in range(len(edges) // 2):
            sub[i, kk] = edges[2 * i], edges[2 * i + 1]
    with h5py.File(path, "r+") as f:
        tx = f["science/LSAR/RRSD/swaths/frequencyA/txH"]
        tx["numberOfSubSwaths"] = 5
        for i in range(5):
            tx[f"validSamplesSubSwath{i + 1}"] = sub[i]
        tx["slantRange"] = r0 + dr * np.arange(n_samples)
    return path


def test_a_slip_over_most_of_the_l0b_is_not_a_tag_shift(tmp_path, table):
    """Most echoes tagged with the previous pulse's time: they are early, not relabelled, and nothing is off."""
    true = _true_times(8000, 1000.0, table, phase=37, drift=0.0)
    utc = true.copy()
    utc[3000:] = true[2999:-1]
    pt = l0b.pulse_timing(_gap_limits(_write_l0b(tmp_path / "majority.h5", utc, table, first_line=0), 8000, table, 37))
    assert pt.phase_source == "sub-swaths" and pt.table_phase == 37
    assert [(s["code"], s["first"], s["count"]) for s in pt.segments()] == [(l0b.EARLY, 3000, 5000)]
    assert pt.tag_shift == 0 and "pulses off the radar's sequence" not in pt.summary()
    assert np.max(np.abs(pt.corrected - true)) < 1e-9


def test_a_nan_tag_is_one_inconsistent_echo_and_one_echo_is_unanchored(tmp_path, table):
    true = _true_times(6000, 1000.0, table, drift=0.0)
    utc = true.copy()
    utc[3000] = np.nan
    pt = l0b.pulse_timing(_write_l0b(tmp_path / "nan.h5", utc, table))
    assert pt.anchored and [(s["code"], s["first"], s["count"]) for s in pt.segments()] == [(l0b.INCONSISTENT, 3000, 1)]
    assert np.max(np.abs(pt.corrected - true)) < 1e-9
    with pytest.raises(l0b.UnanchoredTags, match="1 finite tag"):
        l0b.pulse_timing(_write_l0b(tmp_path / "one.h5", true[:1], table))
