"""Refocusing an RSLC from its L0B: the inputs `focus_inputs` prepares, and `run_focus`'s call of ISCE3.

ISCE3 is not imported anywhere: a stand-in interpreter (a shell script) takes
its place, so these run without it.
"""

import os
import stat
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import yaml

import radar_synthetic as rs
from nisar_tools import focus, radar

UNITS = radar.epoch_units(rs.EPOCH)
L0B = "NISAR_L0_PR_RRSD_028_104_D_156S_20260820T101028_20260820T101100_P05023_F_J_001.h5"
GRANULE = "NISAR_L1_PR_RSLC_028_104_D_072_4005_DHDH_A_20260820T101036_20260820T101101_P05023_N_P_J_001"
CAL = ["NISAR_LSAR_EXT_CAL_20250730T000000_04.yaml", "NISAR_LSAR_INT_CAL_20250730T000000_02.h5",
       "NISAR_ANTPAT_20231129T093104_01.h5"]
PRODUCTION = {
    "input_file_group": {"input_file_path": ["/sds/" + L0B], "qa_input_file": "/sds/qa.h5"},
    "dynamic_ancillary_file_group": {"dem_file": "/sds/dem.vrt", "orbit": "/sds/moe.xml", "pointing": "/sds/nrp.xml",
                                     "external_calibration": "/sds/" + CAL[0], "corner_reflector_file": "/sds/cr.csv"},
    "product_path_group": {"product_path": "/sds/out", "scratch_path": "/sds/tmp", "sas_output_file": "/sds/x.h5",
                           "sas_config_file": "/sds/x.yaml", "qa_output_dir": "/sds/qa", "product_counter": 1},
    "primary_executable": {"product_type": "RSLC", "processing_type": "PR",
                           "partial_granule_id": "NISAR_L1_PR_RSLC_028_104_D_072_{MODE}_{POLE}_A_{StartDateTime}"},
    "worker": {"gpu_enabled": True, "gpu_id": 0},
    "processing": {"output_grid": {"output_prf": 1520}, "range_window": {"kind": "Kaiser", "shape": 1.6}},
    "qa": {"workflows": {"validate": True}},
}


def _table():
    return np.random.default_rng(1).uniform(0.423e-3, 0.621e-3, 173)


def _true_utc():
    table = _table()
    k = np.arange(6000)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, k.size // table.size + 3))])
    return 36640.0 + cum[k]


def _write_l0b(path, slip_at=None):
    table = _table()
    k = np.arange(6000)
    utc = _true_utc()
    if slip_at is not None:
        utc[slip_at:] = utc[slip_at - 1:-1].copy()         # tagged with the previous pulse's time
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/granuleId"] = np.bytes_(path.stem)
        tx = f.create_group("science/LSAR/RRSD/swaths/frequencyA/txH")
        tx["UTCtime"] = utc
        tx["UTCtime"].attrs["units"] = np.bytes_(UNITS)
        tx["rangeLineIndex"] = (5000 + k).astype(np.uint32)
        tx["PRISequence"] = table
        tx["calType"] = np.zeros(utc.size, np.uint8)
    return path


@pytest.fixture
def rslc(tmp_path):
    orbit = rs.orbit()
    grid = rs.radar_grid(length=50, width=40)
    path = tmp_path / f"{GRANULE}.h5"
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"], ident["orbitPassDirection"] = np.bytes_("Left"), np.bytes_("Descending")
        ident["listOfFrequencies"] = np.array([b"A"])
        ident["granuleId"] = np.bytes_(GRANULE)
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(UNITS)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        sw.create_group("frequencyA")["slantRange"] = grid.starting_range + np.arange(grid.width) * 3.0
        o = root.create_group("metadata/orbit")
        o["time"] = orbit.time
        o["time"].attrs["units"] = np.bytes_(UNITS)
        o["position"], o["velocity"] = orbit.position, orbit.velocity
        o["orbitType"] = np.bytes_("MOE")
        a = root.create_group("metadata/attitude")
        a["time"] = orbit.time[:5]
        a["time"].attrs["units"] = np.bytes_(UNITS)
        q = np.random.default_rng(2).normal(size=(5, 4))
        a["quaternions"] = q / np.linalg.norm(q, axis=1, keepdims=True)
        a["attitudeType"] = np.bytes_("NRP")
        info = root.create_group("metadata/processingInformation")
        info["inputs/l0bGranules"] = np.array([np.bytes_("/sds/" + L0B)])
        info["inputs/auxcalFiles"] = np.array([np.bytes_("/sds/" + n) for n in CAL])
        info["parameters/runConfigurationContents"] = np.bytes_(yaml.safe_dump(PRODUCTION))
    return path


@pytest.fixture
def inputs(tmp_path):
    l0b_dir, anc = tmp_path / "l0b", tmp_path / "anc"
    l0b_dir.mkdir()
    anc.mkdir()
    _write_l0b(l0b_dir / L0B, slip_at=4000)
    for name in CAL:
        (anc / name).write_text("")
    return l0b_dir, anc


def test_runconfig_is_production_with_local_inputs(rslc, inputs, tmp_path):
    l0b_dir, anc = inputs
    window = {"start_time": "2026-08-20T10:10:52", "end_range": 1040000}
    path = focus.focus_inputs(rslc, tmp_path / "work", l0b_dir=l0b_dir, ancillary_dir=anc,
                              dem=tmp_path / "dem.tif", download=False, window=window)
    groups = yaml.safe_load(path.read_text())["runconfig"]["groups"]
    work = tmp_path / "work"
    assert groups["input_file_group"] == {"input_file_path": [str(l0b_dir / L0B)]}
    anc_group = groups["dynamic_ancillary_file_group"]
    assert anc_group["dem_file"] == str(tmp_path / "dem.tif") and anc_group["corner_reflector_file"] is None
    assert anc_group["orbit"] == str(work / "orbit.xml") and anc_group["pointing"] == str(work / "attitude.xml")
    assert [anc_group[k] for k in ("external_calibration", "internal_calibration", "antenna_pattern")] == \
        [str(anc / n) for n in CAL]
    out = groups["product_path_group"]
    assert out["sas_output_file"] == str(work / "output" / f"{GRANULE.replace('_PR_', '_OD_')}.h5")
    assert out["scratch_path"] == str(work / "scratch") and (work / "scratch").is_dir()
    assert groups["worker"] == {"gpu_enabled": False} and "qa" not in groups
    assert groups["primary_executable"]["processing_type"] == "OD"
    assert groups["primary_executable"]["partial_granule_id"].startswith("NISAR_L1_OD_RSLC_028_104_D_072_")
    assert groups["processing"]["output_grid"] == {"output_prf": 1520, "start_time": "2026-08-20T10:10:52",
                                                   "end_range": 1040000.0}
    assert groups["processing"]["range_window"] == PRODUCTION["processing"]["range_window"]
    timing = (work / "pulse_timing.txt").read_text()
    assert "2000 echoes tagged with the previous" in timing
    # the repaired times the run hands to ISCE3: the slipped echoes back on their pulses
    repaired = np.load(work / "pulse_times.npz")
    assert list(repaired) == [f"{L0B}__A__txH"]
    np.testing.assert_allclose(repaired[f"{L0B}__A__txH"], _true_utc(), rtol=0, atol=1e-9)


def test_unknown_window_keys_are_refused(rslc, inputs, tmp_path):
    l0b_dir, anc = inputs
    with pytest.raises(ValueError, match="start_line"):
        focus.focus_inputs(rslc, tmp_path / "work", l0b_dir=l0b_dir, ancillary_dir=anc, dem="dem.tif",
                           download=False, window={"start_line": 0})


def test_missing_l0b_without_download(rslc, tmp_path):
    with pytest.raises(FileNotFoundError, match=L0B):
        focus.focus_inputs(rslc, tmp_path / "work", dem="dem.tif", download=False)


def test_orbit_and_attitude_xml(rslc, tmp_path):
    from nisar_tools.rslc import RSLC

    with RSLC(rslc) as r:
        focus.write_orbit_xml(r, tmp_path / "orbit.xml")
        focus.write_attitude_xml(r, tmp_path / "attitude.xml")
    orbit = rs.orbit()
    vectors = ET.parse(tmp_path / "orbit.xml").getroot().find("orbitStateVectorList")
    assert int(vectors.get("count")) == orbit.time.size == len(vectors)
    first = vectors[0]
    assert np.datetime64(first.find("utc").text) == rs.EPOCH + np.timedelta64(int(round(orbit.time[0] * 1e9)), "ns")
    assert float(first.find("x").text) == orbit.position[0, 0] and float(first.find("vz").text) == orbit.velocity[0, 2]
    pointing = ET.parse(tmp_path / "attitude.xml").getroot()
    assert pointing.find("productInformation/fileType").text == "NRP"
    states = pointing.find("LSAR/radarPointingStateVectorList")
    assert len(states) == 5 and states[2].find("q3") is not None


def _fake_isce3(tmp_path, fails=False):
    """An interpreter stand-in: records its arguments (NUL-separated) and environment, writes nothing."""
    script = tmp_path / "python"
    script.write_text("#!/bin/sh\n"
                      f'printf "%s\\0" "$@" > "{tmp_path}/args.bin"\n'
                      f'echo "$OMP_NUM_THREADS $KMP_DUPLICATE_LIB_OK" > "{tmp_path}/env.txt"\n'
                      f"exit {3 if fails else 0}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _args(tmp_path):
    return (tmp_path / "args.bin").read_bytes().decode().split("\0")[:-1]


def _runconfig(tmp_path, times=True):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"runconfig": {"groups": {
        "product_path_group": {"sas_output_file": str(tmp_path / "out.h5")}}}}))
    if times:
        np.savez(tmp_path / "pulse_times.npz", x=np.zeros(3))
    return path


def test_run_focus_hands_the_repaired_times_to_isce3(tmp_path):
    out = focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path), threads=6, log=lambda *a: None)
    assert out == tmp_path / "out.h5"
    assert _args(tmp_path) == ["-c", focus._FOCUS_WRAPPER, "-", str(tmp_path / "pulse_times.npz"), "develop",
                               str(tmp_path / "run.yaml")]
    assert (tmp_path / "env.txt").read_text().split() == ["6", "TRUE"]
    assert (tmp_path / "focus.log").exists()


def test_run_focus_selects_polarizations_and_can_keep_the_tags(tmp_path):
    focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path), polarizations=["HH"], repair=False,
                    log=lambda *a: None)
    assert _args(tmp_path)[2:] == ["HH", "-", "-", str(tmp_path / "run.yaml")]


def test_run_focus_needs_an_interpreter_and_the_times(tmp_path, monkeypatch):
    monkeypatch.delenv(focus.ISCE3_PYTHON_ENV, raising=False)
    with pytest.raises(RuntimeError, match="pass python="):
        focus.run_focus(_runconfig(tmp_path), log=lambda *a: None)
    monkeypatch.setenv(focus.ISCE3_PYTHON_ENV, _fake_isce3(tmp_path))
    (tmp_path / "pulse_times.npz").unlink()
    with pytest.raises(FileNotFoundError, match="pulse times"):
        focus.run_focus(_runconfig(tmp_path, times=False), log=lambda *a: None)


def test_run_focus_reports_a_failed_run(tmp_path):
    with pytest.raises(RuntimeError, match="exit 3"):
        focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path, fails=True), log=lambda *a: None)


def _fake_nisar(monkeypatch, tags):
    """Stand-ins for the ISCE3 modules the wrapper touches: the focus workflow and the L0B reader."""
    import datetime
    import sys
    import types

    class Channel:
        def __init__(self, freq_id, pol):
            self.freq_id, self.pol = freq_id, pol

    class RawBase:
        frequencies = ["A", "B"]
        polarizations = {"A": ["HH", "HV"], "B": ["HH"]}

        def __init__(self, filename):
            self.filename = filename

        def getPulseTimes(self, frequency=None, tx=None, epoch=None):
            file_epoch = datetime.datetime(2026, 1, 1)
            t = tags.copy()
            if epoch is None:
                return file_epoch, t
            return epoch, t + (file_epoch - epoch).total_seconds()

    class Raw(RawBase):
        pass

    seen = {}
    focus_mod = types.ModuleType("nisar.workflows.focus")
    focus_mod.get_common_mode = lambda raws: set(Channel(f, p) for f in "AB" for p in ("HH", "HV"))
    focus_mod.main = lambda argv: seen.update(argv=list(sys.argv), mode=focus_mod.get_common_mode([]))
    raw_mod = types.ModuleType("nisar.products.readers.Raw")
    raw_mod.Raw = Raw
    for name, module in (("nisar", types.ModuleType("nisar")), ("nisar.workflows", types.ModuleType("nisar.workflows")),
                         ("nisar.workflows.focus", focus_mod), ("nisar.products", types.ModuleType("nisar.products")),
                         ("nisar.products.readers", types.ModuleType("nisar.products.readers")),
                         ("nisar.products.readers.Raw", raw_mod)):
        monkeypatch.setitem(sys.modules, name, module)
    return seen, Raw


def test_the_wrapper_keeps_only_the_named_channels(monkeypatch):
    import sys

    seen, _ = _fake_nisar(monkeypatch, np.zeros(3))
    monkeypatch.setattr(sys, "argv", ["-c", "HH", "-", "-", "run.yaml"])
    exec(focus._FOCUS_WRAPPER, {})
    assert seen["argv"] == ["-c", "run.yaml"]
    assert sorted((c.freq_id, c.pol) for c in seen["mode"]) == [("A", "HH"), ("B", "HH")]


def test_the_wrapper_hands_the_reader_the_repaired_times(monkeypatch, tmp_path):
    import datetime
    import sys

    tags, fixed = np.array([10.0, 10.5, 10.5]), np.array([10.0, 10.5, 11.0])
    np.savez(tmp_path / "times.npz", **{"l0b.h5__A__txH": fixed})
    seen, Raw = _fake_nisar(monkeypatch, tags)
    monkeypatch.setattr(sys, "argv", ["-c", "-", str(tmp_path / "times.npz"), "-", "run.yaml"])
    exec(focus._FOCUS_WRAPPER, {})
    assert seen["argv"] == ["-c", "run.yaml"]
    np.testing.assert_array_equal(Raw("/data/l0b.h5").getPulseTimes("A", "H")[1], fixed)
    np.testing.assert_array_equal(Raw("/data/l0b.h5").getPulseTimes()[1], fixed)          # the defaults
    epoch, t = Raw("/data/l0b.h5").getPulseTimes("A", "H", epoch=datetime.datetime(2025, 12, 31, 23, 59))
    np.testing.assert_array_equal(t, fixed + 60.0)
    np.testing.assert_array_equal(Raw("/data/other.h5").getPulseTimes("A", "H")[1], tags)   # not repaired


# -- the focusing fixes of ISCE3's develop branch, patched in at run time -------------------------
_BEAMFORMER = """
import numpy as np
from scipy.interpolate import interp1d


def get_calib_range_line_idx(cal_path_mask):
    m = np.asarray(cal_path_mask)                     # 0 HPA, 1 LNA, 2 BYPASS; LNA and BYPASS lines are noise-only
    return np.flatnonzero(m == 0), np.flatnonzero(m == 2), np.flatnonzero(m == 1), np.flatnonzero(m > 0)


def compute_transmit_pattern_weights(tx_trm_info, norm=False):
    hcal_lines_idx, bcal_lines_idx, _, noise_lines_idx = get_calib_range_line_idx(tx_trm_info.cal_path_mask)
    active_tx_idx = np.asarray(tx_trm_info.channels) - 1
    tx_weights = tx_trm_info.correlator_tap2[:, active_tx_idx].astype(complex)
    if bcal_lines_idx.size:
        for line_start, line_stop in zip(bcal_lines_idx[:-1], bcal_lines_idx[1:]):
            bcal_rel = tx_weights[line_start] / tx_weights[line_start, 0]
            tx_weights[line_start:line_stop] /= bcal_rel
        bcal_rel = tx_weights[bcal_lines_idx[-1]] / tx_weights[bcal_lines_idx[-1], 0]
        tx_weights[bcal_lines_idx[-1]:] /= bcal_rel
    nearest = interp1d(hcal_lines_idx, tx_weights[hcal_lines_idx], kind="nearest", fill_value="extrapolate",
                       assume_sorted=True, axis=0)
    tx_weights[noise_lines_idx] = nearest(noise_lines_idx)
    if norm:
        norm_vals = np.linalg.norm(tx_weights, axis=1)
        norm_vals[np.isclose(norm_vals, 0)] = 1
        tx_weights /= norm_vals[:, None]
    if tx_trm_info.tx_phase is None:
        return tx_weights
    return abs(tx_weights) * np.exp(1j * tx_trm_info.tx_phase[:, active_tx_idx])
"""

_VALID_REGIONS = """
import numpy as np


def rasterize_subswath_polygons(pieces, threshold=2):
    swaths = np.zeros((1, len(pieces), 2), dtype=np.uint32)
    later = np.zeros((1, len(pieces), 2), dtype=np.uint32)
    for itime, (first, second) in enumerate(pieces):
        swaths[0, itime], later[0, itime] = first, second
        i0, i1 = swaths[0, itime]
        j0, j1 = later[0, itime]
        if abs(int(j0) - i1) <= threshold:
            swaths[0, itime] = (i0, j1)
        elif abs(int(i0) - j1) <= threshold:
            swaths[0, itime] = (j0, i1)
        elif (j1 - j0) > (i1 - i0):
            swaths[0, itime] = (j0, j1)
    return swaths


def get_focused_sub_swaths(pieces):
    return rasterize_subswath_polygons(pieces)
"""

_RAW = """
import numpy as np


def get_dwp_change_indices(rd, wd, wl):
    start = np.min(rd + wd, axis=1).astype(np.int64)
    end = np.max(rd + wd + wl, axis=1).astype(np.int64)
    return np.where(np.any(np.diff(np.vstack((start, end)), axis=1) != 0, axis=0))[0] + 1


class RawBase:
    polarizations = {"A": ["HH"]}

    def __init__(self, limits, wd, times):
        self.limits, self.wd, self.times = limits, wd, times

    def getSubSwaths(self, frequency="A", tx="H"):
        return self.limits

    def getRdWdWl(self, frequency="A", polarization=None):
        return np.zeros_like(self.wd), self.wd, np.full_like(self.wd, 100)

    def getSubSwathBboxes(self, frequency, polarization=None, epoch=None, num_ignore=0):
        changes = get_dwp_change_indices(*self.getRdWdWl(frequency, polarization))
        breaks = np.hstack(([0], changes, [self.times.size - 1]))
        out = []
        for a, b in zip(breaks[:-1], breaks[1:]):
            j0, j1 = self.limits[0, a]
            if j1 > j0:
                out.append([(self.times[a], self.times[b], int(j0), int(j1))])
        return out


class Raw(RawBase):
    pass
"""


def _fake_isce3_modules(monkeypatch, tmp_path):
    """Stand-ins, in files (the fixes read their source), for the modules the develop fixes patch."""
    import importlib.util
    import sys
    import types

    def load(name, text):
        path = tmp_path / (name.replace(".", "_") + ".py")
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    seen = {}
    focus_mod = types.ModuleType("nisar.workflows.focus")
    focus_mod.get_common_mode = lambda raws: []
    focus_mod.main = lambda argv: seen.update(argv=list(sys.argv))
    for name in ("nisar", "nisar.workflows", "nisar.products", "nisar.products.readers", "nisar.antenna",
                 "isce3", "isce3.focus"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nisar.workflows.focus", focus_mod)
    raw = load("nisar.products.readers.Raw", _RAW)
    beamformer = load("nisar.antenna.beamformer", _BEAMFORMER)
    valid_regions = load("isce3.focus.valid_regions", _VALID_REGIONS)
    monkeypatch.setattr(sys, "argv", ["-c", "-", "-", "develop", "run.yaml"])
    return seen, raw, beamformer, valid_regions


def _tx_weights_fixed(info, norm, beamformer, original):
    """develop's compute_transmit_pattern_weights: the lines before the first BCAL line normalised by it too
    (here by dividing the correlator values of those lines first, which the rest of the function then keeps)."""
    import types

    _, bcal, _, _ = beamformer.get_calib_range_line_idx(info.cal_path_mask)
    tap = info.correlator_tap2.copy()
    active = np.asarray(info.channels) - 1
    tap[:bcal[0], active] /= tap[bcal[0], active] / tap[bcal[0], active[0]]
    return original(types.SimpleNamespace(**dict(vars(info), correlator_tap2=tap)), norm)


def test_the_tx_weights_before_the_first_bcal_line_are_normalised(monkeypatch, tmp_path):
    import types

    _, _, beamformer, _ = _fake_isce3_modules(monkeypatch, tmp_path)
    original = beamformer.compute_transmit_pattern_weights
    exec(focus._FOCUS_WRAPPER, {})
    patched = beamformer.compute_transmit_pattern_weights
    assert patched is not original
    rng = np.random.default_rng(3)
    mask = np.zeros(400, int)
    mask[rng.choice(400, 60, replace=False)] = 1                        # LNA lines
    mask[[48, 49, 52]] = 1                                              # beside the first BCAL line
    mask[50::100] = 2                                                   # BYPASS (BCAL), the first after 50 lines
    tap = (rng.normal(1, 0.3, (400, 8)) + 1j * rng.normal(0, 0.3, (400, 8))) * np.exp(1j * rng.uniform(-3, 3, 8))
    for phase in (None, rng.uniform(-3, 3, (400, 8))):
        info = types.SimpleNamespace(cal_path_mask=mask, channels=[1, 2, 3, 4, 6, 7], correlator_tap2=tap,
                                     tx_phase=phase)
        for norm in (False, True):
            got = patched(info, norm)
            want = _tx_weights_fixed(info, norm, beamformer, original)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
            assert not np.allclose(original(info, norm)[:50], want[:50])


def test_overlapping_sub_swath_pieces_are_joined(monkeypatch, tmp_path):
    _, _, _, valid_regions = _fake_isce3_modules(monkeypatch, tmp_path)
    pieces = [((10, 100), (99, 300)), ((10, 100), (102, 300)), ((10, 100), (150, 160))]
    assert valid_regions.get_focused_sub_swaths(pieces)[0, 0].tolist() == [99, 300]   # NumPy 2: the bigger wins
    exec(focus._FOCUS_WRAPPER, {})
    out = valid_regions.get_focused_sub_swaths(pieces)
    assert out.dtype == np.uint32 and out[0].tolist() == [[10, 300], [10, 300], [10, 100]]
    assert valid_regions.np is np


def test_a_segment_starting_without_echo_keeps_its_valid_region(monkeypatch, tmp_path):
    seen, raw, _, _ = _fake_isce3_modules(monkeypatch, tmp_path)
    times = 100.0 + np.arange(900) / 1900.0
    limits = np.tile(np.array([[[5, 95]]], np.uint32), (1, 900, 1))
    limits[0, :50] = np.iinfo(np.uint32).max                            # the datatake's empty first pulses
    wd = np.full((900, 4), 4720, np.int64)
    wd[300:] -= 5
    wd[600:] -= 5
    reader = raw.Raw(limits, wd, times)
    assert reader.getSubSwathBboxes("A")[0][0][0] == times[300]          # v0.25.16 loses the first segment
    exec(focus._FOCUS_WRAPPER, {})
    assert seen["argv"] == ["-c", "run.yaml"]
    boxes = reader.getSubSwathBboxes("A")
    assert [b[0][:2] for b in boxes] == [(times[50], times[300]), (times[300], times[600]), (times[600], times[899])]
    assert raw.get_dwp_change_indices(*reader.getRdWdWl()).tolist() == [300, 600]   # put back after the call
    whole = raw.Raw(np.tile(np.array([[[5, 95]]], np.uint32), (1, 900, 1)), wd, times)
    assert [b[0][0] for b in whole.getSubSwathBboxes("A")] == [times[0], times[300], times[600]]


def test_refocused_name():
    assert focus.refocused_name(GRANULE).startswith("NISAR_L1_OD_RSLC_028_104_D")
    assert "_OD_RSLC_" in focus.refocused_name(GRANULE.replace("_PR_", "_UR_"))


@pytest.mark.skipif(not os.environ.get("NISAR_TEST_FOCUS_RSLC"), reason="set NISAR_TEST_FOCUS_RSLC to run")
def test_real_rslc_runconfig(tmp_path):
    """A real production RSLC's runconfig parses and names its L0B and calibration files."""
    from nisar_tools.rslc import RSLC

    with RSLC(os.environ["NISAR_TEST_FOCUS_RSLC"]) as r:
        config = focus.production_runconfig(r)
        assert config["input_file_group"]["input_file_path"]
        assert focus._input_names(r, "l0bGranules")[0].startswith("NISAR_L0_")
        assert any("_EXT_CAL_" in n for n in focus._input_names(r, "auxcalFiles"))
