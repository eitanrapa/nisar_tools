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


def _write_l0b(path, slip_at=None):
    table = _table()
    k = np.arange(6000)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, k.size // table.size + 3))])
    utc = 36640.0 + cum[k]
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


def _fake_isce3(tmp_path, repairs=True, fails=False):
    """An interpreter stand-in: records its arguments and environment, writes nothing."""
    script = tmp_path / "python"
    script.write_text("#!/bin/sh\n"
                      f'if [ "$1" = "-c" ] && [ "$#" -eq 2 ]; then exit {0 if repairs else 1}; fi\n'
                      f'echo "$@" > "{tmp_path}/args.txt"\n'
                      f'echo "$OMP_NUM_THREADS $KMP_DUPLICATE_LIB_OK" > "{tmp_path}/env.txt"\n'
                      f"exit {3 if fails else 0}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _runconfig(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"runconfig": {"groups": {
        "product_path_group": {"sas_output_file": str(tmp_path / "out.h5")}}}}))
    return path


def test_run_focus_calls_the_workflow(tmp_path):
    out = focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path), threads=6, log=lambda *a: None)
    assert out == tmp_path / "out.h5"
    assert (tmp_path / "args.txt").read_text().split() == ["-m", "nisar.workflows.focus", str(tmp_path / "run.yaml")]
    assert (tmp_path / "env.txt").read_text().split() == ["6", "TRUE"]
    assert (tmp_path / "focus.log").exists()


def test_run_focus_selects_polarizations(tmp_path):
    focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path), polarizations=["HH"], log=lambda *a: None)
    args = (tmp_path / "args.txt").read_text().split()
    assert args[0] == "-c" and args[-2:] == ["HH", str(tmp_path / "run.yaml")]


def test_run_focus_refuses_isce3_without_the_repair(tmp_path, monkeypatch):
    monkeypatch.delenv(focus.ISCE3_PYTHON_ENV, raising=False)
    with pytest.raises(RuntimeError, match="pass python="):
        focus.run_focus(_runconfig(tmp_path), log=lambda *a: None)
    stock = _fake_isce3(tmp_path, repairs=False)
    with pytest.raises(RuntimeError, match="does not repair pulse time tags"):
        focus.run_focus(_runconfig(tmp_path), python=stock, log=lambda *a: None)
    monkeypatch.setenv(focus.ISCE3_PYTHON_ENV, stock)
    assert focus.run_focus(_runconfig(tmp_path), require_fork=False, log=lambda *a: None) == tmp_path / "out.h5"


def test_run_focus_reports_a_failed_run(tmp_path):
    with pytest.raises(RuntimeError, match="exit 3"):
        focus.run_focus(_runconfig(tmp_path), python=_fake_isce3(tmp_path, fails=True), log=lambda *a: None)


def test_polarization_wrapper_keeps_only_the_named_channels(monkeypatch):
    """The wrapper's own code, with ISCE3's module replaced by a stand-in."""
    import sys
    import types

    class Channel:
        def __init__(self, freq_id, pol):
            self.freq_id, self.pol = freq_id, pol

    seen = {}
    fake = types.ModuleType("nisar.workflows.focus")
    fake.get_common_mode = lambda raws: set(Channel(f, p) for f in "AB" for p in ("HH", "HV"))
    fake.main = lambda argv: seen.update(argv=list(sys.argv), mode=fake.get_common_mode([]))
    pkg, workflows = types.ModuleType("nisar"), types.ModuleType("nisar.workflows")
    pkg.workflows, workflows.focus = workflows, fake
    monkeypatch.setitem(sys.modules, "nisar", pkg)
    monkeypatch.setitem(sys.modules, "nisar.workflows", workflows)
    monkeypatch.setitem(sys.modules, "nisar.workflows.focus", fake)
    monkeypatch.setattr(sys, "argv", ["-c", "HH", "run.yaml"])
    exec(focus._FOCUS_POLS, {})
    assert seen["argv"] == ["-c", "run.yaml"]
    assert sorted((c.freq_id, c.pol) for c in seen["mode"]) == [("A", "HH"), ("B", "HH")]


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
