"""Refocus an RSLC from its L0B with ISCE3, run as a separate program.

Most defects of a delivered RSLC can be undone afterwards, in geocoding: a
whole-pulse azimuth slip over the area of interest, a band's range offset, a
calibration constant. What cannot is what changed how the pulses were added up:

- a wrong PRF (time-scale error in the L0B's tags; `nisar_tools.l0b`), which
  defocuses the whole frame;
- the edge of a one-pulse slip, where one synthetic aperture mixes good and
  slipped echoes;
- echo removed or smeared by RFI mitigation.

Those need the L0B refocused. nisar_tools does not import ISCE3: `run_focus`
runs ISCE3's focusing workflow with the interpreter of an environment that has
it, from `python=` or the `NISAR_TOOLS_ISCE3_PYTHON` variable. No GPU is
needed; ISCE3's CPU backprojection is used.

Stock ISCE3 v0.25.16 focuses the pulse tags as stored, so it reproduces
production, defects included. Repairing slipped and squeezed tags needs ISCE3
patched to model them from the commanded pulse sequence (our branch of
v0.25.16 adds `nisar.products.readers.Raw.model_pulse_times` and uses it in
`getPulseTimes`). `run_focus` refuses an ISCE3 without it unless
`require_fork=False`.

`focus_inputs` prepares what production used for an RSLC:

- the L0B(s) it names in `l0bGranules`, downloaded from ASF (10-90 GB each);
- the calibration files of its `auxcalFiles`, from ASF's `NISAR_ANC_AUX`;
- its orbit and attitude, written as NISAR XML from the RSLC itself, because the
  SDS's MOE/NRP files are not distributed and an L0B holds only a forecast
  orbit;
- the NISAR DEM v1.2 over the frame;
- production's own run configuration with these paths, CPU processing and a
  local output;
- `pulse_timing.txt`, what `nisar_tools.l0b.pulse_timing` finds in each L0B.
"""

import datetime as _dt
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import radar

#: Environment variable naming the Python interpreter of an ISCE3 environment.
ISCE3_PYTHON_ENV = "NISAR_TOOLS_ISCE3_PYTHON"
#: Imports only in an ISCE3 that repairs pulse time tags.
_REPAIR_PROBE = "from nisar.products.readers.Raw import model_pulse_times"
#: ISCE3's focus restricted to some polarizations (v0.25.16 has no option for it).
_FOCUS_POLS = """
import sys
import nisar.workflows.focus as focus
pols = set(sys.argv.pop(1).split(","))
common = focus.get_common_mode
def some(raws):
    mode = common(raws)
    return type(mode)(c for c in mode if c.pol in pols)
focus.get_common_mode = some
focus.main(sys.argv[1:])
"""
_META = "science/LSAR/RSLC/metadata"


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _utc_strings(epoch, seconds):
    return [str(epoch + np.timedelta64(int(round(float(s) * 1e9)), "ns")) for s in np.atleast_1d(seconds)]


def write_orbit_xml(rslc, path):
    """Write an RSLC's orbit as a NISAR orbit-ephemeris XML (`load_orbit_from_xml`'s format)."""
    import xml.etree.ElementTree as ET

    g = rslc._file[f"{_META}/orbit"]
    epoch = radar.parse_epoch(g["time"].attrs["units"])
    kind = _decode(g["orbitType"][()]) if "orbitType" in g else "Custom"
    times = _utc_strings(epoch, g["time"][()])
    root = ET.Element("orbitEphemeris")
    info = ET.SubElement(root, "productInformation")
    ET.SubElement(info, "productName").text = f"orbit of {rslc.granule_id}"
    ET.SubElement(info, "productType").text = kind
    ET.SubElement(info, "validityStartDateTime").text = times[0]
    ET.SubElement(info, "validityEndDateTime").text = times[-1]
    svl = ET.SubElement(root, "orbitStateVectorList", count=str(len(times)))
    for i, (t, x, v) in enumerate(zip(times, g["position"][()], g["velocity"][()]), start=1):
        node = ET.SubElement(svl, "orbitStateVector", index=str(i))
        ET.SubElement(node, "utc").text = t
        for name, value in zip(("x", "y", "z", "vx", "vy", "vz"), (*x, *v)):
            ET.SubElement(node, name).text = repr(float(value))
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")
    return Path(path)


def write_attitude_xml(rslc, path):
    """Write an RSLC's attitude as a NISAR radar-pointing XML (`load_attitude_from_xml`'s format)."""
    import xml.etree.ElementTree as ET

    g = rslc._file[f"{_META}/attitude"]
    epoch = radar.parse_epoch(g["time"].attrs["units"])
    times = _utc_strings(epoch, g["time"][()])
    root = ET.Element("radarPointing")
    info = ET.SubElement(root, "productInformation")
    ET.SubElement(info, "productName").text = f"attitude of {rslc.granule_id}"
    ET.SubElement(info, "fileType").text = _decode(g["attitudeType"][()]) if "attitudeType" in g else "Custom"
    svl = ET.SubElement(ET.SubElement(root, "LSAR"), "radarPointingStateVectorList", count=str(len(times)))
    for i, (t, q) in enumerate(zip(times, g["quaternions"][()]), start=1):
        node = ET.SubElement(svl, "radarPointingStateVector", index=str(i))
        ET.SubElement(node, "utc").text = t
        for name, value in zip(("q0", "q1", "q2", "q3"), q):
            ET.SubElement(node, name).text = repr(float(value))
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")
    return Path(path)


def production_runconfig(rslc):
    """The run configuration the SDS focused an RSLC with, as a dict."""
    import yaml

    text = _decode(rslc._file[f"{_META}/processingInformation/parameters/runConfigurationContents"][()])
    return yaml.safe_load(text)


def refocused_name(granule_id):
    """An RSLC name marked as Science On-Demand (`OD`), so it cannot pass for an SDS product."""
    return granule_id.replace("_PR_RSLC_", "_OD_RSLC_").replace("_UR_RSLC_", "_OD_RSLC_")


def _input_names(rslc, key):
    names = rslc._file[f"{_META}/processingInformation/inputs/{key}"][()]
    return [Path(_decode(n)).name for n in np.atleast_1d(names)]


def _points(coords):
    """The (lon, lat) pairs of a GeoJSON `coordinates` value, however deeply nested."""
    if coords and not isinstance(coords[0], (list, tuple)):
        yield coords[:2]
        return
    for item in coords:
        yield from _points(item)


def _frame_bbox(config, rslc):
    """`(lon_min, lon_max, lat_min, lat_max)` of the frame polygon, else the swath footprint."""
    import json

    polygon = (config.get("geometry") or {}).get("track_frame_polygon")
    if polygon:
        pts = np.array(list(_points(json.loads(polygon)["coordinates"])), float)
    else:
        pts = rslc.footprint().reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def focus_inputs(rslc, workdir, *, l0b_dir=None, ancillary_dir=None, dem=None, download=True, window=None):
    """Prepare everything to refocus an RSLC the way production did. Returns the runconfig path.

    Parameters
    ----------
    rslc : str, Path or `nisar_tools.rslc.RSLC`
        The production RSLC to reproduce or repair.
    workdir : str or Path
        Gets the orbit and attitude XML, the DEM (unless `dem=`), `run.yaml`,
        and `output/` and `scratch/` for the run.
    l0b_dir, ancillary_dir : optional
        Where the L0Bs and the calibration files are, or are downloaded to
        (default: `workdir`).
    dem : optional
        A DEM covering the frame (default: the NISAR DEM v1.2, staged).
    download : bool
        Fetch missing L0Bs and calibration files from ASF.
    window : dict, optional
        Focus only part of the frame: any of `start_time`, `end_time` (UTC,
        ISO strings or datetimes) and `start_range`, `end_range` (m). CPU
        backprojection is slow (see the guide's costs), so a window around the
        area of interest is the usual way to refocus.
    """
    import yaml

    from . import download as dl
    from .dem import stage_nisar_dem
    from .l0b import pulse_timing
    from .rslc import RSLC

    own = not isinstance(rslc, RSLC)
    r = RSLC(rslc) if own else rslc
    try:
        workdir = Path(workdir).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        l0b_dir = Path(l0b_dir or workdir).expanduser()
        ancillary_dir = Path(ancillary_dir or workdir).expanduser()
        config = production_runconfig(r)

        l0bs = [l0b_dir / n for n in _input_names(r, "l0bGranules")]
        missing = [p.name for p in l0bs if not p.exists()]
        if missing:
            if not download:
                raise FileNotFoundError(f"missing L0B(s) in {l0b_dir}: {', '.join(missing)}")
            dl.download_l0b(l0b_dir, missing)
        (workdir / "pulse_timing.txt").write_text("\n".join(pulse_timing(p).summary() for p in l0bs) + "\n")
        calibration = {}
        for name in _input_names(r, "auxcalFiles"):
            path = ancillary_dir / name
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f"missing calibration file {path}")
                dl.download_ancillary(ancillary_dir, [name])
            key = ("external_calibration" if "_EXT_CAL_" in name else "internal_calibration" if "_INT_CAL_" in name
                   else "antenna_pattern" if "_ANTPAT_" in name else None)
            if key:
                calibration[key] = str(path)
        orbit = write_orbit_xml(r, workdir / "orbit.xml")
        attitude = write_attitude_xml(r, workdir / "attitude.xml")
        if dem is None:
            dem = stage_nisar_dem(workdir / "dem.tif", _frame_bbox(config, r))

        name = refocused_name(r.granule_id or Path(r.path).stem)
        out = workdir / "output"
        config["input_file_group"]["input_file_path"] = [str(p) for p in l0bs]
        config["input_file_group"].pop("qa_input_file", None)
        anc = config["dynamic_ancillary_file_group"]
        anc.update(dem_file=str(dem), orbit=str(orbit), pointing=str(attitude), corner_reflector_file=None,
                   **calibration)
        paths = config["product_path_group"]
        paths.update(product_path=str(out), scratch_path=str(workdir / "scratch"),
                     sas_output_file=str(out / f"{name}.h5"), sas_config_file=str(out / "run.yaml"),
                     qa_output_dir=str(out))
        executable = config.setdefault("primary_executable", {})
        executable["processing_type"] = "OD"
        if executable.get("partial_granule_id"):          # the granuleId written into the product
            executable["partial_granule_id"] = refocused_name(executable["partial_granule_id"])
        config["worker"] = {"gpu_enabled": False}
        if window:
            unknown = set(window) - {"start_time", "end_time", "start_range", "end_range"}
            if unknown:
                raise ValueError(f"unknown window keys: {', '.join(sorted(unknown))}")
            grid = config.setdefault("processing", {}).setdefault("output_grid", {})
            grid.update({k: (str(v) if "time" in k else float(v)) for k, v in window.items()})
        config.pop("qa", None)
        for d in (out, workdir / "scratch"):
            d.mkdir(parents=True, exist_ok=True)
        path = workdir / "run.yaml"
        # the product stores the groups only; the workflow wants them under runconfig/groups
        path.write_text(yaml.safe_dump({"runconfig": {"groups": config}}, sort_keys=False))
        return path
    finally:
        if own:
            r.close()


def run_focus(runconfig, *, python=None, threads=None, polarizations=None, require_fork=True, log=print):
    """Run ISCE3's focusing on `runconfig`. Returns the output RSLC path.

    Parameters
    ----------
    runconfig : str or Path
        From `focus_inputs`. ISCE3's log goes to `focus.log` beside it.
    python : str, optional
        The interpreter of an environment with ISCE3 (default: the
        `NISAR_TOOLS_ISCE3_PYTHON` variable).
    threads : int, optional
        OpenMP's thread count (default: all).
    polarizations : list of str, optional
        Focus only these, such as `["HH"]`, in every band (default: all the
        L0B holds). Each polarization costs as much as the first.
    require_fork : bool
        Refuse an ISCE3 that does not repair pulse time tags. False runs stock
        ISCE3, which reproduces production, defects included.
    """
    import yaml

    python = python or os.environ.get(ISCE3_PYTHON_ENV)
    if not python:
        raise RuntimeError(f"no ISCE3 interpreter: pass python= or set {ISCE3_PYTHON_ENV}")
    runconfig = Path(runconfig)
    config = yaml.safe_load(runconfig.read_text())
    groups = config.get("runconfig", {}).get("groups", config)
    output = Path(groups["product_path_group"]["sas_output_file"])
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    if threads:
        env["OMP_NUM_THREADS"] = str(int(threads))
    if require_fork and subprocess.run([python, "-c", _REPAIR_PROBE], capture_output=True, env=env).returncode:
        raise RuntimeError(f"the ISCE3 of {python} does not repair pulse time tags, so it would focus slipped "
                           "and squeezed tags as stored; use a patched build, or require_fork=False to "
                           "reproduce production")
    if polarizations:
        command = [python, "-c", _FOCUS_POLS, ",".join(polarizations), str(runconfig)]
    else:
        command = [python, "-m", "nisar.workflows.focus", str(runconfig)]
    logfile = runconfig.with_name("focus.log")
    start = time.perf_counter()
    log(f"focusing {runconfig} with {python} ({_dt.datetime.now():%H:%M})")
    with open(logfile, "w") as fh:
        proc = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=runconfig.parent)
    if proc.returncode:
        raise RuntimeError(f"focus failed (exit {proc.returncode}); see {logfile}")
    log(f"wrote {output} in {time.perf_counter() - start:.0f} s")
    return output


def refocus(rslc, workdir, *, python=None, threads=None, polarizations=None, require_fork=True, log=print,
            **inputs):
    """`focus_inputs` then `run_focus`. Returns the refocused RSLC's path."""
    return run_focus(focus_inputs(rslc, workdir, **inputs), python=python, threads=threads,
                     polarizations=polarizations, require_fork=require_fork, log=log)


if __name__ == "__main__":     # python -m nisar_tools.focus RSLC WORKDIR
    print(refocus(sys.argv[1], sys.argv[2]))
