"""`.grd` export across the product stages.

`GSLCStack`, `InterferogramStack`, `UnwrappedStack` and `LOSStack` all share one
`RasterStackMixin.to_grd`: each reprojects its fields to lon/lat and writes one
single-variable GMT `.grd` per field, per slice along the stack dim. Complex
products (SLC, interferogram) are split into `amplitude` and wrapped `phase`.
(`LOSStack`'s own export is covered in `test_los.py`.)
"""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLC, GSLCStack, UnwrappedStack


def _gslc_stack(gslc_factory):
    p1 = gslc_factory(ny=48, nx=40, seed=0,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=48, nx=40, seed=1,
                      datetime_str="2025-12-10T02:32:50.000000000")
    return GSLCStack.from_gslcs([GSLC(p1), GSLC(p2)])


def test_gslc_stack_exports_amplitude_per_time(gslc_factory, tmp_path):
    stack = _gslc_stack(gslc_factory)
    outdir = tmp_path / "slc"

    written = stack.to_grd(outdir)

    # Default is amplitude only (absolute SLC phase is not meaningful), one file
    # per time slice, reprojected to lon/lat.
    assert {p.name for p in written} == {"amplitude_time0.grd", "amplitude_time1.grd"}
    amp = xr.open_dataarray(outdir / "amplitude_time0.grd")
    assert amp.dims == ("lat", "lon")
    finite = np.isfinite(amp.values)
    assert finite.any()
    assert (amp.values[finite] >= 0).all()

    # Phase is on the menu and comes back wrapped to [-pi, pi].
    ph = stack.to_grd(tmp_path / "slc_ph", fields=["phase"], indices=[0])
    assert {p.name for p in ph} == {"phase_time0.grd"}
    phase = xr.open_dataarray(tmp_path / "slc_ph" / "phase_time0.grd").values
    m = np.isfinite(phase)
    assert m.any() and np.all(np.abs(phase[m]) <= np.pi + 1e-4)


def test_interferogram_stack_exports_phase_and_coherence(gslc_factory, tmp_path):
    stack = _gslc_stack(gslc_factory)
    igrams = stack.form_interferograms(pairs="sequential", looks=5, downsample=True)
    outdir = tmp_path / "ig"

    written = igrams.to_grd(outdir)

    # Default: wrapped phase + coherence per pair (one pair here).
    assert {p.name for p in written} == {"phase_pair0.grd", "coherence_pair0.grd"}
    phase = xr.open_dataarray(outdir / "phase_pair0.grd").values
    m = np.isfinite(phase)
    assert m.any() and np.all(np.abs(phase[m]) <= np.pi + 1e-4)
    coh = xr.open_dataarray(outdir / "coherence_pair0.grd").values
    mc = np.isfinite(coh)
    assert mc.any() and np.all((coh[mc] >= -1e-4) & (coh[mc] <= 1 + 1e-4))

    # The interferogram amplitude is available on request.
    amp = igrams.to_grd(tmp_path / "ig_amp", fields=["amplitude"])
    assert {p.name for p in amp} == {"amplitude_pair0.grd"}


def test_unwrapped_stack_exports_every_layer(gunw_factory, tmp_path):
    # A GUNW carries unw + coherence + conncomp + phase_screen + subswath_mask.
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=64))
    outdir = tmp_path / "unw"

    written = u.to_grd(outdir)
    names = {p.name for p in written}

    layers = [v for v in ("unw", "coherence", "phase_screen", "conncomp",
                          "subswath_mask") if v in u.ds.data_vars]
    assert names == {f"{v}_pair0.grd" for v in layers}
    assert {"unw_pair0.grd", "coherence_pair0.grd"} <= names

    unw = xr.open_dataarray(outdir / "unw_pair0.grd")
    assert unw.dims == ("lat", "lon")
    assert np.isfinite(unw.values).any()


def test_to_grd_unknown_field_reports_the_menu(gunw_factory, tmp_path):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=32, nx=32))
    with pytest.raises(KeyError, match="unknown field 'bogus'"):
        u.to_grd(tmp_path / "x", fields=["bogus"])
