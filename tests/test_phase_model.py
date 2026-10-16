"""A phase model taken out before multilooking and added back after unwrapping (2026-10-14).

For a coseismic pair the near-fault fringes are so dense that they cancel inside the looks and leave SNAPHU a step
to jump; a model from pixel offsets removed at full resolution keeps the looks coherent and the residual smooth.
"""
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLCStack, Workspace

STEP = 5.0


def _pair(phase):
    """A two-date stack whose interferogram ref * conj(sec) is |s|^2 exp(i phase)."""
    rng = np.random.default_rng(0)
    ny, nx = phase.shape
    s = (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))).astype(np.complex64)
    data = np.stack([s, s * np.exp(-1j * phase).astype(np.complex64)])
    ds = xr.Dataset({"slc": (("time", "y", "x"), da.from_array(data, chunks=(1, 64, 64)))},
                    coords={"time": np.arange(2).astype("datetime64[s]"),
                            "y": 1000.0 - STEP * np.arange(ny), "x": 2000.0 + STEP * np.arange(nx)},
                    attrs={"epsg": 32611, "direction": "Ascending", "x_spacing": STEP, "y_spacing": -STEP})
    return GSLCStack(ds)


def _ramp(ny=160, nx=160, cycles_per_px=0.23):
    return 2 * np.pi * cycles_per_px * (np.arange(nx)[None] * np.ones((ny, 1)))


def test_the_model_keeps_steep_fringes_coherent():
    phase = _ramp()                                       # a 10-look window spans 2.3 cycles
    stack = _pair(phase)
    model = xr.DataArray(phase, dims=("y", "x"), coords={"y": stack.y, "x": stack.x}, attrs={"description": "ramp"})
    plain = stack.form_interferograms(looks=10, convolution="Gaussian")
    guided = stack.form_interferograms(looks=10, convolution="Gaussian", phase_model=model)
    assert np.nanmedian(plain.ds["coherence"].values) < 0.3
    assert np.nanmedian(guided.ds["coherence"].values) > 0.95
    assert np.nanstd(np.angle(guided.ds["igram"].values[0])) < 0.05          # the residual is flat
    pm = guided.ds["phase_model"].values                                      # kept on the multilooked grid
    want = 2 * np.pi * 0.23 * (guided.ds.x.values - 2000.0) / STEP
    np.testing.assert_allclose(pm[5], want, atol=1e-3)
    assert "sha256" in guided.ds.attrs["phase_model"] and "ramp" in guided.ds.attrs["phase_model"]
    assert "phase_model" not in plain.ds


def test_unwrapping_adds_the_model_back(tmp_path):
    phase = _ramp()
    stack = _pair(phase)
    model = xr.DataArray(phase, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})
    guided = stack.form_interferograms(looks=10, convolution="Gaussian", phase_model=model)
    unw = guided.unwrap(Workspace(tmp_path), name="unw", nproc=1)
    u = unw.ds["unw"].values[0]
    full = u - u[0, 0] - (guided.ds["phase_model"].values - guided.ds["phase_model"].values[0, 0])
    assert np.nanmax(np.abs(full)) < 0.1                                       # residual + model = the full phase
    assert np.nanmax(np.abs(np.diff(u, axis=1))) > 10.0                        # the store holds the steep phase


def test_cost_mode_is_checked_and_defo_runs(tmp_path):
    phase = _ramp(cycles_per_px=0.01)
    igrams = _pair(phase).form_interferograms(looks=10, convolution="Gaussian")
    with pytest.raises(ValueError, match="cost"):
        igrams.unwrap(Workspace(tmp_path), name="bad", cost="topo")
    defo = igrams.unwrap(Workspace(tmp_path), name="defo", nproc=1, cost="defo")
    smooth = igrams.unwrap(Workspace(tmp_path), name="smooth", nproc=1)
    assert np.isfinite(defo.ds["unw"].values).all()
    import json

    assert json.loads(defo.ds.attrs["params"])["cost"] == "defo"
    assert "cost" not in json.loads(smooth.ds.attrs["params"])          # the default keeps old stores' hashes
