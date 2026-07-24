"""Stitching adjacent unwrapped frames with 2*pi continuity across the seam.

`UnwrappedStack.merge` reuses `GSLCStack.merge`'s union-grid padding (self wins
in the overlap) and adds the unwrapped-phase fix: the two frames share the
identical wrapped phase in the overlap but were unwrapped independently, so they
sit a whole number of 2*pi cycles apart -- merge reads that integer from the
overlap and removes it.
"""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import Workspace
from nisar_tools.unwrap import UnwrappedStack

TWO_PI = 2.0 * np.pi
REF = np.datetime64("2025-11-28T02:32:16")
SEC = np.datetime64("2025-12-10T02:32:16")


def _phi(y, x, y0=1000.0):
    """A smooth phase field in absolute coords -- identical where frames overlap."""
    yy, xx = np.meshgrid(np.asarray(y), np.asarray(x), indexing="ij")
    return (0.03 * xx + 0.02 * (y0 - yy)).astype(np.float32)


def _frame(row_shift=0, cycle_offset=0, coh=0.8, cc=1, ny=8, nx=6, dx=10.0,
           dy=10.0, y0=1000.0, direction="Descending", epsg=32611,
           phase_screen=False, source_files=None):
    """A one-pair UnwrappedStack on a north-down grid shifted ``row_shift`` rows
    south, its ``unw`` offset by ``cycle_offset`` whole 2*pi cycles."""
    x = np.arange(nx) * dx
    y = (y0 - row_shift * dy) - dy * np.arange(ny)
    unw = (_phi(y, x, y0) + cycle_offset * TWO_PI)[None].astype(np.float32)
    data = {
        "unw": (("pair", "y", "x"), unw),
        "coherence": (("pair", "y", "x"), np.full((1, ny, nx), coh, np.float32)),
        "conncomp": (("pair", "y", "x"), np.full((1, ny, nx), cc, np.uint32)),
    }
    if phase_screen:
        data["phase_screen"] = (("pair", "y", "x"),
                                np.full((1, ny, nx), 0.1, np.float32))
    ds = xr.Dataset(
        data,
        coords={"pair": [0], "y": y, "x": x,
                "ref_time": ("pair", [REF]), "sec_time": ("pair", [SEC])},
    ).rio.write_crs(f"EPSG:{epsg}")
    ds.attrs.update(epsg=epsg, direction=direction, source="snaphu")
    if source_files is not None:
        ds.attrs.update(source="gunw", source_files=source_files)
    return UnwrappedStack(ds)


def test_merge_removes_the_2pi_step_and_is_continuous():
    a = _frame(row_shift=0, cycle_offset=0)
    b = _frame(row_shift=4, cycle_offset=3)   # 4-row overlap, 3 cycles apart

    m = a.merge(b)

    # Union grid: self's 8 rows + 4 new southern rows; self's direction kept.
    assert m.sizes == {"pair": 1, "y": 12, "x": 6}
    assert m.y[0] == 1000.0 and m.y[-1] == 890.0

    # After alignment both frames equal the same phi, so the merged phase is that
    # one continuous field everywhere -- the 3-cycle jump is gone.
    got = m.ds["unw"].isel(pair=0).values
    np.testing.assert_allclose(got, _phi(m.y, m.x), atol=1e-4)

    # No 2*pi discontinuity across the seam: adjacent-row steps stay at the phi
    # gradient (~0.2 rad), never near a cycle (~6.28 rad).
    assert np.nanmax(np.abs(np.diff(got, axis=0))) < 1.0


def test_merge_self_precedence_and_distinct_components():
    a = _frame(row_shift=0, cycle_offset=0, coh=0.8, cc=1)
    b = _frame(row_shift=4, cycle_offset=3, coh=0.3, cc=1)

    m = a.merge(b)
    coh = m.ds["coherence"].isel(pair=0).values
    cc = m.ds["conncomp"].isel(pair=0).values

    # Rows 0..7 are self's (including the 4-row overlap): self wins.
    assert np.allclose(coh[:8], 0.8) and np.all(cc[:8] == 1)
    # Rows 8..11 are other-only: its coherence, and a label shifted clear of 1.
    assert np.allclose(coh[8:], 0.3) and np.all(cc[8:] == 2)


def test_merge_carries_phase_screen_and_merges_source_files():
    a = _frame(row_shift=0, cycle_offset=0, phase_screen=True, source_files=["a.h5"])
    b = _frame(row_shift=4, cycle_offset=1, phase_screen=True, source_files=["b.h5"])

    m = a.merge(b)
    assert "phase_screen" in m.ds
    assert m.ds["phase_screen"].isel(pair=0).values.shape == (12, 6)
    # Both frames' files are kept so a merged GUNW's to_los samples each cube.
    assert m.ds.attrs["source_files"] == ["a.h5", "b.h5"]


def test_merge_rejects_mismatches():
    a = _frame()
    with pytest.raises(ValueError, match="different UTM zones"):
        a.merge(_frame(epsg=32612))
    with pytest.raises(ValueError, match="pass directions"):
        a.merge(_frame(direction="Ascending"))

    two = _frame()
    two.ds = xr.concat([two.ds, two.ds.assign_coords(pair=[1])], dim="pair")
    with pytest.raises(ValueError, match="different pair counts"):
        a.merge(two)


def test_merged_stack_persists_distinctly(tmp_path):
    a = _frame(cycle_offset=0)
    b = _frame(row_shift=4, cycle_offset=3)
    ws = Workspace(tmp_path / "ws")

    a.persist(ws, "plain")
    a.merge(b).persist(ws, "merged")
    # The merge is provenance folded into the hash, so it stores distinctly.
    assert ws.stored_params_hash("plain") != ws.stored_params_hash("merged")

    back = UnwrappedStack.from_zarr(ws.path("merged"))
    assert back.sizes["y"] == 12 and "merged" in back.ds.attrs
