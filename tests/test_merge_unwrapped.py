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


# -- across a UTM zone boundary ---------------------------------------------
#
# A track crossing 114W is gridded in zone 11 on one side and zone 12 on the
# other. `merge` warps `other` onto `self`'s grid first, then reads the 2*pi
# offset on the shared grid. These use realistic UTM coordinates -- the `_frame`
# fixture above sits at x~0, which is nowhere inside zone 11, so reprojecting it
# is meaningless.

ZONE_A, ZONE_B = 32611, 32612


def _ground_phase(x11, y11):
    """Phase as a function of *zone-11 ground position*, so both frames carry the
    identical field wherever they overlap, whatever grid they are stored on."""
    return (0.004 * (x11 - 700_000.0) + 0.003 * (3_900_000.0 - y11)).astype(np.float32)


def _zone_frames(row_shift=20, cycle_offset=3, ny=48, nx=40, step=100.0):
    """Two overlapping frames of the same ground strip, one per UTM zone.

    Frame A is a regular zone-11 grid. Frame B is a regular *zone-12* grid over
    the same strip shifted ``row_shift`` rows south, filled by evaluating the same
    ground-referenced phase at each of its pixels -- so warping B back into zone 11
    must reproduce A's field in the overlap, up to resampling error and the
    injected ``cycle_offset``.
    """
    import pyproj

    to_b = pyproj.Transformer.from_crs(
        f"EPSG:{ZONE_A}", f"EPSG:{ZONE_B}", always_xy=True
    )
    to_a = pyproj.Transformer.from_crs(
        f"EPSG:{ZONE_B}", f"EPSG:{ZONE_A}", always_xy=True
    )

    # Frame A: zone 11, anchored near the 114W boundary.
    ax = 700_000.0 + step * np.arange(nx)
    ay = 3_900_000.0 - step * np.arange(ny)          # north-down, as products are
    axx, ayy = np.meshgrid(ax, ay)
    a = _stack(_ground_phase(axx, ayy), ax, ay, ZONE_A, cc=1)

    # Frame B: a regular zone-12 grid over the same strip, shifted south.
    shifted_y = ay - row_shift * step
    cx, cy = to_b.transform(
        [ax[0], ax[-1], ax[0], ax[-1]],
        [shifted_y[0], shifted_y[0], shifted_y[-1], shifted_y[-1]],
    )
    bx = np.arange(min(cx), max(cx), step)
    by = np.arange(max(cy), min(cy), -step)
    bxx, byy = np.meshgrid(bx, by)
    b_x11, b_y11 = to_a.transform(bxx, byy)          # back to zone 11 to evaluate
    b_phase = _ground_phase(b_x11, b_y11) + cycle_offset * TWO_PI
    b = _stack(b_phase.astype(np.float32), bx, by, ZONE_B, cc=7)
    return a, b


def _stack(phase, x, y, epsg, cc):
    ny, nx = phase.shape
    ds = xr.Dataset(
        {
            "unw": (("pair", "y", "x"), phase[None]),
            "coherence": (("pair", "y", "x"), np.full((1, ny, nx), 0.8, np.float32)),
            "conncomp": (("pair", "y", "x"), np.full((1, ny, nx), cc, np.uint32)),
        },
        coords={"pair": [0], "y": y, "x": x,
                "ref_time": ("pair", [REF]), "sec_time": ("pair", [SEC])},
    ).rio.write_crs(f"EPSG:{epsg}")
    ds.attrs.update(epsg=epsg, direction="Descending", source="snaphu")
    return UnwrappedStack(ds)


def test_cross_zone_merge_lands_on_selfs_grid():
    a, b = _zone_frames()
    m = a.merge(b)

    assert m.epsg == ZONE_A
    assert m.ds.rio.crs.to_epsg() == ZONE_A
    # Self's lattice is preserved exactly -- same spacing *and* grid phase, which
    # is what lets the union be reached by padding rather than reindexing.
    ux, uy = m.x, m.y
    assert np.isin(a.x, ux).all() and np.isin(a.y, uy).all()
    np.testing.assert_allclose(np.diff(ux), np.diff(a.x)[0])
    np.testing.assert_allclose(np.diff(uy), np.diff(a.y)[0])
    # And it grew southward to take in the warped frame.
    assert uy.min() < a.y.min()
    assert m.ds.attrs["merged"][-1]["other_epsg"] == ZONE_B
    assert m.ds.attrs["merged"][-1]["resampling"] == "bilinear"


def test_cross_zone_merge_removes_the_2pi_step():
    """The whole point: continuity across the seam despite the reprojection."""
    a, b = _zone_frames(row_shift=20, cycle_offset=3)
    m = a.merge(b)
    unw = m.ds["unw"].isel(pair=0).values
    uy, ux = m.y, m.x

    # Compare the merged field against the ground truth it should reproduce
    # (frame A's field, extended over the union) wherever the merge has data.
    xx, yy = np.meshgrid(ux, uy)
    truth = _ground_phase(xx, yy)
    good = np.isfinite(unw)
    assert good.mean() > 0.5, "the warp should cover most of the union"
    residual = unw[good] - truth[good]
    # No 2*pi step anywhere: the residual is resampling error, well under a cycle.
    assert np.abs(residual).max() < 0.5 * TWO_PI, (
        f"max |residual| = {np.abs(residual).max():.2f} rad -- a 2*pi step survived"
    )
    # Tighter: the residual is bilinear resampling error only (measured ~0.19 rad
    # max, 0.016 rms). Without the cycle fix the warped rows would be ~18.8 rad off.
    assert np.abs(residual).max() < 0.5

    # Non-vacuous: rows the warped frame alone supplies must be there and correct.
    only_other = ~np.isin(uy, a.y) & good.any(axis=1)
    assert only_other.sum() > 5, "no rows came from the warped frame"


def test_cross_zone_merge_keeps_labels_discrete():
    """A connected-component label must never be interpolated -- a blended label
    is not a label -- so integer layers are warped with nearest and refilled."""
    a, b = _zone_frames()
    m = a.merge(b)
    cc = m.ds["conncomp"].isel(pair=0).values

    assert cc.dtype == np.uint32
    base = int(a.ds["conncomp"].max())
    # Only self's label, other's shifted label, and the 0 fill. No blends.
    assert set(np.unique(cc)) <= {0, 1, 7 + base}
    assert (cc == 1).any() and (cc == 7 + base).any()


def test_cross_zone_merge_honours_nearest_resampling():
    a, b = _zone_frames()
    m = a.merge(b, resampling="nearest")
    assert m.ds.attrs["merged"][-1]["resampling"] == "nearest"
    unw = m.ds["unw"].isel(pair=0).values
    assert np.isfinite(unw).any()


def test_cross_zone_merge_then_to_los(gslc_factory):
    """The stage after the stitch: `to_los` takes one granule per frame, and on a
    cross-zone merge the second frame's geometry cube is in the *other* zone -- so
    `sample_look_geometry` has to transform the merged grid into each cube's CRS."""
    from nisar_tools import GSLC

    kw = dict(ny=64, nx=64, dx=100.0, dy=100.0, write_geometry=True)
    pa = gslc_factory(epsg=ZONE_A, x0=700_000.0, y0=3_900_000.0, **kw)
    pb = gslc_factory(epsg=ZONE_B, x0=153_000.0, y0=3_898_000.0, **kw)

    def _from_granule(path, cc):
        g = GSLC(path)
        x, y, epsg = g.x_coords, g.y_coords, g.epsg
        g.close()
        return _stack(np.zeros((len(y), len(x)), np.float32), x, y, epsg, cc)

    a = _from_granule(pa, cc=1)
    b = _from_granule(pb, cc=5)
    merged = a.merge(b)
    assert merged.epsg == ZONE_A

    los = merged.to_los([str(pa), str(pb)], dem=None)
    inc = np.deg2rad(los.ds["incidence_angle"].values)
    up = los.ds["los_up"].values
    ok = np.isfinite(inc) & np.isfinite(up)
    assert ok.mean() > 0.5
    # The invariant that says the geometry survived the reprojection intact.
    np.testing.assert_allclose(up[ok], np.cos(inc[ok]), rtol=1e-3, atol=1e-3)

    # The zone-12 cube really contributed: rows only that frame covers have
    # geometry. Without sampling every source cube they would be blank.
    only_b = ~np.isin(merged.y, a.y)
    assert only_b.sum() > 5
    assert np.isfinite(inc)[only_b].mean() > 0.5


def test_same_zone_merge_records_no_resampling():
    """A same-zone merge runs no warp, so its provenance -- and therefore its
    persist hash -- is unchanged from before cross-zone support existed."""
    m = _frame().merge(_frame(row_shift=4, cycle_offset=3))
    assert "resampling" not in m.ds.attrs["merged"][-1]


def test_merge_rejects_mismatches():
    a = _frame()
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
