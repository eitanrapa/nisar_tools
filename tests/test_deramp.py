"""Tests for the unwrapped-phase cleaning pipeline and its kernels.

Covers the scipy port of ``filt_gunw.csh`` / ``h52grd.py``: edge masking,
tension-spline residual outlier rejection, polynomial/spline deramping, and
phase-screen (ionosphere) subtraction -- all as lazy ``UnwrappedStack`` methods,
plus the underlying numpy kernels.
"""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import UnwrappedStack, Workspace
from nisar_tools._kernels import (
    deramp,
    deramp_poly_dask,
    fit_surface,
    halo_planes,
    mask_edges,
    mask_edges_planes,
    poly_surface,
    remove_outliers,
    remove_outliers_depth,
    remove_outliers_planes,
    smooth_surface,
)


# -- chunk decomposition ---------------------------------------------------
#
# These kernels used to force chunk({"pair": 1, "y": -1, "x": -1}), so a stack ran
# on as many cores as it had pairs -- one or two, typically -- and each task held a
# whole plane. They now decompose spatially; these tests pin that the answer did
# not move when they did.

def _ragged_field(shape=(3, 64, 80), seed=0):
    """A stack with a sheared invalid footprint and an interior hole."""
    rng = np.random.default_rng(seed)
    ny, nx = shape[-2:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    field = rng.normal(size=shape) + 0.01 * xx + 0.02 * yy
    field = np.where((xx + 0.7 * yy < 8) | (xx - 0.6 * yy > nx - 10), np.nan, field)
    field[..., ny // 2, nx // 3] = np.nan
    return field


@pytest.mark.parametrize("edge_pixels", [0, 1, 3, 8])
@pytest.mark.parametrize("chunks", [(1, 16, 20), (1, 64, 80), (1, 13, 11)])
def test_mask_edges_halo_is_exact(edge_pixels, chunks):
    """``edge_pixels`` cross erosions reach exactly ``edge_pixels`` px, so a
    matching halo gives the identical result, seams and all."""
    da = pytest.importorskip("dask.array")
    field = _ragged_field()
    ref = mask_edges_planes(field, edge_pixels=edge_pixels)
    got = halo_planes(
        mask_edges, da.from_array(field, chunks=chunks),
        depth=max(1, edge_pixels), edge_pixels=edge_pixels,
    ).compute()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_array_equal(got[np.isfinite(ref)], ref[np.isfinite(ref)])


@pytest.mark.parametrize("scale,iterations", [(4.0, 1), (4.0, 2), (8.0, 2)])
def test_remove_outliers_halo_is_exact(scale, iterations):
    """scipy's Gaussian truncates at 4 sigma, so the smooth surface has *finite*
    support and ``remove_outliers_depth`` covers it exactly -- per iteration,
    because each pass re-smooths the previous pass's NaN pattern."""
    da = pytest.importorskip("dask.array")
    field = _ragged_field(shape=(2, 96, 96), seed=1) * 0.2
    field[:, 40:44, 40:44] += 12.0  # spikes to reject
    ref = remove_outliers_planes(
        field, scale=scale, threshold=1.0, iterations=iterations
    )
    got = halo_planes(
        remove_outliers, da.from_array(field, chunks=(1, 48, 48)),
        depth=remove_outliers_depth(scale, iterations),
        scale=scale, threshold=1.0, iterations=iterations,
    ).compute()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got[np.isfinite(ref)], ref[np.isfinite(ref)])


def test_halo_planes_falls_back_when_the_halo_exceeds_the_raster():
    """A globally supported kernel (spline deramp at its default scale) has no
    decomposition; it must still run, as one plane per task."""
    da = pytest.importorskip("dask.array")
    field = _ragged_field(shape=(2, 32, 32), seed=2)
    ref = remove_outliers_planes(field, scale=16.0, threshold=1.0, iterations=2)
    got = halo_planes(
        remove_outliers, da.from_array(field, chunks=(1, 8, 8)),
        depth=remove_outliers_depth(16.0, 2),  # 128 >> 32
        scale=16.0, threshold=1.0, iterations=2,
    ).compute()
    np.testing.assert_allclose(got, ref, equal_nan=True)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("chunks", [(1, 16, 20), (1, 13, 11), (1, 64, 80)])
@pytest.mark.parametrize("masked", [False, True])
def test_deramp_poly_chunked_matches_whole_plane(degree, chunks, masked):
    """The fit is a sum over pixels, so accumulating the normal equations per
    chunk and solving once is the same fit -- to normal-equation precision."""
    da = pytest.importorskip("dask.array")
    field = _ragged_field(seed=3)
    ny, nx = field.shape[-2:]
    exclude = None
    if masked:
        exclude = np.zeros((ny, nx), bool)
        exclude[ny // 2:, nx // 2:] = True

    ref = np.stack([
        deramp(plane, degree=degree, method="poly", exclude=exclude)
        for plane in field
    ])
    got = deramp_poly_dask(
        da.from_array(field, chunks=chunks), degree, exclude=exclude
    ).compute()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    finite = np.isfinite(ref)
    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-9, atol=1e-9)


def test_deramp_poly_chunked_handles_an_unconstrainable_plane():
    """Fewer valid pixels than coefficients gives an all-NaN plane, matching
    ``poly_surface``'s whole-plane guard."""
    da = pytest.importorskip("dask.array")
    field = np.full((1, 20, 20), np.nan)
    field[0, 0, 0] = 1.0
    got = deramp_poly_dask(da.from_array(field, chunks=(1, 10, 10)), 2).compute()
    assert np.all(np.isnan(got))


def test_cleaning_methods_do_not_collapse_spatial_chunks():
    """The point of the exercise: a one-pair stack must expose more than one
    runnable task per stage.

    A whole-plane input is the realistic case -- a persisted stack arrives on the
    2048-px disk chunk, and a multilooked raster is smaller than that -- so this
    starts from one chunk and checks the stage splits it. The exact block count is
    a function of ``os.cpu_count()``, so only "more than one" is asserted.
    """
    da = pytest.importorskip("dask.array")
    ny = nx = 512
    ds = xr.Dataset(
        {
            "unw": (("pair", "y", "x"),
                    da.zeros((1, ny, nx), chunks=(1, ny, nx), dtype=np.float32)),
            "conncomp": (("pair", "y", "x"),
                         da.ones((1, ny, nx), chunks=(1, ny, nx), dtype=np.uint32)),
        },
        coords={"pair": [0], "y": np.arange(float(ny)), "x": np.arange(float(nx))},
        attrs={"epsg": 32611},
    )
    stack = UnwrappedStack(ds)
    assert stack.ds["unw"].data.numblocks[-2:] == (1, 1)  # one plane going in
    for label, out in (
        ("mask_edges", stack.mask_edges(edge_pixels=4)),
        ("remove_outliers", stack.remove_outliers(scale=4.0, iterations=1)),
        ("deramp(poly)", stack.deramp(degree=1, method="poly")),
    ):
        blocks = out.ds["unw"].data.numblocks[-2:]
        assert blocks[0] * blocks[1] > 1, f"{label} stayed one block: {blocks}"

    # The spline deramp is globally supported, so it stays whole-plane on purpose.
    spline = stack.deramp(method="spline")
    assert spline.ds["unw"].data.numblocks[-2:] == (1, 1)


# -- kernels ---------------------------------------------------------------
def test_poly_surface_fits_plane_exactly():
    ny, nx = 20, 30
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    truth = 3.0 + 0.5 * xx - 0.2 * yy
    field = truth.copy()
    field[5, 5] = np.nan  # a hole must not change an exact plane fit
    assert np.allclose(poly_surface(field, degree=1), truth, atol=1e-6)


def test_smooth_surface_is_nan_aware():
    field = np.ones((30, 30), float)
    field[10:20, 10:20] = np.nan  # a NaN block must not drag the surface to 0
    s = smooth_surface(field, scale=3.0)
    assert np.nanmax(np.abs(s[np.isfinite(s)] - 1.0)) < 1e-6


def test_mask_edges_kernel_erodes_border():
    field = np.full((20, 20), 1.0, np.float32)
    out = mask_edges(field, edge_pixels=3)
    assert np.all(np.isnan(out[:3])) and np.all(np.isnan(out[-3:]))
    assert np.all(np.isfinite(out[3:-3, 3:-3]))


def test_remove_outliers_kernel_nulls_spike():
    yy, xx = np.mgrid[0:40, 0:40].astype(np.float32)
    field = 0.01 * xx  # gentle ramp
    field[20, 20] = 50.0  # one gross outlier
    out = remove_outliers(field, scale=5.0, threshold=1.0, iterations=3)
    assert np.isnan(out[20, 20])
    assert np.isfinite(out[5, 5])


def test_deramp_kernel_flattens_and_validates_method():
    yy, xx = np.mgrid[0:30, 0:40].astype(np.float64)
    field = 2.0 + 0.3 * xx - 0.1 * yy
    flat = deramp(field, degree=1, method="poly")
    assert np.nanstd(flat) < 1e-6
    with pytest.raises(ValueError, match="method must be"):
        deramp(field, method="nope")


def test_deramp_is_field_minus_fit_surface():
    """deramp subtracts exactly what fit_surface (the phase screen) returns."""
    rng = np.random.default_rng(0)
    field = np.cumsum(rng.normal(size=(40, 40)), axis=1)  # smooth-ish
    for kw in ({"method": "poly", "degree": 2}, {"method": "spline", "scale": 5.0}):
        assert np.allclose(deramp(field, **kw), field - fit_surface(field, **kw),
                           equal_nan=True)


# -- UnwrappedStack methods (on ingested GUNW) -----------------------------
def test_remove_phase_screen_subtracts(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=40, nx=40, nan_border=2,
                                                   iono_amp=0.7))
    assert "phase_screen" in u.ds
    unw0 = u.ds["unw"].isel(pair=0).values
    screen = u.ds["phase_screen"].isel(pair=0).values

    out = u.remove_phase_screen()
    unw1 = out.ds["unw"].isel(pair=0).values
    valid = np.isfinite(unw0) & np.isfinite(screen)
    assert np.allclose(unw1[valid], (unw0 - screen)[valid], atol=1e-5)
    assert out.ds.attrs["phase_screen_removed"] is True

    with pytest.raises(ValueError, match="already been removed"):
        out.remove_phase_screen()


def test_remove_phase_screen_requires_layer(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("phase_screen")
    with pytest.raises(ValueError, match="carries no"):
        u.remove_phase_screen()


def test_mask_edges_trims_footprint(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=40, nx=40, nan_border=2))
    before = np.isfinite(u.ds["unw"].isel(pair=0).values)
    out = u.mask_edges(edge_pixels=3)
    after = np.isfinite(out.ds["unw"].isel(pair=0).values)

    assert after.sum() < before.sum()
    row = after.shape[0] // 2  # centre row loses 3 px on each side
    assert before[row].sum() - after[row].sum() == 6
    assert out.ds.attrs["edges_masked"]["edge_pixels"] == 3


def test_mask_edges_min_coherence_needs_coherence(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("coherence")
    with pytest.raises(ValueError, match="coherence"):
        u.mask_edges(min_coherence=0.5)


def test_mask_edges_builtin_mask_nulls_invalid_subswath(gunw_factory):
    # An interior stripe is finite in unw but flagged out-of-subswath by the
    # product mask: erosion alone keeps it, the built-in mask removes it.
    u = UnwrappedStack.from_gunw_file(
        gunw_factory(ny=40, nx=40, nan_border=2, mask_invalid_cols=3)
    )
    assert "subswath_mask" in u.ds
    unw = u.ds["unw"].isel(pair=0).values
    stripe = np.zeros_like(unw, bool)
    stripe[:, 20:23] = True
    assert np.isfinite(unw[stripe]).any()  # the stripe is real data in unw

    ero = u.mask_edges(edge_pixels=0).ds["unw"].isel(pair=0).values
    assert np.isfinite(ero[stripe]).any()  # erosion-only leaves it

    out = u.mask_edges(edge_pixels=0, use_builtin_mask=True)
    got = out.ds["unw"].isel(pair=0).values
    assert np.all(np.isnan(got[stripe]))                      # mask removes it
    assert np.isfinite(got[np.isfinite(unw) & ~stripe]).all()  # valid land kept
    assert out.ds.attrs["edges_masked"]["use_builtin_mask"] is True


def test_mask_edges_builtin_mask_requires_layer(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("subswath_mask")
    with pytest.raises(ValueError, match="subswath"):
        u.mask_edges(use_builtin_mask=True)


def test_remove_outliers_nulls_spikes(gunw_factory):
    u = UnwrappedStack.from_gunw_file(
        gunw_factory(ny=48, nx=48, nan_border=2, spikes=8, seed=1)
    )
    unw = u.ds["unw"].isel(pair=0).values
    spikes = unw > 20.0  # injected +50 outliers tower over the [-3, 3] ramp
    assert spikes.sum() >= 1

    cleaned = u.remove_outliers(scale=6.0, threshold=2.0, iterations=3)
    out = cleaned.ds["unw"].isel(pair=0).values
    assert np.all(np.isnan(out[spikes]))                       # spikes gone
    interior = np.isfinite(unw) & ~spikes
    assert np.isfinite(out[interior]).mean() > 0.9             # ramp survives
    assert cleaned.ds.attrs["outliers_removed"]["threshold"] == 2.0


def test_deramp_poly_flattens_ramp(gunw_factory):
    # unwrappedPhase is a linear ramp in x; a degree-1 deramp flattens it.
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=48, nan_border=2))
    before = u.ds["unw"].isel(pair=0).values
    out = u.deramp(degree=1).ds["unw"].isel(pair=0).values
    valid = np.isfinite(before)
    assert np.nanstd(before[valid]) > 1.0     # was a big ramp
    assert np.nanstd(out[valid]) < 0.05       # now flat


def test_deramp_enters_persist_hash(gunw_factory, tmp_path):
    """Applying deramp changes the persist identity (same stage name, two stores)."""
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    ws_a = Workspace(tmp_path / "a")
    ws_b = Workspace(tmp_path / "b")

    u.persist(ws_a, "s")
    u.deramp(degree=2).persist(ws_b, "s")
    assert ws_a.stored_params_hash("s") != ws_b.stored_params_hash("s")


def test_pipeline_chains_and_preserves_type(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=48, nan_border=2,
                                                  spikes=5, seed=2))
    out = (u.remove_phase_screen()
             .mask_edges(edge_pixels=2)
             .remove_outliers(threshold=2.0)
             .deramp(degree=1))
    assert isinstance(out, UnwrappedStack)
    # every step recorded its provenance
    for key in ("phase_screen_removed", "edges_masked", "outliers_removed", "deramp"):
        assert key in out.ds.attrs
    # still convertible to LOS from its own cube
    los = out.to_los()
    assert los.sizes == out.sizes


# -- deramp with a masked-out signal region --------------------------------
def test_poly_surface_exclude_ignores_masked_region():
    ny, nx = 30, 40
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    truth = 2.0 + 0.5 * xx - 0.3 * yy         # the ramp we want back
    field = truth.copy()
    field[5:15, 5:15] += 20.0                 # a strong signal bump on top
    exclude = np.zeros((ny, nx), bool)
    exclude[5:15, 5:15] = True

    # Excluding the bump recovers the true ramp; including it biases the fit.
    assert np.allclose(poly_surface(field, 1, exclude=exclude), truth, atol=1e-6)
    assert not np.allclose(poly_surface(field, 1), truth, atol=1.0)


def test_smooth_surface_exclude_fills_from_neighbours():
    field = np.ones((41, 41))
    field[18:23, 18:23] = 100.0               # a region to keep out of the fit
    exclude = np.zeros_like(field, bool)
    exclude[18:23, 18:23] = True
    s = smooth_surface(field, scale=6.0, exclude=exclude)
    # The masked block is filled from the surrounding 1.0 field, not its 100s.
    assert np.all(np.abs(s[18:23, 18:23] - 1.0) < 1e-3)


def test_deramp_kernel_exclude_removes_ramp_and_keeps_signal():
    ny, nx = 30, 40
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    field = (0.5 * xx - 0.3 * yy)
    field[10:20, 10:20] += 5.0                # localized signal over the ramp
    exclude = np.zeros((ny, nx), bool)
    exclude[10:20, 10:20] = True

    out = deramp(field, degree=1, method="poly", exclude=exclude)
    outside = ~exclude
    assert np.nanstd(out[outside]) < 1e-6                    # ramp gone far-field
    assert np.allclose(out[10:20, 10:20], 5.0, atol=1e-6)    # signal preserved


def _ramp_stack(bump=5.0, ny=30, nx=40, x=None, y=None):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    field = (0.5 * xx - 0.3 * yy)
    field[10:20, 10:20] += bump
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), field[None].astype(np.float32))},
        coords={"pair": [0],
                "y": np.arange(ny, dtype=float) if y is None else y,
                "x": np.arange(nx, dtype=float) if x is None else x},
    ).rio.write_crs("EPSG:32611")
    ds.attrs.update(epsg=32611, direction="Descending", source="snaphu")
    return UnwrappedStack(ds)


def test_deramp_mask_array_excludes_the_signal():
    u = _ramp_stack(bump=5.0)
    mask = np.zeros((30, 40), bool)
    mask[10:20, 10:20] = True

    out = u.deramp(degree=1, mask=mask).ds["unw"].isel(pair=0).values
    outside = ~mask
    assert np.nanstd(out[outside]) < 1e-4                    # far field flat
    assert np.allclose(out[10:20, 10:20], 5.0, atol=1e-4)    # signal kept

    # Without the mask the signal biases the fit, so the far field is not flat.
    plain = u.deramp(degree=1).ds["unw"].isel(pair=0).values
    assert np.nanstd(plain[outside]) > 0.01

    with pytest.raises(ValueError, match="does not match"):
        u.deramp(mask=np.zeros((5, 5), bool))


def test_deramp_mask_bbox_and_provenance():
    from nisar_tools import geo

    ny = nx = 40
    dx = dy = 100.0
    x = 400000.0 + dx * np.arange(nx)
    y = 3_800_000.0 - dy * np.arange(ny)      # descending (north-down)
    u = _ramp_stack(bump=8.0, ny=ny, nx=nx, x=x, y=y)

    # A lon/lat bbox over the signal block (rows/cols 10..19).
    bbox = geo.native_bbox_to_lonlat(
        x[10] - dx / 2, x[19] + dx / 2, y[19] - dy / 2, y[10] + dy / 2, 32611
    )
    out = u.deramp(degree=1, mask=bbox)
    got = out.ds["unw"].isel(pair=0).values

    # Rows 25.. are well clear of the block: the bbox-excluded fit flattens them,
    # and the plain deramp (biased by the bump) does not.
    far = got[25:, :]
    plain_far = u.deramp(degree=1).ds["unw"].isel(pair=0).values[25:, :]
    assert np.nanstd(far) < 0.05
    assert np.nanstd(far) < np.nanstd(plain_far)
    assert out.ds.attrs["deramp"]["mask"] == list(bbox)


def test_deramp_mask_changes_persist_hash(tmp_path):
    u = _ramp_stack()
    mask = np.zeros((30, 40), bool)
    mask[10:20, 10:20] = True
    ws = Workspace(tmp_path / "ws")

    u.deramp(degree=1).persist(ws, "plain")
    u.deramp(degree=1, mask=mask).persist(ws, "masked")
    assert ws.stored_params_hash("plain") != ws.stored_params_hash("masked")
