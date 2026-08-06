"""Tests for sub-pixel pixel offsets (the GMTSAR ``xcorr`` port).

The kernel has no legacy oracle, so it is pinned three ways: against the
brute-force time-domain correlator in ``offset_reference.py`` (the ``-time`` mode
that is deliberately not shipped), against planted integer and sub-pixel shifts
it must recover, and by its behaviour on invalid and decorrelated data.

The sub-pixel test is the acceptance test. Its accuracy is set by how much
independent texture a correlation window holds, not by the interpolator, so it
uses a band-limited field and asserts an *rms* bound rather than a per-location
one.
"""

import warnings

import dask.array as da
import matplotlib
import numpy as np
import pytest
from offset_reference import correlation_surface, peak_offset
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")  # before pyplot is imported anywhere

from nisar_tools import GSLC, GSLCStack, PixelOffsetStack, Workspace  # noqa: E402
from nisar_tools import _kernels  # noqa: E402
from nisar_tools.workspace import WorkspaceError  # noqa: E402

# Small enough that a whole test is milliseconds; the shape of the algorithm does
# not depend on the window size.
SMALL = dict(nx_corr=32, ny_corr=32, xsearch=16, ysearch=16)
SMALL_WINDOW = SMALL["nx_corr"] + 2 * SMALL["xsearch"]
TINY = dict(nx_corr=8, ny_corr=8, xsearch=4, ysearch=4)
TINY_WINDOW = TINY["nx_corr"] + 2 * TINY["xsearch"]


def _speckle(ny, nx, seed=0):
    """A complex SLC-like field: uniform amplitude, uniform random phase."""
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 1.5, (ny, nx))
    phase = rng.uniform(-np.pi, np.pi, (ny, nx))
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _texture(ny, nx, seed=0, scale=1.5):
    """A band-limited real field, so a *fractional* shift of it is well defined."""
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=(ny, nx)), scale) + 5.0


def _fft_shift(field, dy, dx):
    """Shift a real field so its content moves to *larger* indices by ``(dy, dx)``.

    Real in, real out: the kernel correlates ``abs()`` of its inputs, and
    ``abs()`` of a complex-valued shift is not the shift of ``abs()``.
    """
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.fftfreq(field.shape[1])[None, :]
    spec = np.fft.fft2(field) * np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.fft.ifft2(spec).real


def _lattice(ny, nx, window, **kwargs):
    return (_kernels.offset_locations(ny, window, **kwargs),
            _kernels.offset_locations(nx, window, **kwargs))


def _shifted_gslcs(gslc_factory, dy=2, dx=-3, ny=320, nx=320, seed=0):
    """Two granules on one grid, the second a whole-pixel roll of the first."""
    img = _speckle(ny, nx, seed)
    p1 = gslc_factory(ny=ny, nx=nx, data=img,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=ny, nx=nx, data=np.roll(img, (dy, dx), axis=(0, 1)),
                      datetime_str="2025-12-10T02:32:50.000000000")
    return GSLC(p1), GSLC(p2)


def _offsets_in_memory(gslc_factory, **kwargs):
    """A computed :class:`PixelOffsetStack`, with the granules already closed.

    The dask graph holds live HDF5 handles, so anything that closes the granules
    has to force the compute first -- exactly the rule the pipeline follows when
    it persists before closing.
    """
    g1, g2 = _shifted_gslcs(gslc_factory, **kwargs)
    try:
        offsets = GSLCStack.from_gslcs([g1, g2]).pixel_offsets(step=64, **SMALL)
        return PixelOffsetStack(offsets.ds.compute())
    finally:
        g1.close()
        g2.close()


# -- the location lattice ----------------------------------------------------


def test_offset_locations_keeps_every_window_inside_the_raster():
    for origins in (_kernels.offset_locations(500, 64, step=32),
                    _kernels.offset_locations(500, 64, count=7)):
        assert origins[0] >= 0
        assert origins[-1] + 64 <= 500


def test_offset_locations_is_exactly_uniform_for_both_specs():
    # rioxarray needs a uniform grid to build an affine transform, so a lattice
    # that merely spans the raster is not good enough.
    for origins in (_kernels.offset_locations(500, 64, step=37),
                    _kernels.offset_locations(500, 64, count=9),
                    _kernels.offset_locations(501, 64, count=7)):
        gaps = np.diff(origins)
        assert len(set(gaps.tolist())) == 1, origins


def test_offset_locations_honours_the_count_it_is_given():
    for count in (1, 2, 5, 13):
        assert len(_kernels.offset_locations(500, 64, count=count)) == count


def test_offset_locations_leaves_room_for_the_a_priori_shift():
    # The secondary's window sits at origin + shift and must also fit.
    for shift in (-40, 0, 40):
        origins = _kernels.offset_locations(500, 64, step=16, shift=shift)
        assert origins[0] >= 0 and origins[0] + shift >= 0
        assert origins[-1] + 64 <= 500
        assert origins[-1] + shift + 64 <= 500


@pytest.mark.parametrize("kwargs, match", [
    (dict(), "exactly one"),
    (dict(count=4, step=8), "exactly one"),
    (dict(step=0), "step must be"),
    (dict(count=0), "count must be"),
    (dict(count=1000), "do not fit"),
])
def test_offset_locations_rejects_bad_specs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _kernels.offset_locations(500, 64, **kwargs)


def test_offset_locations_rejects_a_window_that_cannot_fit():
    with pytest.raises(ValueError, match="does not fit"):
        _kernels.offset_locations(100, 128, step=8)


def test_offset_centre_pixels_is_the_window_centre():
    origins = np.array([0, 32, 64])
    np.testing.assert_array_equal(
        _kernels.offset_centre_pixels(origins, 64), [32, 64, 96]
    )


# -- kernel properties -------------------------------------------------------


def test_integer_peak_and_correlation_match_the_brute_force_oracle():
    img = _speckle(96, 96, seed=4)
    sec = np.roll(img, (2, -3), axis=(0, 1))
    origins = np.array([0, 40])

    got_x, got_y, got_c = _kernels.pixel_offsets(
        img, sec, origins, origins, subpixel=False, **TINY
    )
    for i, oy in enumerate(origins):
        for j, ox in enumerate(origins):
            cut = (slice(oy, oy + TINY_WINDOW), slice(ox, ox + TINY_WINDOW))
            want_x, want_y, want_c = peak_offset(img[cut], sec[cut], **TINY)
            assert got_x[i, j] == want_x
            assert got_y[i, j] == want_y
            assert got_c[i, j] == pytest.approx(want_c, rel=1e-5)


def test_correlation_surface_peak_agrees_with_the_direct_sum():
    """The FFT surface is the circular correlation, not merely peaked like it."""
    img = _speckle(64, 64, seed=5)
    sec = np.roll(img, (1, 2), axis=(0, 1))
    cut = (slice(0, TINY_WINDOW), slice(0, TINY_WINDOW))
    surface = correlation_surface(img[cut], sec[cut], **TINY)

    i, j = np.unravel_index(int(surface.argmax()), surface.shape)
    # lag = ref_position - sec_position, offset = -lag.
    assert (-(j - TINY["xsearch"]), -(i - TINY["ysearch"])) == (2, 1)


@pytest.mark.parametrize("dy, dx", [(0, 0), (3, 5), (-4, 7), (-1, -6)])
def test_recovers_a_planted_integer_shift(dy, dx):
    img = _speckle(320, 320, seed=6)
    sec = np.roll(img, (dy, dx), axis=(0, 1))
    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=3)

    x_off, y_off, corr = _kernels.pixel_offsets(
        img, sec, yo, xo, subpixel=False, **SMALL
    )
    np.testing.assert_array_equal(x_off, dx)
    np.testing.assert_array_equal(y_off, dy)
    assert corr.min() > 99.0


@pytest.mark.parametrize("dy, dx", [(0, 0), (3, 5), (-4, 7)])
def test_subpixel_refinement_does_not_move_an_integer_answer(dy, dx):
    """Speckle is the interpolator's worst case, and it still stays put.

    A fully developed speckle field is white, so its correlation peak is one
    pixel wide and interpolating it (through the reference's 0.25 power, over an
    8x8 window) is the least well conditioned this ever gets -- the sinc
    sidelobes are comparable to the peak. Real SAR amplitude is oversampled
    relative to its resolution cell, hence band-limited, hence easier; the
    fractional-shift test below is the one that measures accuracy.
    """
    img = _speckle(320, 320, seed=6)
    sec = np.roll(img, (dy, dx), axis=(0, 1))
    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=3)

    x_off, y_off, _ = _kernels.pixel_offsets(img, sec, yo, xo, **SMALL)
    np.testing.assert_allclose(x_off, dx, atol=0.2)
    np.testing.assert_allclose(y_off, dy, atol=0.2)


def test_recovers_a_planted_subpixel_shift():
    """The acceptance test: fractional offsets, unbiased and better than 0.1 px rms.

    Bounds the *population*, not each location. Individual windows scatter by up
    to ~0.4 px on this field even at correlation 97, and the correlation does not
    single them out -- every window here scores 96.6 to 97.1. So a high
    correlation says the match is real, not that the offset is accurate to a
    hundredth of a pixel, and 81 locations are used so the statistic is stable.
    """
    field = _texture(768, 768, seed=7)
    dy, dx = 0.37, -0.62
    sec = _fft_shift(field, dy, dx)
    window = 128 + 2 * 64
    yo, xo = _lattice(768, 768, window, step=window // 4)

    x_off, y_off, corr = _kernels.pixel_offsets(field, sec, yo, xo)
    ex, ey = (x_off - dx).ravel(), (y_off - dy).ravel()
    assert abs(ex.mean()) < 0.05 and abs(ey.mean()) < 0.05      # unbiased
    assert np.sqrt((ex ** 2).mean()) < 0.1
    assert np.sqrt((ey ** 2).mean()) < 0.1
    assert corr.min() > 90.0


def test_the_default_subpixel_window_is_the_less_biased_one():
    """GMTSAR's 8-sample refinement window is measurably biased; ours is 16.

    The refinement FFT-interpolates the window, i.e. treats it as periodic, and at
    8 samples the correlation peak has not decayed by its edges. Guards the
    default against being "restored" to the reference's value.
    """
    field = _texture(768, 768, seed=7)
    window = 128 + 2 * 64
    yo, xo = _lattice(768, 768, window, step=window)

    narrow, wide = [], []
    for frac in (0.25, 0.62, 0.75):
        sec = _fft_shift(field, frac, frac)
        narrow.append((_kernels.pixel_offsets(
            field, sec, yo, xo, subpixel_window=8)[0] - frac).mean())
        wide.append((_kernels.pixel_offsets(
            field, sec, yo, xo, subpixel_window=16)[0] - frac).mean())

    assert max(np.abs(narrow)) > 3 * max(np.abs(wide))
    assert max(np.abs(wide)) < 0.05


def test_offset_sign_is_secondary_minus_reference():
    """A feature at a LARGER index in the secondary gives a POSITIVE offset.

    The bright squares are placed inside the *secondary's template* -- the
    central ``ny_corr x nx_corr`` of the patch -- because everything outside it is
    masked to zero before the correlation and simply does not participate.
    """
    origin, half = 16, SMALL_WINDOW // 2
    img = _speckle(160, 160, seed=8) * 0.05
    sec = _speckle(160, 160, seed=9) * 0.05
    img[origin + half - 2:origin + half + 2,
        origin + half - 6:origin + half - 2] = 10.0
    sec[origin + half + 3:origin + half + 7,
        origin + half + 2:origin + half + 6] = 10.0  # 5 rows down, 8 cols right

    x_off, y_off, _ = _kernels.pixel_offsets(
        img, sec, np.array([origin]), np.array([origin]), subpixel=False, **SMALL
    )
    assert x_off[0, 0] == 8.0
    assert y_off[0, 0] == 5.0


def test_the_a_priori_shift_is_added_back():
    """Searching around a known shift finds the same total offset."""
    img = _speckle(320, 320, seed=9)
    dy, dx = 10, -12
    sec = np.roll(img, (dy, dx), axis=(0, 1))
    # One lattice both calls can use: inset enough for the shifted windows too.
    yo = _kernels.offset_locations(320, SMALL_WINDOW, count=2, shift=dy)
    xo = _kernels.offset_locations(320, SMALL_WINDOW, count=2, shift=dx)

    plain = _kernels.pixel_offsets(img, sec, yo, xo, subpixel=False, **SMALL)
    primed = _kernels.pixel_offsets(
        img, sec, yo, xo, subpixel=False, x_shift=dx, y_shift=dy, **SMALL
    )
    np.testing.assert_array_equal(plain[0], primed[0])
    np.testing.assert_array_equal(plain[1], primed[1])
    # The correlation value is scored over the secondary's template, which the
    # shift cuts from different pixels -- so it agrees closely, not exactly.
    np.testing.assert_allclose(plain[2], primed[2], rtol=1e-3)
    # ... and it is the right answer, found from both starting points.
    np.testing.assert_array_equal(primed[0], dx)
    np.testing.assert_array_equal(primed[1], dy)


def test_invalid_windows_are_nan_and_do_not_leak():
    img = _speckle(320, 320, seed=10)
    sec = np.roll(img, (2, -3), axis=(0, 1))
    blanked = img.copy()
    blanked[:80, :80] = np.nan  # swallows the first location's window only

    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=3)
    x_off, y_off, corr = _kernels.pixel_offsets(blanked, sec, yo, xo, **SMALL)

    assert np.isnan(x_off[0, 0]) and np.isnan(y_off[0, 0])
    assert corr[0, 0] == 0.0
    # Every other location is untouched: NaN does not spread.
    assert np.isfinite(x_off[1:, :]).all() and np.isfinite(x_off[:, 1:]).all()
    np.testing.assert_allclose(x_off[1:, 1:], -3, atol=1.0 / 16)


def test_a_partly_invalid_window_still_measures_above_the_threshold():
    img = _speckle(320, 320, seed=11)
    sec = np.roll(img, (2, -3), axis=(0, 1))
    holed = img.copy()
    holed[::4, ::4] = np.nan  # 6.25% invalid, well above min_valid_fraction

    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=3)
    x_off, y_off, corr = _kernels.pixel_offsets(holed, sec, yo, xo, **SMALL)
    assert np.isfinite(x_off).all()
    np.testing.assert_allclose(x_off, -3, atol=1.0 / 16)
    assert corr.min() > 50.0


def test_decorrelated_pair_has_low_correlation():
    img = _speckle(320, 320, seed=12)
    other = _speckle(320, 320, seed=13)  # independent, nothing in common
    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=3)

    _, _, corr = _kernels.pixel_offsets(img, other, yo, xo, **SMALL)
    assert corr.max() < 25.0


def test_oversampling_the_input_keeps_a_clean_answer():
    img = _speckle(320, 320, seed=14)
    sec = np.roll(img, (4, -6), axis=(0, 1))
    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=2)

    x_off, y_off, _ = _kernels.pixel_offsets(
        img, sec, yo, xo, oversample=2, **SMALL
    )
    np.testing.assert_allclose(x_off, -6, atol=0.1)
    np.testing.assert_allclose(y_off, 4, atol=0.1)


@pytest.mark.parametrize("kwargs, match", [
    (dict(nx_corr=31), "nx_corr must be"),
    (dict(ny_corr=0), "ny_corr must be"),
    (dict(xsearch=15), "xsearch must be"),
    (dict(subpixel_window=7), "subpixel_window must be"),
    (dict(interp_factor=0), "interp_factor must be"),
    (dict(oversample=0), "oversample must be"),
    (dict(min_valid_fraction=1.5), "min_valid_fraction must be"),
])
def test_pixel_offsets_rejects_bad_params(kwargs, match):
    img = _speckle(96, 96)
    with pytest.raises(ValueError, match=match):
        _kernels.pixel_offsets(img, img, np.array([0]), np.array([0]),
                               **{**TINY, **kwargs})


def test_pixel_offsets_rejects_windows_outside_the_raster():
    img = _speckle(96, 96)
    with pytest.raises(ValueError, match="run outside the raster"):
        _kernels.pixel_offsets(img, img, np.array([90]), np.array([0]), **TINY)


def test_pixel_offsets_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        _kernels.pixel_offsets(_speckle(96, 96), _speckle(96, 64),
                               np.array([0]), np.array([0]), **TINY)


# -- the dask path -----------------------------------------------------------


@pytest.mark.parametrize("chunk", [64, 128, 320])
@pytest.mark.parametrize("tile", [1, 2, 16])
def test_dask_matches_numpy_for_every_chunking_and_tile(chunk, tile):
    img = _speckle(320, 320, seed=15)
    sec = np.roll(img, (2, -3), axis=(0, 1))
    stack_ref = np.stack([img, img])
    stack_sec = np.stack([sec, np.roll(img, (1, 1), axis=(0, 1))])
    yo, xo = _lattice(320, 320, SMALL_WINDOW, step=48)

    want = _kernels.pixel_offsets_dask(stack_ref, stack_sec, yo, xo, **SMALL)
    got = _kernels.pixel_offsets_dask(
        da.from_array(stack_ref, chunks=(1, chunk, chunk)),
        da.from_array(stack_sec, chunks=(1, chunk, chunk)),
        yo, xo, locations_per_tile=tile, **SMALL,
    )
    for a, b in zip(want, got):
        assert b.shape == a.shape
        np.testing.assert_array_equal(np.asarray(b), a)


def test_dask_planes_do_not_mix():
    """Each pair is correlated on its own; a bad pair does not touch its neighbour."""
    img = _speckle(320, 320, seed=16)
    stack_ref = np.stack([img, img])
    stack_sec = np.stack([np.roll(img, (2, -3), axis=(0, 1)),
                          np.roll(img, (-5, 6), axis=(0, 1))])
    yo, xo = _lattice(320, 320, SMALL_WINDOW, count=2)

    x_off, y_off, _ = _kernels.pixel_offsets_dask(
        da.from_array(stack_ref, chunks=(1, 128, 128)),
        da.from_array(stack_sec, chunks=(1, 128, 128)), yo, xo, **SMALL,
    )
    np.testing.assert_allclose(np.asarray(x_off[0]), -3, atol=1.0 / 16)
    np.testing.assert_allclose(np.asarray(x_off[1]), 6, atol=1.0 / 16)
    np.testing.assert_allclose(np.asarray(y_off[0]), 2, atol=1.0 / 16)
    np.testing.assert_allclose(np.asarray(y_off[1]), -5, atol=1.0 / 16)


# -- stack integration -------------------------------------------------------


def test_pixel_offsets_matches_the_kernel(gslc_factory):
    g1, g2 = _shifted_gslcs(gslc_factory, dy=2, dx=-3)
    try:
        stack = GSLCStack.from_gslcs([g1, g2])
        offsets = stack.pixel_offsets(step=64, **SMALL)
        assert isinstance(offsets, PixelOffsetStack)
        # Lazy until asked, like every stage but the unwrap.
        assert _kernels._is_dask(offsets.ds["x_offset"].data)

        yo, xo = _lattice(320, 320, SMALL_WINDOW, step=64)
        want = _kernels.pixel_offsets(
            stack.ds["slc"].isel(time=0).compute().values,
            stack.ds["slc"].isel(time=1).compute().values,
            yo, xo, **SMALL,
        )
        for name, expected in zip(("x_offset", "y_offset", "correlation"), want):
            np.testing.assert_allclose(
                offsets.ds[name].isel(pair=0).compute().values, expected,
                rtol=1e-5, atol=1e-5,
            )
    finally:
        g1.close()
        g2.close()


def test_offsets_carry_the_grid_and_the_source_pixel_indices(gslc_factory):
    g1, g2 = _shifted_gslcs(gslc_factory)
    try:
        stack = GSLCStack.from_gslcs([g1, g2])
        offsets = stack.pixel_offsets(step=64, **SMALL)

        yo, _ = _lattice(320, 320, SMALL_WINDOW, step=64)
        centres = _kernels.offset_centre_pixels(yo, SMALL_WINDOW)
        np.testing.assert_array_equal(offsets.ds["y_pixel"].values, centres)
        # The coarse axis is the source coordinate of exactly those pixels.
        np.testing.assert_array_equal(offsets.ds["y"].values, stack.y[centres])
        assert offsets.epsg == stack.epsg
        assert offsets.ds.attrs["nx_corr"] == SMALL["nx_corr"]
        assert offsets.ds.attrs["step"] == 64
        assert offsets.ds.attrs["pairs"] == [[0, 1]]
    finally:
        g1.close()
        g2.close()


def test_north_offset_is_positive_northward(gslc_factory):
    """y_spacing is stored signed, so the row->northing flip comes out free."""
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    y_spacing = offsets.ds.attrs["y_spacing"]
    assert y_spacing < 0  # north-up descending grid

    # +2 rows is southward, so the northing must come out negative.
    assert float(offsets.ds["y_offset"].mean()) > 0
    assert float(offsets.north_offset.mean()) < 0
    np.testing.assert_allclose(
        offsets.north_offset.values, offsets.ds["y_offset"].values * y_spacing
    )
    np.testing.assert_allclose(
        offsets.east_offset.values,
        offsets.ds["x_offset"].values * offsets.ds.attrs["x_spacing"],
    )


def test_pixel_offsets_rejects_ambiguous_lattice_specs(gslc_factory):
    g1, g2 = _shifted_gslcs(gslc_factory)
    try:
        stack = GSLCStack.from_gslcs([g1, g2])
        with pytest.raises(ValueError, match="not both"):
            stack.pixel_offsets(step=64, nx=4, ny=4, **SMALL)
        with pytest.raises(ValueError, match="both nx= and ny="):
            stack.pixel_offsets(nx=4, **SMALL)
    finally:
        g1.close()
        g2.close()


def test_offsets_persist_and_reopen(gslc_factory, tmp_path):
    g1, g2 = _shifted_gslcs(gslc_factory, dy=2, dx=-3)
    try:
        ws = Workspace(tmp_path / "ws")
        stack = GSLCStack.from_gslcs([g1, g2])
        stored = stack.pixel_offsets(step=64, **SMALL).persist(ws, "offsets")
        assert isinstance(stored, PixelOffsetStack)

        reopened = PixelOffsetStack.from_zarr(ws.path("offsets"))
        assert reopened.epsg == stack.epsg
        assert reopened.ds.attrs["xsearch"] == SMALL["xsearch"]
        np.testing.assert_array_equal(
            reopened.ds["x_pixel"].values, stored.ds["x_pixel"].values
        )
        np.testing.assert_allclose(
            reopened.ds["x_offset"].values, stored.ds["x_offset"].values
        )
        # The CRS coordinate survives the Zarr round trip.
        assert reopened.ds.rio.crs is not None
    finally:
        g1.close()
        g2.close()


def test_offset_params_change_the_stage_hash(gslc_factory, tmp_path):
    g1, g2 = _shifted_gslcs(gslc_factory)
    try:
        ws = Workspace(tmp_path / "ws")
        stack = GSLCStack.from_gslcs([g1, g2])
        stack.pixel_offsets(step=64, **SMALL).persist(ws, "offsets")
        with pytest.raises(WorkspaceError):
            stack.pixel_offsets(step=32, **SMALL).persist(ws, "offsets")
    finally:
        g1.close()
        g2.close()


def test_offsets_export_to_grd(gslc_factory, tmp_path):
    g1, g2 = _shifted_gslcs(gslc_factory)
    try:
        stack = GSLCStack.from_gslcs([g1, g2])
        offsets = stack.pixel_offsets(step=64, **SMALL)
        paths = offsets.to_grd(tmp_path / "grd")
        assert sorted(p.name for p in paths) == [
            "correlation_pair0.grd", "x_offset_pair0.grd", "y_offset_pair0.grd",
        ]
        extra = offsets.to_grd(tmp_path / "grd2", fields=["east_offset"])
        assert extra[0].name == "east_offset_pair0.grd"
    finally:
        g1.close()
        g2.close()


# -- ASCII export ------------------------------------------------------------


def test_to_text_reproduces_the_xcorr_format(gslc_factory, tmp_path):
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    paths = offsets.to_text(tmp_path / "txt")
    assert [p.name for p in paths] == ["freq_xcorr_pair0.dat"]

    lines = paths[0].read_text().splitlines()
    assert len(lines) == offsets.ds["x_offset"].isel(pair=0).size
    # print_results.c: fprintf(file, " %d %6.3f %d %6.3f %6.2f \n", ...)
    x_pixel = int(offsets.ds["x_pixel"].values[0])
    y_pixel = int(offsets.ds["y_pixel"].values[0])
    x_off = float(offsets.ds["x_offset"].values[0, 0, 0])
    y_off = float(offsets.ds["y_offset"].values[0, 0, 0])
    corr = float(offsets.ds["correlation"].values[0, 0, 0])
    assert lines[0] == f" {x_pixel} {x_off:6.3f} {y_pixel} {y_off:6.3f} {corr:6.2f} "


def test_to_text_columns_are_the_ones_fitoffset_reads(gslc_factory, tmp_path):
    """fitoffset.csh: awk '{if ($5 > SNR) print $1, $3, $2}' -- x, y, x_offset."""
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    path = offsets.to_text(tmp_path / "txt")[0]

    table = np.loadtxt(path)
    assert table.shape[1] == 5
    np.testing.assert_array_equal(
        np.unique(table[:, 0]), offsets.ds["x_pixel"].values
    )
    np.testing.assert_array_equal(
        np.unique(table[:, 2]), offsets.ds["y_pixel"].values
    )
    np.testing.assert_allclose(table[:, 1], -3, atol=0.2)    # x_offset
    np.testing.assert_allclose(table[:, 3], 2, atol=0.2)     # y_offset
    assert (table[:, 4] > 20).all()                          # the SNR column


def test_to_text_drops_nan_and_low_correlation(gslc_factory, tmp_path):
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    ds = offsets.ds.copy()
    ds["x_offset"] = ds["x_offset"].copy()
    ds["x_offset"].values[0, 0, 0] = np.nan
    ds["correlation"] = ds["correlation"].copy()
    ds["correlation"].values[0, 0, 1] = 5.0
    ds.attrs.update(offsets.ds.attrs)
    trimmed = PixelOffsetStack(ds)

    total = ds["x_offset"].isel(pair=0).size
    assert len(np.loadtxt(trimmed.to_text(tmp_path / "a")[0])) == total - 1
    kept = np.loadtxt(trimmed.to_text(tmp_path / "b", min_correlation=20.0)[0])
    assert len(kept) == total - 2


def test_to_text_warns_when_too_few_points_survive(gslc_factory, tmp_path):
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    with pytest.warns(RuntimeWarning, match="fitoffset"):
        path = offsets.to_text(tmp_path / "txt", min_correlation=101.0)[0]
    assert path.read_text() == ""


def test_to_text_writes_an_optional_comment_header(gslc_factory, tmp_path):
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    path = offsets.to_text(tmp_path / "txt", comment="from nisar_tools")[0]
    lines = path.read_text().splitlines()
    assert lines[0] == "# from nisar_tools"
    # Everything after the header still parses as the plain five columns.
    assert np.loadtxt(path).shape[1] == 5


# -- plotting ----------------------------------------------------------------


def test_plot_offsets_renders_two_panels(gslc_factory):
    offsets = _offsets_in_memory(gslc_factory, dy=2, dx=-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = offsets.plot_offsets(pair=0, min_correlation=10.0)
    assert len(axes) == 2
    assert axes[0].get_title() == "Map x offset"
    matplotlib.pyplot.close(fig)


def test_plot_offsets_rejects_unknown_units(gslc_factory):
    offsets = _offsets_in_memory(gslc_factory)
    with pytest.raises(ValueError, match="units must be"):
        offsets.plot_offsets(units="metres_per_year")
