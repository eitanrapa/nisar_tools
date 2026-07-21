"""NaN-aware multilooking: the invalid-pixel footprint must not spread.

The legacy formula hands NaN straight to ``scipy.ndimage``, whose filters are
not NaN-aware, so a single invalid sample contaminates its whole filter
footprint -- and for ``uniform_filter``, which is a running sum, everything
downstream of it along both axes. Real GSLCs are NaN outside the swath and
every merged union grid is NaN in the corners, so this quietly destroyed data.

These tests pin the replacement: multilook over the valid samples only, keep an
output pixel when enough of its filter weight was valid, and carry the
resulting footprint through unwrapping.
"""

import dask.array as da
import numpy as np
import pytest

import legacy_reference as legacy
from nisar_tools import GSLC, GSLCStack, InterferogramStack, Workspace, _kernels

CONVOLUTIONS = ["Uniform", "Gaussian"]
LOOKS = 5


def _max_xy(ny, nx, looks):
    return nx // looks * looks, ny // looks * looks


def _pair(ny=120, nx=100, seed=3):
    rng = np.random.default_rng(seed)
    c1 = (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx)))
    c2 = c1 * np.exp(1j * 0.05 * np.arange(nx)[None, :])
    return c1.astype(np.complex64), c2.astype(np.complex64)


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
@pytest.mark.parametrize("downsample", [False, True])
def test_nan_aware_matches_legacy_phase_and_coherence(convolution, downsample):
    """On NaN-free input the only change is the amplitude near the border.

    The smoothed mask cancels exactly in the coherence ratio and dividing by a
    positive real cannot move the phase, so both must match the oracle. The
    interferogram amplitude legitimately differs within a filter radius of the
    edge, where the normalization removes the zero-padding bias, so amplitude
    is compared on the interior only.
    """
    ny, nx = 120, 100
    c1, c2 = _pair(ny, nx)
    max_x, max_y = _max_xy(ny, nx, LOOKS)

    igram_ref, corr_ref = legacy._calculate_multilooked_interferograms(
        c1, c2, max_x, max_y, LOOKS, downsample, convolution
    )
    igram_new, corr_new = _kernels.igram_coherence(
        c1, c2, max_x, max_y, LOOKS, downsample, convolution, nan_aware=True
    )

    np.testing.assert_allclose(corr_new, corr_ref, rtol=1e-4, atol=1e-4)
    dphase = np.angle(igram_new * np.conj(igram_ref))
    np.testing.assert_allclose(dphase, 0.0, atol=1e-5)

    # Strip the border ring the zero-padding taper reaches into.
    radius = _kernels._overlap_depth(LOOKS, convolution)
    trim = radius // LOOKS + 1 if downsample else radius
    inner = (slice(trim, -trim), slice(trim, -trim))
    assert igram_new[inner].size > 0
    np.testing.assert_allclose(
        np.abs(igram_new[inner]), np.abs(igram_ref[inner]), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
def test_nan_footprint_is_not_dilated(convolution):
    """The kept region tracks the true valid region, it does not erode."""
    ny, nx = 300, 300
    c1, c2 = _pair(ny, nx)
    # A sheared swath, as a geocoded NISAR frame actually is.
    yy, xx = np.mgrid[0:ny, 0:nx]
    swath = (xx + 0.5 * yy > 60) & (xx + 0.5 * yy < nx + 90)
    c1 = np.where(swath, c1, np.nan).astype(np.complex64)
    c2 = np.where(swath, c2, np.nan).astype(np.complex64)
    max_x, max_y = _max_xy(ny, nx, LOOKS)

    igram, corr = _kernels.igram_coherence(
        c1, c2, max_x, max_y, LOOKS, True, convolution
    )
    kept = np.isfinite(igram)

    # Ground truth: an output pixel is valid when at least half of its filter
    # footprint was valid, which is exactly what the threshold asks for.
    expected = _kernels.multilook(
        swath.astype(np.float64), max_x, max_y, LOOKS, True, convolution
    ) / _kernels.boundary_response(
        ny, nx, max_x, max_y, LOOKS, True, convolution
    ) >= 0.5
    np.testing.assert_array_equal(kept, expected)

    # And the footprint is genuinely preserved, not merely self-consistent.
    truth = _kernels.multilook(
        swath.astype(np.float64), max_x, max_y, LOOKS, True, "Uniform"
    ) >= 0.5
    assert abs(kept.mean() - truth.mean()) < 0.02
    assert corr[kept].size > 0
    assert np.all(corr[~kept] == 0.0)


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
def test_one_bad_pixel_does_not_blank_its_footprint(convolution):
    """A single invalid sample used to take out its whole filter footprint.

    For Uniform it was worse than the footprint: ``uniform_filter`` is a
    running sum, so the NaN poisoned every sample after it along both axes.
    """
    ny, nx = 300, 300
    c1, c2 = _pair(ny, nx)
    c1 = c1.copy()
    c1[150, 150] = np.nan
    max_x, max_y = _max_xy(ny, nx, LOOKS)

    igram, _ = _kernels.igram_coherence(
        c1, c2, max_x, max_y, LOOKS, True, convolution
    )
    assert np.isfinite(igram).mean() > 0.99

    legacy_igram, _ = _kernels.igram_coherence(
        c1, c2, max_x, max_y, LOOKS, True, convolution, nan_aware=False
    )
    assert np.isfinite(legacy_igram).mean() < np.isfinite(igram).mean()


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
def test_nan_aware_is_chunk_independent(convolution):
    """The dask path must agree with numpy even when the input has NaN.

    ``_overlap_depth`` promises chunk-layout independence. The legacy path
    broke that promise for Uniform, because the halo bounds the running-sum
    contamination to a block instead of to the whole array.
    """
    ny, nx = 300, 300
    c1, c2 = _pair(ny, nx)
    c1 = c1.copy()
    c1[:80, :80] = np.nan
    c1[200, 200] = np.nan
    max_x, max_y = _max_xy(ny, nx, LOOKS)

    igram_np, corr_np = _kernels.igram_coherence(
        c1, c2, max_x, max_y, LOOKS, True, convolution
    )
    d1 = da.from_array(c1, chunks=(64, 64))
    d2 = da.from_array(c2, chunks=(64, 64))
    igram_dk, corr_dk = _kernels.igram_coherence(
        d1, d2, max_x, max_y, LOOKS, True, convolution
    )
    igram_dk = np.asarray(igram_dk)
    corr_dk = np.asarray(corr_dk)

    np.testing.assert_array_equal(np.isfinite(igram_np), np.isfinite(igram_dk))
    np.testing.assert_allclose(corr_np, corr_dk, rtol=1e-4, atol=1e-4)
    finite = np.isfinite(igram_np)
    np.testing.assert_allclose(
        igram_np[finite], igram_dk[finite], rtol=1e-4, atol=1e-4
    )


def test_boundary_response_matches_full_filter():
    """The separable shortcut must equal the real 2D filter of an all-ones raster."""
    ny, nx = 120, 100
    for convolution in CONVOLUTIONS:
        for downsample in (False, True):
            max_x, max_y = _max_xy(ny, nx, LOOKS)
            got = _kernels.boundary_response(
                ny, nx, max_x, max_y, LOOKS, downsample, convolution
            )
            ref = _kernels.multilook(
                np.ones((ny, nx)), max_x, max_y, LOOKS, downsample, convolution
            )
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_min_valid_fraction_controls_the_edge():
    """A stricter threshold keeps strictly fewer pixels."""
    ny, nx = 200, 200
    c1, c2 = _pair(ny, nx)
    c1 = np.where(np.mgrid[0:ny, 0:nx][1] > 60, c1, np.nan).astype(np.complex64)
    max_x, max_y = _max_xy(ny, nx, LOOKS)

    kept = []
    for frac in (0.1, 0.5, 0.9):
        igram, _ = _kernels.igram_coherence(
            c1, c2, max_x, max_y, LOOKS, True, "Gaussian",
            min_valid_fraction=frac,
        )
        kept.append(int(np.isfinite(igram).sum()))
    assert kept[0] > kept[1] > kept[2]

    with pytest.raises(ValueError, match="min_valid_fraction"):
        _kernels.igram_coherence(
            c1, c2, max_x, max_y, LOOKS, True, "Gaussian", min_valid_fraction=1.5
        )


def test_merge_then_form_interferograms(gslc_factory):
    """The union grid of two frames is NaN in its corners; igrams must survive it.

    No test chained merge -> form_interferograms before, which is exactly the
    combination that quietly returned an all-NaN interferogram.
    """
    ny, nx = 80, 60
    common = dict(ny=ny, nx=nx, dx=10.0, dy=10.0)
    times = ["2025-11-28T02:32:50.000000000", "2025-12-10T02:32:50.000000000"]

    first = [GSLC(gslc_factory(x0=400000.0, y0=4_000_000.0, seed=i,
                               datetime_str=t, **common))
             for i, t in enumerate(times)]
    # The adjacent along-track frame is offset in x *and* y, so the union
    # rectangle is a staircase with two empty corners -- as in a real track.
    second = [GSLC(gslc_factory(x0=400000.0 + 10.0 * nx,
                                y0=4_000_000.0 - 10.0 * (ny // 2), seed=10 + i,
                                datetime_str=t, **common))
              for i, t in enumerate(times)]

    merged = GSLCStack.from_gslcs(first).merge(GSLCStack.from_gslcs(second))
    assert merged.sizes["x"] == 2 * nx
    assert merged.sizes["y"] == ny + ny // 2
    valid_fraction = float(
        np.isfinite(merged.ds["slc"].isel(time=0).compute().values).mean()
    )
    assert 0.5 < valid_fraction < 0.8  # genuinely NaN in the corners

    igrams = merged.form_interferograms(
        pairs="sequential", looks=LOOKS, downsample=True, convolution="Gaussian"
    )
    igram = igrams.ds["igram"].isel(pair=0).compute().values
    assert abs(np.isfinite(igram).mean() - valid_fraction) < 0.05

    legacy_stack = merged.form_interferograms(
        pairs="sequential", looks=LOOKS, downsample=True,
        convolution="Gaussian", nan_aware=False,
    )
    legacy_igram = legacy_stack.ds["igram"].isel(pair=0).compute().values
    assert np.isfinite(legacy_igram).mean() < np.isfinite(igram).mean()

    for g in first + second:
        g.close()


def test_nan_aware_params_change_stage_hash(gslc_factory, tmp_path):
    """The workspace must not hand back a store built with other settings."""
    ws = Workspace(tmp_path / "ws")
    gslcs = [
        GSLC(gslc_factory(ny=60, nx=60, seed=i, datetime_str=t))
        for i, t in enumerate(
            ["2025-11-28T02:32:50.000000000", "2025-12-10T02:32:50.000000000"]
        )
    ]
    stack = GSLCStack.from_gslcs(gslcs)

    stack.form_interferograms(looks=LOOKS).persist(ws, "igrams")
    default_hash = ws.stored_params_hash("igrams")

    for changed in (dict(min_valid_fraction=0.9), dict(nan_aware=False)):
        stack.form_interferograms(looks=LOOKS, **changed).persist(
            ws, "igrams", overwrite=True
        )
        assert ws.stored_params_hash("igrams") != default_hash

    for g in gslcs:
        g.close()


def test_unwrap_masks_invalid_pixels(gslc_factory, tmp_path):
    """Invalid ground must stay identifiable after SNAPHU, not come back finite."""
    ny, nx = 80, 80
    gslcs = [
        GSLC(gslc_factory(ny=ny, nx=nx, seed=i, datetime_str=t))
        for i, t in enumerate(
            ["2025-11-28T02:32:50.000000000", "2025-12-10T02:32:50.000000000"]
        )
    ]
    stack = GSLCStack.from_gslcs(gslcs)
    igrams = stack.form_interferograms(looks=LOOKS, downsample=True)

    # Blank a corner of the multilooked interferogram.
    igram = igrams.ds["igram"].compute()
    igram[:, :4, :4] = np.nan
    igrams.ds["igram"] = igram

    ws = Workspace(tmp_path / "ws")
    unwrapped = igrams.unwrap(ws, name="unw")

    invalid = ~np.isfinite(igram.isel(pair=0).values)
    unw = unwrapped.ds["unw"].isel(pair=0).compute().values
    conncomp = unwrapped.ds["conncomp"].isel(pair=0).compute().values

    np.testing.assert_array_equal(np.isnan(unw), invalid)
    assert np.all(conncomp[invalid] == 0)
    assert np.isfinite(unw[~invalid]).all()

    for g in gslcs:
        g.close()


def test_crop_is_available_at_every_stage(gslc_factory, tmp_path):
    """crop() must work after interferogram formation, not only before it."""
    import pyproj

    ny, nx = 80, 80
    gslcs = [
        GSLC(gslc_factory(ny=ny, nx=nx, seed=i, datetime_str=t))
        for i, t in enumerate(
            ["2025-11-28T02:32:50.000000000", "2025-12-10T02:32:50.000000000"]
        )
    ]
    stack = GSLCStack.from_gslcs(gslcs)
    igrams = stack.form_interferograms(looks=LOOKS, downsample=True)

    # Ask for the middle half of the igram grid, in lon/lat.
    x, y = igrams.x, igrams.y
    tr = pyproj.Transformer.from_crs(
        f"EPSG:{igrams.epsg}", "EPSG:4326", always_xy=True
    )
    qx = [x[len(x) // 4], x[3 * len(x) // 4]]
    qy = [y[len(y) // 4], y[3 * len(y) // 4]]
    lons, lats = tr.transform(qx * 2, sorted(qy) * 2)

    cropped = igrams.crop(min(lons), max(lons), min(lats), max(lats))
    assert isinstance(cropped, InterferogramStack)
    assert 0 < cropped.sizes["x"] < igrams.sizes["x"]
    assert 0 < cropped.sizes["y"] < igrams.sizes["y"]
    assert cropped.sizes["pair"] == igrams.sizes["pair"]
    assert cropped.epsg == igrams.epsg
    assert cropped.ds.attrs["looks"] == igrams.ds.attrs["looks"]

    ws = Workspace(tmp_path / "ws")
    unwrapped = igrams.unwrap(ws, name="unw")
    unw_cropped = unwrapped.crop(min(lons), max(lons), min(lats), max(lats))
    assert type(unw_cropped) is type(unwrapped)
    assert unw_cropped.sizes["x"] == cropped.sizes["x"]
    assert unw_cropped.sizes["y"] == cropped.sizes["y"]

    for g in gslcs:
        g.close()
