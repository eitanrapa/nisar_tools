"""Water and swath-edge masking on the SLC stage.

Both operations already existed downstream (``InterferogramStack.mask_water``,
``UnwrappedStack.mask_edges``); doing them before multilooking is what keeps the
blanked samples out of the *estimate* as well as the product, since
``form_interferograms`` is NaN-aware.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLC, GSLCStack, InterferogramStack, Workspace


def _sheared_swath(ny, nx, seed=0):
    """A GSLC-like plane: valid data in a swath that shears across the rows."""
    rng = np.random.default_rng(seed)
    data = (rng.uniform(0.5, 1.5, (ny, nx))
            * np.exp(1j * rng.uniform(-np.pi, np.pi, (ny, nx)))).astype(np.complex64)
    for i in range(ny):
        lo = 5 + i // 6
        data[i, :lo] = np.nan
        data[i, nx - lo:] = np.nan
    return data


@pytest.fixture
def swath_stack(gslc_factory):
    """A two-date stack on a 500 m grid, with a sheared valid swath."""
    data = _sheared_swath(60, 80)
    paths = [
        gslc_factory(ny=60, nx=80, dx=500.0, dy=500.0, data=data,
                     datetime_str=f"2024-01-0{k + 1}T00:00:00.000000000")
        for k in range(2)
    ]
    return GSLCStack.from_gslcs([GSLC(p) for p in paths])


def _valid(stack, t=0):
    return np.isfinite(stack.ds["slc"].isel(time=t).compute().values)


# -- edges --------------------------------------------------------------------
def test_mask_edges_trims_only_the_range_edges(swath_stack):
    """Each row crosses the swath once, so its first and last valid samples are
    the two range boundaries -- ``edge_pixels`` off each, and nothing else."""
    before = _valid(swath_stack)
    after = _valid(swath_stack.mask_edges(edge_pixels=3))

    assert (after <= before).all()  # only ever removes
    # Every row here is non-empty, so every row loses exactly 2 * edge_pixels.
    np.testing.assert_array_equal(
        before.sum(axis=1) - after.sum(axis=1), np.full(60, 6)
    )
    # The azimuth ends are *not* range edges, so the first and last rows keep
    # their interior -- an isotropic erosion would have eaten into them.
    assert after[0].sum() > 0 and after[-1].sum() > 0


def test_mask_edges_all_erodes_more_than_along_track(swath_stack):
    along = _valid(swath_stack.mask_edges(edge_pixels=3))
    everywhere = _valid(swath_stack.mask_edges(edge_pixels=3, edges="all"))
    assert everywhere.sum() < along.sum()
    assert (everywhere <= along).all()
    # The isotropic erosion reaches the frame's azimuth ends; the default does not.
    assert everywhere[:3].sum() == 0 and along[0].sum() > 0


def test_mask_edges_zero_is_a_no_op(swath_stack):
    before = swath_stack.ds["slc"].isel(time=0).compute().values
    after = swath_stack.mask_edges(edge_pixels=0).ds["slc"].isel(time=0).compute().values
    np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
    finite = np.isfinite(before)
    np.testing.assert_array_equal(before[finite], after[finite])


@pytest.mark.parametrize("edges", ["along_track", "all"])
def test_mask_edges_dask_matches_numpy(swath_stack, edges):
    """The row split (along_track) and the halo (all) are both exact, so the
    chunked result has to equal the whole-plane one bit for bit."""
    lazy = _valid(swath_stack.mask_edges(edge_pixels=3, edges=edges, target_blocks=8))
    eager = np.isfinite(
        GSLCStack(swath_stack.ds.compute())
        .mask_edges(edge_pixels=3, edges=edges)
        .ds["slc"].isel(time=0).values
    )
    np.testing.assert_array_equal(lazy, eager)


def test_mask_edges_masks_complex_to_nan(swath_stack):
    """Complex data must land on the same nan+0j the unwritten out-of-swath
    chunks of a real GSLC already decode to, or downstream sees a new gap kind."""
    out = swath_stack.mask_edges(edge_pixels=3).ds["slc"].isel(time=0).compute()
    assert out.dtype == np.complex64
    dropped = out.values[~_valid(swath_stack.mask_edges(edge_pixels=3))]
    assert np.isnan(dropped.real).all() and (dropped.imag == 0).all()


def test_mask_edges_rejects_an_unknown_mode(swath_stack):
    with pytest.raises(ValueError, match="along_track"):
        swath_stack.mask_edges(edges="sideways")


def test_mask_edges_feeds_the_nan_aware_multilook(swath_stack):
    """The point of masking here rather than later: the trimmed samples are gone
    before the multilook window averages them into igram *and* coherence."""
    plain = swath_stack.form_interferograms(looks=5)
    trimmed = swath_stack.mask_edges(edge_pixels=5).form_interferograms(looks=5)

    keep_plain = np.isfinite(plain.ds["igram"].compute().values)
    keep_trim = np.isfinite(trimmed.ds["igram"].compute().values)
    assert keep_trim.sum() < keep_plain.sum()
    assert (keep_trim <= keep_plain).all()
    # Same grid either way -- masking changes values, never the lattice.
    np.testing.assert_array_equal(plain.x, trimmed.x)
    np.testing.assert_array_equal(plain.y, trimmed.y)


# -- edges, one stage later ---------------------------------------------------
def test_igram_mask_edges_puts_both_layers_on_one_footprint(swath_stack):
    """Coherence is exactly 0.0 outside the swath, not NaN, so it cannot be
    edge-scanned itself -- it has to follow the igram's footprint."""
    igrams = swath_stack.form_interferograms(looks=5)
    trimmed = igrams.mask_edges(edge_pixels=2)

    ig = trimmed.ds["igram"].isel(pair=0).compute().values
    coh = trimmed.ds["coherence"].isel(pair=0).compute().values
    np.testing.assert_array_equal(np.isfinite(ig), np.isfinite(coh))

    before = np.isfinite(igrams.ds["igram"].isel(pair=0).compute().values)
    assert 0 < np.isfinite(ig).sum() < before.sum()
    assert (np.isfinite(ig) <= before).all()


def test_igram_mask_edges_scans_the_igram_not_the_coherence(swath_stack):
    """The trap this guards: coherence's out-of-swath 0.0 is *finite*, so a row
    scan over it would call column 0 the near-range edge on every row and trim
    the raster's own border instead of the swath's."""
    igrams = swath_stack.form_interferograms(looks=5)
    coh = igrams.ds["coherence"].isel(pair=0).compute().values
    ig = igrams.ds["igram"].isel(pair=0).compute().values
    # Precondition: the two layers really do disagree about what is valid.
    assert np.isfinite(coh).all() and not np.isfinite(ig).all()

    kept = np.isfinite(
        igrams.mask_edges(edge_pixels=2).ds["igram"].isel(pair=0).compute().values
    )
    # A row whose data starts well inside the raster must lose samples at its
    # *swath* edge, leaving the columns outside the swath as they were.
    row = np.isfinite(ig).sum(axis=1).argmax()
    first = np.isfinite(ig)[row].argmax()
    assert first > 0                     # the swath does not reach column 0
    assert not kept[row, first]          # trimmed at the swath edge
    assert not kept[row, first - 1]      # still out of swath, unchanged


def test_igram_mask_edges_min_coherence(swath_stack):
    igrams = swath_stack.form_interferograms(looks=5)
    coh = igrams.ds["coherence"].isel(pair=0).compute().values
    thr = float(np.nanmedian(coh[np.isfinite(coh)]))

    out = igrams.mask_edges(edge_pixels=0, min_coherence=thr)
    ig = out.ds["igram"].isel(pair=0).compute().values
    kept = np.isfinite(ig)
    assert kept.sum() < np.isfinite(igrams.ds["igram"].isel(pair=0).compute()).sum()
    # Nothing below the threshold survives, in either layer.
    assert (coh[kept] >= thr).all()
    np.testing.assert_array_equal(
        kept, np.isfinite(out.ds["coherence"].isel(pair=0).compute().values)
    )


@pytest.mark.parametrize("edges", ["along_track", "all"])
def test_igram_mask_edges_dask_matches_numpy(swath_stack, edges):
    igrams = swath_stack.form_interferograms(looks=5)
    lazy = np.isfinite(
        igrams.mask_edges(edge_pixels=2, edges=edges, target_blocks=8)
        .ds["igram"].isel(pair=0).compute().values
    )
    eager = np.isfinite(
        InterferogramStack(igrams.ds.compute())
        .mask_edges(edge_pixels=2, edges=edges)
        .ds["igram"].isel(pair=0).values
    )
    np.testing.assert_array_equal(lazy, eager)


def test_igram_mask_edges_reaches_the_stage_hash(swath_stack, tmp_path):
    from nisar_tools.workspace import hash_params

    ws = Workspace(tmp_path / "ws")
    igrams = swath_stack.form_interferograms(looks=5)
    igrams.persist(ws, "igrams")
    igrams.mask_edges(edge_pixels=2).persist(ws, "igrams_trimmed")
    assert ws.stored_params_hash("igrams") != ws.stored_params_hash("igrams_trimmed")

    # An untrimmed stack keeps the exact hash it had before this existed.
    attrs = igrams.ds.attrs
    assert ws.stored_params_hash("igrams") == hash_params({
        "stage": "igrams", "epsg": igrams.epsg,
        **{k: attrs.get(k) for k in ("looks", "downsample", "align_looks",
                                     "convolution", "nan_aware",
                                     "min_valid_fraction", "pairs")},
    })


def test_igram_mask_edges_rejects_an_unknown_mode(swath_stack):
    with pytest.raises(ValueError, match="along_track"):
        swath_stack.form_interferograms(looks=5).mask_edges(edges="sideways")


# -- water --------------------------------------------------------------------
@pytest.fixture
def fake_coastline(monkeypatch):
    """Stub GMT: the western third of the requested region is water."""
    pygmt = pytest.importorskip("pygmt")

    def fake_grdlandmask(region, spacing, maskvalues, resolution, registration):
        lon = np.linspace(region[0], region[1], 60)
        lat = np.linspace(region[2], region[3], 50)
        data = np.ones((50, 60))
        data[:, :20] = np.nan
        return xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))

    monkeypatch.setattr(pygmt, "grdlandmask", fake_grdlandmask)


def test_mask_water_blanks_water_on_the_slc(swath_stack, fake_coastline):
    before = _valid(swath_stack)
    after = _valid(swath_stack.mask_water(resolution="i"))
    assert 0 < after.sum() < before.sum()   # some water blanked, some land kept
    assert (after <= before).all()          # and the swath's own NaN is untouched


def test_mask_water_spacing_tracks_the_native_pixel(swath_stack, fake_coastline):
    """Recorded resolved, not as a placeholder None -- it feeds the stage hash.

    500 m pixels, 2x oversampled. At a real GSLC's 5 m posting this would read
    "2.5e", which is what ``make_water_mask`` warns about.
    """
    masked = swath_stack.mask_water(resolution="i")
    assert masked.ds.attrs["water_mask"] == {"resolution": "i", "spacing": "250e"}


def test_mask_water_is_lazy(swath_stack, fake_coastline, tmp_path):
    """``mask_cache`` caches the coastline, not the masked data."""
    ws = Workspace(tmp_path / "ws")
    masked = swath_stack.mask_water(mask_cache=ws, resolution="i")
    cached = [p.name for p in ws.workdir.iterdir() if p.name.startswith("water_mask")]
    assert len(cached) == 1
    # Nothing of the stack itself was written.
    assert not ws.exists("slc_stack")


def test_masks_apply_to_a_zarr_reloaded_stack(swath_stack, fake_coastline, tmp_path):
    """Zarr has no coord/variable distinction, so ``spatial_ref`` comes back as a
    *data variable*. The mask's own rio bookkeeping then collides with it unless
    it has been stripped -- the same MergeError the UnwrappedStack path hit."""
    ws = Workspace(tmp_path / "ws")
    swath_stack.persist(ws, "slc_stack")
    reloaded = GSLCStack.from_zarr(ws.path("slc_stack"))
    assert "spatial_ref" in reloaded.ds.coords or "spatial_ref" in reloaded.ds.data_vars

    masked = reloaded.mask_water(resolution="i").mask_edges(edge_pixels=3)
    valid = np.isfinite(masked.ds["slc"].isel(time=0).compute().values)
    assert 0 < valid.sum() < _valid(swath_stack).sum()


def test_masks_compose_and_both_reach_the_stage_hash(swath_stack, fake_coastline,
                                                     tmp_path):
    ws = Workspace(tmp_path / "ws")
    both = swath_stack.mask_water(resolution="i").mask_edges(edge_pixels=3)
    assert "water_mask" in both.ds.attrs and "edges_masked" in both.ds.attrs

    swath_stack.persist(ws, "plain")
    both.persist(ws, "both")
    assert ws.stored_params_hash("plain") != ws.stored_params_hash("both")

    # Round-trips: the masked values are what came back.
    back = GSLCStack.from_zarr(ws.path("both"))
    np.testing.assert_array_equal(
        np.isfinite(back.ds["slc"].compute().values),
        np.isfinite(both.ds["slc"].compute().values),
    )


def test_an_unmasked_stack_keeps_the_hash_it_had_before(swath_stack, tmp_path):
    """The new keys are recorded only once masked, so existing slc_stack stores
    are not invalidated by this feature merely existing."""
    from nisar_tools.workspace import hash_params

    ws = Workspace(tmp_path / "ws")
    swath_stack.persist(ws, "slc_stack")
    # Exactly the params dict persist() built before water_mask/edges_masked.
    assert ws.stored_params_hash("slc_stack") == hash_params(
        {"stage": "slc_stack", "epsg": swath_stack.epsg}
    )


# -- the native-resolution guard ----------------------------------------------
def test_the_guard_fires_at_a_native_posting_and_not_a_multilooked_one():
    """A 28 km crop off the Venezuelan coast: 2.7e5 nodes multilooked at 150 m,
    2.4e8 at the native 5 m. The budget has to separate those two."""
    from nisar_tools.mask import _warn_if_oversized, grid_spacing_arg

    for pixel, should_warn in ((150.0, False), (50.0, False), (5.0, True)):
        n = int(28_000 / pixel)
        x = 500_000.0 + pixel * np.arange(n)
        y = 1_160_000.0 - pixel * np.arange(n)
        spacing = grid_spacing_arg(x, y, 32619)
        # The lon/lat region make_water_mask derives, buffer included.
        region = [-67.0 - 0.05, -66.74 + 0.05, 10.45 - 0.05, 10.71 + 0.05]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_oversized(region, spacing)
        assert bool(caught) is should_warn, (pixel, spacing,
                                             [str(w.message) for w in caught])

    # An increment whose unit cannot be priced is left alone, not guessed at.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_oversized([-67.0, -66.0, 10.0, 11.0], "1+n")
    assert not caught


def test_make_water_mask_emits_the_guard(monkeypatch):
    """Wiring check: the guard is reached on the real call path.

    The increment is passed explicitly so the *target* grid stays 40x40 -- what
    makes a native-resolution mask expensive is the node count, not the stack.
    """
    from nisar_tools import mask as mask_mod

    pygmt = pytest.importorskip("pygmt")
    monkeypatch.setattr(
        pygmt, "grdlandmask",
        lambda region, spacing, maskvalues, resolution, registration: xr.DataArray(
            np.ones((8, 8)),
            coords={"y": np.linspace(region[2], region[3], 8),
                    "x": np.linspace(region[0], region[1], 8)},
            dims=("y", "x"),
        ),
    )
    x = 500_000.0 + 500.0 * np.arange(40)
    y = 1_160_000.0 - 500.0 * np.arange(40)

    with pytest.warns(RuntimeWarning, match="GMT nodes"):
        mask_mod.make_water_mask(x, y, 32619, resolution="i", spacing="0.5e")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask_mod.make_water_mask(x, y, 32619, resolution="i", spacing="250e")
    assert not [w for w in caught if "GMT nodes" in str(w.message)]


def test_increment_parsing_covers_gmt_unit_suffixes():
    """``e`` is metres, ``s`` arc-seconds, bare is degrees -- the same suffix
    confusion that once made the mask 639x too slow."""
    from nisar_tools.mask import _parse_increment

    assert _parse_increment("250e") == (250.0, "m")
    assert _parse_increment("2.5e") == (2.5, "m")
    assert _parse_increment("0.001") == (0.001, "deg")
    value, unit = _parse_increment("36s")
    assert unit == "deg" and value == pytest.approx(0.01)
    # Units this cannot price are left alone rather than guessed at.
    assert _parse_increment("5k") is None
    assert _parse_increment("100+n") is None
