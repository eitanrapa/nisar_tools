"""Tests for dropping the regions the mainland cannot reach.

An unwrapper propagates phase along arcs between adjacent pixels, so a region
with no path to the main body carries no recoverable ambiguity however large it
is -- the criterion is connectivity, not size. On the real coastal frame that
motivated this, a 4-connected labelling found 29 components: one mainland of
36,018,513 px and 28 strays of 16..1530 px totalling 6,369 px (0.018% of valid).
The distribution is bimodal with a four-order-of-magnitude gap, which is why
keep-the-largest is unambiguous there and why a size threshold was the wrong
instrument (a threshold of 32 would have removed 4 components and 80 pixels).
"""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import InterferogramStack, UnwrappedStack, Workspace
from nisar_tools._kernels import (
    halo_planes,
    remove_unconnected_regions,
    remove_unconnected_regions_planes,
    unconnected_regions_depth,
)


def _two_bodies_and_speckle(ny=40, nx=40):
    """A 324-px mainland, a 36-px second body, and two 1-px specks."""
    field = np.full((ny, nx), np.nan, np.float32)
    field[2:20, 2:20] = 1.0     # 18*18 = 324
    field[25:31, 25:31] = 2.0   # 6*6 = 36
    field[35, 35] = 3.0
    field[37, 5] = 4.0
    return field


# -- the kernel ------------------------------------------------------------

def test_keep_largest_drops_every_other_component():
    field = _two_bodies_and_speckle()
    assert np.isfinite(field).sum() == 324 + 36 + 1 + 1

    out = remove_unconnected_regions(field, max_drop_fraction=1.0)

    assert np.isfinite(out).sum() == 324
    # The mainland survives untouched, values and all.
    mainland = np.isfinite(field) & (field == 1.0)
    np.testing.assert_array_equal(out[mainland], field[mainland])


def test_min_size_keeps_both_real_bodies_and_drops_the_speckle():
    """The escape hatch for a scene that genuinely is two landmasses."""
    field = _two_bodies_and_speckle()

    out = remove_unconnected_regions(field, min_size=10, max_drop_fraction=1.0)

    assert np.isfinite(out).sum() == 324 + 36


@pytest.mark.parametrize("min_size, expected", [
    (35, 324 + 36),  # strictly greater than min_size survives
    (36, 324),       # a component of exactly min_size is dropped
])
def test_min_size_boundary_is_strictly_greater(min_size, expected):
    field = _two_bodies_and_speckle()
    out = remove_unconnected_regions(field, min_size=min_size,
                                     max_drop_fraction=1.0)
    assert np.isfinite(out).sum() == expected


def test_connectivity_decides_whether_a_diagonal_touch_connects():
    """4-connected is the default because a diagonal is not an arc for the
    solver, so a diagonally-attached region is as unreachable as a detached one."""
    field = np.full((10, 10), np.nan, np.float32)
    field[1:5, 1:5] = 1.0
    field[5, 5] = 2.0  # touches the block corner-to-corner only

    four = remove_unconnected_regions(field, max_drop_fraction=1.0)
    eight = remove_unconnected_regions(field, connectivity=2,
                                       max_drop_fraction=1.0)

    assert np.isfinite(four).sum() == 16   # the diagonal pixel is its own region
    assert np.isfinite(eight).sum() == 17  # ... and here it is part of the body


def test_single_component_is_an_exact_no_op():
    field = np.arange(25, dtype=np.float32).reshape(5, 5)
    np.testing.assert_array_equal(remove_unconnected_regions(field), field)


def test_a_lone_component_below_min_size_is_still_dropped():
    """"Only one component" short-circuits keep-largest, where the sole component
    is trivially the largest -- but under ``min_size`` it still has to clear the
    threshold. Under a halo each block sees only its own fragment, so a shortcut
    taken here silently keeps everything (it did, until this test)."""
    field = np.full((20, 20), np.nan, np.float32)
    field[5:8, 5:8] = 1.0  # a single 9-px component

    assert np.isfinite(
        remove_unconnected_regions(field, max_drop_fraction=1.0)
    ).sum() == 9
    assert np.isfinite(
        remove_unconnected_regions(field, min_size=100, max_drop_fraction=1.0)
    ).sum() == 0


def test_all_nan_plane_does_not_raise():
    field = np.full((5, 5), np.nan, np.float32)
    out = remove_unconnected_regions(field)
    assert np.isnan(out).all()


def test_rejects_unknown_connectivity():
    with pytest.raises(ValueError, match="connectivity must be 1 or 2"):
        remove_unconnected_regions(_two_bodies_and_speckle(), connectivity=3)


# -- the safeguard ---------------------------------------------------------

def test_guard_refuses_to_blank_a_second_landmass():
    field = _two_bodies_and_speckle()  # second body is 36/362 = 9.9% of valid

    with pytest.raises(ValueError, match="max_drop_fraction") as excinfo:
        remove_unconnected_regions(field)

    message = str(excinfo.value)
    assert "36 pixels" in message      # names the offending component
    assert "9.94%" in message          # ... and its share
    assert "min_size=" in message      # ... and both ways out
    assert "max_drop_fraction=" in message


@pytest.mark.parametrize("kwargs", [
    {"max_drop_fraction": 1.0},   # confirm the removal
    {"min_size": 10},             # or keep both bodies
    {"max_drop_fraction": None},  # or disable the guard outright
])
def test_guard_can_be_satisfied_either_way(kwargs):
    remove_unconnected_regions(_two_bodies_and_speckle(), **kwargs)


def test_guard_ignores_a_large_total_spread_over_tiny_components():
    """The guard is on the largest *single* dropped component, not the total.

    A scene shredded into thousands of specks has a large total but no large
    component, and those specks are exactly what this function is for -- a
    total-based guard would fire on the case it exists to handle.
    """
    field = np.full((200, 200), np.nan, np.float32)
    field[0:60, 0:60] = 1.0  # 3600 px mainland
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(70, 199, 1500), rng.integers(70, 199, 1500)
    field[ys, xs] = 1.0

    speckle = int(np.isfinite(field).sum()) - 3600
    assert speckle / np.isfinite(field).sum() > 0.20  # far above the 1% default

    out = remove_unconnected_regions(field)  # must not raise
    assert np.isfinite(out).sum() == 3600


# -- chunk decomposition ---------------------------------------------------
#
# min_size=N is chunkable with a halo of exactly N; keep-largest is a global
# property and cannot be haloed at all.

def _seamed_field(seed=0):
    """A mainland plus blobs parked on the 32-px chunk seams."""
    rng = np.random.default_rng(seed)
    field = np.full((128, 128), np.nan, np.float32)
    field[10:100, 5:60] = rng.normal(size=(90, 55))
    for _ in range(40):
        cy = int(rng.choice([32, 64, 96]) + rng.integers(-3, 4))
        cx = int(rng.choice([64, 96]) + rng.integers(-3, 4))
        h, w = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        field[cy:cy + h, cx:cx + w] = rng.normal(size=field[cy:cy + h,
                                                            cx:cx + w].shape)
    return field


@pytest.mark.parametrize("min_size", [1, 4, 16, 32])
@pytest.mark.parametrize("chunks", [(32, 32), (17, 23), (128, 128)])
@pytest.mark.parametrize("connectivity", [1, 2])
def test_min_size_halo_is_exact(min_size, chunks, connectivity):
    """A component of <= N pixels has diameter <= N-1, so an N-pixel halo sees
    all of it; one that reaches the padded edge spanned N+1 getting there and is
    correctly kept. So the chunked answer is the whole-plane answer exactly."""
    da = pytest.importorskip("dask.array")
    field = _seamed_field()
    kwargs = dict(min_size=min_size, connectivity=connectivity,
                  max_drop_fraction=1.0)

    ref = remove_unconnected_regions(field, **kwargs)
    got = halo_planes(
        remove_unconnected_regions, da.from_array(field, chunks=chunks),
        depth=unconnected_regions_depth(min_size, *field.shape), **kwargs,
    ).compute()

    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_array_equal(got[np.isfinite(ref)], ref[np.isfinite(ref)])


def test_a_smaller_halo_really_does_break_it():
    """Pins that the halo is load-bearing rather than gratuitous: without it a
    component larger than the threshold looks small inside a single block."""
    da = pytest.importorskip("dask.array")
    field = np.full((80, 80), np.nan, np.float32)
    field[38:42, 20:60] = 1.0  # a 160-px bar straddling the 40-px seam
    kwargs = dict(min_size=100, max_drop_fraction=1.0)

    ref = remove_unconnected_regions(field, **kwargs)
    assert np.isfinite(ref).sum() == 160  # 160 > 100, so whole-plane keeps it

    arr = da.from_array(field, chunks=(40, 40))
    starved = halo_planes(remove_unconnected_regions, arr, depth=1,
                          **kwargs).compute()
    assert np.isfinite(starved).sum() == 0  # each block sees only a fragment

    ok = halo_planes(remove_unconnected_regions, arr,
                     depth=unconnected_regions_depth(100, *field.shape),
                     **kwargs).compute()
    np.testing.assert_array_equal(np.isnan(ok), np.isnan(ref))


def test_keep_largest_depth_spans_the_raster():
    """Which component is largest is global, so there is no halo that works --
    the depth has to collapse ``halo_planes`` to one whole plane per task."""
    assert unconnected_regions_depth(None, 120, 140) >= 140
    assert unconnected_regions_depth(32, 120, 140) == 32


def test_planes_wrapper_handles_a_stacked_axis():
    field = np.stack([_two_bodies_and_speckle(), _two_bodies_and_speckle()[::-1]])
    out = remove_unconnected_regions_planes(field, max_drop_fraction=1.0)
    assert out.shape == field.shape
    assert [int(np.isfinite(p).sum()) for p in out] == [324, 324]


# -- the stage methods -----------------------------------------------------

def _igram_stack(chunks=(1, 32, 32), ny=120, nx=140):
    da = pytest.importorskip("dask.array")
    igram = np.full((2, ny, nx), np.nan, np.complex64)
    igram[:, 5:80, 5:90] = 1 + 1j        # 75*85 = 6375 px mainland
    igram[:, 100:104, 120:124] = 2 + 0j  # 16 px stray
    igram[:, 110, 130] = 3 + 0j          # 1 px stray
    coherence = np.where(np.isfinite(igram.real), 0.8, 0.0).astype(np.float32)
    ds = xr.Dataset(
        {
            "igram": (("pair", "y", "x"), da.from_array(igram, chunks=chunks)),
            "coherence": (("pair", "y", "x"),
                          da.from_array(coherence, chunks=chunks)),
        },
        coords={"pair": [0, 1], "y": np.arange(ny) * 50.0,
                "x": np.arange(nx) * 50.0},
        attrs={"epsg": 32611, "looks": 10, "pairs": [[0, 1]]},
    )
    return InterferogramStack(ds)


@pytest.mark.parametrize("chunks", [(1, 120, 140), (1, 32, 32), (1, 17, 23)])
def test_interferogram_method_is_chunk_independent(chunks):
    out = _igram_stack(chunks).remove_unconnected_regions()
    igram = out.ds["igram"].isel(pair=0).values
    assert np.isfinite(igram.real).sum() == 6375


def test_interferogram_method_masks_coherence_with_the_igram():
    """Both variables have to follow one footprint, as ``mask_water`` does --
    coherence carries no NaN of its own (it is exactly 0.0 outside the swath),
    so it follows the igram rather than being labelled separately."""
    out = _igram_stack().remove_unconnected_regions()
    igram = out.ds["igram"].isel(pair=0).values
    coherence = out.ds["coherence"].isel(pair=0).values
    np.testing.assert_array_equal(np.isfinite(igram.real), np.isfinite(coherence))


def test_keep_largest_collapses_to_whole_planes_but_min_size_does_not():
    stack = _igram_stack(chunks=(1, 32, 32))
    largest = stack.remove_unconnected_regions()
    sized = stack.remove_unconnected_regions(min_size=4, max_drop_fraction=1.0)

    assert largest.ds["igram"].chunks[1:] == ((120,), (140,))
    assert len(sized.ds["igram"].chunks[1]) > 1  # still decomposed


def test_the_method_is_lazy_and_the_guard_raises_at_compute_time():
    """An eager check would cost a whole extra pass over the stack, so the guard
    lives in the kernel and surfaces when the graph runs."""
    stack = _igram_stack().remove_unconnected_regions(max_drop_fraction=0.001)
    with pytest.raises(ValueError, match="max_drop_fraction"):
        stack.ds["igram"].isel(pair=0).compute()


def test_unwrapped_method_drops_unreachable_regions():
    da = pytest.importorskip("dask.array")
    ny, nx = 120, 140
    unw = np.full((2, ny, nx), np.nan, np.float32)
    unw[:, 5:80, 5:90] = 3.0
    unw[:, 100:104, 120:124] = 7.0
    ds = xr.Dataset(
        {
            "unw": (("pair", "y", "x"), da.from_array(unw, chunks=(1, 32, 32))),
            "conncomp": (("pair", "y", "x"),
                         da.from_array(np.ones((2, ny, nx), np.uint32),
                                       chunks=(1, 32, 32))),
        },
        coords={"pair": [0, 1], "y": np.arange(ny) * 50.0,
                "x": np.arange(nx) * 50.0},
        attrs={"epsg": 32611, "source": "snaphu"},
    )
    out = UnwrappedStack(ds).remove_unconnected_regions()
    assert np.isfinite(out.ds["unw"].isel(pair=0).values).sum() == 6375


# -- provenance ------------------------------------------------------------

def test_params_reach_the_attrs():
    out = _igram_stack().remove_unconnected_regions(
        min_size=4, connectivity=2, max_drop_fraction=0.5
    )
    assert out.ds.attrs["unconnected_removed"] == {
        "min_size": 4, "connectivity": 2, "max_drop_fraction": 0.5,
    }


def test_untouched_stack_keeps_its_persist_hash(tmp_path):
    """Backward compatibility: the provenance is folded into the hash only once
    applied, so every existing igrams store stays readable."""
    ws = Workspace(tmp_path)
    stack = _igram_stack()
    assert "unconnected_removed" not in stack.ds.attrs

    stack.persist(ws, "igrams")
    before = sorted(p.name for p in tmp_path.iterdir())

    # Re-persisting the same untouched stack must land on the same store.
    stack.persist(ws, "igrams")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_cleaning_changes_the_persist_hash(tmp_path):
    """A cleaned stack must not silently reuse the untouched store. The stage
    name is fixed, so a changed hash surfaces as a refusal to overwrite -- loud,
    which is the contract the rest of the workspace relies on."""
    from nisar_tools.workspace import WorkspaceError

    ws = Workspace(tmp_path)
    _igram_stack().persist(ws, "igrams")

    with pytest.raises(WorkspaceError, match="different parameters"):
        _igram_stack().remove_unconnected_regions().persist(ws, "igrams")
