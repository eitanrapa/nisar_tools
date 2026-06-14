"""Equivalence tests: dask kernels vs verbatim numpy vs the legacy module.

These are the load-bearing tests for the refactor. If the lazy ``map_overlap``
multilook matches the original global-filter result across chunk layouts, the
rest of the pipeline can rely on the kernels being correct.
"""

import dask.array as da
import numpy as np
import pytest

import legacy_reference as legacy  # the original procedural module, as oracle
from nisar_tools import _kernels


CONVOLUTIONS = ["Uniform", "Gaussian"]
DOWNSAMPLE = [False, True]
# Include a non-square, non-multiple-of-looks shape to exercise truncation.
SHAPES = [(64, 64), (70, 53)]


def _max_xy(ny, nx, looks):
    return nx // looks * looks, ny // looks * looks


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
@pytest.mark.parametrize("downsample", DOWNSAMPLE)
@pytest.mark.parametrize("shape", SHAPES)
def test_multilook_numpy_matches_legacy(convolution, downsample, shape):
    rng = np.random.default_rng(0)
    ny, nx = shape
    arr = rng.standard_normal(shape).astype(np.float64)
    max_x, max_y = _max_xy(ny, nx, 5)

    out = _kernels.multilook(arr, max_x, max_y, 5, downsample, convolution)
    ref = legacy._multilook_array(arr, max_x, max_y, 5, downsample, convolution)

    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
@pytest.mark.parametrize("downsample", DOWNSAMPLE)
@pytest.mark.parametrize("shape", SHAPES)
def test_multilook_dask_matches_numpy_real(convolution, downsample, shape):
    rng = np.random.default_rng(1)
    ny, nx = shape
    arr = rng.standard_normal(shape).astype(np.float64)
    max_x, max_y = _max_xy(ny, nx, 5)

    # Deliberately tiny chunks so many chunk seams fall inside the filter reach.
    darr = da.from_array(arr, chunks=(16, 16))

    out = _kernels.multilook_dask(darr, max_x, max_y, 5, downsample, convolution).compute()
    ref = _kernels.multilook(arr, max_x, max_y, 5, downsample, convolution)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
@pytest.mark.parametrize("downsample", DOWNSAMPLE)
@pytest.mark.parametrize("shape", SHAPES)
def test_igram_coherence_dask_matches_legacy(convolution, downsample, shape):
    rng = np.random.default_rng(2)
    ny, nx = shape
    c1 = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    c2 = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    max_x, max_y = _max_xy(ny, nx, 5)

    igram_ref, corr_ref = legacy._calculate_multilooked_interferograms(
        c1, c2, max_x, max_y, 5, downsample, convolution
    )

    d1 = da.from_array(c1, chunks=(16, 16))
    d2 = da.from_array(c2, chunks=(16, 16))
    igram_d, corr_d = _kernels.igram_coherence(d1, d2, max_x, max_y, 5, downsample, convolution)
    igram_d = np.asarray(igram_d)
    corr_d = np.asarray(corr_d)

    np.testing.assert_allclose(igram_d, igram_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(corr_d, corr_ref, rtol=1e-4, atol=1e-4)
    assert corr_d.dtype == np.float32


@pytest.mark.parametrize("convolution", CONVOLUTIONS)
@pytest.mark.parametrize("downsample", DOWNSAMPLE)
def test_multilook_dask_3d_filters_only_spatial(convolution, downsample):
    # A stacked (pair, y, x) array must be filtered slice-by-slice: each slice
    # must match the 2D result, with no mixing across the leading axis.
    rng = np.random.default_rng(7)
    npair, ny, nx = 3, 40, 32
    arr = rng.standard_normal((npair, ny, nx))
    max_x, max_y = _max_xy(ny, nx, 5)

    darr = da.from_array(arr, chunks=(npair, 16, 16))  # whole stack in one chunk
    out = _kernels.multilook_dask(darr, max_x, max_y, 5, downsample, convolution).compute()

    for k in range(npair):
        ref = _kernels.multilook(arr[k], max_x, max_y, 5, downsample, convolution)
        np.testing.assert_allclose(out[k], ref, rtol=1e-6, atol=1e-6)


def test_snaphu_params_matches_legacy():
    for shape in [(1000, 4000), (4000, 1000), (512, 512)]:
        for nproc in [1, 4, 40]:
            assert _kernels.snaphu_params(shape, nproc) == legacy._calculate_snaphu_params(
                shape, nproc
            )


def test_snaphu_nlooks_matches_legacy():
    assert _kernels.snaphu_nlooks(5, 5, 10.0, 10.0, 8, 3) == legacy._calculate_snaphu_nlooks(
        5, 5, 10.0, 10.0, 8, 3
    )
