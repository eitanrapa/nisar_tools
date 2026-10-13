"""The numpy port of ISCE3's PyCuAmpcor (``nisar_tools.ampcor``).

Each stage is checked against a direct evaluation of what the CUDA kernel computes, and the
whole correlator against a known sub-pixel shift of band-limited speckle. The end-to-end
agreement with a real NISAR ROFF (99.7-99.9% of windows bit-identical in pixels, the rest
one 1/64-pixel step apart) is in the docstring of ``nisar_tools.roff``; it needs the granules.
"""

import numpy as np
import pytest

from nisar_tools import ampcor as ac
from nisar_tools.roff import GOFF_LAYERS, OffsetLattice


def test_xcorr_is_the_valid_correlation():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((3, 8, 6)).astype(np.float32)
    s = rng.standard_normal((3, 15, 11)).astype(np.float32)
    c = ac._xcorr(t, s, 1)
    assert c.shape == (3, 8, 6)
    direct = np.array([[[np.sum(t[n] * s[n, u:u + 8, v:v + 6]) for v in range(6)]
                        for u in range(8)] for n in range(3)])
    np.testing.assert_allclose(c, direct, rtol=2e-5, atol=2e-5)


def test_normalise_is_the_ncc():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((2, 8, 8)).astype(np.float32)
    t -= t.mean(axis=(1, 2), keepdims=True)
    s = (rng.standard_normal((2, 14, 14)) + 3.0).astype(np.float32)
    corr = ac._normalise(ac._xcorr(t, s, 1), t, s)
    for n in range(2):
        for u in range(7):
            for v in range(7):
                w = s[n, u:u + 8, v:v + 8].astype(float)
                ncc = np.sum(t[n] * w) / np.sqrt(np.sum(t[n] ** 2) * np.sum((w - w.mean()) ** 2))
                assert corr[n, u, v] == pytest.approx(ncc, abs=1e-5)


def test_adjust_offset_follows_dev_adjustOffset():
    # cuOffset.cu's examples (old range 32, new range 4); note the CODE gives shift +2 at the
    # left edge where its comment says -2 -- the port keeps the code.
    start, shift = ac._adjust(np.array([40, 2, 64]), 32, 4)
    assert start.tolist() == [36, 0, 56]
    assert shift.tolist() == [0, 2, 4]


def test_argmax_tie_rule_is_the_cuda_reduction():
    v = np.zeros((1, 1200), np.float32)
    v[0, [2, 513, 1100]] = 1.0            # gid % 512 = 2, 1, 76: thread 1 wins
    k, val = ac._argmax_cuda(v, np.arange(1200))
    assert k[0] == 513 and val[0] == 1.0
    v[0, 1] = 1.0                          # now thread 1 holds gid 1 and 513: the first
    k, _ = ac._argmax_cuda(v, np.arange(1200))
    assert k[0] == 1
    v[0, 3] = np.nan                       # NaN never wins
    k, _ = ac._argmax_cuda(v, np.arange(1200))
    assert k[0] == 1


def test_fft_oversample_keeps_the_samples():
    rng = np.random.default_rng(2)
    z = (rng.standard_normal((2, 16, 12)) + 1j * rng.standard_normal((2, 16, 12))).astype(np.complex64)
    o = ac._fft_oversample(z, 2, 1)
    assert o.shape == (2, 32, 24)
    np.testing.assert_allclose(o[:, ::2, ::2], z, rtol=1e-4, atol=1e-4)


def test_sinc_table():
    filt, ntap = ac.sinc_filter()
    assert ntap == 8 and filt.size == 8 * 4096 + 1 and filt.dtype == np.float32
    K = ac._sinc_matrix(8, 32, np.arange(0, 256, 7))
    np.testing.assert_allclose(K.sum(axis=1), 1.0, rtol=1e-12)


def test_lattice_matches_the_pge():
    # NISAR_L1_UR_ROFF_..._D_085 (P05023): 15200 x 52585 reference -> 973 x 3465, start 266, labels 298
    lat = OffsetLattice.for_grid(15200, 52585, GOFF_LAYERS)
    assert (lat.az_start, lat.rg_start, lat.length, lat.width) == (266, 266, 973, 3465)
    assert lat.label_lines(0) == 298 and lat.label_samples(1) == 313


def _speckle(rng, shape, band=0.8):
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    F = np.fft.fft2(z)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return F * ((np.abs(fy) < band / 2) & (np.abs(fx) < band / 2))


def _shifted(F, dy, dx):
    fy = np.fft.fftfreq(F.shape[0])[:, None]
    fx = np.fft.fftfreq(F.shape[1])[None, :]
    return np.fft.ifft2(F * np.exp(-2j * np.pi * (fy * dy + fx * dx))).astype(np.complex64)


@pytest.mark.parametrize("method", ["sinc", "fft"])
def test_recovers_a_known_subpixel_shift(method):
    rng = np.random.default_rng(3)
    p = ac.AmpcorParams(window_range=32, window_azimuth=32, half_search_range=8,
                        half_search_azimuth=8, correlation_surface_oversampling_method=method)
    truth = rng.uniform(-2.5, 2.5, size=(12, 2))
    ref, sec = [], []
    for dy, dx in truth:
        F = _speckle(rng, (64, 64))
        a = np.fft.ifft2(F).astype(np.complex64)
        b = _shifted(F, dy, dx)                  # feature at +d in the secondary
        ref.append(a[16:48, 16:48])
        sec.append(b[8:56, 8:56])
    out = ac.ampcor_windows(np.array(ref), np.array(sec), p)
    err = np.c_[out["down"], out["across"]] - truth
    assert np.abs(err).max() < 0.03
    assert np.all(out["peak"] > 0.8) and np.all(out["snr"] > 10)
