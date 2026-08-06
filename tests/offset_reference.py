"""Brute-force time-domain reference for the pixel-offset kernel.

A direct transcription of the correlation GMTSAR's ``xcorr`` computes -- the
exhaustive search of ``do_time_int_xcorr.c``, the ``-time`` mode
:mod:`nisar_tools._kernels` deliberately does not ship, kept here as the oracle
its FFT path is checked against. It is ``O(search^2 * window^2)`` per location,
so tests use small windows.

The sums are written with ``np.roll``, which makes them *circular* -- matching
what an FFT correlation actually computes, rather than the linear correlation a
naive nested loop would give. That is the whole point of GMTSAR masking the
secondary down to its central template: the wrap-around lands on zeros.

Imported bare, like ``legacy_reference``: pytest's default import mode puts
``tests/`` on ``sys.path``.
"""

import numpy as np


def prepare(patch):
    """Amplitude, demeaned over the finite samples, invalid filled with zero."""
    valid = np.isfinite(patch)
    amp = np.abs(np.where(valid, patch, 0))
    mean = float(amp[valid].mean()) if valid.any() else 0.0
    return np.where(valid, amp - mean, 0.0), valid


def correlation_surface(ref_patch, sec_patch, nx_corr, ny_corr, xsearch, ysearch):
    """``|sum_n ref[n + lag] * sec_template[n]|`` at every lag, by direct sum.

    Laid out with lag ``(0, 0)`` at ``[ysearch, xsearch]``, so entry ``[i, j]``
    is the lag ``(i - ysearch, j - xsearch)`` -- the same layout the kernel's
    ``fftshift``-and-crop produces.
    """
    a, _ = prepare(ref_patch)
    b, _ = prepare(sec_patch)
    template = np.zeros_like(b)
    template[ysearch:ysearch + ny_corr, xsearch:xsearch + nx_corr] = 1.0
    b = b * template

    out = np.zeros((2 * ysearch, 2 * xsearch))
    for i, lag_y in enumerate(range(-ysearch, ysearch)):
        for j, lag_x in enumerate(range(-xsearch, xsearch)):
            # np.roll(a, -lag)[n] == a[n + lag].
            shifted = np.roll(a, (-lag_y, -lag_x), axis=(0, 1))
            out[i, j] = abs(float(np.sum(shifted * b)))
    return out


def peak_offset(ref_patch, sec_patch, nx_corr, ny_corr, xsearch, ysearch):
    """``(x_offset, y_offset, correlation)`` at the surface's integer peak.

    The correlation is GMTSAR's ``calc_time_corr``:
    ``100 * |sum a*b| / sqrt(sum a^2 * sum b^2)`` over the template, with the
    reference shifted by the peak lag.
    """
    surface = correlation_surface(
        ref_patch, sec_patch, nx_corr, ny_corr, xsearch, ysearch
    )
    i, j = np.unravel_index(int(surface.argmax()), surface.shape)
    lag_y, lag_x = i - ysearch, j - xsearch

    a, a_valid = prepare(ref_patch)
    b, b_valid = prepare(sec_patch)
    r0, c0 = ysearch + lag_y, xsearch + lag_x
    rows, cols = slice(r0, r0 + ny_corr), slice(c0, c0 + nx_corr)
    trows, tcols = slice(ysearch, ysearch + ny_corr), slice(xsearch, xsearch + nx_corr)

    both = a_valid[rows, cols] & b_valid[trows, tcols]
    av, bv = a[rows, cols][both], b[trows, tcols][both]
    denom = np.sqrt(float(av @ av) * float(bv @ bv))
    corr = 0.0 if denom == 0.0 else 100.0 * abs(float(av @ bv)) / denom
    return float(-lag_x), float(-lag_y), corr
