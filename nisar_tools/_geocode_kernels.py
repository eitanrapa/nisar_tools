"""Numerical kernels for geocoding an RSLC: a port of ISCE3's ``geocodeSlc.cpp``.

Pure numpy, one output block at a time, no I/O -- the workflow that tiles a
frame, reads the RSLC and writes the product lives in
:mod:`nisar_tools.gslc_geocode`. Split out so the numerics can be tested against
synthetic point targets without an HDF5 file in sight.

The per-pixel algorithm (ISCE3 v0.25.16, what ASF's GSLCs are made with):

1. Map pixel centre -> lon/lat -> DEM height (ISCE3's spline, float32).
2. Zero-Doppler ``geo2rdr`` from mid-swath -> ``(t, r)``; keep ``r`` as the
   *uncorrected* range for flattening.
3. Timing corrections: ``t += dt_az(t, r)``, then ``r += dr(t, r)`` **evaluated
   at the corrected t** -- the order matters at the millimetre level.
4. Reject points outside the radar grid (half a pixel beyond its edges) or the
   Doppler LUT; fractional line/sample indices otherwise.
5. Subswath mask: the subswath holding the floor index, 0 if any corner of the
   +-4 sample chip is invalid, 255 outside the radar grid.
6. 9x9 chip -> azimuth **Doppler demodulation** relative to the chip's integer
   row -> 8x8 windowed-sinc interpolation at the fractional position -> the
   Doppler phase of that fraction added back.
7. Flatten: multiply by ``exp(+j 4 pi r_uncorrected / lambda)``.

Step 6 is not optional on NISAR: the Doppler centroid is ~950 Hz against a
1520 Hz line rate, so the azimuth spectrum straddles the Nyquist frequency and
interpolating it without basebanding would fold it.
"""

from functools import lru_cache

import numpy as np

from . import radar

SINC_LEN = 8
SINC_SUB = 8192
SINC_HALF = 4
SINC_ONE = 9


@lru_cache(maxsize=1)
def sinc_kernel():
    """ISCE3's ``Sinc2dInterpolator(8, 8192)`` weight table, ``(8192, 8)``.

    ``_sinc_coef(beta=1, pedestal=0, weight=1)``: a raised-cosine (Hann) window
    over the whole 65536-tap filter times ``sinc(floor(i - soff) / 8192)`` --
    note the floor in the sinc argument but not in the window -- then each of
    the 8192 sub-sample rows normalised to unit sum.
    """
    n = SINC_SUB * SINC_LEN
    i = np.arange(n, dtype=np.float64)
    soff = (n - 1.0) / 2.0
    wgthgt = 0.5
    wgt = (1.0 - wgthgt) + wgthgt * np.cos((np.pi * (i - soff)) / soff)
    s = np.floor(i - soff) / SINC_SUB
    fct = np.where(s != 0.0, np.sin(np.pi * s) / np.where(s != 0.0, np.pi * s, 1.0), 1.0)
    filt = fct * wgt
    # filter[ifrac + decfactor * j] for kernel row ifrac, column j
    kernel = filt.reshape(SINC_LEN, SINC_SUB).T.copy()
    return kernel / kernel.sum(axis=1, keepdims=True)


def subswath_number(valid_samples, az_index, rg_index, length, width):
    """``SubSwaths::getSampleSubSwath`` for integer index arrays.

    ``valid_samples`` is the list of ``(lines, 2)`` ``[start, end)`` arrays; an
    empty list means "all samples valid, subswath 1".
    """
    az_index = np.asarray(az_index, np.int64)
    rg_index = np.asarray(rg_index, np.int64)
    inside = (rg_index >= 0) & (rg_index < width) & (az_index >= 0) & (az_index < length)
    out = np.zeros(az_index.shape, np.uint8)
    if not valid_samples:
        out[inside] = 1
        return out
    az = np.where(inside, az_index, 0)
    unassigned = inside.copy()
    for s, vs in enumerate(valid_samples, start=1):
        if vs.size == 0:
            out[unassigned] = s
            return out
        start, end = vs[az, 0], vs[az, 1]
        hit = unassigned & (rg_index >= start) & (rg_index < end)
        out[hit] = s
        unassigned &= ~hit
    return out


def radar_indices(lon, lat, height, radar_grid, orbit, native_doppler,
                  valid_samples, az_correction=None, rg_correction=None,
                  threshold=1e-8, maxiter=25, flatten_with_corrected_range=False):
    """Steps 2-5 for a set of output pixels.

    ``lon``/``lat`` in radians and ``height`` in metres, all 1-D. Returns
    ``(az_index, rg_index, flatten_range, mask)``: fractional full-grid indices
    (NaN where the pixel is not geocoded), the slant range to flatten with, and
    the uint8 subswath mask (255 = not in the radar grid).
    """
    n = lon.size
    llh = np.stack([lon, lat, height], axis=1)
    t, r, ok = radar.geo2rdr(llh, orbit, None, radar_grid.wavelength,
                             radar_grid.look_side, t0=radar_grid.sensing_mid,
                             threshold=threshold, maxiter=maxiter, delta_range=1e-8)
    flatten_range = np.where(ok, r, np.nan)
    t = np.where(ok, t, np.nan)
    r = np.where(ok, r, np.nan)
    if az_correction is not None and az_correction.have_data:
        inside = ok & az_correction.contains(t, r)
        t = np.where(inside, t + az_correction.eval(np.where(inside, t, az_correction.y_start),
                                                     np.where(inside, r, az_correction.x_start)), t)
    if rg_correction is not None and rg_correction.have_data:
        inside = ok & rg_correction.contains(t, r)
        r = np.where(inside, r + rg_correction.eval(np.where(inside, t, rg_correction.y_start),
                                                     np.where(inside, r, rg_correction.x_start)), r)
    if flatten_with_corrected_range:
        flatten_range = r
    good = ok & radar_grid.contains(t, r) & native_doppler.contains(t, r)
    az_index = np.where(good, (t - radar_grid.sensing_start) * radar_grid.prf, np.nan)
    rg_index = np.where(good, (r - radar_grid.starting_range) / radar_grid.range_pixel_spacing, np.nan)

    mask = np.full(n, 255, np.uint8)
    sel = np.flatnonzero(good)
    if sel.size:
        iaz = np.floor(az_index[sel]).astype(np.int64)
        irg = np.floor(rg_index[sel]).astype(np.int64)
        L, W = radar_grid.length, radar_grid.width
        centre = subswath_number(valid_samples, iaz, irg, L, W)
        half = SINC_ONE // 2
        corner_bad = np.zeros(sel.size, bool)
        for daz in (-half, half):
            for drg in (-half, half):
                corner_bad |= subswath_number(valid_samples, iaz + daz, irg + drg, L, W) == 0
        mask[sel] = np.where((centre != 0) & corner_bad, 0, centre)
    return az_index, rg_index, flatten_range, mask


@lru_cache(maxsize=1)
def _kernel_columns():
    """The float32 sinc table as ``(8, 8192)``, so a gather gives ``(8, P)`` rows."""
    return np.ascontiguousarray(sinc_kernel().astype(np.float32).T)


def _mul_cf(ar, ai, br, bi):
    """``std::complex<float>`` multiply, one ufunc per operation.

    Separate calls keep each product and sum a plain IEEE float32 operation:
    a single expression could be contracted into fused multiply-adds, which
    round differently from the build that made the reference products.
    """
    return np.subtract(ar * br, ai * bi), np.add(ar * bi, ai * br)


def interpolate_slc(block, first_line, first_sample, az_index, rg_index,
                    flatten_range, radar_grid, native_doppler, flatten=True,
                    batch=16384):
    """Steps 6-7: sinc-interpolate a radar block at fractional indices.

    ``block`` is the complex RSLC window whose ``[0, 0]`` sample is
    ``(first_line, first_sample)`` of the full grid -- or a sequence of
    same-shaped windows (one per polarization), which share every
    polarization-independent step and come back as a list. Pixels whose 9x9
    chip would leave the *block* stay NaN -- ISCE3 checks against the block it
    was handed, which is the radar grid itself at the swath edges. Returns
    complex64.

    The arithmetic is ISCE3's, operation for operation, in float32: each chip
    sample demodulated by ``complex<float>(cos, -sin)`` of its row's Doppler
    phase, then ``sum_i sum_j (chip * ky_i) * kx_j`` accumulated in that order
    (``Sinc2dInterpolator::_sinc_eval_2d``), then the Doppler phase of the
    fractional line and the flattening phase multiplied back on. A float32 sum
    is not associative, so evaluating the same weights separably (a range pass
    then an azimuth pass) changes ~0.1% of pixels after mantissa truncation.
    """
    single = not isinstance(block, (list, tuple))
    blocks = [np.asarray(block)] if single else [np.asarray(b) for b in block]
    if len({b.shape for b in blocks}) != 1:
        raise ValueError("interpolate_slc: blocks must share one shape")
    npix = az_index.size
    outs = [np.full(npix, np.nan + 1j * np.nan, dtype=np.complex64) for _ in blocks]
    L, W = blocks[0].shape
    az_rel = az_index - first_line
    rg_rel = rg_index - first_sample
    valid = np.isfinite(az_rel) & np.isfinite(rg_rel)
    int_az = np.zeros(npix, np.int64)
    int_rg = np.zeros(npix, np.int64)
    int_az[valid] = np.trunc(az_rel[valid]).astype(np.int64)
    int_rg[valid] = np.trunc(rg_rel[valid]).astype(np.int64)
    valid &= (int_rg >= SINC_HALF) & (int_rg < W - SINC_HALF)
    valid &= (int_az >= SINC_HALF) & (int_az < L - SINC_HALF)
    sel = np.flatnonzero(valid)
    if sel.size:
        t_full = radar_grid.sensing_start + az_index[sel] / radar_grid.prf
        r_full = radar_grid.starting_range + rg_index[sel] * radar_grid.range_pixel_spacing
        keep = native_doppler.contains(t_full, r_full)
        sel, t_full, r_full = sel[keep], t_full[keep], r_full[keep]
    if sel.size == 0:
        return outs[0] if single else outs

    f32 = np.float32
    parts = [(np.ascontiguousarray(b.real, dtype=f32).reshape(-1),
              np.ascontiguousarray(b.imag, dtype=f32).reshape(-1)) for b in blocks]
    results = [(np.empty(sel.size, f32), np.empty(sel.size, f32)) for _ in blocks]
    for start in range(0, sel.size, batch):
        stop = min(start + batch, sel.size)
        s = sel[start:stop]
        shared = _chip_geometry(s, az_rel, rg_rel, int_az, int_rg, W,
                                t_full[start:stop], r_full[start:stop], flatten_range,
                                radar_grid, native_doppler, flatten)
        for (block_re, block_im), (res_re, res_im) in zip(parts, results):
            res_re[start:stop], res_im[start:stop] = _sinc_chips(block_re, block_im, shared)
    for out, (res_re, res_im) in zip(outs, results):
        out.real[sel] = res_re
        out.imag[sel] = res_im
    return outs[0] if single else outs


def _chip_geometry(s, az_rel, rg_rel, int_az, int_rg, W, t_full, r_full,
                   flatten_range, radar_grid, native_doppler, flatten):
    """Everything about a batch of pixels that does not depend on the image."""
    f32 = np.float32
    kernel = _kernel_columns()
    frac_az = az_rel[s] - int_az[s]
    frac_rg = rg_rel[s] - int_rg[s]
    dop = native_doppler.eval(t_full, r_full) * 2.0 * np.pi / radar_grid.prf
    # Sinc2dInterpolator::interp_impl on the chip at (4 + frac_rg, 4 + frac_az).
    x = 4.0 + frac_rg
    y = 4.0 + frac_az
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    in_kernel = (ix <= SINC_ONE - SINC_HALF - 1) & (iy <= SINC_ONE - SINC_HALF - 1)
    ifx = np.clip((fx * SINC_SUB).astype(np.int64), 0, SINC_SUB - 1)
    ify = np.clip((fy * SINC_SUB).astype(np.int64), 0, SINC_SUB - 1)
    # cos(-a) == cos(a) and sin(-a) == -sin(a) exactly, and dop * -k is
    # exactly -(dop * k): five offsets' trigonometry serves all eight.
    phase = {k: dop * k for k in range(SINC_HALF + 1)}
    cos = {k: np.cos(v).astype(f32) for k, v in phase.items()}
    sin = {k: np.sin(v).astype(f32) for k, v in phase.items()}
    # Chip row i (0..7) is line int_az + 4 - i; column j is sample int_rg + 4 - j.
    rows = []
    base = int_az[s] * W + int_rg[s]
    offsets = SINC_HALF - np.arange(SINC_LEN)
    for k in offsets:
        # chip = sample * complex<float>(cos(k dop), -sin(k dop))
        rows.append((base + (k * W + offsets[:, None]),
                     cos[abs(k)], -sin[k] if k >= 0 else sin[-k]))
    back = dop * frac_az
    ramps = [(np.cos(back).astype(f32), np.sin(back).astype(f32))]
    if flatten:
        phi = 4.0 * (np.pi / radar_grid.wavelength) * flatten_range[s]
        ramps.append((np.cos(phi).astype(f32), np.sin(phi).astype(f32)))
    # Fancy indexing, not np.take: ~4x faster for multi-dimensional indices.
    return dict(rows=rows, kx=kernel[:, ifx], ky=kernel[:, ify],
                outside=~in_kernel, ramps=ramps)


def _sinc_chips(block_re, block_im, g):
    """Interpolate one image's chips for a batch described by :func:`_chip_geometry`."""
    f32 = np.float32
    kx, ky = g["kx"], g["ky"]
    acc_re = np.zeros(kx.shape[1], f32)
    acc_im = np.zeros(kx.shape[1], f32)
    for i, (idx, c, sn) in enumerate(g["rows"]):
        re, im = _mul_cf(block_re[idx], block_im[idx], c, sn)
        re *= ky[i]
        re *= kx
        im *= ky[i]
        im *= kx
        for j in range(SINC_LEN):
            acc_re += re[j]
            acc_im += im[j]
    acc_re[g["outside"]] = 0
    acc_im[g["outside"]] = 0
    for c, sn in g["ramps"]:
        acc_re, acc_im = _mul_cf(acc_re, acc_im, c, sn)
    return acc_re, acc_im


def truncate_mantissa(z, significant_bits=10):
    """``isce3.core.types.truncate_mantissa``, in place on float or complex arrays.

    Zeroes all but the top ``significant_bits`` mantissa bits so gzip has
    something to work with -- the reason ASF's GSLC ``max_real_value`` reads
    like 95.75 rather than 95.7493. NaN stays NaN (its top mantissa bit is kept).
    """
    if np.iscomplexobj(z):
        truncate_mantissa(z.real, significant_bits)
        truncate_mantissa(z.imag, significant_bits)
        return z
    finfo = np.finfo(z.dtype)
    nzero = finfo.nmant - significant_bits
    if nzero <= 0:
        return z
    bits = z.dtype.itemsize * 8
    mask = ((1 << bits) - 1) ^ ((1 << nzero) - 1)
    u = z.view(np.dtype(f"u{z.dtype.itemsize}"))
    u &= np.array(mask, dtype=u.dtype)
    return z
