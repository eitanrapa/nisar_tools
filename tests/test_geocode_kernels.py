"""The per-pixel numerics of RSLC -> GSLC geocoding (``_geocode_kernels``).

Two kinds of check. Physical ones on synthetic signals: a band-limited image is
interpolated to within the 8-tap kernel's error, *only* when its Doppler is
removed first, and a point target geocodes to its own map position with its
phase flattened to zero. And one literal one: the vectorised interpolator must
reproduce a scalar transcription of ISCE3's ``interpolate`` +
``Sinc2dInterpolator`` loop bit for bit, because float32 sums are not
associative and ASF's products are made by that loop.
"""

import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import radar
from nisar_tools._geocode_kernels import (
    SINC_SUB, interpolate_slc, radar_indices, sinc_kernel, subswath_number,
    truncate_mantissa)

f32 = np.float32


def _isce3_pixel(block, az_index, rg_index, grid, doppler_hz, flatten_range):
    """``geocodeSlc.cpp::interpolate`` + ``carrierPhaseRerampAndFlatten``, one pixel."""
    kernel = sinc_kernel()
    trig = lambda fn, v: f32(fn(np.array([v]))[0])        # noqa: E731
    az, rg = float(az_index), float(rg_index)
    int_az, int_rg = int(az), int(rg)
    frac_az, frac_rg = az - int_az, rg - int_rg
    dopp = doppler_hz * 2 * np.pi / grid.prf
    chip = {}
    for ii in range(9):
        phase = dopp * (ii - 4)
        c, s = trig(np.cos, phase), f32(-np.sin(np.array([phase]))[0])
        for jj in range(9):
            v = block[int_az + ii - 4, int_rg + jj - 4]
            vr, vi = f32(v.real), f32(v.imag)
            chip[ii, jj] = (vr * c - vi * s, vr * s + vi * c)
    x, y = 4.0 + frac_rg, 4.0 + frac_az
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - ix, y - iy
    re, im = f32(0.0), f32(0.0)
    if 3 <= ix <= 4 and 3 <= iy <= 4:
        ifx = min(max(0, int(fx * SINC_SUB)), SINC_SUB - 1)
        ify = min(max(0, int(fy * SINC_SUB)), SINC_SUB - 1)
        for i in range(8):
            ky = f32(kernel[ify, i])
            for j in range(8):
                kx = f32(kernel[ifx, j])
                cr, ci = chip[iy + 4 - i, ix + 4 - j]
                re = re + (cr * ky) * kx
                im = im + (ci * ky) * kx
    back = dopp * frac_az
    c, s = trig(np.cos, back), trig(np.sin, back)
    re, im = re * c - im * s, re * s + im * c
    phi = 4.0 * (np.pi / grid.wavelength) * flatten_range
    c, s = trig(np.cos, phi), trig(np.sin, phi)
    return np.complex64(complex(re * c - im * s, re * s + im * c))


def _tones(shape, freqs, amps):
    """A sum of complex exponentials, and the continuous signal it samples."""
    lines, samples = np.mgrid[0:shape[0], 0:shape[1]].astype(float)

    def signal(l, m):
        return sum(a * np.exp(2j * np.pi * (fa * l + fr * m)) for (fa, fr), a in zip(freqs, amps))

    return signal(lines, samples).astype(np.complex64), signal


# -- the kernel table -------------------------------------------------------------
def test_sinc_table_rows_are_normalised_and_centred():
    k = sinc_kernel()
    assert k.shape == (SINC_SUB, 8)
    assert np.allclose(k.sum(axis=1), 1.0, atol=1e-12)
    # Row 0 is an on-sample position: column 4 is the sample itself.
    assert k[0, 4] > 0.99 and np.abs(np.delete(k[0], 4)).max() < 1e-3
    # Half-way between two samples the two neighbours share the weight.
    half = k[SINC_SUB // 2]
    assert half[3] == pytest.approx(half[4], rel=1e-3) and half[3] > 0.5


# -- ISCE3's operation order ----------------------------------------------------------
def test_interpolation_is_isce3s_float32_loop_bit_for_bit():
    rng = np.random.default_rng(7)
    grid = rs.radar_grid(length=60, width=50)
    block = (rng.standard_normal((60, 50)) + 1j * rng.standard_normal((60, 50))).astype(np.complex64) * 40
    n = 64
    az = rng.uniform(5, 54, n)
    rg = rng.uniform(5, 44, n)
    az[:4] = np.floor(az[:4])                     # exactly on a line
    rng_flat = rng.uniform(grid.starting_range, grid.end_range, n)
    doppler_hz = 951.37
    lut = rs.constant_doppler(grid, doppler_hz)
    out = interpolate_slc(block, 0, 0, az, rg, rng_flat, grid, lut)
    for k in range(n):
        ref = _isce3_pixel(block, az[k], rg[k], grid, doppler_hz, rng_flat[k])
        assert out[k].tobytes() == ref.tobytes(), k


def test_interpolation_does_not_depend_on_batching_or_polarization_count():
    rng = np.random.default_rng(8)
    grid = rs.radar_grid(length=80, width=70)
    a = (rng.standard_normal((80, 70)) + 1j * rng.standard_normal((80, 70))).astype(np.complex64)
    b = a[::-1, ::-1].copy()
    az = rng.uniform(0, 80, 3000)
    rg = rng.uniform(0, 70, 3000)
    az[::97] = np.nan
    rng_flat = rng.uniform(grid.starting_range, grid.end_range, 3000)
    lut = rs.constant_doppler(grid, 900.0)
    ref_a = interpolate_slc(a, 0, 0, az, rg, rng_flat, grid, lut)
    ref_b = interpolate_slc(b, 0, 0, az, rg, rng_flat, grid, lut, batch=17)
    both = interpolate_slc([a, b], 0, 0, az, rg, rng_flat, grid, lut, batch=1000)
    assert both[0].tobytes() == ref_a.tobytes()
    assert both[1].tobytes() == ref_b.tobytes()
    with pytest.raises(ValueError, match="share one shape"):
        interpolate_slc([a, b[:-1]], 0, 0, az, rg, rng_flat, grid, lut)


def test_pixels_whose_chip_leaves_the_block_or_the_doppler_lut_are_nan():
    grid = rs.radar_grid(length=40, width=40)
    block = np.ones((40, 40), np.complex64)
    lut = rs.constant_doppler(grid, 0.0)
    az = np.array([3.9, 4.0, 35.99, 36.0, 20.0, 20.0])
    rg = np.array([20.0, 20.0, 20.0, 20.0, 3.5, 20.0])
    out = interpolate_slc(block, 0, 0, az, rg, np.full(6, grid.starting_range), grid, lut, flatten=False)
    assert np.isnan(out[[0, 3, 4]]).all()
    assert np.isfinite(out[[1, 2, 5]]).all()
    # The same pixel against a LUT that stops short of it.
    t = np.array([grid.sensing_start - 1.0, grid.sensing_start + 0.001])
    r = np.array([grid.starting_range - 10.0, grid.end_range + 10.0])
    short = radar.LUT2d.from_coords(r, t, np.zeros((2, 2)))
    out = interpolate_slc(block, 0, 0, np.array([30.0]), np.array([20.0]),
                          np.array([grid.starting_range]), grid, short, flatten=False)
    assert np.isnan(out).all()


# -- physics ------------------------------------------------------------------------
def test_band_limited_signal_is_reconstructed_between_samples():
    grid = rs.radar_grid(length=64, width=64)
    freqs = [(0.11, -0.2), (-0.27, 0.05), (0.3, 0.24)]
    block, signal = _tones((64, 64), freqs, [1.0, 0.7, 0.4])
    rng = np.random.default_rng(3)
    az, rg = rng.uniform(10, 54, 500), rng.uniform(10, 54, 500)
    out = interpolate_slc(block, 0, 0, az, rg, np.zeros(500), grid,
                          rs.constant_doppler(grid, 0.0), flatten=False)
    err = np.abs(out - signal(az, rg)) / 2.1
    assert np.median(err) < 5e-3 and err.max() < 3e-2


def test_doppler_is_removed_before_interpolating_and_restored_after():
    """NISAR's ~950 Hz centroid at 1520 Hz puts the azimuth spectrum across Nyquist."""
    grid = rs.radar_grid(length=64, width=64)
    fd = 950.0
    carrier = fd / grid.prf                                   # 0.625 cycles / line
    freqs = [(carrier + 0.2, 0.1), (carrier - 0.25, -0.15), (carrier, 0.0)]
    block, signal = _tones((64, 64), freqs, [1.0, 0.6, 0.8])
    rng = np.random.default_rng(4)
    az, rg = rng.uniform(10, 54, 500), rng.uniform(10, 54, 500)
    truth = signal(az, rg)
    right = interpolate_slc(block, 0, 0, az, rg, np.zeros(500), grid,
                            rs.constant_doppler(grid, fd), flatten=False)
    wrong = interpolate_slc(block, 0, 0, az, rg, np.zeros(500), grid,
                            rs.constant_doppler(grid, 0.0), flatten=False)
    assert np.median(np.abs(right - truth) / 2.4) < 5e-3
    assert np.median(np.abs(wrong - truth) / 2.4) > 0.1


def test_point_target_geocodes_onto_itself_with_flattened_phase():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    fd = 950.0
    doppler = rs.constant_doppler(grid, fd)
    line0, sample0, height = 31.37, 30.81, 250.0
    t0 = grid.sensing_start + line0 / grid.prf
    r0 = grid.starting_range + sample0 * grid.range_pixel_spacing
    llh, ok = radar.rdr2geo([t0], [r0], orbit, height=height, wavelength=grid.wavelength,
                            look_side=grid.look_side, threshold=1e-9)
    assert ok.all()
    # The focused response of a point at (line0, sample0): band-limited in both
    # axes, the Doppler centroid as an azimuth carrier, and ISCE3's
    # exp(-j 4 pi R / lambda) propagation phase.
    lines, samples = np.mgrid[0:64, 0:64].astype(float)
    bw = 0.8
    block = (np.sinc(bw * (lines - line0)) * np.sinc(bw * (samples - sample0))
             * np.exp(2j * np.pi * fd / grid.prf * (lines - line0))
             * np.exp(-4j * np.pi * r0 / grid.wavelength)).astype(np.complex64)
    lon, lat, h = llh[:, 0], llh[:, 1], llh[:, 2]
    az, rg, flat_r, mask = radar_indices(lon, lat, h, grid, orbit, doppler, [])
    assert abs(az[0] - line0) < 1e-4 and abs(rg[0] - sample0) < 1e-4
    assert abs(flat_r[0] - r0) < 1e-4 and mask[0] == 1
    value = interpolate_slc(block, 0, 0, az, rg, flat_r, grid, doppler)[0]
    # An 8-tap kernel droops a few percent at 0.4 cycles/sample (bw 0.8); the
    # phase is what geocoding must not disturb.
    assert abs(abs(value) - 1.0) < 0.05
    assert abs(np.angle(value)) < 0.01


def test_timing_corrections_move_the_sample_but_not_the_flattening():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 0.0)
    t0 = grid.sensing_start + 30.0 / grid.prf
    r0 = grid.starting_range + 30.0 * grid.range_pixel_spacing
    llh, _ = radar.rdr2geo([t0], [r0], orbit, height=0.0, wavelength=grid.wavelength,
                           look_side=grid.look_side, threshold=1e-9)
    ts = np.array([grid.sensing_start - 1, grid.sensing_stop + 1])
    rs_ = np.array([grid.starting_range - 100, grid.end_range + 100])
    dt, dr = 2.0 / grid.prf, 3.0 * grid.range_pixel_spacing
    az_lut = radar.LUT2d.from_coords(rs_, ts, np.full((2, 2), dt))
    rg_lut = radar.LUT2d.from_coords(rs_, ts, np.full((2, 2), dr))
    a0, g0, f0, _ = radar_indices(llh[:, 0], llh[:, 1], llh[:, 2], grid, orbit, doppler, [])
    a1, g1, f1, _ = radar_indices(llh[:, 0], llh[:, 1], llh[:, 2], grid, orbit, doppler, [],
                                  az_lut, rg_lut)
    assert a1[0] - a0[0] == pytest.approx(2.0, abs=1e-6)
    assert g1[0] - g0[0] == pytest.approx(3.0, abs=1e-6)
    assert f1[0] == f0[0]


# -- subswath mask ------------------------------------------------------------------
def test_subswath_number_assigns_half_open_ranges():
    lines = 10
    sw1 = np.column_stack([np.full(lines, 5), np.full(lines, 20)])
    sw2 = np.column_stack([np.full(lines, 25), np.full(lines, 40)])
    az = np.array([0, 3, 3, 3, 3, 9, -1, 10])
    rg = np.array([4, 5, 19, 20, 25, 39, 10, 10])
    out = subswath_number([sw1, sw2], az, rg, lines, 50)
    assert out.tolist() == [0, 1, 1, 0, 2, 2, 0, 0]
    assert subswath_number([], az, rg, lines, 50).tolist() == [1, 1, 1, 1, 1, 1, 0, 0]


def test_mask_is_255_outside_zero_at_chip_edges_and_the_subswath_inside():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=100, width=100)
    valid = [np.column_stack([np.full(100, 10), np.full(100, 90)])]
    lines = np.array([50.0, 50.0, 50.0, 50.0, -5.0])
    samples = np.array([50.0, 11.0, 15.0, 86.5, 50.0])
    t = grid.sensing_start + lines / grid.prf
    r = grid.starting_range + samples * grid.range_pixel_spacing
    llh, ok = radar.rdr2geo(t, r, orbit, height=0.0, wavelength=grid.wavelength,
                            look_side=grid.look_side, threshold=1e-9)
    _, _, _, mask = radar_indices(llh[:, 0], llh[:, 1], llh[:, 2], grid, orbit,
                                  rs.constant_doppler(grid, 0.0), valid)
    assert mask.tolist() == [1, 0, 1, 0, 255]


# -- mantissa truncation ------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_truncate_mantissa_keeps_the_top_bits(dtype):
    rng = np.random.default_rng(9)
    x = (rng.standard_normal(10000) * 10.0 ** rng.integers(-3, 4, 10000)).astype(dtype)
    y = truncate_mantissa(x.copy(), significant_bits=10)
    exponent = np.floor(np.log2(np.abs(x)))
    quantum = 2.0 ** (exponent - 10)
    expected = np.sign(x) * np.floor(np.abs(x.astype(float)) / quantum) * quantum
    assert np.array_equal(y.astype(float), expected)
    assert np.array_equal(truncate_mantissa(y.copy()), y)             # idempotent


def test_truncate_mantissa_on_complex_keeps_nan():
    z = np.array([95.7493 - 3.14159j, complex(np.nan, np.nan)], np.complex64)
    truncate_mantissa(z)
    assert z[0] == np.complex64(95.6875 - 3.140625j)
    assert np.isnan(z[1].real) and np.isnan(z[1].imag)
