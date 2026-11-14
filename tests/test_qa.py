"""RSLC checks: isolated mask lines, data-window changes, pairing, and band B's
range offset against band A.

The band-offset RSLC is synthetic: both bands image one ground texture with
their own speckle; band B (a fifth of A's bandwidth, sampled 8x coarser) sees
it a known number of A samples further, by a different amount in each of two
data windows.
"""

import h5py
import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import qa, radar


def _swath(lines, start, end):
    return np.column_stack([np.full(lines, start), np.full(lines, end)]).astype(np.int64)


def test_isolated_drops_and_their_repair():
    a, b = _swath(40, 10, 90), _swath(40, 100, 190)
    a[12] = (10, 40)                          # 50 of 170 samples gone on one line
    a[25] = (10, 85)                          # 5 of 170: under the 10% threshold
    b[30:] = (0, 0)                           # a partial frame's empty tail does not count
    lines, frac = qa.isolated_drops([a, b])
    assert lines.tolist() == [12] and frac[0] == pytest.approx(50 / 170)
    fixed = qa.repair_drops([a, b])
    assert fixed[0][12].tolist() == [10, 90] and a[12].tolist() == [10, 40]
    assert qa.isolated_drops(fixed)[0].size == 0


def _start_mask(first, lines, offset=10):
    """Per-line `[start, end)` of a swath whose sample `j` is valid from line `first[j]` on."""
    v = np.zeros((lines, 2), np.int64)
    for i in range(lines):
        ok = np.flatnonzero(first <= i)
        if ok.size:
            v[i] = (offset + ok[0], offset + ok[-1] + 1)
    return v


def test_restore_start_moves_each_samples_first_line_back():
    shift, first = 150, 20 + np.arange(100) // 10          # far range fully focused last
    true, masked = _start_mask(first, 400), _start_mask(first + shift, 400)
    masked[300:, 0] = 11                                   # a later data-window update: left alone
    (fixed,) = qa.restore_start([masked], shift)
    np.testing.assert_array_equal(fixed[:300], true[:300])
    np.testing.assert_array_equal(fixed[300:], masked[300:])
    full = _start_mask(np.zeros(100, int), 400)            # a frame that starts after the start
    np.testing.assert_array_equal(qa.restore_start([full], shift)[0], full)
    empty = np.zeros((50, 2), np.int64)
    np.testing.assert_array_equal(qa.restore_start([empty], shift)[0], empty)
    np.testing.assert_array_equal(qa.restore_start([masked], 0)[0], masked)


def test_window_changes_hold_and_merge():
    v = _swath(3000, 10, 90)
    v[1000:, 0] = 12                          # a data-window update...
    v[1300:, 1] = 95                          # ...moves the far edge 300 lines later: one change
    v[2000, 1] = 80                           # a one-line blip is no change
    v[2500:, 0] = 14
    assert qa.window_changes([v]) == [1000, 2500]


def test_check_pair_names_what_differs():
    base = dict(release="P05023", isce3="0.25.16",
                bands={"A": dict(prf=1909.64, common_delay=-57.125), "B": dict(prf=1909.64, common_delay=-19.65)})
    other = dict(base, release="X05009", bands={"A": dict(prf=2066.98, common_delay=-60.823),
                                                "B": dict(prf=1909.64, common_delay=-19.65)})
    assert qa.check_pair(base, base) == []
    found = qa.check_pair(base, other)
    assert found[0].startswith("release") and any("commonDelay" in s for s in found)
    assert any("PRF" in s and s.startswith("frequency A") for s in found)


def _texture(rng, n):
    kernel = np.ones(40) / 40
    return np.exp(6 * np.convolve(rng.normal(size=n), kernel, "same"))


PAD = 512


def _band(rng, texture, shift, lines, bandwidth, fs, keep):
    """Speckled lines of the ground `texture`, seen `shift` A samples further, band-limited.

    Lines are made `PAD` samples longer on each side and cut, so the FFT's
    wrap-around stays out of the kept part.
    """
    x = np.arange(texture.size, dtype=float)
    out = np.empty((lines, keep), np.complex64)
    for i in range(lines):
        sc = np.sqrt(np.interp(x - shift, x, texture)) * (rng.normal(size=x.size) + 1j * rng.normal(size=x.size))
        spectrum = np.fft.fft(sc)
        spectrum[np.abs(np.fft.fftfreq(x.size, 1 / fs)) > bandwidth / 2] = 0
        out[i] = np.fft.ifft(spectrum)[PAD:PAD + keep]
    return out


@pytest.fixture
def two_band_rslc(tmp_path):
    rng = np.random.default_rng(3)
    lines, width_b, factor = 1280, 512, 8
    width_a = width_b * factor
    grid = rs.radar_grid(length=lines, width=width_a, spacing=qa.CLOCK_PERIOD_M)
    fs = radar.SPEED_OF_LIGHT / (2 * qa.CLOCK_PERIOD_M)
    texture = _texture(rng, width_a + 2 * PAD)
    a = _band(rng, texture, 0.0, lines, 40e6, fs, width_a)
    b = np.vstack([_band(rng, texture, 2.0, 640, 5e6, fs, width_a)[:, ::factor],     # 6.25 m further
                   _band(rng, texture, 5.0, 640, 5e6, fs, width_a)[:, ::factor]])    # 15.6 m further
    path = tmp_path / "rslc.h5"
    epoch = radar.epoch_units(rs.EPOCH)
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"], ident["orbitPassDirection"] = np.bytes_("Left"), np.bytes_("Descending")
        ident["granuleId"] = np.bytes_("NISAR_L1_PR_RSLC_TEST")
        ident["listOfFrequencies"] = np.array([b"A", b"B"])
        sw = f.create_group("science/LSAR/RSLC/swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(lines) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(epoch)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        for name, data, spacing, bw in (("A", a, qa.CLOCK_PERIOD_M, 40e6), ("B", b, factor * qa.CLOCK_PERIOD_M, 5e6)):
            g = sw.create_group(f"frequency{name}")
            g["listOfPolarizations"] = np.array([b"HH"])
            g["slantRange"] = grid.starting_range + np.arange(data.shape[1]) * spacing
            g["processedCenterFrequency"] = 1239e6 if name == "A" else 1293.5e6
            g["processedRangeBandwidth"] = bw
            g["HH"] = data
            n = data.shape[1]
            valid = _swath(lines, 0, n)
            valid[640:, 0] = factor if name == "A" else 1       # the data window changes at line 640
            g["numberOfSubSwaths"] = 1
            g["validSamplesSubSwath1"] = valid.astype(np.uint32)
    return path


def test_band_offset_recovers_each_data_windows_shift(two_band_rslc):
    off = qa.band_offset(two_band_rslc, step=128 / 1520.0, bins=2, window=1024, lines=128, group=32,
                         min_peak=0.3, margin=0.02)
    assert off.offset.shape == (2, 2) and off.measured.all()
    np.testing.assert_allclose(off.offset[0], 2 * qa.CLOCK_PERIOD_M, atol=0.35)
    np.testing.assert_allclose(off.offset[1], 5 * qa.CLOCK_PERIOD_M, atol=0.35)
    t = np.array([off.times[0] + 0.01, off.times[1] + 0.01])
    r = np.full(2, 0.5 * (off.ranges[0] + off.ranges[-1]))
    np.testing.assert_allclose(off.shift(t, r), [off.offset[0, 1], off.offset[1, 1]])
    assert "data window" not in off.summary() and off.k.dtype.kind == "i"


def test_band_offset_needs_band_b(tmp_path):
    from nisar_tools.rslc import RSLC

    path = tmp_path / "a_only.h5"
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"], ident["orbitPassDirection"] = np.bytes_("Left"), np.bytes_("Descending")
        ident["listOfFrequencies"] = np.array([b"A"])
        sw = f.create_group("science/LSAR/RSLC/swaths")
        sw["zeroDopplerTime"] = np.arange(4.0)
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(radar.epoch_units(rs.EPOCH))
        sw["zeroDopplerTimeSpacing"] = 1.0
    with RSLC(path) as r, pytest.raises(ValueError, match="no band B"):
        qa.band_offset(r)


def _blocks(cells):
    """Block measurements: {(window, bin): [A - B values]} on windows [0, 10), [10, 20) s and bins [0, 1), [1, 2) m."""
    return [dict(t=10.0 * i + 2.0 + 0.5 * n, r=j + 0.5, a_minus_b=v, peak=0.9)
            for (i, j), values in cells.items() for n, v in enumerate(values)]


def test_cell_offsets_ignore_a_lone_false_peak():
    times, ranges = np.array([0.0, 10.0, 20.0]), np.array([0.0, 1.0, 2.0])
    good = [-10.3, -10.5, -10.4, -10.6]
    blocks = _blocks({(0, 0): good, (0, 1): [-26.0],                   # one false peak, alone in its cell
                      (1, 0): good, (1, 1): [-1.0, -10.4, -18.0, -10.5, -10.3]})   # outliers among good blocks
    values, measured = qa._cell_offsets(blocks, times, ranges, margin=0.5, agree=0.75, min_blocks=2)
    assert measured.tolist() == [[True, False], [True, True]]
    np.testing.assert_allclose(values[1], [-10.45, -10.4])


def test_cell_offsets_keep_a_supported_change_across_range():
    times, ranges = np.array([0.0, 10.0]), np.array([0.0, 1.0, 2.0])
    deviant = qa._cell_offsets(_blocks({(0, 0): [-4.0, -4.2], (0, 1): [-0.2, -0.3, -0.1]}), times, ranges,
                               margin=0.5, agree=0.75, min_blocks=2)
    supported = qa._cell_offsets(_blocks({(0, 0): [-4.0, -4.2, -4.1], (0, 1): [-0.2, -0.3, -0.1]}), times, ranges,
                                 margin=0.5, agree=0.75, min_blocks=2)
    assert deviant[1].tolist() == [[False, True]]          # two cells disagree: the better supported stays
    np.testing.assert_allclose(supported[0][0], [-4.1, -0.2])
