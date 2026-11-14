"""Checks of delivered RSLCs for the defects of ISCE3 v0.25.16 that survive into them.

- `check_rslc`: release, calibration delays, PRF, input-data exceptions, and
  `validSamplesSubSwath` lines that drop out of the swath for one line (ISCE3's
  unsigned subtraction under NumPy 2).
- `check_pair`: two dates must have been focused alike before they are
  interferometrically combined.
- `band_offset`: band B's slant ranges are off by whole periods of the 48 MHz
  sampling clock (3.12 m each), constant within a ~10 s data window and the same
  on most dates of a track. It misplaces B against A, which turns the
  deformation's range gradient into split-spectrum screen error; the offset it
  measures is what `geocode_rslc(range_shift=)` corrects B with.

The pulse timing behind an RSLC is checked by `nisar_tools.l0b.timing_report`.
"""

import dataclasses

import numpy as np

from .rslc import RSLC

#: Slant-range length of one period of the 48 MHz clock band B is decimated from (m).
CLOCK_PERIOD_M = 299792458.0 / (2 * 48e6)
#: Band B's slant range measured long against A after every modelled delay, on
#: corner reflectors at two sites and four tracks (m; the bug report's I6-5).
B_CALIBRATION_M = 0.98
#: Dispersive A - B group delay assumed when rounding to whole periods (m; ~0.02 m per slant TECU).
IONOSPHERE_GUESS_M = 0.6


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def isolated_drops(valid_samples, threshold=0.1):
    """Lines whose valid width drops by at least `threshold` against both neighbours.

    `valid_samples` is `RSLC.valid_samples(...)`. Returns `(lines, fractions)`.
    Lines with no valid samples next to them (partial frames) do not count.
    """
    width = sum(np.clip(v[:, 1] - v[:, 0], 0, None) for v in valid_samples).astype(float)
    if width.size < 3:
        return np.array([], int), np.array([])
    mid, before, after = width[1:-1], width[:-2], width[2:]
    hit = (before > 0) & (after > 0) & (mid <= (1 - threshold) * before) & (mid <= (1 - threshold) * after)
    lines = np.flatnonzero(hit) + 1
    return lines, 1 - width[lines] / np.minimum(width[lines - 1], width[lines + 1])


def repair_drops(valid_samples, threshold=0.1):
    """`valid_samples` with each isolated dropped line given its neighbours' extent."""
    lines, _ = isolated_drops(valid_samples, threshold)
    out = [v.copy() for v in valid_samples]
    for v in out:
        for i in lines:
            v[i, 0] = min(v[i - 1, 0], v[i + 1, 0])
            v[i, 1] = max(v[i - 1, 1], v[i + 1, 1])
    return out


def window_changes(valid_samples, persist=50, merge=1000):
    """Lines where the data window changes: a subswath edge steps to a new value and holds it
    for `persist` lines (the end of a one-line blip is no change).

    Changes within `merge` lines of a change are part of it (each update moves both
    edges, ~0.4 s apart). Returns the first line of each.
    """
    lines = set()
    for v in valid_samples:
        for c in (0, 1):
            col = v[:, c]
            for i in np.flatnonzero(np.diff(col)):
                new = col[i + 1]
                if (i + 1 + persist <= col.size and np.all(col[i + 1:i + 1 + persist] == new)
                        and new != col[max(i - 1, 0)]):
                    lines.add(int(i) + 1)
    firsts = []
    for line in sorted(lines):
        if not firsts or line - firsts[-1] > merge:
            firsts.append(line)
    return firsts


def check_rslc(rslc):
    """Metadata checks of one RSLC. Returns a dict with a `problems` list.

    Also reports the release, ISCE3 version, calibration delays and PRF per band
    for `check_pair`.
    """
    own = not isinstance(rslc, RSLC)
    r = RSLC(rslc) if own else rslc
    try:
        f = r._file
        ident = f["science/LSAR/identification"]
        meta = r._root["metadata"]
        out = dict(granule=r.granule_id, release=_decode(ident["compositeReleaseId"][()])
                   if "compositeReleaseId" in ident else None,
                   isce3=_decode(meta["processingInformation/algorithms/softwareVersion"][()])
                   if "processingInformation/algorithms/softwareVersion" in meta else None,
                   input_data_exception=bool(ident["hasInputDataException"][()])
                   if "hasInputDataException" in ident else None,
                   bands={}, problems=[])
        for freq in r.frequencies:
            sw = r.swath(freq)
            cal = f"calibrationInformation/frequency{freq}/commonDelay"
            lines, fractions = isolated_drops(r.valid_samples(freq))
            out["bands"][freq] = dict(
                prf=float(sw["nominalAcquisitionPRF"][()]) if "nominalAcquisitionPRF" in sw else None,
                common_delay=float(meta[cal][()]) if cal in meta else None,
                isolated_drops=lines.tolist(), drop_fractions=np.round(fractions, 3).tolist())
            if lines.size:
                out["problems"].append(
                    f"frequency {freq}: {lines.size} isolated line(s) mark {100 * fractions.max():.0f}% or less of "
                    f"the swath invalid (lines {lines[:5].tolist()}); RSLC.valid_samples(repair=True) fixes them")
        if out["input_data_exception"]:
            out["problems"].append("the L0B had input data exceptions (see inputDataExceptionMask)")
        return out
    finally:
        if own:
            r.close()


def check_pair(reference, secondary, prf_tolerance=1e-4):
    """Differences that make two RSLCs unfit to pair as delivered. Returns a list of strings.

    Different releases or ISCE3 versions, different calibration delays (range
    offsets of metres), or PRFs further apart than `prf_tolerance` (a time-scale
    error in one of them; see `nisar_tools.l0b`).
    """
    a = reference if isinstance(reference, dict) else check_rslc(reference)
    b = secondary if isinstance(secondary, dict) else check_rslc(secondary)
    out = []
    for key in ("release", "isce3"):
        if a[key] != b[key]:
            out.append(f"{key}: {a[key]} vs {b[key]}")
    for freq in sorted(set(a["bands"]) & set(b["bands"])):
        ba, bb = a["bands"][freq], b["bands"][freq]
        if ba["common_delay"] != bb["common_delay"]:
            out.append(f"frequency {freq} commonDelay: {ba['common_delay']} vs {bb['common_delay']} m")
        if ba["prf"] and bb["prf"] and abs(bb["prf"] / ba["prf"] - 1) > prf_tolerance:
            out.append(f"frequency {freq} PRF: {ba['prf']:.4f} vs {bb['prf']:.4f} Hz")
    return out


# -- band B's range offset ------------------------------------------------------------
def _lowpass(a, fs, bandwidth):
    spectrum = np.fft.fft(a, axis=1)
    freqs = np.fft.fftfreq(a.shape[1], 1 / fs)
    return np.fft.ifft(spectrum * (np.abs(freqs) <= bandwidth / 2), axis=1)


def _upsample(b, factor):
    n = b.shape[1]
    spectrum = np.fft.fft(b, axis=1)
    padded = np.zeros((b.shape[0], n * factor), complex)
    half = n // 2
    padded[:, :half], padded[:, -(n - half):] = spectrum[:, :half], spectrum[:, half:]
    return np.fft.ifft(padded, axis=1) * factor


def _log_power(z, group):
    p = np.abs(z) ** 2
    p = p[: p.shape[0] // group * group].reshape(-1, group, p.shape[1]).mean(axis=1)
    q = np.log(p + 1e-12 * np.max(p))
    return q - q.mean(axis=1, keepdims=True)


def _shift_samples(a, b, search):
    """Shift `s` (samples) maximising sum over rows of a(r) b(r + s), refined by a parabola; and the peak."""
    n = a.shape[1]
    taper = np.hanning(n)
    fa, fb = np.fft.rfft(a * taper, 2 * n, axis=1), np.fft.rfft(b * taper, 2 * n, axis=1)
    c = np.fft.irfft(np.conj(fa) * fb, 2 * n, axis=1).sum(axis=0)
    lags = np.r_[np.arange(-search, 0), np.arange(0, search + 1)]
    vals = np.r_[c[-search:], c[:search + 1]]
    i = int(np.argmax(vals))
    d = 0.0
    if 0 < i < vals.size - 1:
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    peak = vals[i] / max(np.sqrt(((a * taper) ** 2).sum() * ((b * taper) ** 2).sum()), 1e-30)
    return lags[i] + d, float(peak)


@dataclasses.dataclass
class BandOffset:
    """Band B's slant-range offset against A, per data window and range bin (`band_offset`).

    `blocks` holds each measurement: zero-Doppler time `t` (s since the RSLC's
    epoch), slant range `r` (m), `a_minus_b` (m; A's range of a feature minus
    B's) and its correlation `peak`. `times` are the data windows' edges and
    `ranges` the range bins' edges. `offset[i, j]` is the metres B's ground lies
    beyond A's in window `i`, bin `j` (the median of `-a_minus_b`, filled from
    the same window, else the nearest, where nothing was measured: `measured`
    says which). `k` is that offset in whole clock periods after B's
    `calibration` and a typical ionosphere, for diagnosis: the correction
    (`shift`) uses the measured offset itself, which also takes out the
    calibration and the dispersive delay.
    """

    granule: str
    blocks: list
    times: np.ndarray
    ranges: np.ndarray
    offset: np.ndarray
    measured: np.ndarray
    calibration: float = B_CALIBRATION_M
    ionosphere: float = IONOSPHERE_GUESS_M

    @property
    def k(self):
        """`offset` in whole periods of the 48 MHz clock (diagnostic)."""
        return np.rint((self.offset - self.calibration + self.ionosphere) / CLOCK_PERIOD_M).astype(int)

    def _cell(self, t, r):
        i = np.clip(np.searchsorted(self.times, np.asarray(t, float), side="right") - 1, 0, self.offset.shape[0] - 1)
        j = np.clip(np.searchsorted(self.ranges, np.asarray(r, float), side="right") - 1, 0,
                    self.offset.shape[1] - 1)
        return i, j

    def shift(self, t, r):
        """Metres to add to band B's slant range at zero-Doppler time `t`, slant range `r`."""
        return self.offset[self._cell(t, r)]

    def summary(self):
        """One line per data window, for logs."""
        rows = [f"{self.granule}: band B lies beyond A by k x {CLOCK_PERIOD_M:.4f} m + ~{self.calibration} m, "
                f"by range bin of {np.diff(self.ranges).mean() / 1e3:.0f} km"]
        for i in range(self.offset.shape[0]):
            cells = " ".join(f"{v:5.1f}{'' if m else '*'}" for v, m in zip(self.offset[i], self.measured[i]))
            rows.append(f"  {self.times[i]:.1f}..{self.times[i + 1]:.1f} s: k {' '.join(str(v) for v in self.k[i])}"
                        f"  offset {cells} m")
        return "\n".join(rows + ["  (* filled from the nearest measured cell)"])


def _fill(values, measured):
    """Fill the unmeasured cells of a (windows, bins) grid: the window's median, else the nearest window's."""
    out = values.copy()
    for i in range(values.shape[0]):
        if measured[i].any():
            out[i, ~measured[i]] = np.median(values[i, measured[i]])
    rows = np.flatnonzero(measured.any(axis=1))
    if rows.size == 0:
        return np.zeros_like(values)
    for i in np.flatnonzero(~measured.any(axis=1)):
        out[i] = out[rows[np.argmin(np.abs(rows - i))]]
    return out


def band_offset(rslc, step=2.0, bins=6, window=4096, lines=512, group=32, min_peak=0.6,
                limits=(-40.0, 5.0), margin=0.5, ionosphere=IONOSPHERE_GUESS_M, calibration=B_CALIBRATION_M):
    """Measure band B's slant-range offset against band A on the same ground.

    Every `step` seconds, a `lines`-line block of A, low-passed to B's
    bandwidth, and of B, upsampled to A's sampling, are reduced to
    azimuth-averaged log power in `window`-sample range windows, one per range
    bin of `bins` across the swath, and cross-correlated in range.
    Measurements with a correlation peak under `min_peak` or an A - B outside
    `limits` (m) are dropped; the rest give a median per data window
    (`window_changes`) and bin, leaving out `margin` seconds at each window's
    edges, where the synthetic aperture mixes the two windows. The offset changes at data-window updates and,
    on some granules, across range within one window.

    Needs both bands, with B's range grid nested in A's (same first range, a
    whole number of A samples per B sample), as in NISAR's 40 + 5 MHz mode.
    About 20 s per frame.
    """
    own = not isinstance(rslc, RSLC)
    r = RSLC(rslc) if own else rslc
    try:
        if "B" not in r.frequencies:
            raise ValueError(f"{r.granule_id} has no band B")
        ga, gb = r.radar_grid("A"), r.radar_grid("B")
        factor = int(round(gb.range_pixel_spacing / ga.range_pixel_spacing))
        if (abs(gb.range_pixel_spacing / ga.range_pixel_spacing - factor) > 1e-6
                or abs(ga.starting_range - gb.starting_range) > 1e-3):
            raise NotImplementedError("band B's range grid is not nested in band A's")
        fs = 299792458.0 / (2 * ga.range_pixel_spacing)
        bandwidth = float(r.swath("B")["processedRangeBandwidth"][()])
        da, db = r.image("A", r.polarizations("A")[0]), r.image("B", r.polarizations("B")[0])
        va = r.valid_samples("A")
        first = np.min([v[:, 0][v[:, 1] > v[:, 0]].min() for v in va if np.any(v[:, 1] > v[:, 0])])
        last = np.max([v[:, 1][v[:, 1] > v[:, 0]].max() for v in va if np.any(v[:, 1] > v[:, 0])])
        ranges = ga.slant_range(np.linspace(first, last, bins + 1))
        t = r.zero_doppler_time
        search = 8 * factor
        quantum = factor * 8                     # whole B samples, a few of them in from each edge
        blocks = []
        for l0 in range(lines, ga.length - 2 * lines, max(int(step * ga.prf), lines)):
            l0 = (l0 // lines) * lines
            lo = int(max(v[l0:l0 + lines, 0].max() for v in va[:1]))
            hi = int(min(v[l0:l0 + lines, 1].min() for v in va[-1:]))
            for centre in 0.5 * (np.linspace(first, last, bins + 1)[:-1] + np.linspace(first, last, bins + 1)[1:]):
                c0 = int(centre - window / 2) // quantum * quantum
                if c0 < lo + quantum or c0 + window > hi - quantum:
                    continue
                a = da[l0:l0 + lines, c0:c0 + window].astype(np.complex128)
                b = db[l0:l0 + lines, c0 // factor:(c0 + window) // factor].astype(np.complex128)
                if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.mean(b == 0) > 0.01:
                    continue
                s, peak = _shift_samples(_log_power(_lowpass(a, fs, bandwidth), group),
                                         _log_power(_upsample(b, factor), group), search)
                a_minus_b = float(-s * ga.range_pixel_spacing)
                if peak >= min_peak and limits[0] <= a_minus_b <= limits[1]:
                    blocks.append(dict(t=float(t[l0 + lines // 2]), r=float(ga.slant_range(c0 + window / 2)),
                                       a_minus_b=a_minus_b, peak=peak))
        edges = [float(t[i]) for i in window_changes(va)]
        times = np.array([float(t[0])] + [e for e in edges if t[0] < e < t[-1]] + [float(t[-1]) + 1.0 / ga.prf])
        medians = np.full((times.size - 1, bins), np.nan)
        for i in range(times.size - 1):
            for j in range(bins):
                inside = [blk["a_minus_b"] for blk in blocks
                          if times[i] + margin <= blk["t"] < times[i + 1] - margin and ranges[j] <= blk["r"] < ranges[j + 1]]
                if inside:
                    medians[i, j] = float(np.median(inside))
        measured = np.isfinite(medians)
        return BandOffset(granule=r.granule_id, blocks=blocks, times=times, ranges=ranges,
                          offset=_fill(-np.nan_to_num(medians), measured), measured=measured,
                          calibration=calibration, ionosphere=ionosphere)
    finally:
        if own:
            r.close()
