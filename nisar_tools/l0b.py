"""Pulse timing of NISAR L0B (RRSD) granules: the check ISCE3 does not make.

An L0B tags every echo with the UTC transmit time of its pulse (`UTCtime`),
numbers the pulses (`rangeLineIndex`) and carries the commanded
pulse-repetition intervals (`PRISequence`, cycled: the dither table of a
dithered mode). ISCE3 v0.25.16 focuses with `UTCtime` as given and takes the
PRF as `(N - 1) / (t_last - t_first)`, so two defects of delivered L0Bs pass
straight into the RSLC:

- **slips**: a run of echoes tagged with the previous (or next) pulse's transmit
  time. That stretch of the image lands one pulse early (late): about 3.5 m along
  track, and ~3 rad of carrier phase in an interferogram that crosses it;
- **time-scale errors**: a run of tags squeezed or stretched in time. The PRF
  derived from the whole datatake is then wrong, and every frame focused from it
  is defocused and misplaced, even where its own tags are right.

`pulse_timing` fits the dither table to the tags and classifies every echo;
`timing_report` places what it finds in an RSLC's zero-Doppler time, and gives
the one-pulse azimuth shift that `geocode_rslc(azimuth_shift=)` corrects a slip
with.
"""

import dataclasses
from pathlib import Path

import numpy as np

from . import radar

RRSD = "science/LSAR/RRSD"
#: A tag this close (s) to its fitted table time counts as consistent.
TOLERANCE = 10e-6

CONSISTENT, EARLY, LATE, INCONSISTENT, RELABELLED = 0, -1, 1, 2, 3
KIND_NAMES = {CONSISTENT: "consistent", EARLY: "previous pulse's time",
              LATE: "next pulse's time", INCONSISTENT: "inconsistent",
              RELABELLED: "time of another pulse"}


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _open(source):
    """`(h5py.File or group, close_after, name)` for a path, an http(s) URL or an open file."""
    import h5py

    if isinstance(source, (h5py.File, h5py.Group)):
        return source, False, Path(str(source.file.filename)).stem
    text = str(source)
    if text.startswith(("http://", "https://")):
        from ._earthdata import RangeFile

        return h5py.File(RangeFile(text), "r"), True, Path(text.split("?", 1)[0]).stem
    return h5py.File(text, "r"), True, Path(text).stem


def _table_times(k, table, phase):
    """Times of pulses `k` (counted from the first) under the cycled table, starting at its `phase`."""
    reps = int(k.max() // table.size) + 4
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, reps))])
    return cum[k + phase] - cum[phase]


class UnanchoredTags(ValueError):
    """No run of an L0B's tags follows its commanded pulse sequence long enough to anchor the sequence.

    The relative times are still the table's, if the radar flew it
    (`tx_gap_phase` checks that from the transmit gaps), but the constant
    offset (where the image lies along track) cannot come from the tags; it has
    to come from elsewhere, such as a registration to another date.
    """


def _fit_table(utc, line_index, table, tolerance=TOLERANCE, anchor="latest", min_anchor=1.0, gap_phase=None):
    """The commanded table's times that the tags fit: `(model, phase, drift, anchored)`.

    Every defect seen in delivered L0Bs puts tags behind their pulses (slips
    carry an earlier pulse's time; squeezed spans, relabelled and random tags
    fall behind and stay behind); none runs ahead. The candidate models are a
    phase (`gap_phase`, from the transmit gaps; else the tags' most common
    phase and two pulses either side) and a stretch of tags at least
    `min_anchor` seconds long on one level, which slipped runs of a dithered
    table never are. With `anchor="latest"` the latest model is taken, with
    `"majority"` the longest stretch's. Without any such stretch (`anchored`
    False) the constant comes from the latest pair of neighbouring tags, so a
    lone stray tag cannot set it. The clock drift is fitted on the agreeing
    tags when they span `min_anchor` seconds.
    """
    if anchor not in ("latest", "majority"):
        raise ValueError(f"anchor must be 'latest' or 'majority', not {anchor!r}")
    k = (line_index - line_index[0]).astype(np.int64)
    if gap_phase is not None:
        phases = [int(gap_phase) % table.size]
    else:
        steps = np.diff(utc)
        errors = [np.median(np.abs(steps - np.diff(_table_times(k, table, p)))) for p in range(table.size)]
        phase = int(np.argmin(errors))
        # the tags must follow the table (at this phase or a slip or two away) for min_anchor seconds, or mostly
        need = min_anchor * table.size / table.sum()
        agree = [np.abs(steps - np.diff(_table_times(k, table, (phase + j) % table.size))) < tolerance
                 for j in (-2, -1, 0, 1, 2)]
        longest = max(_longest_run(a) for a in agree)
        share = float(np.mean(np.any(agree, axis=0)))
        if longest < need and share < 0.5:
            raise UnanchoredTags(
                f"no run of the tags follows the commanded pulse sequence for {min_anchor:g} s (longest "
                f"{longest * table.mean():.2f} s; {100 * share:.1f}% of steps agree within two pulses of one "
                f"phase); the tags span {utc[-1] - utc[0]:.2f} s where the sequence takes "
                f"{_table_times(k, table, phase)[-1]:.2f} s")
        phases = [(phase + j) % table.size for j in (0, 1, -1, 2, -2)]
    stretches = []                           # (level, length, phase): the model's time of the first echo is `level`
    for phase in phases:
        seq = _table_times(k, table, phase)
        lag = utc - seq
        flat = np.abs(np.diff(lag)) < tolerance
        edges = np.flatnonzero(np.diff(np.r_[0, flat.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if seq[b] - seq[a] >= min_anchor:
                stretches.append((float(np.median(lag[a:b + 1])), float(seq[b] - seq[a]), phase))
    if stretches:
        level, _, phase = max(stretches, key=(lambda x: x[0]) if anchor == "latest" else (lambda x: x[1]))
    else:
        phase = phases[0]
        lag = utc - _table_times(k, table, phase)
        level = float(np.minimum(lag[:-1], lag[1:]).max())
    seq = _table_times(k, table, phase)
    model, drift = seq + level, 0.0
    for _ in range(2):
        near = np.abs(utc - model) < tolerance
        if near.sum() < 2 or np.ptp(seq[near]) < min_anchor:
            break
        x = seq - seq[near][0]
        design = np.column_stack([np.ones(int(near.sum())), x[near]])
        (offset, drift), *_ = np.linalg.lstsq(design, (utc - seq)[near], rcond=None)
        model, drift = seq + offset + drift * x, float(drift)
    return model, phase, drift, bool(stretches)


def _longest_run(mask):
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return int(np.max(edges[1::2] - edges[::2]))


@dataclasses.dataclass
class PulseTiming:
    """The echo times of one L0B band, checked against its dither table (`pulse_timing`).

    Times are seconds since `epoch`. `utc` is `UTCtime` as stored; `model` the
    commanded table fitted to it; `kind` one of `CONSISTENT`, `EARLY` (tagged
    with the previous pulse's time), `LATE` (the next pulse's) or
    `INCONSISTENT` per echo, or `RELABELLED` where the tags follow the table
    `tag_shift` pulses off the transmit gaps (`tx_gap_phase`), whose phase the
    model then takes. `anchored` is False when no second of tags agrees with
    the model: it then rests on the latest pair of tags.
    """

    granule: str
    frequency: str
    epoch: np.datetime64
    utc: np.ndarray
    line_index: np.ndarray
    model: np.ndarray
    kind: np.ndarray
    table: np.ndarray
    table_phase: int
    drift: float
    gap_phase: int = None
    gap_match: float = None
    tag_shift: int = 0
    anchored: bool = True
    phase_source: str = "tags"
    limits_phase: int = None
    tx: str = "txH"

    @property
    def residual(self):
        """`utc - model` (s)."""
        return self.utc - self.model

    @property
    def prf_commanded(self):
        """The mean PRF of the commanded table (Hz)."""
        return self.table.size / float(self.table.sum())

    @property
    def prf_from_tags(self):
        """ISCE3's `getNominalPRF`: `(N - 1) / (t_last - t_first)` of the tags (Hz)."""
        return (self.utc.size - 1) / float(self.utc[-1] - self.utc[0])

    @property
    def prf_from_model(self):
        """The same formula on the table times: what ISCE3 would have found from correct tags.

        It differs from `prf_commanded` by up to a few hundred ppm when the
        datatake holds a partial cycle of a dithered table.
        """
        return (self.model.size - 1) / float(self.model[-1] - self.model[0])

    @property
    def corrected(self):
        """The tags, with every one that is not consistent replaced by its table time."""
        return np.where(self.kind == CONSISTENT, self.utc, self.model)

    @property
    def ok(self):
        """No echo departs from the table."""
        return bool(np.all(self.kind == CONSISTENT))

    def segments(self):
        """The runs of echoes that are not consistent, as dicts.

        Each has `kind` (a `KIND_NAMES` value; a run mixing kinds is
        inconsistent), `code`, `first` and `last` echo index, `count`, and
        `start` / `end`: the run's first and last table time, the echoes' true
        transmit times (seconds since `epoch`).
        """
        # runs of consistent, relabelled and other echoes (a run mixing the others is inconsistent)
        group = np.where(self.kind == CONSISTENT, 0, np.where(self.kind == RELABELLED, 1, 2))
        change = np.flatnonzero(np.diff(group)) + 1
        out = []
        for a, b in zip(np.r_[0, change], np.r_[change, group.size]):
            if group[a] == 0:
                continue
            kinds = np.unique(self.kind[a:b])
            code = int(kinds[0]) if kinds.size == 1 else INCONSISTENT
            out.append(dict(kind=KIND_NAMES[code], code=code, first=int(a), last=int(b - 1), count=int(b - a),
                            start=float(self.model[a]), end=float(self.model[b - 1])))
        return out

    def summary(self):
        """One line per finding, for logs."""
        head = (f"{self.granule} {self.frequency}: {self.utc.size} echoes, commanded PRF "
                f"{self.prf_commanded:.4f} Hz, from the tags {self.prf_from_tags:.4f} Hz")
        lines = [head] + [f"  {s['count']} echoes tagged with the {s['kind']}" if s["code"] in (EARLY, LATE, RELABELLED)
                          else f"  {s['count']} echoes {s['kind']}" for s in self.segments()]
        if self.phase_source != "tags":
            lines.append(f"  the table's phase, {self.table_phase}, from the "
                         + ("transmit leakage in the echoes" if self.phase_source == "leakage" else "sub-swath limits"))
        if self.phase_source == "leakage" and self.limits_phase is not None and self.limits_phase != self.table_phase:
            lines.append(f"  the sub-swath limits say phase {self.limits_phase}: they are not the radar's record")
        if self.tag_shift:
            lines.append(f"  the tags follow the table {self.tag_shift} pulses off the radar's sequence")
        behind = float(np.max(self.model - self.utc))
        if behind > 1e-3:
            lines.append(f"  the tags fall up to {behind:.3f} s behind the sequence")
        if not self.anchored:
            lines.append("  no second of tags agrees with the sequence: it is anchored on the latest tags, so check "
                         "the image's along-track position against another date")
        return "\n".join(lines + ([] if not self.ok else ["  every echo matches the table"]))


@dataclasses.dataclass
class GapPhase:
    """`tx_gap_phase`'s answer: the PRI table's phase the radar flew, and how well it predicts the gaps."""

    granule: str
    phase: int
    match: float
    pulses: int
    first_line: int

    def summary(self):
        """One line, for logs."""
        return (f"{self.granule}: the transmit gaps of {self.pulses} pulses follow the PRI table at phase "
                f"{self.phase} ({100 * self.match:.1f}% predicted to 3 samples), counted from line {self.first_line}")


def tx_gap_phase(source, frequency="A", count=519, tolerance=3.0, tx=None):
    """Which phase of the commanded PRI table the radar flew, read from its transmit gaps, not its tags.

    In SweepSAR the radar transmits later pulses while it still receives, and
    each transmit blanks the earlier pulses' receive windows where it falls:
    pulse k + M's transmit lands `(t[k + M] - t[k] - 2 r0 / c) fs` samples into
    pulse k's window, `r0` being the slant range of sample 0. Those gaps (the
    ends of all but the last valid sub-swath) depend on the PRI sequence
    alone, so they time the echoes when the tags cannot (`UnanchoredTags`):
    every echo's time follows from the table, up to one constant.

    Parameters
    ----------
    source : str, Path or h5py.File
        An L0B, as for `pulse_timing`.
    count : int
        Pulses read, from the middle of the datatake.
    tolerance : float
        Samples within which a gap counts as predicted.

    Returns
    -------
    GapPhase
        Its `phase` is `pulse_timing`'s: pulse k, counted from the L0B's first
        line, has interval `PRISequence[(k + phase) % n]`.
    """
    f, close, name = _open(source)
    try:
        band = f[f"science/LSAR/RRSD/swaths/frequency{frequency}"]
        tx = band[tx or sorted(k for k in band if k.startswith("tx"))[0]]
        table = np.atleast_1d(np.asarray(tx["PRISequence"][()], float))
        lines = tx["rangeLineIndex"]
        n = lines.shape[0]
        first = max(0, n // 2 - count // 2)
        sel = slice(first, min(n, first + count))
        line0 = int(lines[0])
        k = np.asarray(lines[sel], np.int64) - line0
        ns = int(np.asarray(tx["numberOfSubSwaths"][()]).ravel()[0])
        sub = np.stack([tx[f"validSamplesSubSwath{i + 1}"][sel] for i in range(ns)]).astype(np.int64)
        r = tx["slantRange"]
        r0, dr = float(r[0]), float(r[1] - r[0])
    finally:
        if close:
            f.close()
    bad = np.iinfo(np.uint32).max
    gaps = [np.array([sub[i, j, 1] for i in range(ns - 1) if sub[i, j, 1] != bad and sub[i + 1, j, 0] != bad], float)
            for j in range(sub.shape[1])]
    c = radar.SPEED_OF_LIGHT
    fs, tau0 = c / (2 * dr), 2 * r0 / c
    lags = np.arange(1, int(np.ceil(4 * tau0 / table.min())) + 2)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, int((k.max() + lags[-1]) // table.size) + 3))])
    best = (0, 0.0)
    for phase in range(table.size):
        resid = []
        for kk, g in zip(k, gaps):
            if g.size:
                pred = (cum[kk + phase + lags] - cum[kk + phase] - tau0) * fs
                resid.extend(g - pred[np.argmin(np.abs(g[:, None] - pred[None, :]), axis=1)])
        if resid:
            resid = np.asarray(resid)
            match = float(np.mean(np.abs(resid - np.median(resid)) <= tolerance))
            if match > best[1]:
                best = (phase, match)
    return GapPhase(granule=name, phase=best[0], match=best[1], pulses=int(sum(g.size > 0 for g in gaps)),
                    first_line=line0)


@dataclasses.dataclass
class LeakagePhase:
    """`tx_leakage_phase`'s answer: the PRI table's phase the radar flew, read from its echoes."""

    granule: str
    phase: int
    ratio: float
    runner_up: float
    offset: float
    pulses: int
    first_line: int

    @property
    def clear(self):
        """The phase stands out: its ratio is at least 1.3 times any other phase's."""
        return self.ratio >= 1.3 * self.runner_up

    def summary(self):
        """One line, for logs."""
        return (f"{self.granule}: the transmit leakage in {self.pulses} echoes follows the PRI table at phase "
                f"{self.phase} (power ratio {self.ratio:.2f}, next {self.runner_up:.2f}; offset {self.offset:+.0f} "
                f"samples), counted from line {self.first_line}")


def _echo_amplitude(dataset, rows, lut=None):
    """|echo| of `rows` of an L0B raw dataset: BFPQ codes through `lut`, or complex samples."""
    raw = dataset[rows]
    if raw.dtype.names and "r" in raw.dtype.names:
        if lut is None:
            return np.hypot(raw["r"].astype(float), raw["i"].astype(float))
        return np.hypot(lut[raw["r"]].astype(float), lut[raw["i"]].astype(float))
    return np.abs(raw).astype(float)


def tx_leakage_phase(source, frequency="A", count=200, tx=None, leak=38.52e-6, search=41.7e-6):
    """Which phase of the commanded PRI table the radar flew, read from the transmit leakage in its echoes.

    While an echo is received the radar transmits later pulses, and each
    transmit leaks into the receive window, several times brighter than the
    echo: pulse k + M's transmit lands `(t[k + M] - t[k] - 2 r0 / c) fs` samples
    into pulse k's window, up to an offset fixed by the hardware. For each
    phase (and offset), the mean amplitude in the predicted windows over the
    mean of all samples picks the radar's phase: on the datatakes checked the
    ratio is about 4 there and at most 2.4 elsewhere, at -250 samples. Unlike
    `tx_gap_phase` it reads the echoes, `count` of them mid-datatake (about
    40 MB), but the L0B's sub-swath limits can be bookkeeping: on one datatake
    they follow the table 44 pulses off, with no leakage inside them.

    Parameters
    ----------
    source : str, Path or h5py.File
        An L0B, as for `pulse_timing`.
    count : int
        Echoes read, from the middle of the datatake.
    tx : str, optional
        The transmit group, such as `"txH"` (default: the band's first).
    leak, search : float
        How long a transmit leaks into the window, and how far either side
        the hardware offset is searched (s).

    Returns
    -------
    LeakagePhase
        Its `phase` is `pulse_timing`'s, counted from the L0B's first line.
    """
    f, close, name = _open(source)
    try:
        band = f[f"science/LSAR/RRSD/swaths/frequency{frequency}"]
        tx = band[tx or sorted(k for k in band if k.startswith("tx"))[0]]
        table = np.atleast_1d(np.asarray(tx["PRISequence"][()], float))
        lines = tx["rangeLineIndex"]
        n = lines.shape[0]
        first = max(0, n // 2 - count // 2)
        sel = slice(first, min(n, first + count))
        line0 = int(lines[0])
        k = np.asarray(lines[sel], np.int64) - line0
        rx = tx[sorted(key for key in tx if key.startswith("rx"))[0]]
        data = rx[sorted(key for key in rx if len(key) == 2 and key.isupper())[0]]
        a = _echo_amplitude(data, sel, rx["BFPQLUT"][()] if "BFPQLUT" in rx else None)
        r = tx["slantRange"]
        r0, dr = float(r[0]), float(r[1] - r[0])
    finally:
        if close:
            f.close()
    ns = a.shape[1]
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    base = a.mean()
    c = radar.SPEED_OF_LIGHT
    fs, tau0 = c / (2 * dr), 2 * r0 / c
    width = max(1, int(round(leak * fs)))
    step = max(1, int(round(search * fs / 50)))
    offsets = np.arange(-50 * step, 50 * step + 1, step)
    lags = np.arange(1, int(np.ceil(4 * tau0 / table.min())) + 2)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(table, int((k.max() + lags[-1]) // table.size) + 3))])
    rows = np.broadcast_to(np.arange(a.shape[0])[:, None], (a.shape[0], lags.size))
    ratios, best_off = np.zeros(table.size), np.zeros(table.size)
    for phase in range(table.size):
        pred = (cum[k[:, None] + phase + lags[None, :]] - cum[k[:, None] + phase] - tau0) * fs
        for off in offsets:
            start = np.round(pred + off).astype(np.int64)
            ok = (start >= 0) & (start + width <= ns)
            if ok.any():
                ratio = (cs[rows[ok], start[ok] + width] - cs[rows[ok], start[ok]]).mean() / width / base
                if ratio > ratios[phase]:
                    ratios[phase], best_off[phase] = ratio, off
    order = np.argsort(ratios)[::-1]
    return LeakagePhase(granule=name, phase=int(order[0]), ratio=float(ratios[order[0]]),
                        runner_up=float(ratios[order[1]]), offset=float(best_off[order[0]]), pulses=int(a.shape[0]),
                        first_line=line0)


def pulse_timing(source, frequency="A", tolerance=TOLERANCE, anchor="latest", min_anchor=1.0, echoes="auto",
                 tx=None):
    """Check an L0B's echo times against its commanded pulse sequence.

    Parameters
    ----------
    source : str, Path or h5py.File
        A local L0B, an `https://` URL of one (read remotely, a few MB of
        metadata; see `nisar_tools.download.l0b_url`), or an open file.
    frequency : str
        The band (`"A"` or `"B"`); both share the pulses.
    tx : str, optional
        The transmit group, such as `"txH"` (default: the band's first).
    tolerance : float
        Seconds within which a tag matches a table time.
    echoes : {"auto", True, False}
        Read the table's phase from the transmit leakage in the echoes
        (`tx_leakage_phase`, about 40 MB of them), the radar's own record:
        `"auto"` does for local files and not for URLs, which fall back on the
        sub-swath limits (`tx_gap_phase`), then on the tags.
    anchor : {"latest", "majority"}
        Which run of tags is taken as right when runs disagree by whole pulses.
        `"latest"`: the run whose tags are latest, among runs of at least
        `min_anchor` seconds. Every slip measured on corner reflectors so far
        put the image early (tags carrying the previous pulse's time), and on
        one of them the slipped part was the larger, so the latest run is the
        right one. `"majority"`: the most common.

    Returns
    -------
    PulseTiming
        `radarTime` is no independent check: it carries the same slips. The
        transmit gaps are (`tx_gap_phase`): where they read the table at another
        phase than the tags, the echoes are re-timed to them (`RELABELLED`).

    Raises
    ------
    UnanchoredTags
        If no run of the tags follows the table for `min_anchor` seconds and the
        L0B has no sub-swaths to time the echoes from instead.
    """
    f, close, name = _open(source)
    try:
        ident = f.get("science/LSAR/identification")
        if ident is not None and "granuleId" in ident:
            name = _decode(ident["granuleId"][()])
        band = f[f"{RRSD}/swaths/frequency{frequency}"]
        tx_name = tx or sorted(k for k in band if k.startswith("tx"))[0]
        tx = band[tx_name]
        times = tx["UTCtime"]
        epoch = radar.parse_epoch(times.attrs["units"])
        utc = times[()].astype(np.float64)
        line_index = tx["rangeLineIndex"][()].astype(np.int64)
        if "PRISequence" in tx:
            table = tx["PRISequence"][()].astype(np.float64)
        else:
            table = np.array([1.0 / float(tx["nominalAcquisitionPRF"][()])])
        try:
            gaps = tx_gap_phase(f, frequency, tx=tx_name) if table.size > 1 else None
        except (KeyError, ValueError, IndexError):
            gaps = None                              # no sub-swaths to read the transmit gaps from
        leak = None
        if table.size > 1 and (echoes is True or (echoes == "auto" and not str(source).startswith("http"))):
            try:
                leak = tx_leakage_phase(f, frequency, tx=tx_name)
            except (KeyError, ValueError, IndexError):
                leak = None                          # no echoes to read the leakage from
    finally:
        if close:
            f.close()
    limits_phase = gaps.phase if gaps is not None and gaps.match >= 0.98 else None
    if leak is not None and leak.clear:
        phase_source = "leakage"
        gaps = GapPhase(granule=name, phase=leak.phase, match=1.0, pulses=leak.pulses, first_line=leak.first_line)
    elif gaps is not None and gaps.match >= 0.98:
        phase_source = "sub-swaths"
    else:
        phase_source, gaps = "tags", None
    model, phase, drift, anchored = _fit_table(utc, line_index, table, tolerance, anchor, min_anchor,
                                               gap_phase=None if gaps is None else gaps.phase)
    k = line_index - line_index[0]
    d = utc - model
    pri_before = np.r_[np.nan, np.diff(model)]               # the interval before each echo
    pri_after = np.r_[np.diff(model), np.nan]
    kind = np.full(utc.size, INCONSISTENT, np.int8)
    kind[np.abs(d + pri_before) < tolerance] = EARLY          # tagged with the previous pulse's time
    kind[np.abs(d - pri_after) < tolerance] = LATE            # the next pulse's
    kind[np.abs(d) < tolerance] = CONSISTENT
    tag_shift = 0
    if gaps is not None and utc.size > 2:
        # tags that follow the table at another phase than their transmits: relabelled
        steps = np.diff(utc)
        errors = [np.median(np.abs(steps - np.diff(_table_times(k, table, p)))) for p in range(table.size)]
        own = int(np.argmin(errors))
        if own != phase:
            follows = np.r_[False, np.abs(steps - np.diff(_table_times(k, table, own))) < tolerance]
            need = min_anchor * table.size / table.sum()          # a second of them, not a chance match
            edges = np.flatnonzero(np.diff(np.r_[0, follows.astype(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= need:
                    run = kind[a:b]
                    run[run == INCONSISTENT] = RELABELLED
                    tag_shift = (phase - own) % table.size
                    tag_shift -= table.size if tag_shift > table.size // 2 else 0
    return PulseTiming(granule=name, frequency=frequency, epoch=epoch, utc=utc, line_index=line_index, model=model,
                       kind=kind, table=table, table_phase=int(phase), drift=drift,
                       gap_phase=None if gaps is None else gaps.phase, gap_match=None if gaps is None else gaps.match,
                       tag_shift=int(tag_shift), anchored=anchored, phase_source=phase_source,
                       limits_phase=limits_phase, tx=tx_name)


def pulse_timings(source, **kwargs):
    """`pulse_timing` for every band and transmit group of an L0B: `{(frequency, tx): PulseTiming}`.

    The bands share their pulses: where band B's tags and line indices are
    band A's, it gets A's timing rather than a second reading of its echoes.
    `kwargs` go to `pulse_timing`.
    """

    f, close, _ = _open(source)
    try:
        swaths = f[f"{RRSD}/swaths"]
        groups = [(k[-1], t) for k in sorted(swaths) if k.startswith("frequency")
                  for t in sorted(swaths[k]) if t.startswith("tx")]
        tags = {key: (swaths[f"frequency{key[0]}/{key[1]}/UTCtime"][()],
                      swaths[f"frequency{key[0]}/{key[1]}/rangeLineIndex"][()]) for key in groups}
    finally:
        if close:
            f.close()
    out = {}
    for frequency, tx in groups:
        same = next((key for key in out if key[1] == tx and np.array_equal(tags[key][0], tags[(frequency, tx)][0])
                     and np.array_equal(tags[key][1], tags[(frequency, tx)][1])), None)
        if same is not None:
            out[(frequency, tx)] = dataclasses.replace(out[same], frequency=frequency)
        else:
            out[(frequency, tx)] = pulse_timing(source, frequency, tx=tx, **kwargs)
    return out


@dataclasses.dataclass
class TimingReport:
    """The pulse timing behind one RSLC, placed in its zero-Doppler time (`timing_report`).

    Times are seconds since the RSLC's epoch. `problems` lists each run of bad
    echoes as a dict with its `kind`, `start` / `end` transmit times, `count`,
    the zero-Doppler span it fully decides (`full`, or None) and the spans where
    the synthetic aperture mixes it with good echoes (`transitions`).
    """

    granule: str
    timings: list
    problems: list
    prf_product: float
    prf_expected: float
    prf_commanded: float
    span: tuple
    lead: tuple
    aperture: float

    @property
    def prf_error(self):
        """Relative error of the PRF the RSLC was focused with, against the same
        formula on correct times (`PulseTiming.prf_from_model`)."""
        return self.prf_product / self.prf_expected - 1.0

    @property
    def verdict(self):
        """`"ok"`, `"correctable"` (whole-pulse slips over parts of the frame, see
        `azimuth_shift`) or `"refocus"` (a wrong PRF, an inconsistent run, or a slip's
        transition inside the frame)."""
        if abs(self.prf_error) > 1e-4:
            return "refocus"
        inside = [p for p in self.problems if _overlaps(p, self.span)]
        if not inside:
            return "ok"
        if any(p["code"] in (INCONSISTENT, RELABELLED) for p in inside):
            return "refocus"
        if any(_overlap(t, self.span) for p in inside for t in p["transitions"]):
            return "refocus"
        return "correctable"

    def azimuth_shift(self, t):
        """Seconds to add to zero-Doppler time `t` to find its content in this RSLC.

        `-PRI` (the table's mean) where a run of echoes tagged with the previous
        pulse's time fully decides the pixel, `+PRI` for the next pulse's, 0
        elsewhere. The table's dither is not undone: a slipped echo's error is
        the interval before it, which varies by about +-0.1 ms around the mean.
        """
        t = np.asarray(t, float)
        out = np.zeros(t.shape)
        for p in self.problems:
            if p["code"] in (EARLY, LATE) and p["full"] is not None:
                a, b = p["full"]
                out[(t >= a) & (t <= b)] = p["code"] / self.prf_commanded
        return out

    def transition(self, t):
        """Whether zero-Doppler time `t` lies where a bad run mixes with good echoes,
        or anywhere an inconsistent run decides."""
        t = np.asarray(t, float)
        out = np.zeros(t.shape, bool)
        for p in self.problems:
            spans = list(p["transitions"]) + ([p["full"]] if p["code"] == INCONSISTENT and p["full"] else [])
            for a, b in spans:
                out |= (t >= a) & (t <= b)
        return out

    def summary(self):
        """Readable lines, for logs."""
        lines = [f"{self.granule}: {self.verdict}; PRF {self.prf_product:.4f} Hz as focused, "
                 f"{self.prf_expected:.4f} Hz from correct times ({1e6 * self.prf_error:+.0f} ppm); "
                 f"table mean {self.prf_commanded:.4f} Hz"]
        for p in self.problems:
            where = "inside the frame" if _overlaps(p, self.span) else "outside the frame"
            lines.append(f"  {p['count']} echoes {p['kind']} at {p['start']:.3f}..{p['end']:.3f} s, {where}")
        return "\n".join(lines)


def _overlap(interval, span):
    return interval is not None and interval[0] <= span[1] and interval[1] >= span[0]


def _overlaps(problem, span):
    return _overlap(problem["full"], span) or any(_overlap(t, span) for t in problem["transitions"])


def _beam_geometry(rslc, frequency):
    """`((lead_min, lead_max), aperture)`: how long before its zero-Doppler time
    the beam centre crosses a target, across the swath, and the synthetic aperture (s)."""
    grid = rslc.radar_grid(frequency)
    orbit = rslc.orbit
    ranges = grid.starting_range + np.array([0.02, 0.5, 0.98]) * (grid.width - 1) * grid.range_pixel_spacing
    t_mid = np.full(ranges.size, grid.sensing_mid)
    llh, _ = radar.rdr2geo(t_mid, ranges, orbit, height=0.0, wavelength=grid.wavelength,
                           look_side=grid.look_side, threshold=1e-5)
    t_beam, _, ok = radar.geo2rdr(llh, orbit, rslc.doppler_centroid(frequency), grid.wavelength,
                                  grid.look_side, t0=t_mid)
    lead = (t_mid - t_beam)[ok]
    _, vel = orbit.interpolate(np.array([grid.sensing_mid]))
    speed = float(np.linalg.norm(vel))
    sw = rslc.swath(frequency)
    bandwidth = float(sw["processedAzimuthBandwidth"][()]) if "processedAzimuthBandwidth" in sw else grid.prf
    aperture = bandwidth * grid.wavelength * float(ranges.max()) / (2.0 * speed ** 2)
    return (float(lead.min()), float(lead.max())), aperture


def timing_report(rslc, l0b=None, frequency="A", tolerance=TOLERANCE):
    """Check the pulse timing behind an RSLC and place each defect in its zero-Doppler time.

    Parameters
    ----------
    rslc : str, Path or `nisar_tools.rslc.RSLC`
    l0b : list, optional
        The L0Bs it was focused from: local paths, URLs or `PulseTiming`s.
        Default: the RSLC's `l0bGranules`, looked up on ASF and read remotely.
    frequency : str
        The band whose geometry places the problems.

    Returns
    -------
    TimingReport
        A pixel whose beam-centre crossing is within half an aperture of a run's
        edge mixes good and bad echoes (`transitions`); one wholly inside a run
        of slipped echoes is just shifted by a pulse (`full`).
    """
    from .rslc import RSLC

    own = not isinstance(rslc, RSLC)
    r = RSLC(rslc) if own else rslc
    try:
        if l0b is None:
            from .download import l0b_url

            names = r._root["metadata/processingInformation/inputs/l0bGranules"][()]
            l0b = [l0b_url(Path(_decode(n)).stem) for n in np.atleast_1d(names)]
        timings = [x if isinstance(x, PulseTiming) else pulse_timing(x, frequency, tolerance) for x in l0b]
        (lead_min, lead_max), aperture = _beam_geometry(r, frequency)
        half = 0.5 * aperture
        span = (float(r.zero_doppler_time[0]), float(r.zero_doppler_time[-1]))
        problems = []
        for pt in timings:
            shift = (pt.epoch - r.ref_epoch) / np.timedelta64(1, "s")
            for s in pt.segments():
                a, b = s["start"] + shift, s["end"] + shift
                full = (a + lead_max + half, b + lead_min - half)
                edges = [(x + lead_min - half, x + lead_max + half) for x in (a, b)]
                problems.append(dict(s, granule=pt.granule, start=a, end=b,
                                     full=full if full[0] < full[1] else None, transitions=edges))
        prf_product = float(r.swath(frequency)["nominalAcquisitionPRF"][()])
        if timings:
            # the L0B ISCE3 took the PRF from is the one whose tags give the product's value
            used = min(timings, key=lambda pt: abs(pt.prf_from_tags - prf_product))
            expected, commanded = used.prf_from_model, used.prf_commanded
        else:
            expected = commanded = prf_product
        return TimingReport(granule=r.granule_id or Path(r.path).stem, timings=timings, problems=problems,
                            prf_product=prf_product, prf_expected=expected, prf_commanded=commanded, span=span,
                            lead=(lead_min, lead_max), aperture=aperture)
    finally:
        if own:
            r.close()
