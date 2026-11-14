# Checking an RSLC before use

ISCE3 v0.25.16, which focused NISAR's provisional products, passes some defects of the raw data into the RSLC without
flagging them. Three checks find them in seconds; run them before an RSLC goes into an interferogram.

```python
from nisar_tools.l0b import timing_report
from nisar_tools.qa import band_offset, check_pair, check_rslc

check_rslc(rslc)["problems"]       # release, PRF, calibration delays, input-data exceptions, mask defects
check_pair(reference, secondary)   # [] when the two were focused alike
print(timing_report(rslc).summary())   # pulse timing, and a datatake start the mask leaves out
print(band_offset(rslc).summary())
```

## Pulse timing

An L0B tags every echo with its pulse's transmit time and carries the commanded pulse sequence (`PRISequence`, the
dither table). ISCE3 focuses with the tags as stored. `pulse_timing` fits the sequence to the tags and finds:

- **slips**: runs of echoes tagged with the previous (or next) pulse's time. That stretch of the image sits a pulse
  (about 3.5 m along track) early or late, and an interferogram across it gains about 3 rad of carrier phase;
- **time-scale errors**: a squeezed span of tags. ISCE3 takes the PRF from the first and last tag, so the whole
  datatake is focused with a wrong PRF, defocused and misplaced, even where its own tags are right.

`timing_report(rslc)` reads the L0Bs the RSLC names from ASF (a few MB of metadata each, with Earthdata credentials in
`~/.netrc`) and places each defect in the RSLC's zero-Doppler time. Its `verdict` is:

- `"ok"`;
- `"correctable"`: slipped spans cover parts of the frame whole, and `geocode_rslc(azimuth_shift=report)` moves them
  back;
- `"refocus"`: a wrong PRF, an inconsistent span, or the edge of a slip inside the frame, where one synthetic aperture
  mixes good and slipped echoes; nothing done to the RSLC undoes it. Use another date, or [refocus the
  L0B](refocusing.md).

The L0B's `radarTime` carries the same slips, so the record alone cannot say which side of a slip is right. The run
whose tags are latest is taken as right: every slip measured against corner reflectors put the image early.

The echoes keep a timing record of their own. While a pulse's echo is received the radar transmits later pulses, and
each transmit leaks into the receive window, several times brighter than the echo. Where it lands depends on the pulse
sequence alone, so `tx_leakage_phase` reads which phase of the commanded sequence the radar flew from the echoes,
without the tags. The L0B's sub-swath limits mark the same windows (`tx_gap_phase`, cheaper, with no echoes to read),
but they can be bookkeeping rather than a record: on one datatake they follow the sequence 44 pulses off, with no
leakage inside them. `pulse_timing` takes the phase from the echoes when it reads them (by default for local files),
and checks the tags against it, which finds two more defects:

- **relabelled tags**: tags that follow the sequence exactly, but a few pulses off the transmits, so every echo carries
  another pulse's time. The error varies from pulse to pulse with the dither, so the RSLC is defocused and only a
  refocus repairs it (`"refocus"`);
- **unanchored tags**: no run of tags follows the sequence at all. The echoes are then timed from the transmit gaps,
  anchored on the first tag, whose own error remains as a constant along-track shift to measure against another date.

## Datatake starts

The data window updates about every 10 s. ISCE3 v0.25.16 takes each window's valid region from its first pulse, and a
datatake's first L0B starts with a few hundred empty pulses, so the datatake's first window gets none:
`validSamplesSubSwath` marks the image fully focused only from half a synthetic aperture after the second window starts,
though the first is focused like the rest. A frame holding a datatake start can lose most of its lines to the mask, and
GSLCs and interferograms made from it drop them. On a map the false edge is a line of constant time, slanted at the
track's heading. ISCE3's development branch has fixed it.

`masked_segments` finds such windows in an L0B. `timing_report(rslc)` reads them with the pulse timing:
`masked_start` is the time the mask loses in the frame, `repair_mask` restores it in a band's valid samples, and
`geocode_rslc(azimuth_shift=report)` does so in the GSLC's mask (unless `isce3_compatible=True`). Only the mask
changes; the pixels were right.

The strip of band A at near range is different. Band B's pulse follows band A's by about 31 µs in one receive window,
both bands share a grid that starts at B's nearer range, and band A's first ~5 km of slant range hold echoes recorded
only in part. ISCE3 marks that strip correctly, and nothing recovers it.

## Band B's range

Band B's slant ranges are off by whole periods of the 48 MHz clock it is decimated from (3.12 m each): constant within
a data window (they update every ~10 s), the same on most dates of a track, and on some granules different across
range. On a pair it does not misregister B against itself, but it misplaces B against A, and a split-spectrum screen
then mistakes the deformation's range gradient for ionosphere (about 12 times the offset times the gradient).
`band_offset` measures the offset per data window and range bin by cross-correlating the two bands' amplitudes;
`geocode_rslc(range_shift={"B": band_offset(rslc).shift})` removes it.
