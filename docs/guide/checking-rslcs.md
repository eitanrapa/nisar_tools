# Checking an RSLC before use

ISCE3 v0.25.16, which focused NISAR's provisional products, passes some defects of the raw data into the RSLC without
flagging them. Three checks find them in seconds; run them before an RSLC goes into an interferogram.

```python
from nisar_tools.l0b import timing_report
from nisar_tools.qa import band_offset, check_pair, check_rslc

check_rslc(rslc)["problems"]       # release, PRF, calibration delays, input-data exceptions, dropped mask lines
check_pair(reference, secondary)   # [] when the two were focused alike
print(timing_report(rslc).summary())
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
each transmit blanks part of the receive window; the L0B's sub-swath limits record where, and the blanked samples are
transmit leakage, several times brighter than the echo. Those positions depend on the pulse sequence alone, so
`tx_gap_phase` reads which phase of the commanded sequence the radar flew without the tags. `pulse_timing` checks the
tags against it, which finds two more defects:

- **relabelled tags**: tags that follow the sequence exactly, but a few pulses off the transmits, so every echo carries
  another pulse's time. The error varies from pulse to pulse with the dither, so the RSLC is defocused and only a
  refocus repairs it (`"refocus"`);
- **unanchored tags**: no run of tags follows the sequence at all. The echoes are then timed from the transmit gaps,
  anchored on the first tag, whose own error remains as a constant along-track shift to measure against another date.

## Band B's range

Band B's slant ranges are off by whole periods of the 48 MHz clock it is decimated from (3.12 m each): constant within
a data window (they update every ~10 s), the same on most dates of a track, and on some granules different across
range. On a pair it does not misregister B against itself, but it misplaces B against A, and a split-spectrum screen
then mistakes the deformation's range gradient for ionosphere (about 12 times the offset times the gradient).
`band_offset` measures the offset per data window and range bin by cross-correlating the two bands' amplitudes;
`geocode_rslc(range_shift={"B": band_offset(rslc).shift})` removes it.
