# Pixel offsets

Offsets measure how far the backscatter pattern moved between two acquisitions. They survive where the interferogram
decorrelates, which is the near field of a large earthquake, at the price of much larger noise. The package makes them
two ways: from a GSLC pair, and from an RSLC pair as NISAR's ROFF product does.

## Amplitude offsets from a GSLC pair

`GSLCStack.pixel_offsets` cross-correlates the amplitude of each pair on a coarse lattice. It is a port of GMTSAR's
`xcorr` in its default frequency-domain mode.

```python
offsets = stack.pixel_offsets().persist(ws, "offsets")   # or step=, or nx=/ny= (xcorr's -nx/-ny)
clean = offsets.robust_offsets("x_offset")              # gated, clipped and centred; see below
fig, axes = offsets.plot_offsets(pair=0, robust=True, vmax=0.5)
```

The result is a `PixelOffsetStack` with `x_offset`, `y_offset` and `correlation` on `(pair, y, x)`; it is a
georeferenced raster like any other stage, so `crop`, `persist` and `to_grd` work on it.

**What the offsets are.** `xcorr` runs on radar-geometry SLCs, where the offsets are a coregistration solution in
range and azimuth. A NISAR GSLC is geocoded: the two dates already share one map grid, so what the correlation
measures is a shift along the map axes, that is, ground displacement. Hence `x_offset` and `y_offset`, in pixels, and
`east_offset` and `north_offset` in metres. An offset is positive when the feature sits at a larger index in the
secondary, as in `xcorr` (see [Conventions](conventions.md#pixel-offsets)).

**Windows and sampling.**

- `nx_corr`, `ny_corr` (64) size the correlation template, which is the resolution of the result. `xsearch`,
  `ysearch` (16) are the search half-widths. That is a 320 m template searched over ±80 m at a 5 m posting. These are
  not GMTSAR's defaults (128 and 64), which are sized for a radar-geometry pair that still has to be coregistered: a
  geocoded pair is already aligned to the whole pixel, so a wide search only adds chances of a false peak.
- `step` (16 pixels) is the spacing of the lattice, not its resolution. The default oversamples the template, so
  neighbouring estimates share most of their data; use `step=nx_corr // 2` for independent estimates. To resolve a
  discontinuity better, shrink the template, not the step.
- `subpixel_window` (8) is the window of the sub-pixel refinement. A wider one reduces a uniform bias of the peak but
  increases the scatter, and `robust_offsets` removes a uniform bias anyway; keep it well below twice the search width.
- `x_shift`, `y_shift` centre the search on a known bulk displacement and are added back to the result. Recentring is
  free; widening the search costs false peaks.
- Windows with less than `min_valid_fraction` (0.5) valid samples return NaN with correlation 0, so the swath edge does
  not contaminate its neighbours.

**Reading the field.** The raw offsets are not yet ground motion. `robust_offsets(field)`, which
`plot_offsets(robust=True)` applies to both components, does three things:

1. drops locations below `min_correlation` (20, the threshold of GMTSAR's `fitoffset.csh`);
2. drops locations more than `n_sigma` (6) robust standard deviations (from the median absolute deviation) from the
   median. Only this removes false peaks: a window that fails to match still reports a maximum, which pins at the
   edge of the search, often with a high correlation;
3. subtracts the median, which is the pair's bulk geolocation offset and can be larger than the deformation.

The removed median and the robust sigma are kept in the result's `attrs`. A high correlation means the match is real,
not that it is precise; judge the population, not one location. A large genuine displacement widens the sigma and so
the cut; on a quiet scene the cut is tight.

`plot_offsets` paints dropped locations grey (`nan_color=`), because on a diverging colour map a zero offset and a
missing one would otherwise look the same.

**Export.**

```python
offsets.to_text("offsets/")                  # freq_xcorr_pair0.dat: xcorr's five columns, then lon and lat
offsets.to_text("offsets/", lonlat=False)    # exactly xcorr's five columns
offsets.to_grd("offsets/", fields=["east_offset", "north_offset", "correlation"])
```

`to_text` writes `x_pixel x_offset y_pixel y_offset correlation` with `xcorr`'s own format string, so
`fitoffset.csh` and `awk` scripts written for `xcorr` work unchanged. Pixel indices are 0-based and index the source
GSLC grid. The appended `lon` and `lat` (in that order) give each location's ground position. `to_grd` writes pixels
by default; ask for `east_offset` and `north_offset` for metres.

## NISAR's ROFF from two RSLCs

`roff.dense_offsets` makes NISAR's ROFF product: it is a port of ISCE3's offsets workflow (geometry offsets from the
orbits and the DEM, a coarse resampling of the secondary, then the PyCuAmpcor correlator, here in numpy). On the same
inputs it reproduces NISAR's own ROFF on almost every window.

```python
from nisar_tools import roff

ds = roff.dense_offsets("data/<date 1 RSLC>.h5", "data/<date 2 RSLC>.h5", "data/dem.tif", workers=8)
ds = roff.open_roff("data/NISAR_L1_..._ROFF_....h5")     # or read a NISAR ROFF product
```

- The layers are NISAR's GOFF/ROFF layers by default (`roff.GOFF_LAYERS`), with the product's window lattice.
  `rows=` and `cols=` select part of the lattice and give the same values as a full run.
- Offsets are in metres: `slantRangeOffset` and `alongTrackOffset`, with their variances (m²), `snr` and
  `correlationSurfacePeak`. They contain what the geometry leaves: deformation, the ionosphere, orbit and timing
  errors.
- `roff.geocode_offsets(ds, reference_rslc, dem, x, y, epsg)` maps them onto a map grid as NISAR's GOFF does;
  `roff.goff_geogrid(path)` returns a GOFF's grid to land on.

For an inversion, `roff.offset_los_stacks(ds, reference_gslc, reference_rslc, dem, x, y, epsg)` returns two
`LOSStack`s that read like interferograms:

- `"range"`: `-slantRangeOffset` on the LOS unit vector, positive toward the sensor. This is the interferogram's own
  quantity, without its 2π ambiguity.
- `"along"`: `+alongTrackOffset` on the along-track unit vector, which is horizontal and points along the flight
  direction.

Each also carries `variance`, `snr` and `correlationSurfacePeak`. The noise of one window is of the order of
decimetres, roughly ten times an interferogram's, so weight offsets by their noise when they are combined with
interferograms. Along-track offsets at L-band can also carry ionospheric streaks.
