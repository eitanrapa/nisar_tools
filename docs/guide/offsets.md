# Pixel offsets

Offsets measure how far the backscatter pattern moved between two acquisitions. They survive where the interferogram
decorrelates, which is the near field of a large earthquake, at the price of much larger noise. The package makes them
from an RSLC pair, as NISAR's ROFF product does.

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
