# Conventions

Signs, grids and units are where InSAR code goes wrong without failing. These are the package's, in one place.

## Bounding boxes

Every bounding box is `(lon_min, lon_max, lat_min, lat_max)` in degrees: `GSLC.crop`, `GSLCStack.from_gslcs(bbox=)`,
the `crop` method of every stage, `deramp(mask=)`, the `download` functions and `dem.stage_nisar_dem`. Where another
library wants `(west, south, east, north)` (earthaccess, sardem), the conversion is done internally.

## Grids

- Stages are on the products' own map grid: a UTM projection for NISAR L2 products, with the EPSG code taken from the
  granule (`stack.epsg`, `stack.crs`). All granules in a stack must share it; merging across a UTM zone boundary
  warps the second stack onto the first one's grid.
- `x` and `y` are pixel-centre coordinates in metres. `x` increases; on a north-up grid `y` decreases, and the stored
  `y_spacing` is negative.
- Dimensions are `(time, y, x)` for a `GSLCStack` and `(pair, y, x)` for the interferogram, unwrapped, LOS and offset
  stacks. Each pair carries `ref_time` and `sec_time` coordinates. Look geometry is one `(y, x)` field per stack.
- `to_grd`, `to_latlon` and the plots reproject to lon/lat; the stages themselves never do.
- A grid imported with `LOSStack.from_grd` in lon/lat is reprojected to UTM on load.
- `rslc.GeoGrid`, used when geocoding RSLCs, follows ISCE3: `start_x`, `start_y` is the top-left *corner* of the grid,
  not a pixel centre.

## Dates and pairs

A stack is sorted by acquisition time. `pairs="sequential"` gives `(0, 1), (1, 2), ...` and `pairs="all"` every `(i, j)`
with `i < j`, so for both the reference is the earlier date. Explicit pairs are used as given.

## Phase

Interferograms are formed as `reference * conj(secondary)`, so the phase is \(\phi = \phi_\mathrm{ref} -
\phi_\mathrm{sec}\). With ISCE3's SLC phase convention, \(\exp(-i\,4\pi r/\lambda)\), ground moving toward the
sensor shortens the range and makes the phase decrease.

## Line-of-sight displacement

\[
d_\mathrm{LOS} = -\frac{\lambda}{4\pi}\,\phi
\]

in metres, **positive toward the sensor** (a decrease in range). The wavelength comes from the granule's
`centerFrequency`. `to_los(sign=-1)` (and `geometry.phase_to_los(sign=-1)`) is for data whose fringe sense is
inverted; whatever `sign` is, the stored `los` is positive toward the sensor, and nothing downstream applies the sign
again.

## Look vectors and angles

- `los_east`, `los_north`, `los_up` are the components of the unit vector from the target to the sensor. So
  `los_up = cos(incidence) > 0`, and the LOS displacement of a ground motion `(e, n, u)` is the plain dot product
  `e * los_east + n * los_north + u * los_up`, with no sign flip.
- NISAR looks left: `los_east` is positive on ascending passes and negative on descending ones.
- `incidence_angle` is measured at the target, from the local vertical. `look_angle` is the off-nadir angle at the
  spacecraft, from the ellipsoid normal, and is always the smaller of the two. Both are in degrees.

## Grids made elsewhere

A `.grd` cannot record its units, its sign or its look-vector convention, so `LOSStack.from_grd` asks for them:
`units` (`"m"`, `"cm"` or `"mm"`), `sign` (`+1` if the grid is already positive toward the sensor, `-1` if it is
positive away, that is, a range increase) and `look_convention` (`"target_to_sensor"` or `"sensor_to_target"`).
The look convention is checked against the data (`los_up` must be positive) and a look vector that is not of unit
length raises. Units and sign cannot be checked: get the sign from the known sense of motion, since a global sign flip
changes the fit of a model very little.

## Pixel offsets

- **Amplitude offsets on GSLC pairs** (`PixelOffsetStack`): `x_offset` and `y_offset` are in pixels of the source grid,
  positive when a feature sits at a larger index in the secondary (secondary minus reference, as in GMTSAR's `xcorr`).
  On a geocoded pair these are shifts along the map axes, that is, ground displacement, not range and azimuth
  offsets. `east_offset` and `north_offset` give metres; the signed `y_spacing` turns a positive `y_offset` (a move
  to a larger row, southward) into a negative northward offset.
- **NISAR ROFF** (`roff`): `slantRangeOffset` is secondary minus reference range, positive away from the sensor;
  `alongTrackOffset` is positive along the flight direction. Both are in metres; variances in m².
  `roff.offset_los_stacks` turns them into LOS-style stacks: `"range"` is `-slantRangeOffset` on the LOS unit vector
  (positive toward the sensor, the same quantity as an interferogram's LOS, without the 2π ambiguity), and `"along"` is
  `+alongTrackOffset` on the horizontal along-track unit vector.

## Ionospheric screens

Screens are phase in radians at band A's carrier, to be subtracted from band A's phase. The corrected phase is band A's
non-dispersive phase and stays at band A's wavelength, so `to_los` needs no adjustment.

## Units and pixel counts

| Quantity | Unit |
|---|---|
| phase, screens | radians |
| LOS displacement, ROFF offsets | metres |
| `x_offset`, `y_offset` | pixels of the source grid |
| `east_offset`, `north_offset` | metres |
| angles | degrees |
| frequencies | Hz |
| `mask_water(spacing=)` | GMT increment: a bare number is degrees, `s` arc-seconds, `e` metres (`"50e"`) |

Pixel counts such as `edge_pixels` are in the stage's own pixels: native GSLC pixels on a `GSLCStack`, multilooked
pixels after `form_interferograms`. `looks` is the number of pixels averaged along each axis.

## Frequencies

A NISAR granule can carry two sub-bands of one acquisition: the main band, frequency A, and a side band, frequency B.
In the common 40 + 5 MHz mode they are centred at 1.239 GHz (40 MHz, 5 m GSLC posting in x) and 1.2935 GHz (5 MHz,
40 m in x). `GSLC(path, frequency="B")` reads the side band. The two bands are not interferometric with each other, so
a stack holds one band; the [ionosphere](ionosphere.md) corrections combine one stack of each.
