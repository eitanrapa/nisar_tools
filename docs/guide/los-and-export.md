# Line of sight and export

## From unwrapped phase to LOS displacement

```python
los = unw.to_los("data/NISAR_L2_PR_GSLC_....h5", dem="data/dem.tif").persist(ws, "los")
```

`to_los` scales the unwrapped phase to metres, positive toward the sensor (see
[Conventions](conventions.md#line-of-sight-displacement)), and attaches the look geometry:

- The wavelength comes from the granule's `centerFrequency` (or `wavelength=`).
- The incidence angle, the look angle and the LOS unit vector are interpolated from the granule's
  `metadata/radarGrid` cube, trilinearly in map position and height. `dem` (a GeoTIFF path or a DataArray of
  ellipsoidal heights) supplies the height; without it the geometry is evaluated at the ellipsoid.
- For a stack built with `GSLCStack.merge`, pass one granule per frame, in the order the frames were merged: each cube
  covers only its own frame.
- A GUNW stack needs no granule: its own cube and centre frequency are used.
- `mask_geometry=True` (the default) blanks the geometry outside the data, since the cube covers the frame's whole
  bounding rectangle.

A `LOSStack` holds `los` on `(pair, y, x)` and one `(y, x)` field each of `incidence_angle`, `look_angle`,
`los_east`, `los_north`, `los_up` and `height`, computed once, since the viewing geometry is shared by the whole
stack.

```python
fig, ax = los.plot(pair=0)
fig, ax = los.plot_incidence()
fig, ax = los.plot_look_angle()
```

## Exporting GMT grids

Every stage has `to_grd(outdir, fields=None, indices=None)`. Each field is reprojected to lon/lat and written as a
single-variable GMT grid that `gmt grdinfo` and `gmt grdimage` read directly. A field on the stack axis is written one
file per slice (`{field}_pair{i}.grd`, or `_time{i}` for SLCs); a shared field once (`{field}.grd`). Complex data is
split into `amplitude` and wrapped `phase`.

| Stage | Default fields |
|---|---|
| `GSLCStack` | `amplitude` (an SLC's absolute phase means nothing on its own; `fields=["phase"]` adds it) |
| `InterferogramStack` | `phase`, `coherence` (`amplitude` on request) |
| `UnwrappedStack` | every layer present: `unw`, `coherence`, `conncomp`, and a GUNW's `phase_screen`, `subswath_mask` |
| `LOSStack` | `los`, `los_east`, `los_north`, `los_up`, `incidence_angle`, `look_angle` (`height` on request) |

Keep the unit vectors: the LOS of an ascending and a descending track can be decomposed into vertical and east-west
motion only with them.

`nisar_tools.geo.read_grd(path)` reads a grid back as a CRS-aware DataArray. One caution when reading grids yourself:
importing `h5py`, which this package always does, breaks pygmt's NetCDF-4 grid reading in the same process. Opening a
grid with `xarray.open_dataarray(..., engine="gmt")` then returns pixel indices instead of longitude and latitude, with
no error. `read_grd` reads NetCDF-4 grids with h5py directly for that reason.

For a GeoTIFF, reproject one pair and write it with rioxarray:

```python
unw.to_latlon(pair=0).rio.to_raster("unwrapped_pair0.tif")
```

## Importing LOS grids made elsewhere

`LOSStack.from_grd` reads a displacement grid (or one per pair) and the three look-vector grids from any other
processor, so they can be used beside NISAR stacks:

```python
from nisar_tools import LOSStack

los = LOSStack.from_grd(
    "los_cm.grd", "look_e.grd", "look_n.grd", "look_u.grd",
    units="cm",                       # "m", "cm" or "mm"
    sign=-1,                          # this grid is positive away from the sensor
    look_convention="target_to_sensor",
    direction="descending", look_direction="right",
)
```

- `units`, `sign` and `look_convention` must be declared; see [Conventions](conventions.md#grids-made-elsewhere).
  The look convention and the vector's length are checked against the data, units and sign are not.
- Geographic grids are reprojected to a metric grid (by default the UTM zone of the scene centre, or `epsg=`, at
  `resolution=` metres), because downstream sampling works in metres.
- Look grids on a different lattice than the displacement are resampled onto it.
- `direction` and `look_direction` are kept as provenance; they let downstream checks compare the sign of `los_east`
  with the pass geometry.

## Into a slip inversion

`los.to_observations(name, trace, frame=None, **kwargs)` samples a `LOSStack` into observations for
[slipsolve](https://github.com/eitanrapa/SlipSolve-Python) (the `slip` extra); the keyword arguments go to
`slipsolve.Observations.from_los`. `roff.offset_los_stacks` makes the same kind of stack from
[pixel offsets](offsets.md#nisars-roff-from-two-rslcs).
