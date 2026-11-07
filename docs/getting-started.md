# Getting started

This example takes two GSLC granules of the same track and frame, acquired on two dates, to a line-of-sight
displacement grid. It needs the `mask` extra for the water mask (see [Installation](installation.md)). Replace the two
granule paths with your own, and the box with one inside both granules.

```python
from nisar_tools import GSLC, GSLCStack, Workspace

paths = [
    "data/NISAR_L2_PR_GSLC_<first date>.h5",     # same track and frame, two dates
    "data/NISAR_L2_PR_GSLC_<second date>.h5",
]
bbox = (-80.75, -80.65, 8.15, 8.25)              # lon_min, lon_max, lat_min, lat_max

ws = Workspace("work")                           # a folder of Zarr stores, one per stage

# 1. Crop both dates into one lazy stack and write it to the workspace.
gslcs = [GSLC(p) for p in paths]
stack = GSLCStack.from_gslcs(gslcs, bbox=bbox).persist(ws, "slc_stack", files=paths, bbox=bbox)
for g in gslcs:
    g.close()                                    # the stack now reads from the workspace
print(stack)

# 2. Interferogram: 10 x 10 looks, water masked, stranded regions dropped, adaptive Goldstein filter.
igrams = (
    stack.form_interferograms(pairs="sequential", looks=10, convolution="Gaussian")
    .mask_water(mask_cache=ws)
    .remove_unconnected_regions()
    .filter_goldstein(alpha="adaptive")
    .persist(ws, "igrams")
)
print(igrams)

# 3. Unwrap with SNAPHU. This step writes its own store, pair by pair.
unw = igrams.unwrap(ws, name="unwrapped", nproc=4)

# 4. Line-of-sight displacement (metres, positive toward the sensor) and look vectors.
los = unw.to_los(paths[0]).persist(ws, "los")
print(los)

# 5. GMT grids, reprojected to lon/lat.
for path in los.to_grd("work/grd"):
    print(path)
```

Run on a pair of production GSLCs, it prints this (SNAPHU also prints its own log while it unwraps):

```text
<GSLCStack EPSG:32617 time=2 y=2213 x=2204>
<InterferogramStack EPSG:32617 pair=1 y=221 x=220>
<LOSStack EPSG:32617 lambda=0.2420m pair=1 y=221 x=220>
work/grd/los_pair0.grd
work/grd/los_east.grd
work/grd/los_north.grd
work/grd/los_up.grd
work/grd/incidence_angle.grd
work/grd/look_angle.grd
```

## What happened

- **Nothing is read until it is needed.** `GSLC` opens the granule and builds a lazy view; `from_gslcs` crops it.
  Each `persist` computes its stage, writes one Zarr store and returns the stage reopened from disk. The granules can
  be closed once the stack is persisted.
- **Interferograms** are `reference * conj(secondary)`, with the earlier date as the reference, multilooked here from
  5 m to 50 m. Masking, region removal and filtering are lazy and are computed by the `persist` that follows them.
- **Unwrapping** is the one step that writes as it goes: SNAPHU needs a whole raster, so each pair is unwrapped and
  written on its own, and a rerun starts at the first unfinished pair.
- **Line of sight** uses the GSLC's own geometry cube for the look vectors and its centre frequency for the
  wavelength. Without `dem=`, the geometry is evaluated at the ellipsoid; pass a DEM to evaluate it at the terrain.
- **Running it again** reuses every finished stage. With different inputs or settings, the first `persist` whose stage
  they change raises a `WorkspaceError` instead of overwriting the earlier results. The unwrapping store keeps its own
  rule; see [Workspaces](guide/workspaces.md#which-steps-write).

## Next steps

- [Conventions](guide/conventions.md): bounding boxes, grids, signs and units.
- [Interferograms](guide/interferograms.md) and [Unwrapping and cleaning](guide/unwrapping.md) for the choices along
  the way.
- The example notebook,
  [`notebooks/nisar_tools.ipynb`](https://github.com/eitanrapa/nisar_tools/blob/main/notebooks/nisar_tools.ipynb),
  with plots at each stage, GUNW input and pixel offsets.
