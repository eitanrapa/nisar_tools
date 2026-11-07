# Unwrapping and cleaning

## SNAPHU

```python
unw = igrams.unwrap(ws, name="unwrapped", nproc=8)
```

`unwrap` runs SNAPHU (through snaphu-py) on each pair and writes the result into its own region of a store in the
workspace, marking each pair done. Peak memory is one pair, and an interrupted run resumes at the first unfinished
pair. The result is an `UnwrappedStack` with `unw` (radians), `conncomp` (SNAPHU's connected-component labels; 0 means
unassigned) and the interferogram's `coherence`.

- `cost="smooth"` (the default) or `"defo"`, SNAPHU's statistical cost mode. `"defo"` makes phase discontinuities
  cheap where the coherence is low.
- `res_az` and `res_rg` (8 and 3) are the single-look resolutions in metres used to compute SNAPHU's effective number
  of looks.
- **Tiling is part of the answer.** SNAPHU's tile grid decides where 2π seams can fall. A raster below
  `max_tile_pixels` (1.5 million) is unwrapped as one tile, which is the best-quality option; lower it, or set
  `ntiles` and `tile_overlap`, to tile. The tiling parameters are recorded in the stage's parameters; `nproc`, which
  only says how many tiles run at once, is not. `single_tile_reoptimize=False` drops the second, serial whole-raster
  pass of a tiled run, its largest cost.
- `pairs_in_flight=N` unwraps `N` pairs at once, sharing `nproc` between them. Even at the default of 1, the next
  pair's inputs are read while SNAPHU works on the current one.
- An interferogram formed with a `phase_model` gets the model added back to each unwrapped pair, so the store holds
  the full phase.

`UnwrappedStack.from_interferograms(igrams, ws, ...)` is the same call.

## Connected components and cycles

Unwrapping recovers phase up to a whole number of cycles, and SNAPHU resolves each connected component separately, so
two components, or two regions kept apart by a mask, can sit whole cycles apart with no way to tell from the phase
alone. When you know the offset (from GNSS, a stable area, pixel offsets, or a neighbouring region that must be
continuous), apply it:

```python
unw = unw.add_cycles(-2, pair=[0], conncomp=[3])     # whole cycles only; lazy
```

The [InSAR pipeline](../insar_pipeline.md#5-tying-cut-off-blocks) page describes two ways of reading such an offset
from the data.

## Cleaning

Three lazy methods, usually chained in this order before `to_los`. They apply to SNAPHU output and to GUNWs alike, and
each records itself in the stage's parameters.

```python
clean = (
    unw.mask_edges(edge_pixels=8)        # decorrelated swath edges
       .remove_outliers(threshold=2.0)   # spikes against a smooth surface, in radians
       .deramp(degree=1)                 # a long-wavelength polynomial
)
```

- `mask_edges` works as on [interferograms](interferograms.md#swath-edges). On a GUNW, `use_builtin_mask=True` first
  applies the product's own validity mask, an exact edge rather than an erosion.
- `remove_outliers(scale=16.0, threshold=1.0, iterations=2)` fits a NaN-aware smooth surface (a Gaussian of `scale`
  pixels) and nulls pixels further than `threshold` radians from it, refitting `iterations` times. Near a mask edge or
  a cut the surface lags the phase gradient, so it can remove good pixels there.
- `deramp(degree=1, method="poly", mask=None)` subtracts a polynomial surface. `mask` (a boolean `(y, x)` grid, or a
  lon/lat box) keeps a deformation region out of the fit, while the ramp is still subtracted everywhere.

!!! warning "Ramps and deformation"
    - A deramp is a fit: where deformation spans the scene it removes part of it. For a slip inversion it is better
      not to pre-remove the ramp at all, and to solve for it with the slip.
    - A mask that spans the full width of the scene and touches one of its edges makes the polynomial extrapolate
      over the masked band, and high degrees then diverge. With such a mask use degree 1 or 2. A mask that stops
      short of the scene's sides does not have the problem, and a mask reaching outside the data is harmless.
    - `method="spline"` subtracts a smooth surface instead. With `scale=None` its width is a quarter of the raster,
      which is both slow and a gentle high-pass that leaves much of a planar ramp behind. Prefer `method="poly"`, or
      set `scale`.

Chain `remove_unconnected_regions` ahead of `remove_outliers` and a spline deramp: both fill gaps from their
neighbours, so a stranded region pulls the fitted surface across the water between it and the mainland.

## NASA GUNW products

```python
from nisar_tools import UnwrappedStack

gunw = UnwrappedStack.from_gunw_file("data/NISAR_L2_PR_GUNW_....h5")
clean = gunw.crop(*bbox).mask_edges(edge_pixels=4, use_builtin_mask=True)
los = clean.to_los()                       # the product's own geometry cube; no GSLC needed
```

A GUNW reads into the same class as SNAPHU output: `unwrappedPhase` becomes `unw`, `coherenceMagnitude` `coherence`
and `connectedComponents` `conncomp`. Two layers are added:

- `phase_screen`, the product's `ionospherePhaseScreen`. The GUNW's `unwrappedPhase` is not corrected for it;
  `remove_phase_screen()` subtracts it. Near a rupture, see the caveats on the [pipeline](../insar_pipeline.md) page
  before using it.
- `subswath_mask`, the product's water and subswath-validity mask, used by `mask_edges(use_builtin_mask=True)`.

`from_gunw_files(paths)` reads several granules of one frame into a `(pair, y, x)` stack.

## Other operations

- `merge` joins frames and removes the 2π step between them; see [Merging frames](merging.md).
- `correct_ionosphere(other)` removes a dual-frequency screen; see [Ionosphere](ionosphere.md).
- `to_latlon(pair)` reprojects one pair to lon/lat (eager), for example to write a GeoTIFF with
  `unw.to_latlon(0).rio.to_raster("unw.tif")`.
- `to_grd(outdir)` writes every layer present, per pair.
- `persist(ws, name)` writes a derived stack (masked, cleaned, cycle-shifted, or a GUNW just read) under a new name.
