# Interferograms

## Forming them

```python
igrams = stack.form_interferograms(
    pairs="sequential",       # or "all", or a list of (i, j) index pairs
    looks=10,                 # pixels averaged along each axis: 5 m -> 50 m
    convolution="Gaussian",   # or "Uniform" (a box)
)
```

The result holds the complex `igram` (`reference * conj(secondary)`) and its `coherence`, estimated from the same
looks, on `(pair, y, x)`.

- `sigma` (Gaussian only) is the filter's standard deviation in input pixels; it defaults to `looks`. A smaller value,
  such as `looks / 2`, keeps steeper phase gradients.
- `nan_aware=True` (the default) averages over valid samples only, so the NaN at a swath edge or a mask does not spread
  across the multilook window. `min_valid_fraction` (0.5) is how much of an output pixel's weight must come from valid
  input for it to be kept.
- `align_looks=True` (the default) anchors the multilook blocks to the absolute native lattice instead of to the corner
  of the crop. Every frame of a track then lands on one shared grid, and frames [merge](merging.md) exactly. It costs at
  most `looks - 1` samples at the near edge of each axis.
- `downsample=False` keeps the native grid (a filtered interferogram at full resolution).
- `phase_model=` takes a `(y, x)` phase model (radians, this band) out of every pair at full resolution, before the
  multilook, and keeps it to be added back after unwrapping. Steep fringes, for example near a rupture, then do not
  cancel inside the looks. A model of the deformation from pixel offsets is one source.

## Masking

Masks set pixels to NaN. They are lazy and are recorded in the stage's parameters.

### Water

`mask_water(mask_cache=ws, resolution="f", spacing=None)` masks with the GSHHG coastline through GMT (the `mask`
extra). `resolution` is GMT's coastline resolution (`"f"` full, then `"h"`, `"i"`, `"l"`, `"c"`); `spacing` is the GMT
increment of the mask grid, by default half the stage's pixel. `mask_cache` caches the mask, keyed on the grid, so GMT
runs once per grid.

### Swath edges

The near- and far-range edges of a swath carry a decorrelated fringe where the antenna pattern rolls off.
`mask_edges(edge_pixels=8, edges="along_track")` trims it: each row is scanned for its first and last valid sample, so
coastlines, masked lakes and the frame's azimuth ends are left alone. `edges="all"` erodes the whole footprint instead,
which also eats a collar along every coastline. Sides can be named against the grid, as `"left"`, `"right"`, `"top"`
and `"bottom"`, a list of them, or a `{side: pixels}` mapping for different depths:

```python
igrams.mask_edges(edge_pixels=4, edges=["left", "bottom"])
igrams.mask_edges(edges={"left": 4, "right": 1})
```

Prefer one call to a chain of several: every side is then measured against the same footprint, and the stage records
only the last call. `InterferogramStack.mask_edges(min_coherence=)` also nulls pixels below a coherence threshold.

### Mask before multilooking where you can

`GSLCStack` has `mask_water` and `mask_edges` too, one stage earlier. Because the multilook is NaN-aware, a pixel masked
on the SLC is dropped from its window's average instead of adding a decorrelated sample to both the interferogram and
the coherence meant to flag it. Two things change at the SLC stage:

- `edge_pixels` counts native pixels: to reproduce a trim of `N` multilooked pixels, pass `N * looks`.
- Pass `spacing` to `mask_water`. The default tracks the grid's pixel, which on a 5 m GSLC grid means a mask with
  orders of magnitude more nodes than the coastline data can resolve (a whole frame reaches tens of gigabytes).
  `spacing="50e"` places the coastline within 50 m. `make_water_mask` warns when a mask would be very large.

```python
stack = stack.mask_water(mask_cache=ws, spacing="50e").mask_edges(edge_pixels=8 * looks)
```

## Regions the unwrapper cannot reach

A coastline mask leaves islets, ships, platforms and decorrelated fragments stranded offshore. An unwrapper propagates
phase between adjacent pixels, so a region with no path to the main body carries no recoverable ambiguity, however
large it is, and left in place it produces artefacts that seem to bridge the water. The criterion is connectivity, not
size:

```python
igrams = igrams.remove_unconnected_regions()               # keep the largest region only
igrams = igrams.remove_unconnected_regions(min_size=2000)  # keep every region above 2000 pixels
```

- `min_size` is for a scene that really is two or more bodies (an island, a block cut off by a mask along a fault):
  the real bodies survive and the speckle goes. Each region kept is unwrapped with its own arbitrary integer, which
  has to be tied later; see [Unwrapping](unwrapping.md#connected-components-and-cycles).
- `max_drop_fraction` (0.01) refuses to drop a region holding more than that share of the valid pixels. It tests the
  largest single dropped region, not the total, because a scene shredded into many specks has a large total and no
  large region, and those specks are what the method is for.
- `connectivity=1` (four neighbours) is the default: a diagonal touch is not a path for the unwrapper.
- The guard raises when the stage is computed, not when the method is called.

Run it after the masks and before the filter and the unwrap. `UnwrappedStack` has the same method.

## Goldstein filter

```python
igrams = igrams.filter_goldstein(alpha="adaptive")   # or a constant strength in [0, 1]
```

The adaptive Goldstein-Werner filter sharpens fringes and suppresses phase noise, which cuts the residues SNAPHU has to
resolve. `alpha="adaptive"` sets the strength per patch to `1 - coherence` (Baran et al., 2003), like GMTSAR's
`phasefilt` with amplitude inputs; a number sets a constant strength, and `0` does nothing. `patch_size` (32),
`overlap` (0.75) and `psd_smooth` (3) set the patches. Only the interferogram is filtered; the coherence is left as
estimated.

## Ionosphere before unwrapping

`InterferogramStack.correct_ionosphere(screen)` rotates the wrapped interferogram by a screen estimated elsewhere, so
that SNAPHU sees fewer fringes. See [Ionosphere](ionosphere.md).

## Looking at them

```python
fig, ax = igrams.plot_wrapped(pair=0)
igrams.to_grd("grd/")                     # phase and coherence per pair; fields=["amplitude"] for more
```
