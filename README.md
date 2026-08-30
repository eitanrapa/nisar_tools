# nisar_tools

Object-oriented, out-of-core tools for cropping, merging NISAR GSLCs and making
interferograms, unwrapping, masking, and plotting.

Also supports slip inversions using a Python-port of [SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve).

Every processing stage is lazy (`xarray` + `dask`) and persisted to a Zarr
`Workspace`, so a full stack of acquisitions never has to fit in memory and an
interrupted run resumes where it left off.

Two runnable end-to-end examples:

- [notebooks/nisar_tools.ipynb](notebooks/nisar_tools.ipynb) — the raster
  pipeline, from granules to line-of-sight displacement.
- [notebooks/slip_inversions.ipynb](notebooks/slip_inversions.ipynb) — the
  inversion, from LOS displacement to slip on a fault.

The notebooks show the path through. **This file is the reference for the parts
that are easy to get wrong**: which knob to reach for, and what quietly breaks
when it is set badly.

## Install

The geospatial stack (h5py, rioxarray, pyproj, snaphu, zarr<3) is easiest from
conda-forge; everything beyond it is optional.

```bash
conda install -c conda-forge h5py "zarr<3" dask rioxarray pyproj snaphu
pip install -e .
```

### Optional dependencies

Every optional dependency is imported lazily, so the package installs and imports
without any of them. Each extra buys exactly one thing:

| extra | brings | needed for | without it |
|---|---|---|---|
| `mask` | `pygmt` | `mask_water` — GSHHG coastlines | water masking raises |
| `download` | `earthaccess`, `sardem` | the `download` module: Earthdata search, DEMs | fetch inputs yourself, or use `method="asf"` |
| `layered` | `pygrnwang` | `run_edgrn`, which generates EDGRN tables from a velocity model | supply tables from your own EDGRN run |
| `gdal` | `gdal` | the `osgeo` bindings and `gdal_*` CLI tools | nothing in the package needs them |
| `dev` | `pytest`, `pytest-xdist`, `cutde` | the test suite | the `cutde` cross-check skips |

```bash
pip install -e ".[mask,download]"          # several at once
conda install -c conda-forge pygmt gdal    # the two with native libraries
```

Three things that trip up a plain `pip install` of these:

- **`earthaccess`'s current release needs Python ≥ 3.12**, and this project's env
  is 3.11. Pin an older earthaccess, or use the direct-by-name download path
  (`method="asf"`), which uses only the standard library.
- **`pygmt` and `gdal` come from conda-forge, not pip.** pip's gdal builds
  against system libgdal headers and commonly fails; conda-forge's does not, and
  `environment.yml` installs it there. Nothing in the package imports `osgeo` —
  `rasterio` already vendors GDAL for it — so the `gdal` extra is only for
  calling the `gdal_*` tools yourself.
- **`layered` compiles Fortran.** `pygrnwang` bundles Rongjiang Wang's
  EDGRN/EDCMP and builds it with gfortran. Skip it unless you want the
  Green's-function tables generated for you; the reference workflow has you run
  EDGRN yourself and hand over the tables.

## Setup on a new machine

The conda environment is **not** part of the repository — `git clone` brings the
code, not the env. To get running on a fresh machine, recreate the environment
from `environment.yml` (which also installs `nisar_tools` editable and
`ipykernel`), then register a Jupyter kernel:

```bash
git clone https://github.com/eitanrapa/nisar_tools.git
cd nisar_tools

# Build the env (named "remote_sensing") + install the package + ipykernel.
conda env create -f environment.yml
conda activate remote_sensing

# Register the kernel, then select "Python (remote_sensing)" in VS Code / Jupyter.
python -m ipykernel install --user --name remote_sensing \
    --display-name "Python (remote_sensing)"
```

`pip install -e .` means the cloned folder *is* the installation — keep it in
place. A `ModuleNotFoundError: No module named 'nisar_tools'` in the notebook
almost always means a different kernel is selected than the env above.

### Two environment variables the kernel needs

This env ships duplicate OpenMP runtimes, and pygmt may otherwise bind to a
system GMT (e.g. a Homebrew install) whose coastline data the conda netCDF stack
can't read — causing water masking to fail with GSHHG errors. Both are fixed by
setting two variables in the kernel's environment. Edit
`~/Library/Jupyter/kernels/<name>/kernel.json` to add:

```json
"env": {
  "KMP_DUPLICATE_LIB_OK": "TRUE",
  "GMT_LIBRARY_PATH": "/path/to/your/env/lib"
}
```

`GMT_LIBRARY_PATH` should point at the `lib` directory of the same env, so pygmt
loads that env's `libgmt` (matched to its netCDF/HDF5). A shell that launches
Jupyter, or a detached job, must export `KMP_DUPLICATE_LIB_OK=TRUE` itself.

## Downloading inputs

The `download` module fetches GSLC granules — all behind lazily-imported optional
dependencies and all needing an [Earthdata Login](https://urs.earthdata.nasa.gov).
Bounding boxes use this package's `(lon_min, lon_max, lat_min, lat_max)` order
(the same `GSLC.crop` takes).

```python
from nisar_tools import download

download.login()                        # earthaccess: netrc / env vars / interactive
bbox = (-120.5, -119.5, 34.0, 35.0)

gslcs = download.download_gslcs("data/gslc", bbox=bbox, temporal=("2025-11", "2025-12"))
gslcs = download.download_gslcs("data/gslc", granules=["NISAR_L2_GSLC_..."])
```

Two backends: **earthaccess** (`method="earthaccess"`, the default — searches CMR
by name, area and time) and the original **direct-by-name** path
(`method="asf"`), which pulls straight from the NISAR bucket using only the
standard library. Use the second where earthaccess can't be installed.

Orbit files are not downloaded and not needed: orbits are embedded in the GSLC,
and the LOS geometry comes from the granule's own `metadata/radarGrid` cube.

## Pipeline

`GSLC` → `GSLCStack` → `InterferogramStack` → `UnwrappedStack` → `LOSStack`,
with `GSLCStack` → `PixelOffsetStack` branching off for amplitude offset tracking.
Every stage is lazy and returns a new object; nothing reaches disk until
`persist`. The sections below are the choices along that path that are worth
making deliberately.

### Masking before multilooking

`mask_water` and `mask_edges` are also available on `GSLCStack`, one stage
earlier than their `InterferogramStack` / `UnwrappedStack` counterparts. Both are
lazy and record an attr that folds into the persist hash only once set, so an
existing `slc_stack` store keeps the hash it already had.

```python
stack.mask_water(mask_cache=ws, spacing="50e").mask_edges(edge_pixels=8 * looks)
```

Masking here rather than later keeps the blanked samples out of the **estimate**
as well as the product. `form_interferograms` is NaN-aware, so a blanked pixel is
dropped from its multilook window's average instead of contributing a
decorrelated sample to both the interferogram *and* the coherence that is
supposed to flag it; windows that then fall below `min_valid_fraction` come out
NaN, so the coastline is widened by the multilook footprint rather than aliased
onto it.

⚠️ **`edge_pixels` is in native GSLC pixels here**, not multilooked ones. 8 px is
40 m at NISAR's 5 m posting — nothing. To reproduce a trim of `N` pixels applied
after multilooking by `looks`, pass `N * looks`.

⚠️ **Pass `spacing` to `mask_water` at this stage.** The default increment tracks
the grid's own pixel, which is right for a multilooked stack and very expensive at
a GSLC's native posting — the node count goes as the inverse square of the
increment:

| region | pixel | GMT increment | nodes | float32 |
|---|---|---|---|---|
| 28 km crop | 150 m | `75e` | 2.7e5 | 1 MB |
| 28 km crop | 50 m | `25e` | 2.5e6 | 10 MB |
| 28 km crop | **5 m (native)** | `2.5e` | **2.4e8** | **978 MB** |
| full frame | 150 m | `75e` | 1.2e7 | 48 MB |
| full frame | **5 m (native)** | `2.5e` | **1.1e10** | **44 GB** |

GSHHG carries nowhere near that much shoreline detail, so the extra nodes buy no
accuracy at all. `make_water_mask` raises a `RuntimeWarning` above ~5e7 nodes
rather than coarsening silently — masking at native resolution is a legitimate
thing to ask for, and quietly moving the coastline by half a pixel to save time
you didn't ask to save is worse than saying so. `spacing="50e"` places the
coastline within 50 m and asks for 20× fewer nodes per axis than the default.
(GMT's `-I`: a bare number is degrees, `s` is arc-seconds, **`e` is metres**.)

`InterferogramStack.mask_edges` exists too, for a stack already persisted past the
SLC stage — same signature plus `min_coherence`, and `edge_pixels` back in
multilooked pixels. Run before `unwrap`: the decorrelated fringe is residues for
SNAPHU to resolve, and resolving them wrongly is what puts a 2π seam into the
interior. Prefer the SLC-stage trim where the SLCs are still to hand, though —
trimming at the igram stage blanks a value the multilook has *already* averaged
the decorrelated samples into, including into the coherence that is meant to flag
them. Both layers come out on one footprint; the edge scan runs on `igram` alone,
because coherence is exactly `0.0` outside the swath rather than NaN and scanning
it for a row's first valid sample would find column 0 on every row.

### Dropping territory the mainland can't reach

A coastline mask leaves things stranded offshore — islets GSHHG calls land,
ships, platforms, decorrelated fragments. An unwrapper propagates phase along
arcs between adjacent pixels, so a region with **no path to the main body carries
no recoverable ambiguity, however large it is**, and left in place it produces
artifacts that appear to bridge open water. The criterion is *connectivity, not
size*: `remove_unconnected_regions()` keeps only the largest component, and it
belongs before the Goldstein filter and the unwrap.

On a real coastal frame that took 29 components down to 1, removing 6,369 px —
**0.018%** of the valid data. That frame is also why a size threshold is the
wrong instrument: its strays run 16 to 1530 px against a 36-million-px mainland,
so a `min_size` of 32 would have removed four of the twenty-eight and looked like
it had worked.

- `min_size=N` keeps every component above `N` pixels instead — for a scene that
  genuinely *is* two landmasses, so both real bodies survive and the speckle
  still goes.
- `max_drop_fraction` (default 0.01) refuses if the largest **single** dropped
  component exceeds that share of the valid pixels. Deliberately not the total: a
  scene shredded into thousands of specks has a large total but no large
  component, and those specks are exactly what the method is for.
- `connectivity=1` (4-connected) by default — a diagonal touch is not an arc for
  the solver, so a diagonally-attached region is as unreachable as a detached one.
- The guard raises at **compute** time, not call time. These methods are lazy and
  an eager check would be a whole extra pass.

Also available on `UnwrappedStack`, where it is worth chaining ahead of
`remove_outliers` and `deramp(method="spline")`: both fit through a normalized
convolution that fills NaN gaps from their neighbours, so a stranded region drags
the fitted surface out across the water separating it from the mainland.

### The ionosphere, measured instead of fitted

NISAR transmits **two sub-bands in every granule** — `frequencyA` at 1.239 GHz
(40 MHz, 5 m posting) and `frequencyB` at 1.2935 GHz (5 MHz, 40 m posting) — and
the ionosphere is dispersive while everything else is not. Unwrap both and
`nisar_tools.ionosphere` separates them:

```python
from nisar_tools import GSLC, GSLCStack, ionosphere

unw = {}
for band, looks in (("A", 16), ("B", 2)):          # -> 80 m in x for both
    stack = GSLCStack.from_gslcs([GSLC(p, frequency=band) for p in paths], bbox=bbox)
    unw[band] = stack.form_interferograms(looks=looks).unwrap(ws, name=f"unw_{band}")

corrected = unw["A"].correct_ionosphere(unw["B"])   # unwrapped-domain
```

The screen is kept as the `phase_screen` layer, `unw` comes back as the
non-dispersive phase **at band A's own carrier** (so `to_los` needs no
adjustment), and the provenance folds into the persist hash. To remove it
*before* unwrapping instead — a large ionospheric ramp is fringes the unwrapper
can lay a seam through — estimate it and apply it to the wrapped interferogram:

```python
screen = ionosphere.dual_frequency_screen(unw["A"], unw["B"], scale=..., min_coherence=0.3)
unw_a = igrams_a.correct_ionosphere(screen).unwrap(ws, name="unw_corrected")
```

The screen is interpolated onto whatever grid it is applied to, so it can be
estimated from a cheaply multilooked pair and removed at full posting.

⚠️ **Both inputs must be unwrapped, and no wrapped-domain estimate exists.** The
combination at NISAR's carriers is `12.12·φ_A − 11.61·φ_B`; an unknown 2π in
either input moves the screen by ~76 rad. The complex product of the two bands
does not help either — the exact identity is
`φ_disp = 0.51·φ_A + 11.61·(φ_A − φ_B)`, so each band's *absolute* phase is
needed, not just their difference. Wrapped is where the correction is
**applied**, never where it is estimated.

⚠️ **The low-pass is not optional.** Those coefficients amplify independent phase
noise by `hypot(12.12, 11.61) = 16.8`, so an unsmoothed screen makes the data
worse. `scale` (pixels of the corrected stack, default 2 km of ground) sets the
Gaussian sigma; each band is smoothed **on its own grid** before combining —
linearity makes that identical on a shared grid and strictly better on the real
pair, where band B is posted 8× coarser in x and resampling its raw phase would
put every interpolation error through the 12× coefficient. Below twice the
coarser band's posting the resampling dominates and it warns.

**Measured against a deramp on the real D_071 pair** (Mojave, 55 × 66 km at 80 m,
12 days, band A `looks=16`, band B `looks=2`):

| | std (rad) | non-planar part |
|---|---|---|
| band A unwrapped phase | 1.393 | 0.318 |
| − dual-frequency screen | 0.497 | 0.348 |
| − `deramp(degree=1)` | **0.318** | 0.318 |
| − screen, then `deramp(degree=1)` | 0.348 | 0.348 |

**The screen is 94% a plane on this pair** (correlation with the best-fit plane
+0.94; it removes 4.68 rad p2p of the data's 5.90). Its amplitude corresponds to
0.32–0.46 TECU of differential TEC, which is a plausible number, and its
non-planar content (0.28 rad std, 4.3 km correlation length) is **8× above the
speckle floor** the two bands' coherence implies (0.038 rad after the 2 km
low-pass) — so it is measuring something, not amplifying noise.

But read the table the right way round. **A deramp always wins on variance**:
it is a least-squares fit to the very field being scored, so it minimises exactly
that number by construction. That is not the reason to prefer either one:

- **A deramp is fitted, so it consumes whatever is under it.** Where deformation
  spans the frame it removes part of the earthquake — the failure documented in
  the next section, and the reason `SlipInversion(ramp=...)` exists. The
  dual-frequency screen *cannot* do that: deformation scales with `f` like the
  orbit does, so it does not appear in the dispersive half at all.
- **Neither touches the troposphere.** In `A − plane` the alluvial fans are
  plainly visible: topography-correlated, non-dispersive, and left behind by both.
  On this scene it is the dominant residual, which is why the ionosphere
  correction improves the *deramped* residual by so little here.
- **Match the two bands' postings.** Re-unwrapping band B at 80 m instead of
  320 m left the plane unchanged (4.68 vs 4.81 rad p2p — the ramp is robust) but
  moved the non-planar part from 0.313 to 0.279 rad and its correlation with band
  A's own residual from +0.22 to +0.32. The ramp survives a coarse second band;
  finer structure does not.

⇒ on a quiet pair, a deramp gets ~95% of what this does, faster and with one
band. Reach for the dual-frequency screen when the scene has signal a deramp
would eat, when the ionosphere is strong enough to matter for unwrapping, or when
you need the correction to be a *measurement* you can defend rather than a fit.

### Cleaning the unwrapped phase

Three lazy `UnwrappedStack` methods, chained before `to_los`:
`mask_edges` → `remove_outliers` → `deramp`. They apply to a SNAPHU result and to
a NASA GUNW alike. Each records an attr that folds into the persist hash, so a
stage cleaned differently cannot silently reuse an old store.

**`mask_edges` trims only the along-track edges by default** (`edges="along_track"`).
Only the near- and far-range boundaries carry the decorrelated antenna-pattern
fringe, and they run along-track, so each row's first and last valid samples *are*
those two edges. The old isotropic behaviour (`edges="all"`) eroded the whole
finite footprint, which on a coastal frame means the coastline and every masked
lake as well: measured on a real Venezuela frame at `edge_pixels=32`, **47.5% of
what it removed was coastline, lakes or the azimuth end** — and the coastal strip
is exactly where the near-fault signal was. A GUNW additionally carries an exact
edge mask; `mask_edges(use_builtin_mask=True)` decodes it.

**The sides can be named one at a time.** `edges` also takes `"left"`, `"right"`,
`"top"`, `"bottom"`, a sequence of them, or a `{side: pixels}` mapping when they
want different depths (`edge_pixels` is then unused):

```python
unw.mask_edges(edge_pixels=32, edges="left")             # one range boundary
unw.mask_edges(edge_pixels=32, edges=["left", "bottom"]) # + an azimuth end
unw.mask_edges(edges={"left": 32, "right": 8})           # uneven depths
```

The names are **grid-relative, not orbit-relative** — which of left/right is
*near* range depends on the pass and the look direction, so naming them after the
grid is the one reading that cannot be got backwards. Each side is still found by
scanning for the first and last valid sample of every line, so naming one spares
the coastlines and lakes `edges="all"` would eat; `top`/`bottom` scan columns
instead of rows, which is the only way to reach an azimuth end without the
isotropic erosion. ⚠️ **Prefer one call over chaining several**: every side is
measured against the same footprint, whereas a second call scans what the first
left behind (on a sheared swath that costs real data), and `attrs["edges_masked"]`
records only the last call — so two chained calls hash as one. The same applies to
`GSLCStack.mask_edges` and `InterferogramStack.mask_edges`, which take the same
`edges`.

**`deramp(mask=...)` keeps a deformation region out of the *fit* but still
subtracts the ramp there**, so real signal doesn't bias the trend and is still
kept. Pass a boolean `(y, x)` grid or a `(lon_min, lon_max, lat_min, lat_max)`
bbox.

⚠️ **A mask that spans the full width *and* touches a frame edge makes the
polynomial extrapolate**, and high degrees then explode. Measured on a real frame
where the mask resolved to all 6748 columns and the top 2481 rows, so the fit saw
normalized `y ∈ [−0.30, 1]` and was evaluated out to `y = −1`:

| degree | 1 | 2 | 3 | 5 | 7 |
|---|---|---|---|---|---|
| max leverage | 4.0 | — | 64 | 1036 | **16477** |
| hold-out rms (rad), extrapolating 1.2° of latitude | 4.23 | 3.71 | 21.8 | 1914 | **14017** |
| \|max\| in the masked band vs a ~20–30 rad signal | 45 | 40 | 37 | 143 | **702** |

⇒ **with an edge-anchored mask use degree 1–2; degree ≥ 4 is unusable.** Bounding
the mask in *longitude* instead drops degree 7's leverage to 31.7, the same as no
mask at all. A mask spilling outside the data bbox is harmless — it is compared
against the stack's own coords — it is full *width* that hurts.

⚠️ **`deramp(method="spline")` is a 23-minute near-no-op at the default scale.**
`scale=None` resolves to a quarter of the raster, and scipy's FIR
`gaussian_filter` then needs a radius of `4σ` = the whole raster: 699 s per pass,
two passes, no parallelism. And a *wider* Gaussian is a *gentler* high-pass —
measured, it leaves **49.5% E-W / 43.4% N-S** of a linear ramp behind, against
0.0% for `poly degree=1`. ⇒ prefer `method="poly"`; if you really want a spline,
set `scale` explicitly.

Where the deformation touches the frame edge the ramp beneath it is
**unconstrained by the data** — different degrees agree to ±6 rad just past the
mask edge and span −14…+442 rad a degree further out. For a slip inversion the
better answer is not to pre-remove it at all: `SlipInversion(..., ramp="linear")`
solves offset and gradients per track jointly with the slip.

### Reading LOS products this package didn't make

`LOSStack.from_grd(los, look_e, look_n, look_u)` is the way in for an existing
GMT-grid product — an ALOS scene, someone else's processing chain — so it can be
inverted beside a NISAR stack. Geographic grids are reprojected to UTM on load,
which is not cosmetic: the quadtree measures `width_min` in **metres** against the
pixel spacing, so a lon/lat lattice would put `width_min=1000` at ~10⁶ columns.

**Three things a `.grd` cannot record must be declared, and each looks plausible
when wrong:**

- `units` — `"m"`, `"cm"` or `"mm"`; everything downstream is metres.
- `sign` — `+1` if the grid is already positive *toward* the sensor, `−1` if
  positive away (a range increase).
- `look_convention` — `target_to_sensor` or `sensor_to_target`.

`look_convention` is **checked against the data** (a target→sensor vector has
`look_u = cos(inc) > 0`) and raises, as does a look triple whose median norm is
outside 0.9–1.1. `units` and `sign` are **silent when wrong**. Get `sign` from the
physics, not from the fit: on the real ALOS D134 scene both routes agree it is
`+1` — the across-fault LOS step implies the south block moves west (dextral, the
known sense of the San Sebastián), and inverting it that way gives 66% dextral
slip. **Variance reduction is 98.9% either way**, because a global flip just flips
the unbounded solution, so VR cannot settle this and only the physical sense can.

⚠️ **`import h5py` silently breaks pygmt's NetCDF-4 grid reading**, and
`nisar_tools` always imports h5py. `xr.open_dataarray(..., engine="gmt")` then
returns pixel *indices* instead of lon/lat, with no error — data right,
georeferencing wrong, nothing downstream notices. That is why the `gmt` engine is
deliberately not used; NetCDF-4 grids are read with h5py directly, verified
bit-identical to pygmt.

### Pixel offsets

Amplitude offset tracking, a port of GMTSAR's `xcorr` in its default
frequency-domain mode. It measures the shift of the backscatter pattern between
two acquisitions by cross-correlating their amplitude on a coarse lattice — the
measurement that survives where the interferogram decorrelates, which is the near
field of a large earthquake.

```python
offsets = stack.pixel_offsets()                   # or step=, or nx=/ny= (xcorr's -nx/-ny)
offsets = offsets.persist(ws, "offsets")
offsets.to_text("offsets/")                       # freq_xcorr_pair0.dat, + lon/lat
offsets.to_text("offsets/", lonlat=False)         # the plain five xcorr columns
offsets.to_grd("offsets/")                        # x_offset, y_offset, correlation

# Almost always what you want to look at -- see "Reading the field" below.
fig, axes = offsets.plot_offsets(pair=0, robust=True, vmax=0.5)
clean = offsets.robust_offsets("x_offset")        # the same numbers, as a DataArray
```

`PixelOffsetStack` is a georeferenced raster like any other stage, so `crop`,
`persist` and `to_grd` work on it unchanged.

**What the offsets mean, and what they are not.** `xcorr` runs on radar-geometry
SLCs, where the two images are unaligned and the offsets are a *coregistration*
solution in range and azimuth. A NISAR GSLC is **geocoded**: the pair already
shares one map grid, so there is nothing to coregister, and what comes out is a
shift in map x/y — ground displacement. That is why the fields are `x_offset` and
`y_offset` rather than range and azimuth. They are not literally range/azimuth
offsets and converting them to a 3-D displacement vector needs the geocoding
geometry, which this does not attempt.

- **Sign**: an offset is positive when the feature sits at a **larger index** in
  the secondary — `sec_position - ref_position`, the same convention as `xcorr`.
  `east_offset` and `north_offset` give metres. The northing flip is free, since
  `y_spacing` is stored signed and is negative on a north-up grid, so a positive
  `y_offset` (southward) comes out as a negative northing.
- ⚠️ **A high correlation does not mean an accurate offset.** The `correlation`
  column is `xcorr`'s own normalised value ×100, the number `fitoffset.csh`
  thresholds at 20. On a band-limited synthetic field every window scored 96–97
  while individual offsets scattered by up to **0.4 px**; the correlation says
  the match is real, not that it is precise. Judge the population, not a location.
  This is why `robust_offsets` exists.
- **Out-of-swath NaN is handled**; `xcorr` has no concept of it. A window with
  less than `min_valid_fraction` (0.5) finite samples returns NaN with correlation
  0, and does not contaminate its neighbours.
- ⚠️ **The window defaults are not GMTSAR's.** `xcorr` uses `nx_corr=ny_corr=128`
  with `xsearch=ysearch=64`, sized for a radar-geometry pair that still has to be
  coregistered. A geocoded pair is already aligned to the whole pixel, so most of
  that search covers lags that cannot occur and each is another chance at a false
  peak. The defaults here are **`nx_corr=ny_corr=64`, `xsearch=ysearch=16`** — a
  320 m template searched over ±80 m at NISAR's 5 m posting — which on real data
  halved the false-peak rate (4.2% → 2.5%) and left the measured displacement
  gradient unchanged. The transformed patch is `ny_corr + 2*ysearch` by
  `nx_corr + 2*xsearch`.
- ⚠️ **`nx_corr` is the resolution; `step` is only the sampling.** A denser lattice
  gives more samples of the same template-wide average, so to localise a
  discontinuity better you shrink `nx_corr`, not `step`. `step` defaults to **16**
  (80 m), which deliberately oversamples the 64 px template — adjacent estimates
  share 83% of their data and are not independent. It makes a readable map; it
  does not add information. Use `step=nx_corr//2` for independent estimates at a
  quarter of the cost.
- **`subpixel_window` follows `xcorr` at 8**, and the reason changed with the
  window sizes above. The refinement FFT-interpolates that window, i.e. treats it
  as periodic, and at 8 samples the correlation peak has not decayed by its edges,
  which *biases* the result. A wider window cuts that bias, and at `nx_corr=128,
  xsearch=64` it did so for free — which is why 16 used to be the default. At a
  64 px template it is not free:

  | `subpixel_window` | max abs bias | scatter after centring |
  |---|---|---|
  | **8** | 0.136 px | **0.094 px** |
  | 16 | 0.079 px | 0.193 px |

  The scatter is what matters, because `robust_offsets` subtracts the field's own
  median and so removes a uniform bias exactly while leaving scatter untouched.
  Pass `subpixel_window=16` when the *absolute* offset matters rather than its
  variation, and keep it well under `2*search` — the window is clipped to the
  correlation surface and zero-padded beyond it, so one as wide as the surface
  biases every peak that is not already centred (this now raises).
- `x_shift`/`y_shift` centre the search on a known bulk displacement (`xcorr`'s
  PRM `rshift`/`ashift`) and are added back into the answer. Prefer them to a
  wider `xsearch`: recentring the search costs nothing, enlarging it costs false
  peaks.

**Reading the field: `robust_offsets`.** The raw offsets are not yet ground
motion. `offsets.robust_offsets("x_offset")` — and `plot_offsets(robust=True)`,
which applies it to both components — does three things, each removing something
that is not displacement:

1. drops locations below `min_correlation` (default 20, `fitoffset.csh`'s own);
2. drops locations more than `n_sigma` (default 6) **MAD**-based robust standard
   deviations from the median. ⚠️ This is the only thing that removes the **false
   peaks**, and no correlation cut finds them: a window that fails to match still
   reports an argmax, which pins at the edge of the search and comes back as
   ±(search − 0.5) px *with a high correlation*. Measured at **3–4%** of locations
   on a real A061 crop;
3. subtracts the median, which is the pair's **bulk geolocation offset** — −1.06 px
   (−5.3 m) between two real D_071 granules, agreeing between crops 20 km apart to
   0.06 px. It is several times the deformation in the same scene and constant
   across it, so the raw field plots the difference between two products and hides
   the ground motion.

Statistics are computed per pair and per component, and the removed median and
robust sigma are reported in the result's `attrs` — the median is itself a
measurement of how far apart the two products are geolocated.

⚠️ **Invalid samples are painted grey, and that is not cosmetic.** A diverging
colormap is near-white at its midpoint and matplotlib leaves NaN transparent, so
on a white figure a **zero offset and a dropped location are the same pixel** —
precisely the confusion `robust_offsets` walks into, since it centres the field on
zero and can drop a third of it. `plot_offsets` sets both the colormap's bad
colour and the axes facecolor (the lon/lat reprojection's own margin is not drawn
at all, so the colormap alone does not cover it); `nan_color=` overrides. ⚠️ `n_sigma`
interacts with signal amplitude: the sigma is measured over the whole scene
including the signal, so a large displacement widens the cut (it will not clip
itself) while a quiet scene gets a tight one. Raise `n_sigma` before suspecting
the estimator if a scene has a genuine metre-scale step.

**The ASCII file is byte-compatible with `xcorr`'s.** `to_text` writes
`x_pixel x_offset y_pixel y_offset correlation` under `print_results.c`'s own
format string, leading and trailing space included, so `fitoffset.csh` and any
`awk '{if ($5 > 20) ...}'` on top of it work unchanged. Pixel indices are 0-based
and index the **source** GSLC grid, not the coarse lattice. There is no header by
default, because `xcorr` writes none and an unprefixed one would be compared
numerically by `awk`; `comment=` opts into `#`-prefixed provenance lines.

Two more columns, **`lon` then `lat`** in decimal degrees, follow **by default**
— GMT's x-then-y order, not the spoken "lat/lon" — giving each location's own
ground position, which is the only thing in the file that means anything outside
this granule's grid (`gmt psxy`, `blockmean`, or a join against another track).
They are *appended*, so `$1`…`$5` keep their positions and every `awk` on top of
them reads the same fields; the five-column prefix of a line is byte-for-byte
what `xcorr` writes, which is what lets them be on by default without breaking a
`fitoffset.csh` pipeline. `to_text(..., lonlat=False)` writes the five columns
alone. The transform is pointwise over the location lattice, not two 1-D axis
vectors: a projected grid is rotated against true north away from its central
meridian, so latitude varies along a row and longitude down a column.

Measured on the two real D_071 granules (12 days apart, 5 m posting, a 28 km
crop): 97.4% of locations above correlation 20. Their integer offset is exactly 0
everywhere — the products are co-registered to the whole pixel — over a
**uniform** sub-pixel −0.125 px in x and −0.0625 px in y, i.e. a real 0.63 m /
0.31 m geolocation difference between the two granules rather than anything
moving. Correlating one granule against *itself* returns 0.000 at every location,
and a planted whole-pixel shift comes back as the planted value plus exactly that
same offset. Throughput at the current defaults is **6900 locations/s** on a real
A061 crop (8856 locations in 1.3 s), the narrower 96 px patch being 7× less
transform than the 256 px one the GMTSAR defaults ask for.

### Exporting to GMT `.grd`

`to_grd(outdir)` is on every product stage (`GSLCStack`, `InterferogramStack`,
`UnwrappedStack`, `LOSStack`). Each field is reprojected to lon/lat and written as
a single-variable GMT grid that `grdinfo` and `grdimage` read directly; a field on
the stack axis is written per slice.

Defaults differ per stage because the useful layer does: GSLC exports
`amplitude` only (absolute SLC phase is meaningless), interferograms `phase` +
`coherence`, unwrapped stacks every layer present, and `LOSStack` the LOS
displacement, the ENU unit vectors and the angles. The **unit vectors are the
point** — scalar LOS from an ascending and a descending track can only be
decomposed into vertical and east-west motion if you kept them.

## Merging frames

A track is delivered as a series of frames, and one earthquake rarely sits inside
a single one. Join them at either of two stages:

- **Before interferograms** — `GSLCStack.merge` gives one continuous complex
  stack. The simpler route when both frames have the same acquisitions.
- **After unwrapping** — `UnwrappedStack.merge` additionally removes the 2π seam
  between frames unwrapped independently, and is the only route when the frames
  come from different tracks or different dates. Follow it with
  `to_los([granule_a, granule_b])` — **one granule per frame**, since each cube
  spans only its own frame.

Both grids must lie on the same lattice. Interferograms formed with
`align_looks=True` (the default) do so automatically: multilook blocks are
anchored to the absolute native grid rather than to wherever the crop happened to
start. Without it, two frames cropped to different extents come out a fraction of
a multilooked pixel apart — at `looks=10`, three native pixels of difference is
exactly 0.3 px, and no amount of padding lines them up. `UnwrappedStack.merge`
resamples an off-lattice frame and warns; re-forming both interferograms avoids
the resampling and is exact. Differing *spacing* still raises — that means
different `looks`, which no regridding fixes.

Merging across a **UTM zone boundary** works: a track crossing 114°W is gridded in
two zones, and the second frame is warped onto the first's grid first. Integer
layers (`conncomp`, subswath masks) are always nearest — an interpolated label is
not a label. `GSLCStack.merge` deliberately still refuses an off-lattice pair:
interpolating a complex SLC's carrier aliases the fringe, so a misaligned SLC pair
must be re-cropped, not resampled.

### Different dates

Two frames from different acquisition dates are *different interferograms*, not
one interferogram in two pieces. Pass `time_tolerance=None` to pair them by
position; `tie` then decides how the overlap is reconciled:

| `tie` | removes | right when |
|---|---|---|
| `"cycles"` | a whole number of 2π | one interferogram unwrapped separately per frame — the only thing they *can* differ by |
| `"offset"` | the real-valued median difference | different date pairs, which measure different deformation, so the difference is **not** an integer number of cycles |
| `"none"` | nothing | the two are already on a common datum |
| `"auto"` | picks by whether the times matched | the default |

> ⚠️ **For a slip inversion, do not merge different-date scenes.** A mosaic is one
> raster with one arbitrary constant and one ramp, so a second scene's independent
> nuisance gets absorbed as fictitious slip. Measured on a synthetic recovery: two
> scenes kept apart and combined with `Observations.concat` gave slip correlation
> **0.979** and moment ratio **1.001**, while the same data mosaicked into one
> raster gave **0.553** and **1.671** — with a peak slip 3.5× the truth. Variance
> reduction stayed at 98.4% in both, so it does not warn you. Merge for a
> continuous map or an export; keep the scenes separate for the inversion.

## Slip inversion

`nisar_tools.slip` inverts line-of-sight displacement for coseismic slip on a
fault, discretised into triangular dislocation elements and solved as a bounded,
smoothed linear least-squares problem. It consumes a `LOSStack`, so it picks up
exactly where the InSAR pipeline leaves off. Unlike the raster stages this world
is **eager** and works on points and meshes, not grids.

```
FaultTrace → LocalFrame → FaultMesh → Observations.from_los → SlipInversion → SlipModel
  (.kml)      (shared!)   (the fault)   (quadtree sampling)   (Green's + solve)  (.save)
```

Three modelling choices are independent, and the simplest of each is the default:

| | default | also available |
|---|---|---|
| geometry | vertical (`FaultMesh.vertical`) | curved, one dip per deep segment (`FaultMesh.curved`) |
| medium | homogeneous half-space (`HalfSpaceTDE`) | layered, from EDGRN tables (`LayeredPointSource`) |
| slip basis | constant per element | continuous nodal tent functions (`basis="node"`) |

**Four things that are easy to get wrong:**

- **One `LocalFrame` for everything.** A transverse Mercator centred on the study
  area, *not* UTM, so two tracks in different zones share one x/y. Every object
  stores its frame and checks it; mixing frames is a silent kilometre-scale error
  rather than a crash.
- **Positive `strike_slip` is LEFT-lateral.** A right-lateral fault — San
  Sebastián, Sagaing — needs `polarity=(-1, 0, 0)`. This is measured, not a
  convention chosen: with winding pinned to the trace's left-hand normal, the
  strike vector works out to exactly minus the trace tangent, at any strike.
- **`exclude_within` is required.** A dislocation solution is singular *on* the
  fault surface, so a sample sitting on the trace gives a non-finite Green's
  function; the inversion names the offending samples and refuses to run rather
  than quietly zeroing them.
- **One track per scene, with `ramp=`.** Every unwrapped interferogram carries an
  arbitrary constant and usually an orbital or ionospheric ramp. `ramp="linear"`
  gives each *named* track its own offset and `x`/`y` gradients; without it those
  land in the slip as broad, deep, entirely fictitious patches.

### Measuring the sampling parameters instead of guessing them

`scene_report(los, trace, frame, mesh=mesh)` derives `rms_min`, `width_min` and
`exclude_within` from a scene, and reports the coverage that limits the answer
regardless of them. Inheriting these three from an example is how a setup goes
quietly wrong — each is a statement about the **data**, not a preference:

- **`rms_min` is a noise level.** The quadtree splits while the *pixel* scatter
  inside a cell exceeds it, and that scatter does not shrink as the cell does — so
  set below the noise, cells can never stop and simply run down to `width_min`.
  On real scenes this showed up as a median within-cell scatter of **11.8 mm
  against `rms_min=6 mm`**, with 47% of cells stuck at the floor, and accounted
  for 18,631 samples of mostly atmosphere. `noise_floor` measures it on a *fixed*
  block grid, so the estimate cannot depend on the quadtree it is about to
  configure.
- **`width_min` is discontinuous.** Cells are index rectangles halved at their
  midpoint, so the reachable sizes form a dyadic ladder — per axis, per scene, and
  not guessable. `cell_size_ladder` simulates the descent: 1000 m gave a 1075 m
  terminal cell and 57,246 samples, 1500 m gave 1750 m and 24,429, and two values
  inside the same gap do nothing at all.
- **`exclude_within` only has to stop a *cell* straddling the trace**, so its floor
  is `width_min / 2`; the singularity itself needs only metres. Larger values are a
  judgement about unwrapping error and near-fault model error — worth making
  deliberately, since removing the near field also removes the only data that
  constrains shallow slip.

`ramp_content(obs)` answers the companion question and needs no Green's matrix:
how much of the data a per-track offset explains, and how much more the `x`/`y`
gradients add. The columns are normalised by each track's span, so a gradient
reads directly as **metres of LOS across the scene** — centimetres is orbit, tens
of centimetres is deformation being taken out of the slip model. For a long,
near east-west strike-slip fault the far-field pattern is an arctangent step
across the trace, which over a finite aperture looks a great deal like a gradient
perpendicular to strike, so the two genuinely compete.

#### Is the ramp eating the earthquake?

`ramp_content` overstates the ramp badly, and inherently — nothing in it has a
fault model, so the polynomial is free to claim the coseismic pattern. Two
functions close that gap, and they answer different halves of the question.

```python
from nisar_tools.slip import ramp_leakage, ramp_sweep

leak = ramp_leakage(inversion)                      # no solve; what it *could* absorb
sweep = ramp_sweep(inversion, ("offset", "linear", "quadratic"), **solve_kwargs)
```

`ramp_leakage` is geometric: the fraction of the fault's own predicted field that
lies in the ramp's column space, `||P_ramp (G s)|| / ||G s||`. Zero means the ramp
cannot touch the earthquake whatever its coefficients do; one means it reproduces
it exactly — and since the ramp is unregularised while the slip is smoothed and
bounded, the ramp then wins. **Read `curvature_only`, not the totals**: a per-track
offset always absorbs that track's mean displacement, because an unwrapped
interferogram has an arbitrary constant.

`ramp_sweep` re-solves at each degree and reports what actually happened. It is
affordable because `SlipInversion.with_ramp` shares the Green's matrix — one
*solve* per degree, not one assembly.

From the runners, after stage 1 has written the observations:

```bash
KMP_DUPLICATE_LIB_OK=TRUE NISAR_RAMP_KINDS=offset,linear,quadratic,cubic \
nohup python -u scripts/run_ramp_check.py > workdir/ramp_check.log 2>&1 &
```

writing `ramp_leakage.json`, `ramp_sweep.csv` and `figures/ramp_check.png`, and
printing a verdict. Outputs gain a `_bootstrap` suffix when the observations came
from the coarse round-0 sampling, as the L-curve's do.

- ⚠️ **Variance reduction cannot answer this, which is the whole problem.** Extra
  free columns can only fit better, so a ramp absorbing coseismic signal looks like
  a **better fit with a smaller earthquake**. `ramp_sweep` flags exactly that
  conjunction — moment down by more than 20% while VR is up — and warns.
- **The conjunction matters, not either half.** On the starved synthetic fixture the
  *linear* ramp also loses 31% of the moment, but it fits *worse* while doing so:
  that is a worse model, not absorption, and it is deliberately not flagged.
- **Measured, two synthetic apertures.** Full aperture with the near field kept:
  `curvature_only` +0.006, moment ratio 1.003, nothing flagged. Everything within
  60 km of the trace discarded: quadratic leakage **0.992**, and the quadratic solve
  recovers **13% of the moment while VR rises** — flagged.
- **Leakage says *can*, not *did*.** The healthy fixture leaks 0.55 at linear order
  and its moment is stable to 0.3%. A high leakage is a reason to run the sweep, not
  a verdict on its own.

**`plot_fit` draws the ramp on its own, beside the model without it.** When the
inversion solved nuisance columns the figure has two rows: `Data | Model |
Residual` on top, and `Ramp | Data − ramp | Model − ramp (slip only)` below. The
top row is the fit; the bottom row is *what it was fitted with*. That separation
is the only place in the figures where a ramp drifting into the earthquake is
visible at all — variance reduction is blind to it by construction, and the
combined `Model` panel is exactly the sum that hides it. The ramp panel
deliberately shares the **data** panel's colour scale so its size can be read off
directly, the two deramped panels share their own, and the title reports how many
metres of the data the ramp is carrying. `plot_fit(model, ramp=False)` keeps the
old three-panel form; a model solved with `ramp="none"` gets it anyway.
`model.ramp_prediction` and `model.fault_prediction` are the same two fields as
arrays, and a model reloaded with `SlipModel.load` still splits — the ramp
columns are rebuilt from the observations, exactly as the smoothing operator is.

**None of it fixes coverage, which is usually what binds.** The along-strike
profile is the point of the report: on the Venezuela scenes an aggregate "19% of
samples north of the trace" read like thin two-sided coverage, while the profile
showed the north block was absent along the eastern *two-thirds* — exactly where
the largest signal was. `plot_coverage(report)` draws it.

Aim for a few thousand samples. Recovery flattens off around 5× the parameter
count: on a synthetic test spanning a 6.6× change in sampling density, 2.4× the
parameters already put the L-curve corner within one grid step of the
fully-sampled answer. Sampling harder than the noise supports buys noisier cells
and a slower solve, not information.

### Sub-sampling from the model instead of from the data

Everything above picks quadtree cells from the **observed** displacement, which is
noise-driven for the reason in `rms_min` above. `iterate_sampling` implements the
Wang & Fialko (2015, §2) alternative: quadtree a **synthetic** interferogram from
a preliminary model, fill those cells from the real data, re-solve, and repeat
until the model stops changing (default: no parameter moves by more than 1 cm).

**Resample first, and not only for tidiness.** The reachable cell sizes are a
dyadic ladder *set by the pixel size*, so two scenes at different resolutions land
on different ladders, one `width_min` means two different things, and the
per-track sample counts diverge for a reason that has nothing to do with the data
— which `Observations.concat` then turns into an unintended reweighting.
`resample_all(scenes, frame)` puts every track on one grid in the shared frame,
defaulting to 10 arc-seconds (309 m): the usual ALOS-2 posting, and already ~20×
finer than a fault element.

**`refine_within` is not optional.** An initial model with little shallow slip
predicts a smooth near field, so the quadtree coarsens exactly where shallow slip
needs constraining and the next round is free to invent it — the paper's "spurious
shallow slip", and its own remedy was "a relatively dense sampling around the
fault trace was retained through all iterations". `iterate_sampling` holds that
band through every round.

**`rms_min` changes meaning** under model-based sampling: a threshold on model
*curvature*, not a noise level, so the `scene_report` value does not apply.
`model_rms_min` derives it from the predicted field's own scatter, and
`iterate_sampling` calls it for you.

⚠️ **Not unconditionally better.** Measured on the synthetic fixture at 12 mm of
noise, data-driven vs model-driven correlation with the truth: **0.944/0.934** for
white noise, **0.895/0.874** correlated over 15 km, **0.852/0.918** at 30 km. The
gain arrives only once the noise is long-wavelength enough that extra samples
inside one atmospheric cell carry no independent information; the sample count
moves in either direction. What it reliably does is stop the sampler chasing
noise. On the real ALOS-2 D134 scene: 610 samples in round 0, then 1246 → 1311 →
1312, converging in three rounds (max parameter change 1.38 → 0.045 → 0.001 m) at
VR 97.8%.

⚠️ **A degenerate model makes the loop "converge" on nothing**: a flat model
predicts a flat field, the quadtree splits on float noise, no parameter moves, and
the loop stops declaring success. A `RuntimeWarning` fires before each
model-driven round if peak slip is under 1 mm — believe it over `converged`.

`plot_mesh(mesh, trace=trace)` draws the discretisation itself — map view beside
the unrolled section, coloured by element area — and takes a bare `FaultMesh`, so
it is available before any model exists. Read it against `plot_samples`: elements
much smaller than the data constraining them are what the smoothing weight then
has to paper over.

`plot_slip(model)` draws the **unrolled** fault, along-strike distance against
depth. That is the right default — the unrolling is exact, since the fault is
parameterized by exactly those two coordinates — but it projects the dip away by
construction. On a dipping or curved mesh use `plot_slip_3d(model, trace=trace)`,
which shades the same per-element field on the same colour scale so the two can be
read against each other:

```python
from nisar_tools.slip.plot import plot_slip_3d

fig, ax = plot_slip_3d(model, component="strike", trace=trace,
                       exaggeration=4.0, view=(22, -70))
```

- **`exaggeration` stretches every axis except the longest one**, and it is not
  cosmetic: the Venezuela mesh is 264 × 25 × 40 km, so at true scale
  (`exaggeration=1`) it renders as an unreadable sliver. Stretching the two short
  axes *together* preserves the apparent dip whenever the fault strikes near a
  grid axis; on a diagonally striking fault the dip is distorted like any
  vertically exaggerated section.
- **`view=(60, -90)` looks down on the surface**, which is where a dip *reversal*
  is easiest to see — the case a bottom trace produces and the unrolled panel
  cannot show at all.

### Weighting by a measured noise model

InSAR noise is spatially correlated — atmospheric delay over kilometres to tens of
kilometres, the same scale as the deformation. Without a model for that, an
observation's weight is `1.0`, scaled per track by `1/sqrt(n)`, so samples are
weighted by *how many their track contributed* and nothing else.

`nisar_tools.slip.covariance` fits the correlation from the far field and whitens
the system:

```python
from nisar_tools.slip import covariance as C, SlipInversion

model, variogram = C.scene_covariance(los_stack, trace, frame, min_distance=50e3)
print(model)                     # exponential: sill 20.7 mm, nugget 2.2 mm, range 22.7 km
inversion = SlipInversion(mesh, obs, covariance={"A162": model, ...})
```

Measured on six real Venezuela scenes: sill 13–41 mm, nugget 0.4–3.3 mm, and
**range 22–33 km on all six** despite two different processing chains — that
agreement is the validation. The noise is 15–100× more correlated than white.

⚠️ **`detrend` must match the ramp the inversion solves.** The default is degree 2
because these scenes carry an unremoved ramp and the inversion solves
`ramp="quadratic"`. At degree 1 the fit *collapses* on all four NISAR scenes —
sill 0.0 mm, range 363–816 km against a 108 km maximum lag, i.e. the variogram
never plateaus. `fit_covariance` warns when the fitted range exceeds the largest
lag measured, which is that failure's signature.

⚠️ **A covariance does not improve the slip estimate, and that is measured.**
Swept over 24 smoothing weights and compared at each arm's own optimum, the
whitened solve recovered a planted patch *worse* — 0.147 m against 0.069 m rms, on
5 of 5 seeds, and still worse at SNR 2.0 / 0.4 / 0.1 and at matched effective
rank. Generalised least squares is minimum-variance only among *unbiased*
estimators, and this one is bound-constrained and damped to tens of parameters out
of hundreds, so the regulariser sets the error; an exponential `Cd` up-weights
short wavelengths, exactly what the smoothing exists to suppress.

What it *does* buy is an honest account of where the evidence came from. At
matched rank the per-scene share of `diag(N)` shifts monotonically with each
scene's own sill — quiet scenes gain, noisy lose, D135 ×1.35 through D054 ×0.73 —
and `SlipModel.chi_square` becomes meaningful (it returns `None` without a
covariance, rather than a number in arbitrary units). Solve plain, report
resolution whitened, is the defensible split.

⚠️ **No existing smoothing weight transfers.** Whitening divides by sigma instead
of `1/sqrt(n)`, so a given `lambda` damps far less; on the real problem plain
`lambda=300` and whitened `lambda=1e6` reach the same effective rank. Re-run the
L-curve.

### Choosing the smoothing weight

`l_curve` solves at many weights over the same Green's matrix and tabulates misfit
against roughness; the corner of the curve is the conventional choice. Too much
smoothing and the model can't fit the data; too little and it fits the noise with
rough, deep, physically implausible slip.

⚠️ **Both failure modes are invisible in variance reduction.** Measured on the
real D134 scene, 100 elements, `ramp="offset"`:

| λ | 1000 | 100 | 10 | 3 | 1 | 0.3 | 0.1 | 0.03 |
|---|---|---|---|---|---|---|---|---|
| VR % | 21.96 | 22.15 | 36.70 | 80.10 | 95.40 | 97.68 | 98.37 | 98.48 |
| max \|slip\| m | 0.000 | 0.004 | 0.308 | 1.678 | 2.862 | 3.926 | 5.569 | **6.000** |

The corner is **0.3–1.0**. Above ~30 the smoothing wins outright and the model
comes back flat zero *while VR still reads 22%* — that is the trap. Below ~0.03
the ±6 m strike bound saturates, so the *bound*, not the data, is setting the
answer. `solve()` normalises the operator by its own row count, so a weight stays
roughly invariant under mesh refinement.

**The sweep is serial on purpose, and the inversion has no parallelism knobs.**
Every weight is an independent solve over the same `G`, which makes a worker pool
the obvious move — it was written, verified to give identical results, and then
removed. Threads measured **1.02×** at 2 workers and *worse* above (0.90× at 8); a
process pool **1.10×**. Two independent reasons:

- **Load imbalance dominates.** The weights cost
  0.78 / 0.75 / 0.54 / 0.84 / 0.61 / 0.91 / **12.15** / 1.59 s — λ=0.02 ran to the
  iteration cap and was **67% of the whole sweep**, so perfect parallelism caps at
  1.50×. That weight's result is meaningless anyway (`converged=False`): drop it,
  don't parallelise it.
- **Threads can't overlap the solver.** scipy's `lsmr` is pure Python, so its
  iteration loop holds the GIL between matrix-vector products. Capping
  `OMP_NUM_THREADS=1` changed nothing, which rules out BLAS oversubscription and
  identifies the GIL.

What *does* make a sweep fast is the inner tolerance (`lsmr_tol="auto"`, the
default) and not paying for weights that never converge. Judge that tolerance over
a whole sweep rather than at one weight: it was 5.8× on the reference problem, but
per-weight it ranges from **2.2× faster to 1.4× slower**, netting ~1.17×. Green's
assembly resists threading too, for a different reason: each element is ~50 small
numpy calls, so dispatch overhead swamps the GIL-free stretch (1 worker 24.6 s, 10
workers 62–79 s).

### Runtime, and checking convergence

Runtime tracks **conditioning far more than size**. On the reference problem (1106
elements, 8000 observations, 10 cores) Green's assembly is ~25 s and the solve a
few seconds — but an underdetermined or badly-scaled setup can spend minutes in
the solver. Two things follow:

- **Always check `model.converged`.** An iteration-capped solve returns a
  plausible-looking model whose variance reduction is meaningless; `to_text`
  refuses to write one.
- **More quadtree samples are not free.** 19,562 samples cost 59 s in the solve
  against 0.8 s for 860, and cannot constrain a mesh that only has a few hundred
  elements.

### A fault that dips

`FaultMesh.vertical` extrudes the trace straight down, and a vertical mesh has
nowhere to put dip-slip signal except into strike-slip or the residual.
`FaultMesh.curved` takes a set of straight **deep segments** in plan view and one
dip each: each is pushed down-dip, a smooth surface is fitted through those bottom
lines *and* the surface trace, and the mesh is that surface sampled on the
`(along-strike, depth)` lattice.

```python
mesh = FaultMesh.curved(trace, frame, segments=segments, dips=[75.0, 80.0],
                        max_depth=25e3, edge_length=3e3, bias_w=1.15)
```

A segment file holds four numbers — `x_begin y_begin x_end y_end`, in **metres in
the local frame**. `FaultSegment.from_trace(trace, frame, 3)` chops the trace into
equal chords instead, which is enough when all you want to say is "the western
third dips 70, the rest 85". For one dip everywhere, `uniform_dip=75.0`.

- **The surface between the trace and the bottom lines is decided by the
  regularizer**, because only two depths carry control points. That is why
  `smoothness` matters and why the reference's specific gridder is ported rather
  than substituted. Pass `depth_control=(x, y, depth)` — three arrays of
  intermediate control points, from relocated seismicity say — to make the
  profile bend with depth.
- **Dips above 90 are meaningful**, not an error: the fault leans the other way.
  The reference's Myanmar configuration uses `[75 75 70 80 85 90 100]`. Note `dip`
  folds to `[0, 90]` on export, so read `dip_direction` to tell the two leans apart.
- **`bias_w` thickens the depth levels downward**, putting fine resolution where
  surface data can actually constrain slip — a patch at 2 km is resolved far more
  sharply than one at 18 km. Measured on the real trace, 8 levels over 20 km:
  `1.15` runs 1.8 km levels at the surface to 4.2 km at the base, `1.3` runs 1.1 to
  5.5 km. It is orthogonal to the geometry, so `curved(uniform_dip=90, bias_w=1.3)`
  is a *vertical* fault with graded depth resolution — `FaultMesh.vertical` does
  not take it. One consequence: the neighbour smoother weights every edge equally,
  so graded levels make it anisotropic with depth.
- **`bias_w` alone does not fix how much grading you get** — `down_dip_levels`
  does. There are `down_dip_levels - 1` intervals with thicknesses
  `bias_w ** (0 … down_dip_levels - 2)`, so the deepest level is
  `bias_w ** (down_dip_levels - 2)` times the shallowest. Left as `None` the count
  comes from `edge_length` (`round(max_depth / edge_length) + 1`, which is **8** at
  20 km / 3 km), so a `bias_w` picked for a target ratio has to be set alongside the
  count it was computed for: `5 ** (1/15)` is a 5× grading at 17 levels and a 1.9×
  grading at the default 8.
- **`uniform_dip=90` reproduces `FaultMesh.vertical` bit for bit**, so switching
  constructor changes nothing already computed. That needs care: `cos(radians(90))`
  is 1.2e-12, not 0, and the triangular-dislocation solution loses *every digit* in
  a narrow band beside vertical — 190× the signal at a ten-thousandth of a degree
  off, against 2e-14 exactly at 90. Both the requested dip and the *fitted* surface
  are snapped onto exact vertical inside that band.
- **The fit is refused** if a down-dip offset exceeds the trace's radius of
  curvature, because the surface would fold back through itself. Real trace: min
  radius 74.4 km, so a 20 km offset at 45° is safe here — not in general.
- **`bias_l` (along-strike grading) raises rather than falling back to Delaunay**,
  whose arbitrary diagonals are the winding hazard the lattice triangulation exists
  to avoid. The reference's own demo sets `biasL = 1`.

#### From a bottom trace instead of dips

If the fault's bottom edge has been mapped, give it directly and skip the angles.
`bottom_trace` takes a `FaultTrace` or a path — the same `.kml`/text reader the
surface trace uses:

```python
mesh = FaultMesh.curved(trace, frame,
                        bottom_trace="fault-bottom.kml",   # map view of the bottom edge
                        bottom_depth=None,                 # None -> max_depth
                        max_depth=40e3, edge_length=5e3)
```

Nothing downstream changes, because a dip was never used as an angle in the first
place: `FaultSegment.project` converts it into a bottom line and only the line is
fitted. A bottom trace *is* that line, already digitised.

- **A KML carries no usable depth** — Google Earth writes every vertex at
  altitude 0 — so `bottom_depth` says what the line means, defaulting to the base
  of the mesh. Setting it **shallower** than `max_depth` is the useful case, not a
  mistake: the levels below the control row are set by the regularizer alone,
  which continues the dip linearly, so a trace digitised at a 15 km locking depth
  still builds a sensible 40 km mesh. Deeper than `max_depth` is refused —
  `gridfit` clips control points into its lattice, so it would be silently
  flattened onto the bottom row instead of ignored.
- **The dip may reverse along strike**, which is the case dips express badly. Where
  the bottom trace crosses the surface trace the fault leans the other way; a dip
  list can only say that with hand-tuned values straddling 90. The San Sebastián
  pair does exactly this — bottom edge ~10 km north of the trace for the western
  33 km, up to 7 km south for the remaining 230 km, giving 75.9°…100.1° at 40 km
  depth. `mesh.attrs["bottom_dip_flips"]` and `dip_range_deg` record it, and the
  runners print both.
- ⚠️ **A bottom trace drawn past the ends of the surface trace is trimmed, and this
  is not cosmetic.** `to_curvilinear` clamps to the polyline, so an overhanging
  point reports its distance to the *endpoint* — along-strike component included —
  and every such point piles onto one arc length. Measured on the real pair, a
  17 km overhang turned a −7.5 km offset into −18.3 km: a 65° dip at the tip of an
  otherwise 76–90° fault, silent and plausible-looking. Samples whose reported
  offset is inflated are dropped with a `RuntimeWarning` naming the count; an end
  vertex that merely reaches a little past the end at a bend is dropped quietly,
  since it constrains nothing its neighbours do not.
- The same folding guard applies, stated as the offset at the base of the mesh
  rather than as a dip. Mutually exclusive with `segments=`/`uniform_dip=`.

### A mesh the data chooses

`FaultMesh.curved` places one node per resampled trace vertex at every depth
level. That is a reasonable default and a poor description of what surface data
can see: resolution decays sharply with depth, so a deep level carries parameters
nothing constrains while a shallow one may be too coarse. Three arguments place
the nodes freely instead, and all three leave the geometry arguments above
untouched — how a node is carried onto the fitted surface does not depend on
where it sits.

```python
mesh = FaultMesh.curved(
    trace, frame, bottom_trace="fault-bottom.kml", edge_length=5e3,
    depths=[0, 1e3, 2.5e3, 4.5e3, 7e3, 10.5e3, 15e3, 21e3, 30e3, 40e3],
    nodes_per_level=lambda depth: max(12, int(90 - 2e-3 * depth)),
)
```

- **`depths=`** lists the levels outright, in place of the geometric
  `bias_w`/`down_dip_levels` progression. Setting both raises rather than letting
  one be silently ignored.
- **`nodes_per_level=`** takes a scalar, one count per level, or a callable of
  depth; **`along_spacing=`** says the same thing in metres. This is the thing
  `bias_l` could not do, and `bias_l` still raises — a single geometric ratio is
  not the same knob.
- **`layout=`** takes a `MeshLayout` built by hand, including a fully unstructured
  one from `layout.lloyd_layout`, which relaxes a point cloud against a target
  element-size field. `FaultMesh.from_layout(layout, trace, frame, ...)` is the
  same thing spelled the other way round.

Rows of unequal count have no quads to split, so those meshes are triangulated by
Delaunay in the `(arc length, depth)` plane. That used to be refused, because a
triangle wound the wrong way negates its dip-slip Green's-function column and
leaves strike-slip alone — a plausible model with a randomly signed dip-slip
field. It is safe now because the winding is fixed in **parameter space**: the
lattice split turns out to give one consistent sign of signed area in `(s, z)`,
and forcing that sign on every simplex reproduces its normals exactly, at any
strike. A uniform layout still takes the quad split, bit for bit, so switching to
these arguments cannot move a mesh you already built.

- **Adjacent levels are held to 2:1** in node count, and the levels themselves to
  2:1 in thickness. Ask for a sharper jump and you get a warning and a refinement,
  because a fine row against a coarse one is what makes Delaunay reach across for
  a long thin triangle.
- **Read a low minimum angle in context.** The real 918-node production mesh
  already sits at 11.3°, from 5 km along strike against 1 km top levels. A graded
  layout is not obviously worse — a 80→14 grading on the same fault measures 14.2°.
- **`plot_mesh(mesh, color=values, label=...)`** draws any per-element field, which
  is how model resolution is worth looking at before committing to a mesh.

### Sizing the mesh from model resolution

Variance reduction measures fit, and a slip model fits happily on a discretisation
the data never constrained — the spare freedom goes to the noise. Model resolution
measures the other thing.

```python
from nisar_tools.slip.resolution import (
    resolution_diagonal, element_resolution, target_element_size, adapt_layout)

resolution = resolution_diagonal(model.inversion, smoothing=0.3)
per_element = element_resolution(model.inversion, resolution)
layout = adapt_layout(mesh, target_element_size(mesh, per_element))
mesh = FaultMesh.from_layout(layout, trace, frame, bottom_trace=bottom)
```

`resolution_diagonal` is the diagonal of `R = (A'A + S'S + B'B)⁻¹A'A`, assembled
from exactly what `solve` stacks. One dense Cholesky; seconds at 918 nodes. `1`
means the estimate came from the data, `0` means the smoothing supplied it.

- ⚠️ **`solve` is bound-constrained and `R` is not.** The Venezuela run had 86% of
  elements on a bound, where a linear resolution matrix describes a different
  problem than the one solved. Use these numbers to size a mesh, never to state an
  uncertainty. `R` also depends on the smoothing weight, so evaluate it at the one
  the run uses.
- ⚠️ **`budget` is what makes an adaptive loop converge rather than drift.**
  Resolution is near zero over most of a fault, so a size field that merely
  equalises it grows the mean element every round: measured, 330 elements to 44
  over three rounds with variance reduction falling to **0.74%** — converging, and
  returning nothing. `budget="conserve"` holds the element count and lets the
  resolution decide only *where* the parameters go.
- ⚠️ **There is no cheap column-norm shortcut, and it is worth knowing why.**
  `‖G[:, k]‖` correlates with depth at −0.036 where resolution correlates at
  −0.742; it stays flat from the surface to 20 km while resolution falls from 0.34
  to 2e-6. A deeper element radiates a broader but weaker pattern whose norm is
  conserved, and resolution collapses because deep elements trade off with *each
  other* — which a column norm structurally cannot see.

#### Depth levels, or no levels at all

`adapt_layout(..., structure=)` chooses how much freedom the node set has.

```python
layout = adapt_layout(mesh, target_element_size(mesh, per_element),
                      structure="unstructured")     # or "rows", the default
```

**`"rows"` keeps depth levels**: one depth for the whole fault, with each level's
node count and positions free. That last part is the catch — a *level* is a single
depth, so the layering itself has to be read off **one vertical, at the fault's
midpoint**. Node density varies along strike; the levels cannot. A resolution field
that wants fine layering at one end of the rupture and coarse layering at the other
has no way to say so.

**`"unstructured"` drops the levels entirely.** The size field goes to
`lloyd_layout` as a density and the nodes relax to it in two dimensions, so a fine
patch can sit at one along-strike position with a coarse one beside it at the same
depth. Measured on exactly the case it exists for — samples over the western half
only, so resolution collapses in the east and the target size varies 2.6× along
strike:

| structure | elements | corr(size, target) | median \|rel err\| | min angle |
|---|---|---|---|---|
| `"rows"` | 357 | +0.926 | 15.3% | 9.0° |
| `"unstructured"` | 381 | +0.945 | **10.4%** | **19.1°** |

It follows the field 1.5× more closely *and* the elements are better shaped — a
coarse row against a fine one is what makes Delaunay reach for a sliver, and
dropping the rows removes the cause. On a field that varies mostly with **depth**
the two are equivalent (+0.930 vs +0.926), so this buys nothing where along-strike
coverage is even. Uneven coverage along the fault is the case: on the Venezuela
scenes the north block is absent along the eastern two-thirds.

- **The fault's outline stays a rectangle in `(s, z)`.** Freeform means the
  discretisation, not the extent: `boundary_node_mask` finds the four sides by
  exact equality and the zero-slip constraint depends on it, so an undulating top
  or bottom row would silently unconstrain the rim.
- ⚠️ **The relaxation honours the size field only because it is density-weighted.**
  An area-weighted centroid is the centroidal-Voronoi iteration for a *uniform*
  density, so it relaxes toward equal element sizes and undoes the grading. With
  the `1/h²` weight a 4× down-dip field comes back at a 2.95 size ratio against a
  wanted 2.85; without it, 2.16.
- `iterations` and `seed` are fixed, so a layout is reproducible and
  `FaultMesh.digest()` is stable across runs. A minimum triangle angle below 10°
  warns — the relaxation gives 21–27°, and the uniform production mesh already
  sits at 11.3°, so that threshold only fires on something degenerate.
- In the loop: `mesh_adapt={"geometry": ..., "structure": "unstructured"}`, or
  `NISAR_ADAPT_STRUCTURE=unstructured` for the runners. Both converge; on the
  two-scene acceptance fixture Mw agreed to 0.01 across the two structures while
  the peak moved 1.63 → 2.03 m, the usual "moment is what the data constrains,
  peak is what the discretisation does with it".

### The two-way loop

`iterate_sampling` already samples the data from the model (Wang & Fialko 2015).
`mesh_adapt=` adds the other direction, sizing the mesh from what the data
resolves (Barnhart & Lohman 2010), and `sampling_field="resolution"` puts the
sampling on the same footing (Lohman & Simons 2005) so both directions read one
quantity instead of two heuristics in different currencies.

```python
obs, model, history = iterate_sampling(
    scenes, mesh, trace, frame, sampling,
    mesh_adapt={"geometry": {"bottom_trace": bottom, "edge_length": 5e3}},
    sampling_field="both",
)
```

- **`mesh_adapt` must carry `geometry`.** A layout says where the nodes go, not
  what the fault is, so rebuilding needs the same keywords `curved` took. In the
  runners `slip_config.geometry_kwargs()` is the single source, so an adaptive run
  cannot quietly re-mesh at a different dip than stage 1 built.
- **`sampling_field`** is `"model"` (the default and what every earlier run used),
  `"resolution"`, or `"both"` — the sum of the two standardised fields. The two
  answer different questions: the first avoids spending samples on atmosphere, the
  second spends them where the model is unconstrained. Lohman & Simons needs no new
  sampler, because `from_los(field=)` already accepts an arbitrary decision field.
- **It moves samples rather than adding them.** Measured on a two-scene synthetic
  pair: `"model"` placed 1974 samples with 33.5% inside 20 km of the trace,
  `"resolution"` placed 890 with **74.4%**, `"both"` 1663 with 39.8%. Do not expect
  `"both"` to sample more than either — `rms_min` is a fraction of the field's own
  global scatter, so a field that is large but smooth raises the split threshold
  everywhere without raising any cell's local scatter.
- ⚠️ **A changing mesh invalidates the obvious convergence test.**
  `max |element_slip − previous|` compares vectors of different length once the
  mesh moves, so it is measured on a fixed `(s, z)` probe lattice instead. With the
  mesh held fixed the element-space number is kept verbatim, so an existing run
  prints exactly what it printed before.
- ⚠️ **The mesh stops adapting before the last round, on purpose.** Resampling a
  piecewise-constant slip field onto a fixed lattice differs by about the
  element-boundary contrast however little the model moved — two models fitting the
  *same* data on 240 and 232 elements agree to a 2 mm median and still differ by
  0.97 m at the worst point — so a centimetre tolerance is unreachable while the
  mesh moves. `mesh_adapt["rounds"]` defaults to `max_rounds - 1` so the run
  finishes on a fixed mesh and converges on the exact criterion — measured, a
  four-round run reports 3.50 / 2.14 / 2.22 m while the mesh is moving and
  **0.075 m** on the round that holds it fixed.
- `history` now carries `n_elements` and `n_nodes` per round, and the runner prints
  them: a sample count read beside a silently changed element count means nothing.
- From the runners: `NISAR_ADAPT_ROUNDS=2 NISAR_SAMPLING_FIELD=both python
  scripts/run_sampling.py`. Off by default (`rounds=0`).

### A layered crust, and nodal slip

**A layered medium.** A homogeneous half-space gives the whole crust one rigidity,
and the shallowest few kilometres are much softer than that. Assuming otherwise
means the same surface displacement needs *less* shallow slip and *more* deep slip
— a systematic bias in exactly the quantity a coseismic inversion is for.
`LayeredPointSource(tables)` cuts each element into point sources and looks each
one up in Green's-function tables from Rongjiang Wang's **EDGRN**; pass it as
`SlipInversion(..., engine=engine)`.

You own the Earth model, as in the reference implementation: run EDGRN yourself
and point `EdgrnTables.from_input_file` at its input file. `run_edgrn(crust, dir)`
will drive it for you given the `layered` extra or an `edgrn` on `PATH`. With no
Fortran to hand, `EdgrnTables.homogeneous()` synthesises tables for a *uniform*
medium — the case whose answer must reproduce the half-space engine, which is the
check worth running.

The quadrature order is chosen per element **and per observation** rather than
fixed at the reference's 91 points per triangle: a point source is a good stand-in
for a patch once you are a few patch-widths away, and almost every observation is
far from almost every element. `scripts/bench_greens.py` reproduces the result
from a clean checkout — synthetic fault, synthetic tables, no Fortran — at the
real problem's geometry (median r/L 37.9 against 38):

| assembly | seconds | vs half-space | points / pair |
|---|---|---|---|
| `HalfSpaceTDE` | 14.58 | 1.00× | — |
| layered adaptive `3e-3` | **12.58** | **0.86×** | 1.69 |
| layered reference (91 pt) | 231.0 | 15.9× | 81.0 |

**A layered engine now costs less than the closed-form homogeneous one**, and 48×
fewer quadrature points than the rule it replaces — so the shallow/deep slip bias
above has no remaining computational excuse. Verified on real Crust2.0 tables
against an order-30 rule: median relative error 7.2e-4, and the row sum (what
would rescale slip magnitude) 2.5e-3 against a 3e-3 request.

⚠️ **r/L is the controlling variable, and a small benchmark understates the win.**
A small mesh has large elements, so r/L is small and the rule must ask for high
orders — at 204 elements the reference is only 1.56× the half-space. The script
prints the r/L distribution beside the timings for that reason.

⚠️ **The cap is a silent tolerance violation, and it is now reported.** The order
is capped at `max_points`, which binds within roughly twice an element's own size
— i.e. on near-fault observations of shallow elements, precisely the pairs
carrying the shallow-slip constraint. The guarantee the adaptive rule offers is
never coarser than the fixed rule it replaces, **not** the tolerance. Every
assembly fills `LayeredPointSource.quadrature_report` (pairs, capped,
`capped_fraction`, points per pair, order histogram) and warns above 5%; the real
six-scene run caps 0.00%.

⚠️ **The reference rule is `tolerance=None` *with* `max_points=91`.**
`tolerance=None` alone gives the *cap* — 900 points, order 30 — which is nine
times more work than the rule it is meant to reproduce. From a run, that is
`NISAR_EDGRN_TOLERANCE=none NISAR_EDGRN_MAX_POINTS=91`.

⚠️ **A loaded layered model refuses to `forward()`.** EDGRN tables are megabytes
and are not saved with the model, so it installs an engine that raises and names
what to rebuild rather than silently forward-modelling with half-space physics.

**Nodal slip.** `basis="node"` solves for slip at the mesh *nodes*, with a
continuous piecewise-linear field between them, instead of a constant per
triangle. A real slip distribution is continuous, and a piecewise-constant one
spends resolution representing edges that are not there. There are fewer nodes
than triangles, so it is also a *smaller* problem, and it needs a smoothing
operator defined on the surface (Laplace–Beltrami) rather than on element
adjacency — `solve()` picks the right one automatically. On the test fixture it
recovers a planted patch to **correlation 0.9937 with 328 parameters**, against
0.9860 with 480 for element-constant slip. Slip is still *reported* per element
(`element_slip`, `to_dataset`, `to_text`, `plot_slip`), so nothing downstream
changes with the parameterization.

⚠️ **Depth-dependent rigidity belongs on the `SlipInversion`, not just on
`moment()`.** `moment_magnitude` reads the velocity model off the inversion, so
passing it only to `moment()` reports a layered M₀ beside a 30 GPa Mw in the same
summary — caught in practice as Mw 7.44 against 7.49.

### Saving a result, and the surface field

`model.save(path)` writes **one self-contained file** — the slip vector, the fit,
the mesh and the observations — that can be copied off the machine that produced
it. `SlipModel.load` gives back a model that reports every statistic, re-exports,
plots and forward-models new points. `to_text` writes SlipSolve's ten-column
element table, so existing GMT scripts work unchanged.

`model.to_vertex_text(dir)` writes the two files that let the fault be *rebuilt*,
which the element table cannot — it describes elements only by their centroids:

| file | rows | columns |
|---|---|---|
| `vert_nodes.txt` | one per node | `node_id longitude_deg latitude_deg depth_m along_strike_m strike_slip_m dip_slip_m area_m2 shear_modulus_pa` |
| `vert_elements.txt` | one per element | `element_id node_1 node_2 node_3` |

Same conventions as `slip_model.txt` throughout: tab-delimited, one header line,
depth in metres negative-down, slip in metres with positive strike-slip
left-lateral, **1-based** ids. `element_id` is the same numbering, so row *i* of
`vert_elements.txt` is row *i* of `slip_model.txt` — the two join without a key.

- **Slip is reported at the nodes**, which under `basis="node"` is what was
  actually solved for, rather than the per-element mean `to_text` prints. For an
  element-basis model the values are scattered onto nodes area-weighted instead.
- **`area_m2` is the node's lumped area** — a third of its 1-ring — so the rows
  partition the fault and `sum(shear_modulus × area × |slip|)` reproduces
  `model.moment()` exactly for a nodal model.
- ⚠️ **That sum does not match the same sum over `slip_model.txt`**, and should not
  be expected to: the area-weighted scatter conserves each slip *component*
  exactly (to 3e-16), but a magnitude is not linear — an element's value is the
  mean of three vectors, which is shorter than the mean of their lengths whenever
  they disagree. Measured, the two magnitude sums differ by about 0.9%. Neither is
  a check on the other.

The Green's matrix is deliberately *not* saved: it is the largest object in the
problem and nothing downstream of a solved model needs it. A loaded model
therefore **cannot be re-solved at a new weight** — rebuild the `SlipInversion`
for that.

`model.surface_displacement(spacing=...)` evaluates the full three-component
ground motion on a regular grid (`ux` east, `uy` north, `uz` up, metres), and
`model.to_grd(dir)` writes one GMT grid per component. A radar measures one number
per pixel; what the inversion recovers is the whole vector, which is what lets you
check a model against a second track it was not fitted to, compare with GPS, or
look at the vertical field — small for strike-slip, and the component InSAR sees
worst even though `los_up = cos(inc)` is the largest look-vector component.

- **Grid points on the trace come back NaN.** The solution is genuinely singular
  where the fault meets the free surface, and a very large number there would set
  the colour scale of every plot.
- **It evaluates in blocks**, because the engines build a
  `(points, 3, 2·n_elements)` array — 60,000 points against 1148 elements is
  3.3 GB. Raise `spacing` for a quick look.

## Running an inversion in the background

A fine mesh against a dense quadtree runs for a long time, and a notebook kernel is
a poor place to leave it — closing the browser or losing the SSH session takes the
run with it. `scripts/` holds three ready-to-run stages that detach cleanly, all
configured from one file:

| script | does | cost |
|---|---|---|
| `run_sampling.py` | put every scene on one lattice, then iterate sample → solve → re-sample until the model settles | minutes; the expensive one |
| `run_lcurve.py` | sweep the smoothing weight over those observations | one Green's matrix, many solves |
| `run_inversion.py` | solve once at the chosen weight; write the model, its text table, `summary.json` and the review figures | seconds |
| `archive_run.py` | move the last solve's outputs into a tagged subdirectory, so the next one doesn't overwrite it | instant |

Split three ways because only stage 1 is expensive to redo. They share
`scripts/slip_config.py`, so a mesh or a sampling parameter cannot drift between
them — which would be silent, since each stage's output looks fine on its own. The
Green's matrix is *not* passed between them (it is never saved), so each
re-assembles it: ~25 s at 1106 elements against 8000 observations, far cheaper than
the alternatives.

### Configuring the run

Edit `scripts/slip_config.py` — scenes, mesh, bounds, velocity model. The settings
worth changing between runs also read an environment variable, so a detached job
can be re-launched at a new value without touching the file:

| variable | sets |
|---|---|
| `NISAR_WORK_DIR` | where the workspace and every output live |
| `NISAR_FAULT` | the fault trace (`.kml`, or two-column lon/lat ASCII) |
| `NISAR_EDGE_LENGTH`, `NISAR_MAX_DEPTH` | element size and fault bottom, metres |
| `NISAR_DIP` | `75` for one dip everywhere, `70,80,85` for one per deep segment; unset means vertical |
| `NISAR_SEGMENTS` | segment files, one per dip; unset chops the trace into equal chords |
| `NISAR_BOTTOM` | a digitised bottom trace, which defines the dip instead of `NISAR_DIP` — setting both is refused |
| `NISAR_BOTTOM_DEPTH` | what depth that bottom trace sits at; unset means the base of the mesh |
| `NISAR_BIAS_W`, `NISAR_DOWN_DIP_LEVELS` | depth-level grading, and the level count that makes it mean a definite ratio (see "A fault that dips") |
| `NISAR_ENGINE` | `layered` (default, EDGRN tables from the velocity model) or `halfspace` |
| `NISAR_SMOOTHING` | the weight stage 3 solves at |
| `NISAR_MAX_ROUNDS` | sampling rounds; `0` stops at the coarse data-driven set |

A coarse pass (`NISAR_EDGE_LENGTH=6000`) is how you find out whether the whole
chain works before committing an hour to the fine one. The geometry that actually
ran is printed at the top of every log and recorded in `summary.json`, so a dipping
run is never mistaken for a vertical one later.

Two more settings live only in the file, because they are per-scene:
`SCENE_REPORT` (extra arguments for `scene_report` — `min_distance` in particular,
whose `4 × max_depth` default can exceed the scene on a deep fault) and `SAMPLING`
(pinned per-scene sampling parameters, empty by default so they are measured).

**`NISAR_ENGINE=halfspace` is a smoke test for stages 2 and 3 but a legitimate
choice for stage 1.** Stage 1's model only picks quadtree cells, and the
observations it writes carry no trace of which engine picked them — so sampling
with the half-space and inverting layered is Wang & Fialko's "preliminary model"
argument, and it skips a layered assembly per round.

### Launching

```bash
cd /path/to/nisar_tools
export KMP_DUPLICATE_LIB_OK=TRUE
export NISAR_WORK_DIR=/path/to/workdir
export NISAR_FAULT=/path/to/fault_trace.kml
PY=/path/to/envs/remote_sensing/bin/python   # `which python` with the env active

mkdir -p "$NISAR_WORK_DIR"                   # the log redirect needs it to exist

# 1. sampling — the long one; must finish before the others
nohup $PY -u scripts/run_sampling.py > "$NISAR_WORK_DIR/sampling.log" 2>&1 &
echo $! > "$NISAR_WORK_DIR/sampling.pid"
tail -f "$NISAR_WORK_DIR/sampling.log"       # Ctrl-C stops watching, not the job

# 2. L-curve
nohup $PY -u scripts/run_lcurve.py > "$NISAR_WORK_DIR/lcurve.log" 2>&1 &

# 3. inversion, at the weight you read off the curve
NISAR_SMOOTHING=0.5 nohup $PY -u scripts/run_inversion.py \
     > "$NISAR_WORK_DIR/inversion.log" 2>&1 &
```

Three details that matter:

- **`-u`** is the flag most people miss. Without it Python buffers stdout and the
  log stays **empty until the process exits** — a long run looks hung. For the same
  reason, redirect to a file rather than piping through `tail`, which buffers the
  whole log until the pipe closes.
- **`KMP_DUPLICATE_LIB_OK=TRUE`** is required in this environment (duplicate OpenMP
  runtimes), and `nohup` does not read your shell profile, so the launching shell
  must export it.
- **Use the env's absolute interpreter path.** Same reason: a bare `python` under
  `nohup` is the *system* one, which has neither `nisar_tools` nor its
  dependencies. It also means you do not need `conda activate` inside the job,
  which a detached process does not inherit reliably.

Check on it and collect the result:

```bash
ps -p $(cat "$NISAR_WORK_DIR/sampling.pid")   # still alive?
kill $(cat "$NISAR_WORK_DIR/sampling.pid")    # give up on it
```

```python
from nisar_tools.slip import SlipModel
model = SlipModel.load("workdir/model_sampling/slip_model.slip.zip")   # in a notebook
```

On a shared cluster, prefer the scheduler (`sbatch`, `qsub`) over `nohup` — same
scripts, and they survive a login-node restart.

### The L-curve-first route

`NISAR_MAX_ROUNDS=0` stops stage 1 after its coarse, data-driven round 0, which is
how you get an L-curve *before* letting a model steer the sampling. Stage 2 reads
which sampling it was given off the observations themselves and labels its outputs
`_bootstrap`, so the two curves can't overwrite each other or be confused later.

⚠️ **Read that curve with care.** Round 0 is deliberately under-sampled — on the
test mesh it produced 154 samples against 240 slip parameters — and where there are
fewer observations than parameters the smoothing is supplying missing rank rather
than trading misfit against roughness. Its corner sits at *more* smoothing than the
refined sampling wants. Treat it as a starting point for stage 1, not as the answer.

Stage 2 also flags what misfit and roughness cannot show: weights whose model came
back flat, weights where the slip bound saturated, and weights that hit the
iteration cap.

### Comparing several solves

Stage 3 writes to fixed names, so re-solving at a new weight or a different
`ramp` replaces the previous result. Run `archive_run.py` in between and each
solve keeps its own directory:

```bash
python scripts/archive_run.py                  # -> run_lam1000_offset_1696el/
NISAR_SMOOTHING=500 python -u scripts/run_inversion.py
```

The tag is read off `summary.json` rather than passed in, so it describes the run
that happened rather than what you meant to type. Only stage 3's outputs move —
the sampling and the sweep (`history.csv`, `l_curve*`, the mesh and coverage
figures) describe work that is still current, and stay put.

## Workspaces

A `Workspace` is a directory of per-stage Zarr stores. Each `persist()` writes one
store and records the parameters that produced it, hashed, so a re-run can tell a
finished stage from one that needs recomputing:

```
workdir/
├── workspace.json         # created timestamp
├── slc_stack.zarr         # one store per persisted stage
├── igrams.zarr
├── unwrapped.zarr
└── unwrapped.done.json    # per-pair progress, unwrap only
```

```python
ws = Workspace("workdir/")                 # creates the directory if needed
ws = Workspace("workdir/", create=False)   # open only; never writes on construction
```

### Which steps write, and which don't

Most steps are lazy and return a new stack; nothing reaches disk until you call
`persist`. Unwrapping is the exception — it takes the workspace as its first
argument and writes as it goes, because SNAPHU needs whole rasters, so it works one
pair at a time, writing each into its own region and flagging it done for resume.
By the time it returns, the store already exists.

The rule generalises: **a method taking the workspace *positionally* writes;
passing it by keyword means it does not.** `mask_water(mask_cache=ws)` is lazy —
the cache holds only the coastline mask, keyed on the grid, not the masked data.

### Reloading finished stages

Reopen a stage directly, without touching the granules or recomputing anything.
This is the normal way to pick up a previous session — `from_zarr` gives back a
stage object, `ws.load` gives the raw `xarray.Dataset` underneath:

```python
stack  = GSLCStack.from_zarr(ws.path("slc_stack"))
igrams = InterferogramStack.from_zarr(ws.path("igrams"))
ds     = ws.load("igrams")        # the xr.Dataset, to work with directly
```

Stores are lazy, so reopening a 200 GB stage costs nothing until you compute.

### Clearing and rebuilding

```python
ws.clear("igrams")            # delete one stage (and its .done.json), keep the rest
```

To rebuild a stage in place, pass `overwrite=True` to `persist` — without it,
persisting *different parameters* over an existing stage raises rather than
silently replacing your results. That is the mechanism behind several of the
warnings above: when a default changes in a way that changes the data, existing
stores stop matching and say so loudly. To throw everything away, delete the
directory and construct a new `Workspace`.

⚠️ **Persisting a stage over the store it reads from raises**, rather than racing
the delete against the lazy read. It used to corrupt 0.06–0.28% of pixels,
differently on every run.

## Read throughput

NISAR GSLCs are gzip-compressed, and h5py serialises every call on a single global
lock — across threads, across file handles, and across different files. Decoding a
granule through h5py is therefore effectively single-core, and adding dates or
frames buys no read parallelism at all (three granules read concurrently measured
1.01× versus one).

`GSLC` works around this: it reads each HDF5 chunk's raw compressed bytes, which
needs h5py only for the I/O, and runs the gzip inflate outside the lock where it
releases the GIL. Dask's own worker threads then overlap. Measured ~3× on a real
granule, 2.66× for a full crop-and-persist, with byte-identical output. Granules
whose filter pipeline can't be inverted fall back to plain h5py automatically.

To measure it on your own machine — the win depends on core count and on whether
the granules sit on local disk or a network filesystem:

```bash
python scripts/bench_read.py /path/to/granule.h5
```

Disk is usually not the limit: on a local SSD the raw compressed bytes come off
about 4× faster than they can be decoded.

## Tests

```bash
pytest                     # synthetic GSLC fixtures; no real data needed
NISAR_TEST_GSLC=/path/to/granule.h5 pytest tests/test_real_data.py   # real file
```
