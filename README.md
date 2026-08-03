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

`GSLC` → `GSLCStack` → `InterferogramStack` → `UnwrappedStack` → `LOSStack`.
Every stage is lazy and returns a new object; nothing reaches disk until
`persist`. The sections below are the choices along that path that are worth
making deliberately.

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
far from almost every element. Measured on the real problem (1148 elements, 8000
observations) that is **89× fewer** source-receiver evaluations at 1e-2 accuracy
and 10× at 1e-3. Pass `tolerance=None` for the fixed rule.

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
