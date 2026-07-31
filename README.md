# nisar_tools

Object-oriented, out-of-core tools for cropping, merging NISAR GSLCs and making
interferograms, unwrapping, masking, and plotting.

Every processing stage is lazy (`xarray` + `dask`) and persisted to a Zarr
`Workspace`, so a full stack of acquisitions never has to fit in memory and an
interrupted run resumes where it left off.

## Install

The geospatial stack (h5py, rioxarray, pyproj, snaphu, zarr<3) is easiest from
conda-forge; `pygmt` is optional and only needed for water masking.

```bash
conda install -c conda-forge h5py "zarr<3" dask rioxarray pyproj snaphu pygmt
pip install -e .            # add [dev] for tests, [mask] for pygmt, [download] for earthaccess+sardem
```

The `download` extra (earthaccess + sardem) is only needed for fetching inputs
and is imported lazily, so the package works without it.

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

## Downloading inputs

The `download` module fetches GSLC granules — all behind lazily-imported optional 
dependencies and all needing an [Earthdata Login](https://urs.earthdata.nasa.gov).
Bounding boxes use this package's `(lon_min, lon_max, lat_min, lat_max)` order
(the same `GSLC.crop` takes).

```python
from nisar_tools import download

download.login()                        # earthaccess: netrc / env vars / interactive
bbox = (-120.5, -119.5, 34.0, 35.0)

# GSLCs — search Earthdata by area + time (earthaccess), or by exact name.
gslcs = download.download_gslcs("data/gslc", bbox=bbox, temporal=("2025-11", "2025-12"))
gslcs = download.download_gslcs("data/gslc", granules=["NISAR_L2_GSLC_..."])

Two backends for granules: **earthaccess** (`method="earthaccess"`, the default —
searches CMR by name/area/time) and the original **direct-by-name** path
(`method="asf"`), which pulls straight from the NISAR bucket using only the
standard library — handy where earthaccess can't be installed (its latest
release needs Python ≥ 3.12):

```python
download.download_gslcs("data/gslc", granules=["NISAR_L2_GSLC_..."], method="asf")
```

## Pipeline

A runnable end-to-end example is in [notebooks/nisar_tools.ipynb](notebooks/nisar_tools.ipynb).

### Dropping territory the mainland can't reach

A coastline mask leaves things stranded offshore — islets GSHHG calls land,
ships, platforms, decorrelated fragments. An unwrapper propagates phase along
arcs between adjacent pixels, so a region with **no path to the main body carries
no recoverable ambiguity, however large it is**, and left in place it produces
artifacts that appear to bridge open water. The criterion is *connectivity, not
size*.

```python
igrams = (igrams
    .mask_water(mask_cache=ws, resolution="i")
    .remove_unconnected_regions()          # keeps only the largest component
    .filter_goldstein(alpha="adaptive")
)
```

On a real coastal frame that took 29 components down to 1, removing 6,369 px —
0.018% of the valid data. That frame is also why a size threshold is the wrong
instrument: its strays run 16 to 1530 px against a 36-million-px mainland, so a
`min_size` of 32 would have removed four of the twenty-eight and looked like it
had worked.

- `min_size=N` keeps every component above `N` pixels instead — for a scene that
  genuinely *is* two landmasses, so both real bodies survive and the speckle
  still goes.
- `max_drop_fraction` (default 0.01) refuses if the largest **single** dropped
  component exceeds that share of the valid pixels. Deliberately not the total:
  a scene shredded into thousands of specks has a large total but no large
  component, and those specks are exactly what the method is for.
- `connectivity=1` (4-connected) by default — a diagonal touch is not an arc for
  the solver, so a diagonally-attached region is as unreachable as a detached one.

Also on `UnwrappedStack`, where it is worth chaining ahead of `remove_outliers`
and `deramp(method="spline")`: both fit through a normalized convolution that
fills NaN gaps from their neighbours, so a stranded region drags the fitted
surface out across the water separating it from the mainland.

### Exporting to GMT `.grd`

`geo.project_to_latlon` reprojects any native-grid field to lon/lat, and the
result writes straight to NetCDF, which is what a `.grd` is — GMT's `grdinfo`
and `grdimage` read the output directly. See the notebook's "Export to GMT
`.grd`" section for a `to_grd` helper and a re-import.

## Merging frames

A track is delivered as a series of frames, and one earthquake rarely sits
inside a single one. Two frames can be joined at either of two stages.

**Before interferograms** — `GSLCStack.merge` gives one continuous complex
stack, which is the simpler route when both frames have the same acquisitions:

```python
stack = (GSLCStack.from_gslcs(frame1, bbox=bbox)
         .merge(GSLCStack.from_gslcs(frame2, bbox=bbox)))
```

**After unwrapping** — `UnwrappedStack.merge` additionally removes the 2π seam
between frames that were unwrapped independently, and is the only route when the
frames come from different tracks or different dates:

```python
unw = unw_a.merge(unw_b)          # self wins in the overlap; the 2π step is removed
los = unw.to_los([granule_a, granule_b])   # one granule per frame
```

Both grids must lie on the same lattice. Interferograms formed with
`align_looks=True` (the default) do so automatically: multilook blocks are
anchored to the absolute native grid rather than to whatever the crop happened
to start at. Without it, two frames cropped to different extents come out a
fraction of a multilooked pixel apart — at `looks=10`, three native pixels of
difference is exactly 0.3 px, and no amount of padding will line them up.
`UnwrappedStack.merge` will resample an off-lattice frame onto the first's grid
and warn; re-forming both interferograms avoids the resampling and is exact.

Merging across a **UTM zone boundary** works — a track crossing 114°W is gridded
in two zones, and the second frame is warped onto the first's grid first.

### Different dates

Two frames from different acquisition dates are *different interferograms*, not
one interferogram in two pieces. Pass `time_tolerance=None` to pair them by
position, and `tie` decides how the overlap is reconciled:

```python
unw = unw_a.merge(unw_b, time_tolerance=None)   # tie="auto" → "offset"
```

`tie="cycles"` (the default for same-date frames) removes a whole number of 2π,
which is the *only* thing two frames of one interferogram can differ by.
Different date pairs measure different deformation, so their overlap difference
is not an integer number of cycles and `tie="offset"` removes the real-valued
median instead — rounding it would leave a step of up to π.

> **For a slip inversion, do not merge different-date scenes.** A mosaic is one
> raster with one arbitrary constant and one ramp, so a second scene's
> independent nuisance gets absorbed as fictitious slip. Measured on a synthetic
> recovery: two scenes kept apart and combined with `Observations.concat` gave a
> slip correlation of **0.979** and a moment ratio of **1.001**, while the same
> data mosaicked into one raster gave **0.553** and **1.671** — with a peak slip
> 3.5× the truth. Variance reduction stayed at 98.4% in both, so it does not
> warn you. Merge for a continuous map or an export; keep the scenes separate
> for the inversion.

## Slip inversion

`nisar_tools.slip` inverts line-of-sight displacement for coseismic slip on a
fault, discretised into triangular dislocation elements and solved as a bounded,
smoothed linear least-squares problem. It consumes a `LOSStack`, so it picks up
exactly where the InSAR pipeline leaves off.

Three choices are independent, and the simplest of each is the default:

| | default | also available |
|---|---|---|
| geometry | vertical (`FaultMesh.vertical`) | curved, one dip per deep segment (`FaultMesh.curved`) |
| medium | homogeneous half-space (`HalfSpaceTDE`) | layered, from EDGRN tables (`LayeredPointSource`) |
| slip basis | constant per element | continuous nodal tent functions (`basis="node"`) |

```python
from nisar_tools.slip import FaultTrace, FaultMesh, Observations, SlipInversion

trace = FaultTrace.from_file("fault.kml")     # .kml or two-column ASCII
frame = trace.local_frame()                   # ONE frame shared by everything
mesh  = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

obs = Observations.concat([
    Observations.from_los(los_desc, name="D126", frame=frame, trace=trace,
                          exclude_within=5000.0),
    Observations.from_los(los_asc,  name="A014", frame=frame, trace=trace,
                          exclude_within=5000.0),
])

model = SlipInversion(mesh, obs, ramp="linear").solve(
    smoothing=0.3, polarity=(-1, 0, 0))       # right-lateral
print(model)          # <SlipModel VR=... max_slip=...m Mw=... roughness=...>
model.save("venezuela.slip.zip")
```

Four things that are easy to get wrong:

- **One `LocalFrame` for everything.** A transverse Mercator centred on the study
  area, not UTM, so two tracks in different zones share one x/y. Every object
  stores its frame and checks it; mixing frames is a silent kilometre-scale
  error.
- **Positive `strike_slip` is left-lateral.** A right-lateral fault — San
  Sebastián, Sagaing — needs `polarity=(-1, 0, 0)`.
- **Exclude observations near the trace.** Dislocation solutions are singular on
  the fault surface, so `exclude_within` is required; without a buffer the
  Green's function is non-finite and the inversion refuses to run.
- **One track per scene, with `ramp=`.** An unwrapped interferogram has an
  arbitrary constant and usually an orbital or ionospheric ramp.
  `ramp="linear"` gives each named track its own offset plus `x`/`y` gradients;
  without it those end up in the slip.

### A fault that dips

Give the fault a set of straight *deep segments* in plan view and one dip each.
Each segment is pushed down-dip, a smooth surface is fitted through those bottom
lines and the surface trace, and the mesh is that surface on the `(along-strike,
depth)` lattice:

```python
from nisar_tools.slip import FaultSegment

segments = FaultSegment.from_files(["Segment_001.txt", "Segment_002.txt"])
mesh = FaultMesh.curved(trace, frame, segments=segments, dips=[75.0, 80.0],
                        max_depth=25e3, edge_length=3e3, bias_w=1.15)
```

A segment file holds four numbers — `x_begin y_begin x_end y_end`, in metres in
the local frame. `FaultSegment.from_trace(trace, frame, 3)` chops the trace into
equal chords instead, if you only want "the western third dips 70, the rest 85".
For a single dip everywhere, `FaultMesh.curved(trace, frame, uniform_dip=75.0)`.

- **Dips above 90 are meaningful**, not an error: the fault leans the other way.
- **`bias_w` thickens the depth levels downward**, putting resolution where the
  data can constrain it. `bias_w=1` gives even levels.
- **A dip of exactly 90 reproduces `FaultMesh.vertical` bit for bit**, so nothing
  already computed changes by switching constructor.
- The fit is refused if the down-dip offset exceeds the trace's radius of
  curvature, because the surface would fold back through itself.

### A layered crust

A homogeneous half-space gives the whole crust one rigidity. The shallow crust is
much softer than that, and assuming otherwise biases shallow slip low and deep
slip high. `LayeredPointSource` cuts each element into point sources and looks
each one up in Green's-function tables from Rongjiang Wang's **EDGRN**:

```python
from nisar_tools.slip import EdgrnTables, LayeredPointSource, VelocityModel

tables = EdgrnTables.from_input_file("edgrn.inp")     # tables you generated
engine = LayeredPointSource(tables)
model  = SlipInversion(mesh, obs, engine=engine, basis="node").solve(smoothing=0.3)

crust = VelocityModel.from_file("crust.txt")          # depth vp vs rho
print(model.moment(crust) / 1e18, "x 10^18 N m")      # depth-dependent rigidity
```

You own the Earth model, as in the reference implementation — run EDGRN yourself
and point at its input file. `run_edgrn(crust, workdir)` will drive it for you if
`pip install nisar_tools[layered]` (which brings `pygrnwang` and its bundled
Fortran) or an `edgrn` on `PATH` is available.

The quadrature order is chosen per element and observation rather than fixed at
the reference's 91 points per triangle: a point source is a good stand-in for a
patch once you are a few patch-widths away, and almost every observation is far
from almost every element. Measured on the real problem — 1148 elements, 8000
observations — that is **89× fewer** source-receiver evaluations at 1e-2 accuracy
and 10× at 1e-3. Pass `tolerance=None` for the fixed rule.

### Nodal slip

`basis="node"` solves for slip at the mesh nodes, with a continuous
piecewise-linear field between them, instead of a constant per triangle. There
are fewer nodes than triangles, so it is also a smaller problem, and it needs a
smoothing operator defined on the surface (a Laplace–Beltrami operator) rather
than on element adjacency — `solve()` picks the right one automatically. On the
test fixture it recovers a planted patch to **correlation 0.994 with 328
parameters**, against 0.986 with 480 for element-constant slip.

Slip is still *reported* per element (`model.element_slip`, `to_dataset`,
`to_text`, `plot_slip`), so the output format does not change with the
parameterization.

### Measuring the sampling parameters instead of guessing them

`scene_report` derives `rms_min`, `width_min` and `exclude_within` from a scene,
and reports the coverage that limits the answer regardless of them:

```python
from nisar_tools.slip import scene_report, ramp_content
from nisar_tools.slip.plot import plot_coverage

r = scene_report(los_desc, trace, frame, mesh=mesh)
print(r.attrs["noise_floor"], r.attrs["rms_min"], r.attrs["width_min"])
plot_coverage(r, name="D126")
```

Why each one is a measurement and not a preference:

- **`rms_min` is a noise level.** The quadtree splits while the *pixel* scatter
  inside a cell exceeds it, and that scatter does not shrink as the cell does —
  so set below the noise, cells can never stop and simply run down to
  `width_min`. On real scenes this showed up as a median within-cell scatter of
  **11.8 mm against `rms_min=6 mm`**, with 47% of cells stuck at the floor, and
  it accounted for 18 631 samples of mostly atmosphere. `noise_floor` measures it
  on a *fixed* block grid, so the estimate does not depend on the quadtree it is
  about to configure.
- **`width_min` is discontinuous.** Cells are index rectangles halved at their
  midpoint, so the reachable sizes are a dyadic ladder — per axis, per scene, and
  not guessable. `cell_size_ladder` simulates the descent: 1000 m gave a 1075 m
  terminal cell and 57 246 samples, 1500 m gave 1750 m and 24 429, and two values
  in the same gap do nothing at all.
- **`exclude_within` needs only to stop a *cell* straddling the trace**, so its
  floor is `width_min / 2`. Larger values are a judgement about unwrapping errors
  and near-fault model error — worth making deliberately, since removing the near
  field also removes the only data that constrains shallow slip.

`ramp_content(obs)` answers the companion question, and needs no Green's matrix:
how much of the data a per-track offset explains, and how much more the `x`/`y`
gradients add. The columns are normalised by each track's span, so a gradient
reads directly as **metres of LOS across the scene** — centimetres is orbit, tens
of centimetres is deformation being taken out of the slip model. For a long,
near east-west strike-slip fault the far-field coseismic pattern is an arctangent
step across the trace, which over a finite aperture looks a great deal like a
gradient perpendicular to strike, so the two genuinely compete.

**None of it fixes coverage, which is usually what binds.** The along-strike
profile is the point of the report: on the Venezuela scenes an aggregate "19% of
samples north of the trace" read like thin two-sided coverage, while the profile
showed the north block was absent along the eastern *two-thirds* — exactly where
the largest signal was.

### Sub-sampling from the model instead of from the data

Everything above picks quadtree cells from the **observed** displacement, which is
noise-driven — the split test is the *pixel* scatter in a cell, and that does not
shrink as the cell shrinks. `iterate_sampling` implements the Wang & Fialko (2015)
alternative: quadtree a **synthetic** interferogram from a preliminary model, then
fill those cells from the real data, and repeat until the model stops changing.

```python
from nisar_tools.slip import ARCSEC_10, iterate_sampling, resample_all

# One lattice for every track: 10 arcsec, in the shared LocalFrame.
gridded = resample_all({"D134": los_alos, "D126": los_nisar}, frame, spacing=ARCSEC_10)

obs, model, history = iterate_sampling(
    gridded, mesh, trace, frame,
    {name: dict(rms_min=r.attrs["rms_min"], width_min=r.attrs["width_min"],
                width_max=30_000.0, exclude_within=r.attrs["exclude_within"])
     for name, r in reports.items()},
    inversion_kwargs={"ramp": "linear"}, solve_kwargs={"smoothing": 0.3},
)
```

**Resample first, and not only for tidiness.** A quadtree cell is an integer number
of pixels halved at its midpoint, so the reachable sizes form a dyadic ladder *set by
the pixel size* — per axis, per scene. Two scenes at different resolutions land on
different ladders, one `width_min` means two different things, and the per-track
sample counts diverge for a reason that has nothing to do with the data, which
`Observations.concat` then turns into an unintended reweighting. `resample_all`
defaults to 10 arc-seconds (309 m), the usual ALOS-2 posting and already ~20× finer
than a fault element.

**`refine_within` is not optional.** An initial model with little shallow slip
predicts a smooth near field, so the quadtree coarsens exactly where shallow slip
needs constraining and the next round is free to invent it — the paper's "spurious
shallow slip". `iterate_sampling` holds a dense band along the trace through every
round.

**`rms_min` changes meaning** under model-based sampling: a threshold on model
curvature, not a noise level, so the `scene_report` value does not apply.
`model_rms_min` derives it from the predicted field's own scatter.

⚠️ **Not unconditionally better.** Measured on the synthetic fixture at 12 mm of
noise, data-driven vs model-driven correlation with the truth: **0.944/0.934** for
white noise, **0.895/0.874** correlated over 15 km, **0.852/0.918** at 30 km. The
gain arrives only once the noise is long-wavelength enough that extra samples inside
one atmospheric cell carry no independent information. The sample count moves in
either direction depending on `rms_fraction`. What it reliably does is stop the
sampler chasing noise. On the real ALOS-2 D134 scene: 610 samples in round 0, then
1246 → 1311 → 1312, converging in three rounds (max parameter change 1.38 → 0.045 →
0.001 m) at VR 97.8%.

`plot_mesh(mesh, trace=trace)` draws the discretisation itself — map view beside the
unrolled section, coloured by element area — and takes a bare `FaultMesh`, so it is
available before any model exists. Read it against `plot_samples`: elements much
smaller than the data constraining them are what the smoothing weight then has to
paper over.

### Choosing the smoothing weight

`l_curve` solves at many weights over the same Green's matrix and tabulates
misfit against roughness; the corner of the curve is the conventional choice.

```python
inv = SlipInversion(mesh, obs, ramp="linear")
curve, models = inv.l_curve([2.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01],
                            polarity=(-1, 0, 0))
print(curve[["variance_reduction", "roughness", "iterations", "converged"]]
      .to_dataframe())
```

**The sweep is serial on purpose, and the inversion has no parallelism knobs.**
Every weight is an independent solve over the same `G`, which makes a worker pool
the obvious move — it was written, verified to give identical results, and then
removed. Measured on an 8-weight sweep: threads **1.02×** at 2 workers and *worse*
above (0.90× at 8); a process pool **1.10×**. Two reasons:

- **Load imbalance dominates.** The weights cost
  0.78 / 0.75 / 0.54 / 0.84 / 0.61 / 0.91 / **12.15** / 1.59 s — `λ=0.02` ran to
  the iteration cap and was **67% of the whole sweep**, so no scheduler beats
  1.50×. That weight's result is meaningless anyway (`converged=False`): the fix
  is to drop it, not to parallelise it.
- **Threads can't overlap the solver.** scipy's `lsmr` is pure Python, so its
  iteration loop holds the GIL between matrix-vector products. Capping
  `OMP_NUM_THREADS=1` to rule out BLAS oversubscription changed nothing, which
  points at the GIL rather than core contention. (The dense matrix-vector product
  alone already runs at ~34 GFLOP/s — BLAS is threaded.)

What *does* make a sweep fast is the inner tolerance (`lsmr_tol="auto"`, now the
default) and not paying for weights that never converge. Judge that tolerance
over a whole sweep rather than at one weight: it was **5.8×** on the reference
problem, but per-weight it ranges from **2.2× faster to 1.4× slower** — it loses
at both the smooth and rough ends and wins in the middle, netting ~1.17× across a
sweep. Green's assembly resists parallelism too, for a different reason again:
threading over elements measured uniformly worse (1 worker 24.6 s, 10 workers
62–79 s), because each element is ~50 small numpy calls and dispatch overhead
swamps the GIL-free stretch.

### Runtime, and checking convergence

Runtime tracks **conditioning far more than size**. On the reference problem
(1106 elements, 8000 observations, 10 cores) Green's assembly is ~25 s and the
solve a few seconds — but an underdetermined or badly scaled setup can spend
minutes in the solver. Two things follow:

- **Always check `model.converged`.** An iteration-capped solve returns a
  plausible-looking model whose variance reduction is meaningless; `to_text`
  refuses to write one.
- **More quadtree samples are not free.** 19 562 samples cost 59 s in the solve
  against 0.8 s for 860, and cannot constrain a mesh that only has a few hundred
  elements. Keep samples in the low thousands — that is what the quadtree is for.

### Saving a result

`model.save(path)` writes one self-contained file — the slip vector, the fit,
the mesh and the observations — that can be copied off the machine that ran it:

```python
model.save("venezuela.slip.zip")

from nisar_tools.slip import SlipModel
model = SlipModel.load("venezuela.slip.zip")
model.variance_reduction, model.max_slip, model.moment_magnitude
model.to_text("model.txt")        # the ten-column GMT-ready element table
```

A loaded model reports every statistic, re-exports, plots, and forward-models
new points. The Green's matrix is deliberately *not* saved — it is the largest
object in the problem and nothing downstream of a solved model needs it, so a
loaded model cannot be re-solved at a new weight. Rebuild the `SlipInversion`
for that.

### Running a long inversion in the background

A large mesh against a dense quadtree can run for a long time, so it is worth
detaching it from the terminal. Write a small script that ends in a `save`:

```python
# run_inversion.py
from nisar_tools import LOSStack, Workspace
from nisar_tools.slip import FaultTrace, FaultMesh, Observations, SlipInversion

ws    = Workspace("workdir/", create=False)
trace = FaultTrace.from_file("fault.kml")
frame = trace.local_frame()
mesh  = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

obs = Observations.concat([
    Observations.from_los(LOSStack.from_zarr(ws.path(name)), name=name,
                          frame=frame, trace=trace, exclude_within=5000.0)
    for name in ("los_D126", "los_A014")
])

model = SlipInversion(mesh, obs, ramp="linear").solve(
    smoothing=0.3, polarity=(-1, 0, 0))
print(model, flush=True)
if not model.converged:
    raise SystemExit("solver hit the iteration cap; raise max_iter or smoothing")
model.save("venezuela.slip.zip")
```

Then launch it detached:

```bash
cd /path/to/nisar_tools
KMP_DUPLICATE_LIB_OK=TRUE \
nohup /path/to/envs/remote_sensing/bin/python -u run_inversion.py \
      > inversion.log 2>&1 &
echo $! > inversion.pid          # so you can check on or kill it later
```

- **`-u`** is the important flag: without it Python buffers stdout and the log
  stays empty until the process exits, which makes a long run look hung.
- **`KMP_DUPLICATE_LIB_OK=TRUE`** is required in this environment (duplicate
  OpenMP runtimes) — the shell that launches the job must set it.
- Use the env's **absolute** interpreter path; `nohup` does not run your shell
  profile, so a bare `python` will be the system one.

Watch it, and pick the result up afterwards:

```bash
tail -f inversion.log            # progress
ps -p $(cat inversion.pid)       # still alive?
kill $(cat inversion.pid)        # give up on it
```

```python
from nisar_tools.slip import SlipModel
model = SlipModel.load("venezuela.slip.zip")     # back in the notebook
```

An L-curve sweep launches exactly the same way — the inversion has no
parallelism knobs (see above). If a sweep is slow, the weights that hit the
iteration cap are almost certainly why; print `curve["iterations"]` and
`curve["converged"]` and drop those weights.

## Workspaces

A `Workspace` is a directory of per-stage Zarr stores. Each `persist()` writes
one store and records the parameters that produced it, hashed, so a re-run can
tell a finished stage from one that needs recomputing:

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
argument and writes as it goes, because SNAPHU needs whole rasters, so it works
one pair at a time, writing each into its own region and flagging it done for
resume. By the time it returns, the store already exists.

### Reloading finished stages

Reopen a stage directly, without touching the granules or recomputing anything.
This is the normal way to pick up a previous session — `from_zarr` gives back a
stage object, `ws.load` gives the raw `xarray.Dataset` underneath:

```python
from nisar_tools import GSLCStack, InterferogramStack, UnwrappedStack, LOSStack

stack  = GSLCStack.from_zarr(ws.path("slc_stack"))
igrams = InterferogramStack.from_zarr(ws.path("igrams"))
unw    = UnwrappedStack.from_zarr(ws.path("unwrapped"))

ds = ws.load("igrams")            # the xr.Dataset, if you'd rather work with it directly
```

Stores are lazy, so reopening a 200 GB stage costs nothing until you compute.

### Clearing

```python
ws.clear("igrams")            # delete one stage (and its .done.json), keep the rest
```

To rebuild a stage in place, pass `overwrite=True` — without it, persisting
different parameters over an existing stage raises rather than silently
replacing your results:

```python
igrams = stack.form_interferograms(looks=30).persist(ws, "igrams", overwrite=True)
```

To throw away everything, delete the directory (`shutil.rmtree("workdir/")`) and
construct a new `Workspace`.

### Running the notebook

The notebook must run in the environment where `nisar_tools` and its
dependencies are installed. Register that env as a Jupyter kernel once, then
select **`Python (remote_sensing)`** in the kernel picker:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name remote_sensing --display-name "Python (remote_sensing)"
```

(Replace `remote_sensing` with your env name.) A `ModuleNotFoundError: No
module named 'nisar_tools'` almost always means the notebook is running on a
different kernel than the one the package is installed into.

This env ships duplicate OpenMP runtimes, and pygmt may otherwise bind to a
system GMT (e.g. a Homebrew install) whose coastline data the conda netCDF
stack can't read — causing water masking to fail with GSHHG errors. Both are
fixed by setting two variables in the kernel's environment. Edit
`~/Library/Jupyter/kernels/<name>/kernel.json` to add:

```json
"env": {
  "KMP_DUPLICATE_LIB_OK": "TRUE",
  "GMT_LIBRARY_PATH": "/path/to/your/env/lib"
}
```

`GMT_LIBRARY_PATH` should point at the `lib` directory of the same env, so
pygmt loads that env's `libgmt` (matched to its netCDF/HDF5).

### Read throughput

NISAR GSLCs are gzip-compressed, and h5py serialises every call on a single
global lock — across threads, across file handles, and across different files.
Decoding a granule through h5py is therefore effectively single-core, and adding
dates or frames buys no read parallelism at all (three granules read
concurrently measured 1.01× versus one).

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
