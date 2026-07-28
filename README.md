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
fault: a vertical fault discretised into triangular dislocation elements in a
homogeneous elastic half-space, solved as a bounded, smoothed linear
least-squares problem. It consumes a `LOSStack`, so it picks up exactly where
the InSAR pipeline leaves off.

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

### Choosing the smoothing weight

`l_curve` solves at many weights over the same Green's matrix and tabulates
misfit against roughness; the corner of the curve is the conventional choice.
Each weight is independent, so they can run on a thread pool:

```python
inv = SlipInversion(mesh, obs, ramp="linear")
curve, models = inv.l_curve([2.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01],
                            workers=0,        # 0 = one thread per CPU
                            polarity=(-1, 0, 0))
```

**Expect very little from `workers`.** Measured on an 8-weight sweep: threads
gave **1.02×** at 2 workers and got *worse* above (0.90× at 8); a process pool
managed **1.10×**. Two measured reasons:

- **Load imbalance dominates.** The weights cost
  0.78 / 0.75 / 0.54 / 0.84 / 0.61 / 0.91 / **12.15** / 1.59 s — `λ=0.02` ran to
  the iteration cap and was **67% of the whole sweep**. No scheduler beats 1.50×
  against that, and that weight's result is meaningless anyway
  (`converged=False`).
- **Threads can't overlap the solver.** scipy's `lsmr` is pure Python, so its
  iteration loop holds the GIL between matrix-vector products. Capping
  `OMP_NUM_THREADS=1` to rule out BLAS oversubscription changed nothing, which
  is what points at the GIL rather than core contention.

The things that *do* make a sweep fast are the inner tolerance (`lsmr_tol="auto"`,
now the default — a measured **5.8×** on one solve) and not paying for weights
that never converge. Green's assembly is not parallelisable either: threading
over elements measured uniformly worse (1 worker 24.6 s, 10 workers 62–79 s),
because each element is ~50 small numpy calls and dispatch overhead swamps the
GIL-free stretch.

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

An L-curve sweep needs nothing extra — `workers` buys little (above), so the
usual launch is the same one. If a sweep is slow, the weights that hit the
iteration cap are almost certainly why; print `curve["iterations"]` and
`curve["converged"]` and drop those weights rather than trying to parallelise
around them.

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
