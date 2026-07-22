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

The `download` module fetches the two inputs a run needs — GSLC granules and a
DEM — all behind lazily-imported optional dependencies and all needing an
[Earthdata Login](https://urs.earthdata.nasa.gov). (Orbit ephemeris is *not*
needed: the geometry is read from each GSLC's built-in cube — see below.)
Bounding boxes use this package's `(lon_min, lon_max, lat_min, lat_max)` order
(the same `GSLC.crop` takes).

```python
from nisar_tools import download

download.login()                        # earthaccess: netrc / env vars / interactive
bbox = (-120.5, -119.5, 34.0, 35.0)

# GSLCs — search Earthdata by area + time (earthaccess), or by exact name.
gslcs = download.download_gslcs("data/gslc", bbox=bbox, temporal=("2025-11", "2025-12"))
gslcs = download.download_gslcs("data/gslc", granules=["NISAR_L2_GSLC_..."])

# DEM via sardem — default is the official NISAR reference DEM (needs ~/.netrc);
# pass data_source="COP" for Copernicus (no login).
dem   = download.download_dem("data/dem.tif", bbox)
```

Two backends for granules: **earthaccess** (`method="earthaccess"`, the default —
searches CMR by name/area/time) and the original **direct-by-name** path
(`method="asf"`), which pulls straight from the NISAR bucket using only the
standard library — handy where earthaccess can't be installed (its latest
release needs Python ≥ 3.12):

```python
download.download_gslcs("data/gslc", granules=["NISAR_L2_GSLC_..."], method="asf")
```

## Pipeline

```python
from nisar_tools import GSLC, GSLCStack, Workspace

ws = Workspace("workdir/")

# 1. Crop a region out of each granule and persist the aligned stack.
gslcs = [GSLC(p) for p in paths]
stack = GSLCStack.from_gslcs(gslcs, bbox=(lon0, lon1, lat0, lat1))
stack = stack.persist(ws, "slc_stack", files=paths)   # reopens from Zarr
for g in gslcs:
    g.close()                                          # safe after persist

# 2. Form multilooked interferograms + coherence.
igrams = stack.form_interferograms(pairs="sequential", looks=5).persist(ws, "igrams")

# 3. Goldstein-Werner phase filter (sharpens fringes before unwrapping).
#    alpha=float for constant strength, or "adaptive" for Baran (1-coherence).
igrams = igrams.filter_goldstein(alpha="adaptive").persist(ws, "igrams_filt")

# 4. Unwrap with SNAPHU, one pair at a time (resumable).
unw = igrams.unwrap(ws, nproc=8)

# 5. Convert to line-of-sight displacement + per-pixel look geometry.
los = unw.to_los(paths[0], dem="data/dem.tif").persist(ws, "los")
fig, ax = los.plot(pair=0)            # displacement (m); los.plot_incidence() for angles

# 6. Mask water and plot. Masking is lazy — persist it if you want it kept.
unw = unw.mask_water(mask_cache=ws)          # ws caches the coastline mask
unw = unw.persist(ws, "unwrapped_masked")    # optional; new stage name
fig, ax = unw.plot(pair=0)
```

A runnable end-to-end example is in [notebooks/nisar_tools.ipynb](notebooks/nisar_tools.ipynb).

### Line-of-sight displacement and look angles

`UnwrappedStack.to_los(gslc, dem=...)` scales the unwrapped phase to metres,
`d_los = +(λ/4π)·φ` (positive **toward the sensor**; λ from the GSLC's
`centerFrequency`), and attaches per-pixel look geometry. The geometry comes
from the GSLC's own ISCE3-computed cube at `metadata/radarGrid` (incidence angle
and the target→sensor ENU line-of-sight unit vector, tabulated vs. height),
trilinearly interpolated in `(x, y, DEM-height)`. So it needs a `gslc` (cube +
λ) and a `dem` GeoTIFF — not the orbit ephemeris, which the cube already
encodes. The resulting [`LOSStack`](nisar_tools/los.py) carries `los`
`(pair, y, x)` plus shared `incidence_angle` and `los_east`/`los_north`/`los_up`
`(y, x)` for projecting 3-D motion into LOS. Pass `dem=None` for sea-level
geometry, or `sign=-1` to flip the displacement convention.

### Stitching along-track frames

Adjacent frames from the same pass merge even when the track crosses a UTM
zone boundary: `merge` pairs acquisitions by time (frames along a pass differ
by seconds, never exactly) and, when the CRSs differ, warps the other stack
onto this stack's grid one date at a time:

```python
stitched = stack_a.merge(stack_b)                         # bilinear on I/Q
stitched = stack_a.merge(stack_b, resampling="nearest")   # exact samples
```

The union grid keeps NaN where neither frame has data (including thin wedges
from the zone rotation), exactly as with the NaN fill of a single granule.
Interferogram formation is NaN-aware, so that fill neither spreads nor eats
into the valid data — see [Invalid pixels](#invalid-pixels). Frames on
*different* tracks have no common pass and cannot be stitched at the SLC
level; process each track through unwrapping and mosaic the unwrapped
products instead.

### Invalid pixels

A NISAR GSLC is NaN outside its swath — often around 45% of a granule, since
the radar footprint is a sheared parallelogram on a north-up grid — and every
merged union grid is NaN in its corners.

`form_interferograms` multilooks over the valid samples only (a normalized
convolution: the validity mask is smoothed alongside the data and divided back
out). An output pixel is kept when at least `min_valid_fraction` of its filter
weight came from valid input, so the invalid footprint neither dilates nor
grows — at a straight edge the 0.5 default lands on the true boundary:

```python
igrams = stack.form_interferograms(looks=30, convolution="Gaussian")
igrams = stack.form_interferograms(looks=30, min_valid_fraction=0.9)  # stricter
igrams = stack.form_interferograms(looks=30, nan_aware=False)         # legacy
```

This matters because `scipy.ndimage`'s filters are not NaN-aware: under
`nan_aware=False` a single invalid sample contaminates its whole filter
footprint (a radius of `4 * looks` for Gaussian) and, because `uniform_filter`
is a running sum, everything downstream of it along both axes for Uniform. On
a real block straddling a swath edge at `looks=30`, 51.3% of the input valid:
the legacy path keeps 46.1% (Gaussian) or 32.0% (Uniform); the NaN-aware path
keeps 51.1% either way.

Unwrapping carries the footprint through — invalid pixels are excluded from
SNAPHU's solution rather than silently substituted with zeros, and come back
as NaN with a connected-component label of 0.

Every stage can also be cropped, so a ragged edge can be trimmed after the
fact and not only before:

```python
igrams = igrams.crop(lon_min, lon_max, lat_min, lat_max)
unw = unw.crop(lon_min, lon_max, lat_min, lat_max)
```

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

The rule of thumb: **a method that takes the workspace positionally writes; one
that takes it as a keyword does not.**

| step | writes? | where |
|---|---|---|
| `from_gslcs`, `crop`, `merge` | no | `.persist(ws, "slc_stack")` |
| `form_interferograms`, `filter_goldstein` | no | `.persist(ws, "igrams")` |
| `unwrap(ws, ...)` | **yes** | `unwrapped.zarr`, as it runs |
| `mask_water(mask_cache=ws)` | no | `.persist(ws, "<new name>")` |
| `to_los` | no | `.persist(ws, "los")` |

`mask_water`'s `mask_cache` argument is easy to misread: it caches the
*coastline mask itself* (keyed on the grid, so GMT is not re-run for the same
crop) and has nothing to do with storing your masked data. Masking is lazy, so

```python
unw = unw.mask_water(mask_cache=ws)
```

leaves the mask in that object only — reload the `unwrapped` stage later and the
phase comes back unmasked. Persist it under a **new** stage name to keep it:

```python
unw = unw.mask_water(mask_cache=ws).persist(ws, "unwrapped_masked")
```

A new name is required, not just tidier: persisting back over the store a stack
reads from is refused (see the warning below). Masking also carries into any
later stage you do persist, so `unw.mask_water(ws).to_los(gslc).persist(ws,
"los")` stores masked LOS displacement even without saving the masked phase.

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

You rarely need to check first, because re-running the same code is already the
resume path: `persist()` returns the existing store untouched when the
parameters match, and raises `WorkspaceError` when they don't. If you do want to
look:

```python
ws.exists("igrams")                # True / False
ws.stored_params_hash("igrams")    # hash recorded at write time, or None
ws.pairs_done("unwrapped")         # e.g. {0, 1, 2} — which pairs SNAPHU finished
```

Unwrapping resumes at the first unfinished pair, so an interrupted run picks up
where it stopped just by calling `igrams.unwrap(ws, ...)` again.

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

**Persisting a stage back over the store it reads from raises `WorkspaceError`.**
`overwrite=True` deletes the target directory *before* the write computes, and
the lazy graph is still reading from it, so the two race — historically this did
not fail loudly but produced a plausible-looking array with a varying fraction of
pixels silently corrupted to NaN. It is now refused up front, with the store left
untouched:

```python
stack = GSLCStack.from_zarr(ws.path("slc_stack"))
merged = stack.merge(other)

merged.persist(ws, "slc_merged")                  # new name — fine
merged.persist(ws, "slc_stack", overwrite=True)   # WorkspaceError: still an input
```

The check looks at whether the dataset's *lazy* data still reads the target, so
it also catches stages rebuilt through `merge` (which drops xarray's `encoding`)
and covers the region-written unwrap store. A dataset you have already
`.compute()`d holds nothing open and can be written back freely. Detection is
best effort: on the rare shape it can't inspect it stays out of the way rather
than blocking a valid write, so prefer a fresh stage name regardless.

### Changing a parameter invalidates a stage

The hash covers the arguments that affect the numbers — `looks`, `downsample`,
`convolution`, `nan_aware`, `min_valid_fraction`, the pair list, plus anything
extra you pass to `persist()`. Changing one means the stored result no longer
matches, so `persist()` refuses without `overwrite=True`. Nothing cascades
automatically: downstream stages are not invalidated for you, so after
re-forming interferograms, clear or overwrite the stages built on top of them.

Recording the inputs is worth doing, since they are not otherwise part of the
hash:

```python
stack.persist(ws, "slc_stack", files=paths, bbox=list(bbox))
```

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
pygmt loads that env's `libgmt` (matched to its netCDF/HDF5). If you can't use
the full-resolution coastline, pass a coarser one, e.g.
`unw.mask_water(mask_cache=ws, resolution="i")`.

## Design

| Class | Wraps | Dims |
|-------|-------|------|
| `GSLC` | one HDF5 granule (lazy) | `(y, x)` |
| `GSLCStack` | aligned acquisitions | `(time, y, x)` |
| `InterferogramStack` | igram + coherence | `(pair, y, x)` |
| `UnwrappedStack` | unwrapped phase + conncomp | `(pair, y, x)` |
| `Workspace` | per-stage Zarr stores | — |

- **Lazy everywhere** between disk reads and stage persistence. Cropping and
  merging are coordinate slices (a cross-zone merge warps one date per dask
  task onto the reference lattice); multilooking uses `dask.array.map_overlap`
  over the original scipy filters, so results are independent of chunk layout.
- **Zarr stages** record a parameter hash; `Workspace.has(name, params)` lets a
  re-run skip a finished stage. Unwrapping additionally tracks per-pair "done"
  flags for crash-safe resume. Chunks are `(1, 2048, 2048)` — never more than
  one acquisition/pair along the stack axis.
- **One non-lazy stage**: SNAPHU needs whole rasters, so the *pair* is the unit
  of work; peak memory is a single multilooked pair regardless of stack length.

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

## Numerics

The interferogram/coherence formula and the SNAPHU tile/looks sizing are ported
verbatim from the original procedural module, retained as
`tests/legacy_reference.py` — the oracle the dask kernels are
equivalence-tested against.

The Goldstein–Werner filter (`filter_goldstein`) is new, so it has no legacy
oracle. It runs on the multilooked interferogram, between forming and
unwrapping: each pair is tiled into overlapping `patch_size` windows whose
spectra are scaled by `(smooth(|Z|)/max)**alpha` and recombined by weighted
overlap-add. `alpha` is either a float in `[0, 1]` (constant strength; `0` is an
exact no-op) or `"adaptive"` for the Baran et al. (2003) coherence-adaptive
strength — per patch, `alpha = 1 − mean coherence`, so incoherent areas are
filtered hard and coherent ones barely touched, matching GMTSAR's `phasefilt`
run with `-amp1/-amp2`. Because a multilooked pair is small, the filter runs one
whole plane at a time — the same one-pair-in-memory footprint the unwrap stage
already assumes — and only touches `igram`, leaving `coherence` unchanged. It is
pinned by its properties (exact `alpha=0` round-trip, uniform-coherence adaptive
≡ constant `1−c`, phase-noise reduction, NaN preservation) rather than a legacy
result.

Our algorithm follows the Goldstein–Werner family; versus GMTSAR's `phasefilt`
it additionally smooths the spectrum (closer to Goldstein 1998), tapers each
patch (Welch-style), and normalizes so `alpha=0` is exact — GMTSAR's default
`alpha=0.5` and `psize=32` match our defaults.

## Tests

```bash
pytest                     # synthetic GSLC fixtures; no real data needed
NISAR_TEST_GSLC=/path/to/granule.h5 pytest tests/test_real_data.py   # real file
```
