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
