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
pip install -e .            # add [dev] for tests, [mask] to pull pygmt via pip
```

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

# 3. Unwrap with SNAPHU, one pair at a time (resumable).
unw = igrams.unwrap(ws, nproc=8)

# 4. Mask water and plot.
unw = unw.mask_water(workspace=ws)
fig, ax = unw.plot(pair=0)
```

A runnable end-to-end example is in [notebooks/nisar_tools.ipynb](notebooks/nisar_tools.ipynb).

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
from the zone rotation) — crop to the valid interior before forming
interferograms, exactly as with the NaN fill of a single granule. Frames on
*different* tracks have no common pass and cannot be stitched at the SLC
level; process each track through unwrapping and mosaic the unwrapped
products instead.

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
`unw.mask_water(workspace=ws, resolution="i")`.

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

## Numerics

The interferogram/coherence formula and the SNAPHU tile/looks sizing are ported
verbatim from the original procedural module, retained as
`tests/legacy_reference.py` — the oracle the dask kernels are
equivalence-tested against.

## Tests

```bash
pytest                     # synthetic GSLC fixtures; no real data needed
NISAR_TEST_GSLC=/path/to/granule.h5 pytest tests/test_real_data.py   # real file
```
