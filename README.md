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

# 6. Mask water and plot.
unw = unw.mask_water(workspace=ws)
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
