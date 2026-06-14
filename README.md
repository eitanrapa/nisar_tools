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

## Design

| Class | Wraps | Dims |
|-------|-------|------|
| `GSLC` | one HDF5 granule (lazy) | `(y, x)` |
| `GSLCStack` | aligned acquisitions | `(time, y, x)` |
| `InterferogramStack` | igram + coherence | `(pair, y, x)` |
| `UnwrappedStack` | unwrapped phase + conncomp | `(pair, y, x)` |
| `Workspace` | per-stage Zarr stores | — |

- **Lazy everywhere** between disk reads and stage persistence. Cropping and
  merging are coordinate slices; multilooking uses `dask.array.map_overlap`
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
