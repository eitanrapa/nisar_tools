# Finding and reading products

## Downloading

The `download` module fetches inputs from NASA Earthdata. It needs the `download` extra and an
[Earthdata Login](../installation.md#earthdata-login).

```python
from nisar_tools import download

download.login()                                     # netrc, environment variables, or a prompt
bbox = (-120.5, -119.5, 34.0, 35.0)                  # lon_min, lon_max, lat_min, lat_max

# Search CMR by area and time, then download (earthaccess):
files = download.download_gslcs("data/gslc", bbox=bbox, temporal=("2025-11-01", "2025-12-31"))

# Or by granule name, from the NISAR bucket, with only the standard library:
files = download.download_gslcs("data/gslc", granules=["NISAR_L2_PR_GSLC_..."], method="asf")
```

- `short_name` selects the CMR collection, and by default it follows the granule names (`download.gslc_short_name`):
  production `_PR_` GSLCs are in `NISAR_L2_GSLC_PROVISIONAL_V1`, urgent-response `_UR_` ones in `NISAR_UR_L2`, and
  others in the beta release's `NISAR_L2_GSLC_BETA_V1`. A search by area or time alone uses the production collection;
  pass `short_name=` for another. `download.search(short_name, ...)` searches any collection and returns earthaccess
  results.
- `download_dem(dest, bbox)` builds a DEM with sardem. The default source is the NISAR reference DEM
  (`data_source="NISAR"`); `"COP"` (Copernicus, no login) and `"3DEP"` (United States) also work.
- `download_tec(dest, rslc=...)` (or `start=`, `end=`) fetches the daily NISAR TEC file covering an acquisition, for
  the ionospheric geolocation correction when [making a GSLC](rslc-to-gslc.md). `search_tec` lists the candidates;
  searching needs no login.
- Orbit files are not needed: the orbit is embedded in each granule, and the look geometry comes from the granule's
  `metadata/radarGrid` cube.

To geocode RSLCs, use `dem.stage_nisar_dem` rather than `download_dem`: it copies the NISAR DEM v1.2 pixel for pixel,
which is what interferometric compatibility with ASF's products needs (see [Making GSLCs](rslc-to-gslc.md)).

## GSLC granules

```python
from nisar_tools import GSLC

g = GSLC("data/NISAR_L2_PR_GSLC_....h5", frequency="A", polarization="HH")
g.epsg, g.direction, g.datetime            # projection, pass direction, zero-Doppler start time
g.x_coords, g.y_coords                     # pixel-centre map coordinates
g.wavelength                               # metres, from centerFrequency
patch = g.crop(-115.70, -115.40, 34.55, 34.75)   # lazy (y, x) DataArray of the complex image
```

- A `GSLC` keeps its HDF5 file open for as long as it lives, and its lazy arrays read from that handle. Keep it open
  until the data are computed or persisted; once a stack is [persisted](workspaces.md) it reads from the workspace and
  the granule can be closed (`g.close()`, or use `with GSLC(path) as g:`).
- `GSLCStack.from_gslcs(gslcs, bbox=bbox)` stacks several dates. All granules must share one projection, pass direction,
  sub-band and grid, that is, the same track and frame. For neighbouring frames, build one stack per frame and
  [merge](merging.md) them.
- Reads are fast because NISAR GSLCs are gzip-compressed and h5py decodes on one global lock: the package reads each
  chunk's compressed bytes through h5py and decompresses them outside the lock, so dask's threads decode in parallel.
  The output is byte-identical to reading through h5py, and granules whose filter pipeline it cannot invert fall back
  to h5py automatically.

## RSLC granules

`RSLC(path)` reads an L1 RSLC's metadata eagerly and its imagery on demand. It is the input to
[`geocode_rslc`](rslc-to-gslc.md) and to [`roff.dense_offsets`](offsets.md#nisars-roff-from-two-rslcs);
`RSLC.to_gslc(output, dem, **kwargs)` is the same call as `geocode_rslc`.

## GUNW granules

A NASA GUNW is an interferogram that has already been unwrapped and geocoded. `UnwrappedStack.from_gunw_file(path)`
(or `from_gunw_files(paths)` for a time series on one grid) reads it into the same class that SNAPHU output becomes,
with the product's ionospheric screen and validity mask as extra layers. See
[Unwrapping and cleaning](unwrapping.md#nasa-gunw-products).

## ROFF granules

`roff.open_roff(path)` reads a NISAR ROFF product's pixel offsets in the layout of `roff.dense_offsets`, so the two
can be compared and geocoded with the same functions. See [Pixel offsets](offsets.md).
