# Development

## Setting up

Clone the repository and build the conda environment from `environment.yml` (see [Installation](installation.md)), or
install the package in editable mode with the test extras into an environment that already has the geospatial stack:

```bash
git clone https://github.com/eitanrapa/nisar_tools.git
cd nisar_tools
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The suite runs on synthetic GSLC, GUNW and RSLC files that the fixtures build (`tests/conftest.py`), so it needs no
real data. Every test writes into its own temporary directory, so it also runs in parallel with `pytest-xdist`. One
test measures memory with `tracemalloc` and is marked `serial`; run it on its own:

```bash
pytest -n auto -k "not memory_does_not_scale"
pytest -k memory_does_not_scale
```

Tests that need an optional dependency skip when it is missing: `pygmt` (water masks), `osgeo` (the GSLC writer's
projection metadata) and `slipsolve` (`tests/test_grd_import.py`, which checks LOS grids end to end through a slip
inversion). `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before anything loads OpenMP.

## Tests on real data

A few tests run against real granules and skip unless environment variables point at them:

| Variable | Test file | What it checks |
|---|---|---|
| `NISAR_TEST_GSLC` | `tests/test_real_data.py` | a real GSLC's layout, attributes and lazy reads, and the lazy pipeline at full size |
| `NISAR_TEST_GSLC2` | `tests/test_real_data.py` | with the above: a real two-date interferogram (same frame, another date) |
| `NISAR_TEST_GUNW` | `tests/test_real_gunw.py` | a real GUNW's layout, a crop and its self-contained `to_los` |
| `NISAR_TEST_RSLC`, `NISAR_TEST_DEM` | `tests/test_real_gslc_geocode.py` | geocoding a 2 km box of a real RSLC (ideally against the NISAR DEM v1.2) |
| `NISAR_TEST_TEC` | `tests/test_real_gslc_geocode.py` | with the above: the ionospheric correction from a TEC file |
| `NISAR_TEST_GSLC_REF` | `tests/test_real_gslc_geocode.py` | with the above: a format comparison against any ASF GSLC |
| `NISAR_TEST_DOWNLOAD=1` | `tests/test_download.py` | a live Copernicus DEM download |

```bash
NISAR_TEST_GSLC=/path/to/granule.h5 pytest tests/test_real_data.py
NISAR_TEST_RSLC=rslc.h5 NISAR_TEST_DEM=dem.tif NISAR_TEST_TEC=tec.json \
NISAR_TEST_GSLC_REF=asf_gslc.h5 pytest tests/test_real_gslc_geocode.py
```

## Checking the GSLC format

`tests/gslc_format_diff.py` compares the HDF5 layout of two GSLC files (presence, dtypes, chunking, filters, fill
values and attributes), ignoring their data. It exits non-zero if anything structural differs:

```bash
python tests/gslc_format_diff.py ASF_GSLC.h5 OUR_GSLC.h5 [--values]
```

## Documentation

The site is built with MkDocs Material; the API pages come from the docstrings through mkdocstrings, which reads the
source statically and does not import the package. Docstrings are written in Markdown, with numpy-style sections where
a function has them.

```bash
pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

A GitHub Actions workflow (`.github/workflows/docs.yml`) builds the site on every push and pull request, and deploys
it to GitHub Pages from `main`.
