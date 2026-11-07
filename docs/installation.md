# Installation

`nisar_tools` needs Python 3.10 or later. Its core dependencies are numpy, scipy, h5py, xarray, dask, zarr (version 2),
numcodecs, rioxarray, rasterio, pyproj, snaphu and matplotlib. Everything else is optional and imported only when used.

## With pip, from GitHub

```bash
pip install "nisar-tools @ git+https://github.com/eitanrapa/nisar_tools.git"
```

With optional extras (see [below](#optional-dependencies)):

```bash
pip install "nisar-tools[mask,download,gslc] @ git+https://github.com/eitanrapa/nisar_tools.git"
```

The geospatial stack is easier to install from conda-forge than from pip. A common route is to install the compiled
dependencies with conda and then the package with pip:

```bash
conda install -c conda-forge h5py "zarr<3" dask rioxarray pyproj snaphu
pip install "nisar-tools @ git+https://github.com/eitanrapa/nisar_tools.git"
```

!!! note "zarr 2"
    Stages are stored in the Zarr v2 format, because the v3 specification has no complex data type. The package
    requires `zarr>=2.18,<3`.

## With the conda environment file

The repository ships `environment.yml`, which builds a complete environment from conda-forge (Python 3.11, the core
dependencies, every optional one except `slipsolve-python`, `pytest` and `ipykernel`) and installs the package in
editable mode from the clone:

```bash
git clone https://github.com/eitanrapa/nisar_tools.git
cd nisar_tools
conda env create -f environment.yml            # the environment is named remote_sensing
conda activate remote_sensing
```

Pass `-n <name>` to `conda env create` to choose another name. Because the install is editable, the cloned folder is
the installation: keep it in place, and a `git pull` updates the package.

## Optional dependencies

Each extra enables one part of the package. Without it, everything else still imports and runs.

| Extra | Installs | Enables | Without it |
|---|---|---|---|
| `mask` | `pygmt` | `mask_water`, the GSHHG coastline mask from GMT | water masking raises `ImportError` |
| `download` | `earthaccess`, `sardem`, `requests` | the `download` module: Earthdata search and download, DEMs, TEC files | fetch inputs yourself, or use `download_gslcs(..., method="asf")`, which needs only the standard library |
| `gslc` | `pysolid` | the solid Earth tide correction in `geocode_rslc` | pass `solid_earth_tides=False` |
| `gdal` | `gdal` | the `osgeo` bindings: projection metadata in `geocode_rslc` output, `dem.stage_nisar_dem` | everything except making GSLCs and staging the NISAR DEM |
| `slip` | `slipsolve-python` | `LOSStack.to_observations` | sample LOS stacks with slipsolve directly |
| `dev` | `pytest`, `pytest-xdist`, `dask[diagnostics]` | the test suite | |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | building this documentation | |

Points that trip up a plain `pip install`:

- **`pygmt` and `gdal` are best installed from conda-forge.** Both need native libraries; pip's `gdal` builds against
  system headers and often fails. Only the GSLC writer and `stage_nisar_dem` import `osgeo`; everything else reads and
  writes rasters through rasterio's own GDAL.
- **`earthaccess`'s latest release needs Python 3.12.** On Python 3.11, pip picks an older compatible release, or you
  can use the direct download path (`method="asf"`).

## Earthdata Login

Downloads, the NISAR DEM and the TEC files need a free [Earthdata Login](https://urs.earthdata.nasa.gov). Most routes
read the credentials from `~/.netrc`:

```text
machine urs.earthdata.nasa.gov login <username> password <password>
```

`download.login()` can write that file for you (`strategy="interactive"`, `persist=True`), or read
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (`strategy="environment"`).

## Using the environment from Jupyter

Register the environment as a Jupyter kernel and select it in the notebook:

```bash
conda activate remote_sensing
python -m ipykernel install --user --name remote_sensing --display-name "Python (remote_sensing)"
```

A `ModuleNotFoundError: No module named 'nisar_tools'` in a notebook almost always means that another kernel is
selected.

Two environment variables avoid two failures seen with conda environments that combine these libraries:

- `KMP_DUPLICATE_LIB_OK=TRUE`: an environment can end up with two copies of the OpenMP runtime, and OpenMP then
  aborts the process at import.
- `GMT_LIBRARY_PATH=<environment>/lib`: makes pygmt load the environment's own GMT library. Otherwise it can bind to a
  GMT installed elsewhere on the system, whose netCDF and HDF5 libraries do not match the environment's, and water
  masking fails with errors about the GSHHG coastline files.

Export them in the shell that starts Jupyter (or a batch job), or add them to the kernel's `kernel.json`, whose
folder `jupyter kernelspec list` prints:

```json
"env": {
  "KMP_DUPLICATE_LIB_OK": "TRUE",
  "GMT_LIBRARY_PATH": "/path/to/the/environment/lib"
}
```

## Building this documentation

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what the documentation workflow runs
```
