# nisar_tools

Out-of-core, object-oriented processing of NISAR L-band SAR products: from GSLC, RSLC and GUNW granules to
interferograms, unwrapped phase, ionospheric screens, pixel offsets and line-of-sight displacement. Every stage is a
lazy `xarray` dataset backed by `dask` and persisted to a Zarr workspace, so a full stack of acquisitions never has to
fit in memory and an interrupted run resumes where it stopped.

**Documentation: <https://eitanrapa.github.io/nisar_tools/>**

## Features

- Search and download NISAR GSLCs, TEC files and DEMs from NASA Earthdata.
- Make GSLCs from RSLCs with a port of ISCE3's GSLC workflow, optionally registered to a reference date.
- Interferograms with NaN-aware multilooking, water and swath-edge masks and removal of unreachable regions; the
  Goldstein filter; SNAPHU unwrapping, pair by pair and resumable.
- Ionospheric correction: dual-frequency, and a split-spectrum screen that keeps the deformation out of it.
- Pixel offsets from GSLC pairs (a port of GMTSAR's `xcorr`) and NISAR's ROFF from RSLC pairs (a port of ISCE3's).
- Line-of-sight displacement with per-pixel look vectors, GMT `.grd` export, and frame merging across UTM zones and
  dates.

## Install

```bash
pip install "nisar-tools @ git+https://github.com/eitanrapa/nisar_tools.git"
```

For a conda environment with the optional dependencies too, run `conda env create -f environment.yml` in a clone.
The optional extras (`mask`, `download`, `gslc`, `gdal`, `slip`) are described in the
[installation guide](https://eitanrapa.github.io/nisar_tools/installation/).

## Example

```python
from nisar_tools import GSLC, GSLCStack, Workspace

paths = ["data/<first date GSLC>.h5", "data/<second date GSLC>.h5"]   # same track and frame
bbox = (-80.75, -80.65, 8.15, 8.25)                                   # lon_min, lon_max, lat_min, lat_max
ws = Workspace("work")

stack = GSLCStack.from_gslcs([GSLC(p) for p in paths], bbox=bbox).persist(ws, "slc_stack", files=paths, bbox=bbox)
igrams = (stack.form_interferograms(looks=10, convolution="Gaussian")
          .mask_water(mask_cache=ws).remove_unconnected_regions()
          .filter_goldstein(alpha="adaptive").persist(ws, "igrams"))
unw = igrams.unwrap(ws, name="unwrapped", nproc=4)
los = unw.to_los(paths[0]).persist(ws, "los")      # metres, positive toward the sensor
los.to_grd("work/grd")
```

[Getting started](https://eitanrapa.github.io/nisar_tools/getting-started/) explains each step, and the example
notebook, [`notebooks/nisar_tools.ipynb`](https://github.com/eitanrapa/nisar_tools/blob/main/notebooks/nisar_tools.ipynb),
walks through the same pipeline with plots. For slip inversion of the results, see
[slipsolve-python](https://github.com/eitanrapa/SlipSolve-Python).

## Credits

- The RSLC-to-GSLC geocoding, the GSLC writer and the ROFF offsets port algorithms from
  [ISCE3](https://github.com/isce-framework/isce3) v0.25.16, and the L2 GSLC product specification is copied from it.
- The amplitude offset tracker is a port of [GMTSAR](https://github.com/gmtsar/gmtsar)'s `xcorr`.
- Unwrapping uses [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) through
  [snaphu-py](https://github.com/isce-framework/snaphu-py); water masks use the GSHHG coastlines through GMT and pygmt.

## Licence

MIT; see [`LICENSE`](LICENSE). The ISCE3-derived modules and the vendored product specification remain under
ISCE3's Apache License 2.0, reproduced with ISCE3's notice in
[`nisar_tools/data/ISCE3_LICENSE`](nisar_tools/data/ISCE3_LICENSE). The GSLC pixel-offset tracker is a port of
GMTSAR's `xcorr`, which is GPL-3.0; its authors agreed to its release here under the MIT licence.
