# nisar_tools

`nisar_tools` processes NISAR L-band SAR products into interferograms, unwrapped phase and line-of-sight (LOS)
displacement. Every stage is a lazy `xarray` dataset backed by `dask` and written to a Zarr workspace, so a full stack
of acquisitions never has to fit in memory and an interrupted run resumes where it stopped.

It is written for people who want to control each processing step, or who need products that are not distributed:
a GSLC for an acquisition that only has an RSLC, a secondary GSLC registered to a reference, an interferogram cut
along a rupture, or an ionospheric screen that leaves the deformation alone.

## Features

- **Inputs.** Search and download NISAR GSLCs, TEC files and DEMs through NASA Earthdata (`download`), and read GSLC,
  RSLC and GUNW granules lazily (`GSLC`, `RSLC`, `UnwrappedStack.from_gunw_files`).
- **RSLC to GSLC.** `geocode_rslc` is a port of the ISCE3 GSLC workflow that writes the NISAR L2 GSLC layout, so its
  output reads like any granule from ASF. A `RegistrationField` geocodes a secondary onto a reference's grid at measured
  offsets.
- **Interferograms.** NaN-aware multilooking on a lattice shared by all frames of a track, water and swath-edge masks,
  removal of regions the unwrapper cannot reach, and the Goldstein filter (`GSLCStack`, `InterferogramStack`).
- **Unwrapping.** SNAPHU, one pair at a time and resumable, plus edge, outlier and ramp clean-up (`UnwrappedStack`).
- **Ionosphere.** A dual-frequency correction from two unwrapped sub-bands (`ionosphere`), and a split-spectrum screen
  built from the main band and a deformation-free side-band residual (`split_spectrum`).
- **Pixel offsets.** NISAR's ROFF from two RSLCs (a port of ISCE3's offsets workflow, `roff`), convertible to
  LOS-style stacks.
- **Line of sight.** Displacement in metres with per-pixel look vectors from the products' own geometry cubes
  (`LOSStack`), export to GMT `.grd` from every stage, and import of LOS grids made by other software.
- **Merging.** Frames joined before or after unwrapping, across UTM zones and across dates.
- **Workspaces.** One Zarr store per stage, keyed on the parameters that made it, so reruns reuse finished stages and
  refuse to overwrite results made differently (`Workspace`).

## Where to go next

- [Installation](installation.md): pip, the conda environment file and the optional extras.
- [Getting started](getting-started.md): two GSLCs to a LOS grid in a few lines.
- The example notebook,
  [`notebooks/nisar_tools.ipynb`](https://github.com/eitanrapa/nisar_tools/blob/main/notebooks/nisar_tools.ipynb),
  walks the same path with plots, plus GUNW input and pixel offsets.
- The user guide, starting with the [conventions](guide/conventions.md) for signs, grids and units.
- [The InSAR pipeline](insar_pipeline.md): how these pieces are combined for coseismic pairs, and how that differs
  from the ISCE3 PR products.
- The [API reference](api/index.md), generated from the docstrings.

For slip inversion of the LOS products, see [slipsolve-python](https://github.com/eitanrapa/SlipSolve-Python).
