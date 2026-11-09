# API reference

These pages are generated from the docstrings. The names most scripts need are importable from the package itself:
`GSLC`, `RSLC`, `GSLCStack`, `InterferogramStack`, `UnwrappedStack`, `LOSStack`, `Workspace`,
`WorkspaceError`, `RegistrationField`, `geocode_rslc`, `default_gslc_name`, `product_bounds`, `make_pairs`,
`dense_offsets`, `open_roff`, `geocode_offsets`, `login`, `download_gslcs`, `download_dem`, `download_tec`,
`search_tec`, and the modules `download`, `ionosphere` and `roff`.

| Page | Modules | Contents |
|---|---|---|
| [Products and inputs](products.md) | `gslc`, `rslc`, `download`, `dem` | reading granules, downloading, DEM staging |
| [Processing stages](stages.md) | `stack`, `interferogram`, `unwrap`, `los`, `workspace` | the stage classes and the workspace |
| [Geocoding RSLCs](geocoding.md) | `gslc_geocode`, `gslc_product`, `corrections`, `radar` | RSLC to GSLC, the product writer, timing corrections, radar geometry |
| [Offsets](offsets.md) | `roff`, `ampcor` | NISAR's ROFF pixel offsets from RSLCs |
| [Ionosphere](ionosphere.md) | `ionosphere`, `split_spectrum` | dual-frequency and split-spectrum screens |
| [Utilities](utilities.md) | `geo`, `geometry`, `mask`, `plot` | reprojection and grids, look geometry, water masks, plots |

Every stage class (`GSLCStack`, `InterferogramStack`, `UnwrappedStack`, `LOSStack`) also has
`crop(lon_min, lon_max, lat_min, lat_max)`, `to_grd(outdir, fields=None, indices=None)`, and the properties `epsg`,
`crs`, `direction`, `x`, `y` and `sizes`, shared through a common base class and listed with each class.
