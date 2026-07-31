"""Geographic helpers: bbox transforms, lat/lon reprojection, grid warping.

Kept free of any class so they can be reused by GSLC, the stacks, and the
plotting layer. Reprojection is intentionally eager (rioxarray loads its
input), so it must only ever be handed a single small 2D slice; grid warping
is likewise per-frame and is wrapped in dask tasks by the callers.
"""

import warnings

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from affine import Affine
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import Resampling, reproject, transform_bounds

# rasterio's reproject() warns about bare numpy arrays even when explicit
# src/dst transforms are supplied, and warnings.catch_warnings() is not
# thread-safe under dask's threaded scheduler, so the spurious warning is
# silenced once here, narrowly by message.
warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
    message="Dataset has no geotransform",
)


def bbox_to_native(lon_min, lon_max, lat_min, lat_max, epsg_code):
    """Transform a lon/lat bounding box to native-CRS x/y bounds.

    Returns ``(x_min, x_max, y_min, y_max)`` in the projection given by
    ``epsg_code``. All four corners are transformed (not just two) so the
    bounds are correct even when the box edges are not axis-aligned in the
    native projection.
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)

    corners_lon = [lon_min, lon_max, lon_max, lon_min]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    x_corners, y_corners = transformer.transform(corners_lon, corners_lat)

    return min(x_corners), max(x_corners), min(y_corners), max(y_corners)


def native_bbox_to_lonlat(x_min, x_max, y_min, y_max, epsg_code):
    """Transform a native-CRS bounding box to lon/lat ``(lon_min, lon_max,
    lat_min, lat_max)``. Edge-densified, so the returned box covers the whole
    footprint even where the grid is rotated relative to lon/lat."""
    return transform_native_bbox(x_min, x_max, y_min, y_max, epsg_code, 4326)


def transform_native_bbox(x_min, x_max, y_min, y_max, src_epsg, dst_epsg):
    """Transform a native-CRS bbox into another projected CRS.

    Edge-densified (via :func:`rasterio.warp.transform_bounds`), so the
    returned bounds cover the whole warped footprint even though a projected
    rectangle's edges curve in the target CRS. Returns
    ``(x_min, x_max, y_min, y_max)``.
    """
    left, bottom, right, top = transform_bounds(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", x_min, y_min, x_max, y_max
    )
    return left, right, bottom, top


def grid_transform(x_coords, y_coords):
    """Affine geotransform for a regular grid given its pixel-center coords."""
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    return Affine(
        dx, 0.0, float(x_coords[0]) - dx / 2,
        0.0, dy, float(y_coords[0]) - dy / 2,
    )


def resampling_from_name(name):
    """Resolve a rasterio :class:`Resampling` mode by name (e.g. "bilinear")."""
    try:
        return Resampling[name]
    except KeyError:
        valid = ", ".join(r.name for r in Resampling)
        raise ValueError(
            f"Unknown resampling {name!r}; expected one of: {valid}"
        ) from None


def warp_to_grid(arr, src_transform, src_epsg, dst_transform, dst_epsg,
                 dst_shape, resampling="bilinear"):
    """Warp one 2D array between projected grids. Eager, whole-frame.

    Complex input is resampled on its real and imaginary parts separately
    (for linear kernels this is exactly complex-valued interpolation) and
    recombined as complex64; use ``resampling="nearest"`` to preserve exact
    sample values. Pixels with no source coverage come back NaN.
    """
    resampling = resampling_from_name(resampling)

    def _warp(src):
        """Warp one band, or several stacked on a leading axis, in one call."""
        src = np.ascontiguousarray(src, dtype=np.float32)
        shape = dst_shape if src.ndim == 2 else (src.shape[0],) + tuple(dst_shape)
        dst = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            src,
            dst,
            src_transform=src_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{dst_epsg}",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst

    if np.iscomplexobj(arr):
        # rasterio warps a multi-band array in a single call, so the real and
        # imaginary parts share one coordinate-transform setup instead of paying
        # for it twice.
        bands = _warp(np.stack([arr.real, arr.imag]))
        return (bands[0] + 1j * bands[1]).astype(np.complex64)
    return _warp(arr)


def project_to_latlon(data, x_coords=None, y_coords=None, epsg_code=None):
    """Reproject a 2D grid from its native CRS to WGS84 lon/lat.

    Accepts either a georeferenced :class:`xarray.DataArray` (with a written
    CRS) or a raw 2D array plus ``x_coords``/``y_coords``/``epsg_code``.
    Returns a lon/lat :class:`xarray.DataArray`. Eager: computes its input.
    """
    if isinstance(data, xr.DataArray) and data.rio.crs is not None:
        da = data
    else:
        if x_coords is None or y_coords is None or epsg_code is None:
            raise ValueError(
                "Provide a CRS-aware DataArray, or data with x_coords, "
                "y_coords and epsg_code."
            )
        da = xr.DataArray(
            np.asarray(data), coords={"y": y_coords, "x": x_coords}, dims=["y", "x"]
        ).rio.write_crs(f"EPSG:{epsg_code}")

    return da.rio.reproject("EPSG:4326")


#: xarray engines tried in turn by :func:`read_grd`, best first.
#:
#: A ``.grd`` is single-variable NetCDF, but *which* NetCDF matters: GMT 6 writes
#: HDF5-backed NetCDF-4 by default, which scipy's classic-only backend cannot
#: read, while ``netcdf4``/``h5netcdf`` are optional dependencies this project
#: does not require. NetCDF-4 is handled instead by :func:`_read_grd_hdf5`, using
#: the ``h5py`` this package already depends on.
#:
#: ``pygmt`` registers a ``gmt`` engine that reads both, and it is deliberately
#: **not** in this list: importing ``h5py`` -- which ``nisar_tools`` always does
#: -- makes it drop a NetCDF-4 grid's coordinate variables and return *pixel
#: indices* instead, with no error. Two HDF5 libraries end up loaded in one
#: process and GMT's netCDF layer loses. The data comes back right and only the
#: georeferencing is wrong, so nothing downstream would notice.
_GRD_ENGINES = ("netcdf4", "h5netcdf", "scipy", "rasterio")

#: Leading bytes of an HDF5 file, i.e. of a NetCDF-4 ``.grd``.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def utm_epsg(lon, lat):
    """EPSG code of the WGS84 UTM zone containing ``(lon, lat)``."""
    zone = int((float(lon) + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _read_grd_hdf5(path):
    """Read a NetCDF-4 (HDF5-backed) ``.grd`` with ``h5py``.

    A GMT grid is one 2-D variable plus its two coordinate variables at the file
    root, so the layout can be recovered without a NetCDF library: the data is
    the only 2-D dataset, and its axes are named by the HDF5 dimension scales it
    points at (``DIMENSION_LIST``), falling back to matching 1-D datasets by
    length. The CRS, when present, rides on a ``grid_mapping`` dataset's
    ``spatial_ref``/``crs_wkt`` attribute.
    """
    import h5py

    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(path, "r") as f:
        arrays = {k: v for k, v in f.items() if isinstance(v, h5py.Dataset)}
        grids = [k for k, v in arrays.items() if v.ndim == 2]
        if len(grids) != 1:
            raise ValueError(
                f"{path}: expected exactly one 2-D variable, found {grids or 'none'}"
            )
        var = arrays[grids[0]]

        names = []
        for ref_list in var.attrs.get("DIMENSION_LIST", []):
            names.append(f[ref_list[0]].name.lstrip("/"))
        if len(names) != 2:  # no dimension scales; match by length instead
            names = [
                next(k for k, v in arrays.items() if v.ndim == 1 and v.shape[0] == n)
                for n in var.shape
            ]

        data = np.asarray(var)
        fill = var.attrs.get("_FillValue")
        if fill is not None and np.issubdtype(data.dtype, np.floating):
            data = np.where(data == np.asarray(fill).ravel()[0], np.nan, data)

        da = xr.DataArray(
            data, dims=names,
            coords={n: np.asarray(arrays[n]) for n in names},
            name=grids[0],
        )
        mapping = var.attrs.get("grid_mapping")
        if mapping is not None:
            attrs = arrays.get(text(mapping), None)
            wkt = attrs and (attrs.attrs.get("spatial_ref") or attrs.attrs.get("crs_wkt"))
            if wkt is not None:
                da = da.rio.write_crs(text(wkt))
    return da


def read_grd(path, engine=None):
    """Read a GMT ``.grd`` as a CRS-aware 2-D DataArray -- the inverse of
    :func:`write_grd`.

    Axes are renamed to ``x``/``y``. The CRS comes from the file when it carries
    one; otherwise the grid is taken as geographic if its axes were named
    ``lon``/``lat`` (GMT's own convention for a geographic grid) or if they span
    a plausible lon/lat range. ``engine`` forces one backend instead of trying
    :data:`_GRD_ENGINES` in turn; ``engine="hdf5"`` forces the ``h5py`` reader.

    Eager -- a ``.grd`` is a whole raster in one variable, with no chunking to
    preserve.
    """
    with open(path, "rb") as fh:
        is_hdf5 = fh.read(8) == _HDF5_MAGIC

    engines = [engine] if engine else (
        ["hdf5"] if is_hdf5
        else [e for e in _GRD_ENGINES if e in xr.backends.list_engines()]
    )
    da, errors = None, []
    for name in engines:
        try:
            da = (_read_grd_hdf5(path) if name == "hdf5"
                  else xr.open_dataarray(path, engine=name))
            break
        except Exception as exc:  # unreadable this way; try the next
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if da is None:
        raise ValueError(
            f"Could not read {path} (tried {', '.join(engines)}).\n" + "\n".join(errors)
        )

    da = da.squeeze(drop=True)  # rasterio leaves a length-1 band axis
    geographic = "lon" in da.dims or "lat" in da.dims
    da = da.rename({k: v for k, v in (("lon", "x"), ("lat", "y")) if k in da.dims})
    if da.ndim != 2:
        raise ValueError(f"{path} is not a 2-D grid (dims {da.dims})")

    if da.rio.crs is None:
        x, y = da["x"].values, da["y"].values
        if geographic or (abs(x).max() <= 360.0 and abs(y).max() <= 90.0):
            da = da.rio.write_crs("EPSG:4326")
    return da.rio.write_nodata(np.nan)


def write_grd(field, path):
    """Reproject a 2D native-grid field to lon/lat and write a GMT `.grd`.

    A GMT ``.grd`` is single-variable NetCDF, so the field is reprojected to
    WGS84, its axes renamed ``lat``/``lon`` and the data variable to ``z``.
    Two rioxarray artefacts are stripped or the file is unusable: the length-1
    ``band`` axis a reproject leaves behind (``squeeze``), and the
    ``spatial_ref`` coordinate, which NetCDF would otherwise write as a second
    variable and GMT would then refuse. ``field`` must be CRS-aware (any field
    of a stack is). Eager. Returns ``path``.

    Integer label layers (a connected-component or subswath mask) are cast to
    ``float32`` with NaN outside coverage: a GMT grid is float anyway, uint32
    overflows classic NetCDF's int32, and a NaN edge beats an integer sentinel
    warped in from the reprojection.
    """
    if np.issubdtype(field.dtype, np.integer):
        field = field.astype(np.float32).rio.write_nodata(np.nan)
    g = project_to_latlon(field)                      # eager; EPSG:4326
    g = g.squeeze(drop=True)                           # drop the band axis
    g = g.drop_vars("spatial_ref", errors="ignore")    # else a 2nd variable
    g = g.rename({"y": "lat", "x": "lon"}).rename("z")
    g.to_netcdf(path)
    return path
