"""DEM staging and ISCE3-exact DEM interpolation for geocoding RSLCs.

Two jobs, both in service of :mod:`nisar_tools.rslc`:

- :func:`stage_nisar_dem` fetches the **NISAR DEM v1.2** (the modified Copernicus
  GLO-30 on the WGS84 ellipsoid that every NISAR product is processed with)
  over a lon/lat box, through its global VRT on Earthdata. It is a pure pixel
  copy -- nearest-neighbour onto bounds snapped to source pixel edges, the trick
  sardem uses -- so the heights are the mission's heights, bit for bit. Only the
  on-disk compression differs.
- :class:`DEMInterpolator` evaluates heights the way ISCE3's
  ``DEMInterpolator::interpolateLonLat`` does: its "biquintic" spline in
  float32, reference height 0 m off the DEM.

Why exactness matters here and not in most DEM uses: a GSLC's phase is
*flattened* by the range to the DEM surface, so a height error ``dh`` becomes a
phase error of ``4 pi / lambda * dh * cos(inc)`` -- about **0.4 rad per cm** at
L-band. A GSLC flattened against a different DEM, or the same DEM interpolated
differently, is not interferometrically compatible with ASF's.
"""

import math
import os
from pathlib import Path

import numpy as np

from . import radar

#: The NISAR DEM v1.2 global VRTs, per projection, on Earthdata.
NISAR_DEM_VRT = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/DEM/v1.2/{epsg}/{epsg}.vrt"

#: Source pixel size of the WGS84 NISAR DEM, degrees.
_DEM_RES = 1.0 / 3600.0

#: ``DEMRasterToInterpolator``'s margin, in DEM pixels, around the region.
DEM_MARGIN_PIXELS = 50


def _snap_bounds(west, south, east, north, res=_DEM_RES):
    """Snap outward to source pixel *edges* (centres sit on multiples of res).

    A nearest-neighbour warp whose output centres coincide with input centres is
    a pixel copy; bounds on the centres instead make every sample a rounding tie.
    """
    inv = 1.0 / res
    return ((math.floor(west * inv - 0.5) + 0.5) * res,
            (math.floor(south * inv - 0.5) + 0.5) * res,
            (math.ceil(east * inv - 0.5) + 0.5) * res,
            (math.ceil(north * inv - 0.5) + 0.5) * res)


def stage_nisar_dem(dest, bbox, *, margin_deg=0.1, overwrite=False):
    """Write the NISAR DEM v1.2 over ``bbox`` to a compressed GeoTIFF.

    ``bbox`` is this package's ``(lon_min, lon_max, lat_min, lat_max)``;
    ``margin_deg`` pads it on every side. Needs Earthdata credentials in
    ``~/.netrc`` (GDAL reads them itself). Returns ``dest``.

    Polar products use the EPSG:3413/3031 DEMs; this helper covers the WGS84
    one, which spans 60 S - 60 N where every UTM-projected NISAR frame lies.
    """
    from osgeo import gdal

    dest = Path(dest).expanduser()
    if dest.exists() and not overwrite:
        return dest
    lon_min, lon_max, lat_min, lat_max = bbox
    if not (-60.0 <= lat_min and lat_max <= 60.0):
        raise NotImplementedError("polar NISAR DEMs (EPSG:3413/3031) are not staged here")
    west, south, east, north = _snap_bounds(lon_min - margin_deg, lat_min - margin_deg,
                                            lon_max + margin_deg, lat_max + margin_deg)
    gdal.UseExceptions()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nisar_tools"
    cache.mkdir(parents=True, exist_ok=True)
    cookies = str(cache / "earthdata_cookies.txt")
    options = {
        "GDAL_HTTP_COOKIEFILE": cookies,
        "GDAL_HTTP_COOKIEJAR": cookies,
        "GDAL_HTTP_NETRC": "YES",
        "GDAL_HTTP_AUTH": "BASIC",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    for key, value in options.items():
        gdal.SetConfigOption(key, value)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.tif")
    gdal.Warp(
        str(tmp), "/vsicurl/" + NISAR_DEM_VRT.format(epsg="EPSG4326"),
        options=gdal.WarpOptions(
            format="GTiff", outputBounds=(west, south, east, north),
            xRes=_DEM_RES, yRes=_DEM_RES, resampleAlg="near",
            outputType=gdal.GDT_Float32, multithread=True,
            creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES",
                             "BIGTIFF=IF_SAFER"],
        ),
    )
    tag_nisar_dem(tmp)
    os.replace(tmp, dest)
    return dest


#: Version of the NISAR DEM that :func:`stage_nisar_dem` fetches.
NISAR_DEM_VERSION = "1.2"


def tag_nisar_dem(path, version=NISAR_DEM_VERSION):
    """Record in a GeoTIFF's metadata that it is the NISAR DEM ``version``.

    Lets a GSLC made against it state its DEM source in the mission's own words
    (see ``gslc_geocode._dem_description``) instead of "(NOT SPECIFIED)".
    """
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(path), gdal.GA_Update)
    ds.SetMetadataItem("NISAR_DEM_VERSION", version)
    ds = None


class DEMInterpolator:
    """Heights above the WGS84 ellipsoid from a DEM raster, ISCE3-style.

    Port of ``isce3::geometry::DEMInterpolator`` with ``BIQUINTIC_METHOD``: the
    raster is read a window at a time (:meth:`load`, the ``loadDEM`` index
    arithmetic plus ``DEMRasterToInterpolator``'s 50-pixel margin), and
    :meth:`interpolate_lonlat` / :meth:`interpolate_xy` return the reference
    height wherever the 6x6 spline window would leave the loaded raster.

    ``dem`` is a path to anything rasterio opens (GeoTIFF, VRT), in EPSG:4326 or
    a projected CRS. Thread-safe for concurrent :meth:`heights` calls: each
    call reads its own window through its own dataset handle.
    """

    def __init__(self, dem, ref_height=0.0):
        import rasterio

        self.path = str(dem)
        self.ref_height = float(ref_height)
        with rasterio.open(self.path) as src:
            if src.crs is None:
                raise ValueError(f"{self.path}: DEM has no CRS")
            self.epsg = src.crs.to_epsg()
            gt = src.transform
            if gt.b != 0 or gt.d != 0:
                raise ValueError("rotated DEM geotransforms are not supported")
            self.x0, self.dx = gt.c, gt.a
            self.y0, self.dy = gt.f, gt.e
            self.width, self.length = src.width, src.height
        if self.dx <= 0:
            raise ValueError("DEM x spacing must be positive")
        self._to_dem = None
        if self.epsg != 4326:
            from pyproj import Transformer
            self._to_dem = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}",
                                                always_xy=True)

    @property
    def bounds_lonlat(self):
        """``(lon_min, lon_max, lat_min, lat_max)`` of the raster (EPSG:4326 only)."""
        if self.epsg != 4326:
            raise ValueError("bounds_lonlat is only defined for EPSG:4326 DEMs")
        return (self.x0, self.x0 + self.width * self.dx,
                self.y0 + self.length * self.dy, self.y0)

    def to_dem_xy(self, lon_rad, lat_rad):
        """Radians -> the DEM's own coordinates (degrees for EPSG:4326)."""
        lon = np.degrees(lon_rad)
        lat = np.degrees(lat_rad)
        if self._to_dem is None:
            return lon, lat
        return self._to_dem.transform(lon, lat)

    def load(self, min_x, max_x, min_y, max_y, margin_pixels=DEM_MARGIN_PIXELS):
        """Read the window covering a DEM-coordinate box (``loadDEM``).

        Returns ``(array_float32, xstart, ystart)`` where the starts are the
        *centres* of the first loaded pixel, as ISCE3 stores them.
        """
        import rasterio
        from rasterio.windows import Window

        min_x -= margin_pixels * self.dx
        max_x += margin_pixels * self.dx
        min_y -= margin_pixels * abs(self.dy)
        max_y += margin_pixels * abs(self.dy)
        xf = self.x0 + self.width * self.dx
        yf = self.y0 + self.length * self.dy
        min_x, max_x = max(min_x, self.x0), min(max_x, xf)
        min_y, max_y = max(min_y, yf), min(max_y, self.y0)
        ix0 = int(math.floor((min_x - self.x0) / self.dx))
        ix1 = int(math.ceil((max_x - self.x0) / self.dx))
        iy0 = int(math.floor((max_y - self.y0) / self.dy))
        iy1 = int(math.ceil((min_y - self.y0) / self.dy))
        ix1 = min(ix1, self.width)
        iy1 = min(iy1, self.length)
        if ix1 <= ix0 or iy1 <= iy0:
            return None, None, None
        with rasterio.open(self.path) as src:
            arr = src.read(1, window=Window(ix0, iy0, ix1 - ix0, iy1 - iy0),
                           out_dtype=np.float32)
        xstart = self.x0 + (0.5 + ix0) * self.dx
        ystart = self.y0 + (0.5 + iy0) * self.dy
        return arr, xstart, ystart

    def coarse_sampler(self, bbox_lonlat, max_pixels=2048):
        """A ``(lon_rad, lat_rad) -> height`` callable over a decimated DEM read.

        For corrections whose sensitivity to height is negligible (the solid
        Earth tide range change moves by micrometres for tens of metres of
        height error): one decimated read instead of a full-resolution window
        per ``rdr2geo`` iteration. Heights use the same spline on the coarse
        grid. ``bbox_lonlat`` is ``(lon_min, lon_max, lat_min, lat_max)`` in the
        DEM's own coordinates (degrees for EPSG:4326).
        """
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        lon_min, lon_max, lat_min, lat_max = bbox_lonlat
        with rasterio.open(self.path) as src:
            win = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            scale = max(1.0, max(win.width, win.height) / max_pixels)
            out_shape = (max(8, int(win.height / scale)), max(8, int(win.width / scale)))
            arr = src.read(1, window=win, out_shape=out_shape,
                           resampling=Resampling.average, out_dtype=np.float32)
            wt = src.window_transform(win)
        dx = wt.a * win.width / out_shape[1]
        dy = wt.e * win.height / out_shape[0]
        xstart = wt.c + 0.5 * dx
        ystart = wt.f + 0.5 * dy

        def sample(lon_rad, lat_rad):
            x, y = self.to_dem_xy(np.asarray(lon_rad, float), np.asarray(lat_rad, float))
            return interpolate_dem_window(arr, xstart, ystart, dx, dy, x, y, self.ref_height)

        return sample

    def heights(self, lon_rad, lat_rad):
        """Interpolated heights at geodetic points (radians), float64.

        Loads just the window the points need (plus the margin) and evaluates
        ISCE3's spline in float32. Points off the DEM get the reference height.
        """
        lon_rad = np.asarray(lon_rad, float)
        lat_rad = np.asarray(lat_rad, float)
        x, y = self.to_dem_xy(lon_rad, lat_rad)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.full(x.shape, self.ref_height)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return out
        arr, xstart, ystart = self.load(np.nanmin(x[finite]), np.nanmax(x[finite]),
                                        np.nanmin(y[finite]), np.nanmax(y[finite]))
        if arr is None:
            return out
        out[finite] = interpolate_dem_window(arr, xstart, ystart, self.dx, self.dy,
                                             x[finite], y[finite], self.ref_height)
        return out


def interpolate_dem_window(arr, xstart, ystart, dx, dy, x, y, ref_height=0.0):
    """``DEMInterpolator::interpolateXY`` over a loaded window.

    ``(xstart, ystart)`` is the centre of ``arr[0, 0]``. A point whose floor row
    or column is within 2 of the top/left edge, or within 1 of the bottom/right,
    returns ``ref_height`` -- ISCE3's bounds test, reproduced exactly because it
    decides which pixels near a DEM edge fall back to the ellipsoid.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    row = (y - ystart) / dy
    col = (x - xstart) / dx
    irow = np.floor(row)
    icol = np.floor(col)
    ny, nx = arr.shape
    inside = (irow >= 2) & (irow < ny - 1) & (icol >= 2) & (icol < nx - 1)
    out = np.full(x.shape, float(ref_height))
    if inside.any():
        out[inside] = radar.spline2d_interp(arr, col[inside], row[inside], order=6)
    return out
