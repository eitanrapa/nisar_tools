"""The :class:`GSLC` class: one NISAR GSLC HDF5 granule, read lazily.

A ``GSLC`` keeps its HDF5 file open for its whole lifetime and exposes the
complex image as a chunked, dask-backed :class:`xarray.DataArray`. Nothing is
read from disk until the graph is computed or persisted, so cropping a tiny
region out of a multi-GB granule only ever reads that region.

Lifetime note: because the dask graph holds references to the open ``h5py``
dataset, you must keep the ``GSLC`` (and thus its file handle) alive until the
data is computed or persisted. Persisting a stack reopens it from Zarr and
severs this dependency, after which the granules can be closed.
"""

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr

from . import geo

GRID_PATH = "science/LSAR/GSLC/grids/frequency{frequency}"
IDENT_PATH = "science/LSAR/identification"


def _decode(value):
    """Decode an HDF5 scalar that may be ``bytes`` into a ``str``."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class GSLC:
    """A single GSLC granule (lazy)."""

    def __init__(self, path, frequency="A", polarization="HH", chunks=(2048, 2048)):
        self.path = str(path)
        self.frequency = frequency
        self.polarization = polarization
        self._file = h5py.File(self.path, "r")
        grid = self._file[GRID_PATH.format(frequency=frequency)]
        self._grid = grid

        self._dset = grid[polarization]

        # Coordinates and metadata are tiny: read eagerly.
        self.x_coords = grid["xCoordinates"][:]
        self.y_coords = grid["yCoordinates"][:]
        self.x_spacing = float(grid["xCoordinateSpacing"][()])
        self.y_spacing = float(grid["yCoordinateSpacing"][()])
        self.epsg = int(grid["projection"].attrs["epsg_code"])

        ident = self._file[IDENT_PATH]
        self.direction = _decode(ident["orbitPassDirection"][()])
        if "zeroDopplerStartTime" in ident:
            self.datetime = np.datetime64(_decode(ident["zeroDopplerStartTime"][()]))
        else:
            self.datetime = None

        if self.direction not in ("Ascending", "Descending"):
            raise ValueError(f"Unexpected orbitPassDirection: {self.direction!r}")

        # x must be ascending; y monotonic in the expected pass direction.
        if not _strictly_monotonic(self.x_coords):
            raise ValueError("xCoordinates are not strictly monotonic")
        if not _strictly_monotonic(self.y_coords):
            raise ValueError("yCoordinates are not strictly monotonic")

        # Align dask chunks to the file's internal HDF5 chunking when present,
        # so each dask block read decompresses whole HDF5 chunks.
        self.chunks = _aligned_chunks(self._dset, chunks)

    @property
    def shape(self):
        return self._dset.shape

    @property
    def data(self):
        """Lazy, CRS-aware DataArray ``(y, x)`` of the complex image."""
        # lock=True serialises reads on this file handle (h5py is not
        # thread-safe per handle); different GSLCs still read in parallel.
        arr = da.from_array(self._dset, chunks=self.chunks, lock=True)
        data = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"x": self.x_coords, "y": self.y_coords},
            name=self.polarization,
        )
        return data.rio.write_crs(f"EPSG:{self.epsg}")

    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """Lazily crop to a lon/lat bounding box. Returns a DataArray."""
        x_min, x_max, y_min, y_max = geo.bbox_to_native(
            lon_min, lon_max, lat_min, lat_max, self.epsg
        )
        # Match the slice direction to the coordinate ordering.
        x_slice = (
            slice(x_min, x_max)
            if self.x_coords[0] <= self.x_coords[-1]
            else slice(x_max, x_min)
        )
        y_slice = (
            slice(y_min, y_max)
            if self.y_coords[0] <= self.y_coords[-1]
            else slice(y_max, y_min)
        )
        return self.data.sel(x=x_slice, y=y_slice)

    def close(self):
        """Close the underlying HDF5 file. Persist or compute first."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        date = "" if self.datetime is None else f" {self.datetime}"
        return (
            f"<GSLC {self.polarization} freq{self.frequency} "
            f"EPSG:{self.epsg} {self.direction}{date} shape={self.shape}>"
        )


def _strictly_monotonic(a):
    d = np.diff(a)
    return bool(np.all(d > 0) or np.all(d < 0))


def _aligned_chunks(dset, target):
    """Round target chunks to multiples of the HDF5 internal chunks, capped
    at the array size."""
    ny, nx = dset.shape
    ty, tx = target
    file_chunks = getattr(dset, "chunks", None)
    if file_chunks:
        cy, cx = file_chunks
        ty = max(cy, (ty // cy) * cy) if cy else ty
        tx = max(cx, (tx // cx) * cx) if cx else tx
    return (min(ty, ny), min(tx, nx))
