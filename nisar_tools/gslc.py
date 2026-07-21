"""The :class:`GSLC` class: one NISAR GSLC HDF5 granule, read lazily.

A ``GSLC`` keeps its HDF5 file open for its whole lifetime and exposes the
complex image as a chunked, dask-backed :class:`xarray.DataArray`. Nothing is
read from disk until the graph is computed or persisted, so cropping a tiny
region out of a multi-GB granule only ever reads that region.

Lifetime note: because the dask graph holds references to the open ``h5py``
dataset, you must keep the ``GSLC`` (and thus its file handle) alive until the
data is computed or persisted. Persisting a stack reopens it from Zarr and
severs this dependency, after which the granules can be closed.

Reads go through :class:`DirectChunkReader` where the granule's layout allows
it, because h5py serialises every call on one global lock and NISAR GSLCs are
gzip-compressed: decoding through h5py is single-core no matter how many threads
(or granules) are in play. See that class for the details.
"""

import zlib

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr

from . import geo

GRID_PATH = "science/LSAR/GSLC/grids/frequency{frequency}"
IDENT_PATH = "science/LSAR/identification"

# HDF5 filter codes, in the order they are applied on write.
_H5_DEFLATE = "gzip"


def _decode(value):
    """Decode an HDF5 scalar that may be ``bytes`` into a ``str``."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _is_directly_decodable(dset):
    """True if ``dset``'s filter pipeline is one :class:`DirectChunkReader` inverts.

    Deliberately narrow: anything unexpected falls back to plain h5py rather
    than risking a wrong decode.
    """
    if dset.chunks is None or len(dset.chunks) != 2:
        return False
    if dset.compression != _H5_DEFLATE:
        return False
    if dset.fletcher32 or dset.scaleoffset is not None:
        return False
    if dset.dtype.itemsize <= 0 or dset.dtype.hasobject:
        return False
    return True


class DirectChunkReader:
    """Array-like over a gzip-compressed, chunked 2D HDF5 dataset.

    Exists purely for speed. h5py holds a single global lock across every call,
    including calls on different file handles and different files, so the gzip
    inflate that dominates a GSLC read is effectively single-core: measured on a
    real granule, eight threads read at 429 MB/s versus 398 MB/s for one, and
    three *different* granules read concurrently gained 1%.

    The fix is to keep only the I/O inside h5py. ``read_direct_chunk`` hands back
    a chunk's raw compressed bytes without decoding them, and both
    ``zlib.decompress`` and the unshuffle release the GIL, so dask's own worker
    threads parallelise the expensive part. No thread pool is needed here — one
    dask task decodes its own blocks. Measured 3.2x end to end.

    Output is byte-identical to indexing the dataset through h5py.
    """

    def __init__(self, dset):
        if not _is_directly_decodable(dset):
            raise ValueError("dataset's filter pipeline is not directly decodable")
        self._dset = dset
        self.shape = dset.shape
        self.dtype = dset.dtype
        self.ndim = dset.ndim
        self._cy, self._cx = dset.chunks
        self._fill = dset.fillvalue
        # HDF5 applies shuffle before deflate, so decoding reverses the order.
        if dset.shuffle:
            import numcodecs

            self._unshuffle = numcodecs.Shuffle(elementsize=dset.dtype.itemsize).decode
        else:
            self._unshuffle = None

    def _read_chunk(self, cy, cx):
        """Decode one whole HDF5 chunk, or synthesise it if never written."""
        offset = (cy * self._cy, cx * self._cx)
        # Outside the swath (~45% of a granule) chunks are never allocated;
        # read_direct_chunk raises for those, so test before asking.
        if self._dset.id.get_chunk_info_by_coord(offset).byte_offset is None:
            return np.full((self._cy, self._cx), self._fill, dtype=self.dtype)
        raw = self._dset.id.read_direct_chunk(offset)[1]
        buf = zlib.decompress(raw)
        if self._unshuffle is not None:
            buf = self._unshuffle(buf)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self._cy, self._cx)

    def __getitem__(self, key):
        ys, xs = key
        ny, nx = self.shape
        y0, y1 = (ys.start or 0), (ny if ys.stop is None else min(ys.stop, ny))
        x0, x1 = (xs.start or 0), (nx if xs.stop is None else min(xs.stop, nx))
        out = np.empty((max(y1 - y0, 0), max(x1 - x0, 0)), dtype=self.dtype)
        if out.size == 0:
            return out

        for cy in range(y0 // self._cy, (y1 - 1) // self._cy + 1):
            base_y = cy * self._cy
            for cx in range(x0 // self._cx, (x1 - 1) // self._cx + 1):
                base_x = cx * self._cx
                chunk = self._read_chunk(cy, cx)
                # Overlap of this chunk with the requested window. Edge chunks
                # are stored full-size and padded, so clip to the dataset too.
                ty0, ty1 = max(y0, base_y), min(y1, base_y + self._cy)
                tx0, tx1 = max(x0, base_x), min(x1, base_x + self._cx)
                out[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = chunk[
                    ty0 - base_y:ty1 - base_y, tx0 - base_x:tx1 - base_x
                ]
        return out


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
        self._direct = None  # lazily resolved by _reader()

    @property
    def shape(self):
        return self._dset.shape

    def _reader(self):
        """A :class:`DirectChunkReader` for this granule, or None to use h5py.

        Cached, because the reader holds the unshuffle codec. Any granule whose
        filter pipeline we cannot invert falls back to plain h5py indexing.
        """
        if self._direct is None:
            try:
                self._direct = DirectChunkReader(self._dset)
            except (ValueError, AttributeError, ImportError):
                self._direct = False
        return self._direct or None

    @property
    def data(self):
        """Lazy, CRS-aware DataArray ``(y, x)`` of the complex image."""
        reader = self._reader()
        if reader is not None:
            # DirectChunkReader keeps the gzip inflate outside h5py's lock, so
            # dask's worker threads can actually overlap; it needs no lock of
            # its own because h5py already guards the raw chunk reads.
            arr = da.from_array(reader, chunks=self.chunks, lock=False)
        else:
            # h5py serialises on one global lock across every handle and file,
            # so this path reads at one core's decode rate no matter how many
            # threads or granules are in play. lock=True adds no further cost.
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
