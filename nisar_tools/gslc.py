"""The `GSLC` class: one NISAR GSLC HDF5 granule, read lazily.

A `GSLC` keeps its HDF5 file open for its whole lifetime and exposes the
complex image as a chunked, dask-backed `xarray.DataArray`. Nothing is
read from disk until the graph is computed or persisted, so cropping a tiny
region out of a multi-GB granule only ever reads that region.

Lifetime note: because the dask graph holds references to the open `h5py`
dataset, you must keep the `GSLC` (and thus its file handle) alive until the
data is computed or persisted. Persisting a stack reopens it from Zarr and
severs this dependency, after which the granules can be closed.

Reads go through `DirectChunkReader` where the granule's layout allows
it, because h5py serialises every call on one global lock and NISAR GSLCs are
gzip-compressed: decoding through h5py is single-core no matter how many threads
(or granules) are in play. See that class for the details.
"""

import math
import os
import zlib

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr

from . import geo
from .geometry import SPEED_OF_LIGHT

GRID_PATH = "science/LSAR/GSLC/grids/frequency{frequency}"
IDENT_PATH = "science/LSAR/identification"

# h5py's name for the deflate filter.
_H5_DEFLATE = "gzip"


def _decode(value):
    """Decode an HDF5 scalar that may be `bytes` into a `str`."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _is_directly_decodable(dset):
    """True if `dset`'s filter pipeline is one `DirectChunkReader` inverts.

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
    inflate that dominates a GSLC read is effectively single-core: more threads,
    or several granules read at once, gain almost nothing.

    This keeps only the I/O inside h5py. `read_direct_chunk` hands back
    a chunk's raw compressed bytes without decoding them, and both
    `zlib.decompress` and the unshuffle release the GIL, so dask's own worker
    threads parallelise the expensive part (about 3x faster end to end). No
    thread pool is needed here — one dask task decodes its own blocks.

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
        # Chunks outside the swath (a large part of a granule) are never
        # allocated; read_direct_chunk raises for those, so test before asking.
        if self._dset.id.get_chunk_info_by_coord(offset).byte_offset is None:
            return np.full((self._cy, self._cx), self._fill, dtype=self.dtype)
        raw = self._dset.id.read_direct_chunk(offset)[1]
        buf = zlib.decompress(raw)
        if self._unshuffle is not None:
            buf = self._unshuffle(buf)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self._cy, self._cx)

    def __getitem__(self, key):
        """Index like a 2-D array: slices (with any positive step), integers, or both.

        dask folds a strided read (`data[::4, ::4]`) into the slice it passes here, so a
        step has to be honoured: the window is read whole and then strided.
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 2:
            raise IndexError(f"too many indices for a 2-D dataset: {key!r}")
        key = key + (slice(None),) * (2 - len(key))
        windows, steps, drop = [], [], []
        for axis, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, (int, np.integer)):
                i = int(k) + n if k < 0 else int(k)
                if not 0 <= i < n:
                    raise IndexError(f"index {int(k)} is out of bounds for axis {axis} with size {n}")
                windows.append((i, i + 1))
                steps.append(1)
                drop.append(axis)
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                if step < 1:
                    raise IndexError("DirectChunkReader does not support negative steps")
                windows.append((start, max(start, stop)))
                steps.append(step)
            else:
                raise TypeError(f"DirectChunkReader takes slices and integers, not {type(k).__name__}")
        (y0, y1), (x0, x1) = windows
        out = self._read_window(y0, y1, x0, x1)
        if steps != [1, 1]:
            out = out[::steps[0], ::steps[1]].copy()
        return out.squeeze(axis=tuple(drop)) if drop else out

    def _read_window(self, y0, y1, x0, x1):
        """The contiguous window `[y0:y1, x0:x1]`, decoded chunk by chunk."""
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
    """A single GSLC granule (lazy).

    Opens one `frequency` ("A" or "B") and `polarization` of the granule at
    `path`; `chunks` is the dask read-block size, aligned to the file's HDF5
    chunking.

    Attributes
    ----------
    path : str
        The granule file.
    frequency : str
        The band, `"A"` or `"B"`.
    polarization : str
        The polarization, such as `"HH"`.
    x_coords : ndarray
        x of each column, the granule's `xCoordinates`.
    y_coords : ndarray
        y of each row, the granule's `yCoordinates`.
    x_spacing : float
        Column spacing.
    y_spacing : float
        Row spacing, negative for a north-up grid.
    epsg : int
        The granule's projection.
    center_frequency : float or None
        The band's centre frequency, in Hz.
    direction : str
        The orbit pass direction, `"Ascending"` or `"Descending"`.
    datetime : numpy.datetime64 or None
        The zero-Doppler start time.
    chunks : tuple
        The dask read-block size.
    """

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

        # The sub-band's own carrier. Real granules always carry it; a stripped
        # test fixture may not, so an absent one is None rather than an error --
        # only the dual-frequency ionosphere correction needs it, and that says
        # so itself. Frequencies A and B differ (e.g. 1.239 vs 1.2935 GHz),
        # which is what makes that correction possible.
        self.center_frequency = (
            float(grid["centerFrequency"][()]) if "centerFrequency" in grid else None
        )

        ident = self._file[IDENT_PATH]
        self.direction = _decode(ident["orbitPassDirection"][()])
        if "zeroDopplerStartTime" in ident:
            self.datetime = np.datetime64(_decode(ident["zeroDopplerStartTime"][()]))
        else:
            self.datetime = None

        if self.direction not in ("Ascending", "Descending"):
            raise ValueError(f"Unexpected orbitPassDirection: {self.direction!r}")

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
        """Shape of the image, `(y, x)`."""
        return self._dset.shape

    @property
    def pixel_convention(self):
        """`"isce3"`, or `"fixed"` for a GSLC geocoded with `isce3_compatible=False`.

        ASF's GSLCs, and `geocode_rslc`'s by default, carry ISCE3 v0.25.16's
        pixels. The fixed ones interpolate the DEM and the Doppler differently,
        which changes the phase, so the two do not pair.
        """
        path = "science/LSAR/GSLC/metadata/processingInformation/parameters/runConfigurationContents"
        if path not in self._file:
            return "isce3"
        text = _decode(self._file[path][()])
        ours = text.startswith("# written by nisar_tools.gslc_geocode.geocode_rslc")
        return "fixed" if ours and "\n    isce3_compatible: False\n" in text else "isce3"

    @property
    def wavelength(self):
        """Radar wavelength (m) of this sub-band, or None if unrecorded."""
        if self.center_frequency is None:
            return None
        return SPEED_OF_LIGHT / self.center_frequency

    def _reader(self):
        """A `DirectChunkReader` for this granule, or None to use h5py.

        Cached, because the reader holds the unshuffle codec. Any granule whose
        filter pipeline we cannot invert falls back to plain h5py indexing.
        """
        if self._direct is None:
            try:
                self._direct = DirectChunkReader(self._dset)
            except (ValueError, AttributeError, ImportError):
                self._direct = False
        return self._direct or None

    def read_chunks(self, ny, nx, target_blocks=None):
        """Read-block size for an `(ny, nx)` region of this granule.

        How much one HDF5 read task decompresses is a *separate* question from how
        the result is stored (`SPATIAL_CHUNK` in `_base`). `crop` slices an
        already-chunked array, so with 2048-px read blocks a crop a few thousand
        pixels across would be only a few dask blocks, and most cores would sit
        idle while others ran the gzip inflate.

        The block is sized to give `target_blocks` of them over the region
        (default four per core), rounded *down* to a multiple of the file's own
        chunking and never above `self.chunks`. Not smaller than that: the
        downstream multilook overlaps by `4 * looks` (see
        `_kernels._overlap_depth`), so at `looks=30` a 512-px block does 2.2x
        the work of its interior against 1.25x for a 2048-px one. Enough blocks to
        fill the cores, no more.
        """
        if target_blocks is None:
            target_blocks = 4 * (os.cpu_count() or 1)
        file_chunks = getattr(self._dset, "chunks", None) or (1, 1)
        cap_y, cap_x = self.chunks
        side = max(1, math.isqrt(max(1, (int(ny) * int(nx)) // max(1, target_blocks))))

        def _align(value, unit, cap):
            return min(cap, max(unit, (value // unit) * unit))

        return (_align(side, file_chunks[0], cap_y),
                _align(side, file_chunks[1], cap_x))

    @property
    def data(self):
        """Lazy, CRS-aware DataArray `(y, x)` of the complex image."""
        return self.data_chunked(self.chunks)

    def data_chunked(self, chunks):
        """`data` with an explicit read-block size.

        Separate from `data` because the block size has to be fixed at
        `da.from_array` time -- rechunking afterwards only splits blocks that
        have already been read whole, so it buys no read parallelism.
        """
        reader = self._reader()
        if reader is not None:
            # DirectChunkReader keeps the gzip inflate outside h5py's lock, so
            # dask's worker threads can actually overlap; it needs no lock of
            # its own because h5py already guards the raw chunk reads.
            arr = da.from_array(reader, chunks=chunks, lock=False)
        else:
            # h5py serialises on one global lock across every handle and file,
            # so this path reads at one core's decode rate no matter how many
            # threads or granules are in play. lock=True adds no further cost.
            arr = da.from_array(self._dset, chunks=chunks, lock=True)
        data = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"x": self.x_coords, "y": self.y_coords},
            name=self.polarization,
        )
        return data.rio.write_crs(f"EPSG:{self.epsg}")

    def crop(self, lon_min, lon_max, lat_min, lat_max, target_blocks=None):
        """Lazily crop to a lon/lat bounding box. Returns a DataArray.

        The read blocks are sized for the *crop*, not the granule (see
        `read_chunks`), so a small bbox still splits into enough dask tasks
        to keep every core inflating.
        """
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
        # Size the blocks from the crop's extent, then slice. Selecting on the
        # coordinates (which are already in memory) is what resolves the bbox to a
        # pixel count; no data is read.
        extent = (
            int(((self.y_coords >= min(y_min, y_max))
                 & (self.y_coords <= max(y_min, y_max))).sum()),
            int(((self.x_coords >= min(x_min, x_max))
                 & (self.x_coords <= max(x_min, x_max))).sum()),
        )
        chunks = self.read_chunks(extent[0], extent[1], target_blocks)
        return self.data_chunked(chunks).sel(x=x_slice, y=y_slice)

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
