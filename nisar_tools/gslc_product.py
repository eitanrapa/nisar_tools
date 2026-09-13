"""Write NISAR L2 GSLC products (spec 1.5.0) as ISCE3 0.25.16 lays them out.

The layout is not guessed: every dataset path, dtype, filter pipeline, fill value
and attribute type here reproduces what ISCE3's GSLC workflow and ``GslcWriter``
write, and :func:`decorate_from_spec` applies the official product
specification XML (``data/nisar_L2_GSLC.xml``, copied verbatim from ISCE3
v0.25.16, Apache-2.0) exactly as ``check_and_decorate_product_using_specs_xml``
does. ``scripts/gslc_format_diff.py`` checks a product written here against an
ASF-distributed GSLC.

Traps this reproduces on purpose, because a reader may depend on them:

- Every string attribute is fixed-length ``bytes``, never ``str``.
- The image and ``mask`` datasets carry a real HDF5 fill value (NaN / 255);
  metadata LUTs carry only a ``_FillValue`` *attribute* over a zero fill.
- ``frequencyB/inputDataExceptionMask`` is on **frequency A's** grid: ISCE3's
  writer geocodes it without a frequency, which defaults to the first one.
- Cube layers get a float64 NaN ``_FillValue`` whatever their dtype, and
  float32 ``valid_min``/``valid_max`` (the spec writes those in the dataset's
  own type).

Image chunks are written with ``write_direct_chunk`` after shuffle + deflate in
the caller's thread -- the mirror image of :class:`nisar_tools.gslc.DirectChunkReader`
-- because h5py compresses under its global lock and a frame is ~10 GB of
compressed data per polarization.
"""

import threading
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import h5py
import numpy as np

from . import radar

PRODUCT = "/science/LSAR/GSLC"
RSLC_PRODUCT = "/science/LSAR/RSLC"
IDENT = "/science/LSAR/identification"
SPEC_XML = Path(__file__).with_name("data") / "nisar_L2_GSLC.xml"
SPEC_VERSION = "1.5.0"

CHUNK = 512
PAGE_SIZE = 4 * 1024 * 1024
LEXICOGRAPHIC_BASE_POLS = ["HH", "HV", "VH", "VV"]
COMPACT_POLS = ["RH", "RV"]
CROSSTALK_DATASETS = ["txHorizontalCrosspol", "txVerticalCrosspol",
                      "rxHorizontalCrosspol", "rxVerticalCrosspol"]

#: ``cf_units_dict`` in ISCE3's writer: spec units -> CF/UDUNITS spelling.
CF_UNITS = {
    "1": "1", "DN": "1", "unitless": "1", "seconds": "seconds", "meters": "meters",
    "degrees": "degrees", "hertz": "hertz", "radians": "radians",
    "radians^-1": "radians^-1", "meters / second": "meters / second",
    "meters per second": "meters / second", "meters / second ^ 2": "meters / second ^ 2",
    "meters per second squared": "meters / second ^ 2",
    "radians per second": "radians / second",
}


# ---------------------------------------------------------------------------
# File and small-value helpers
# ---------------------------------------------------------------------------
def create_product_file(path):
    """Open a new product with ISCE3's paged file-space strategy (4 MiB pages).

    Also commits the ``complex64`` named datatype at the root, as
    ``get_complex_output_dtype`` does "to teach tools like GDAL/netCDF" about it.
    """
    f = h5py.File(str(path), "w", fs_strategy="page", fs_persist=True,
                  fs_threshold=1, fs_page_size=PAGE_SIZE)
    ctype = h5py.h5t.py_create(np.dtype(np.complex64))
    ctype.commit(f["/"].id, b"complex64")
    return f


def _to_bytes(value):
    if isinstance(value, bytes):
        return np.bytes_(value)
    return np.bytes_(str(value).encode("utf-8"))


def set_value(f, path, data, format_function=None):
    """``BaseWriterSingleInput.set_value``: strings/bools become fixed bytes."""
    if isinstance(data, (np.bytes_, np.ndarray)) and getattr(data, "dtype", None) is not None \
            and data.dtype.char == "S" and data.shape == ():
        data = np.bytes_(data).decode(errors="replace").rstrip("\x00")
    elif isinstance(data, np.ndarray) and data.dtype.char == "O":
        data = str(data)
    elif isinstance(data, bytes):
        data = data.decode(errors="replace").rstrip("\x00")
    if format_function is not None:
        data = format_function(data)
    if path in f:
        del f[path]
    if isinstance(data, str):
        return f.create_dataset(path, data=_to_bytes(data))
    if isinstance(data, (bool, np.bool_)):
        return f.create_dataset(path, data=np.bytes_(str(bool(data))))
    if isinstance(data, (list, tuple)) and data and all(isinstance(x, str) for x in data):
        return f.create_dataset(path, data=np.array([x.encode() for x in data]))
    return f.create_dataset(path, data=data)


def copy_from(src, f, dst_path, src_path=None, default=None,
              skip_if_not_present=False, format_function=None):
    """``copy_from_input``: value and every attribute (str attributes -> bytes)."""
    src_path = src_path or dst_path
    if src_path not in src:
        if default is None:
            if skip_if_not_present:
                return None
            raise KeyError(f"not in input product: {src_path}")
        return set_value(f, dst_path, default, format_function)
    ds = src[src_path]
    if h5py.check_string_dtype(ds.dtype) and ds.shape == ():
        data = str(ds.asstr()[()])
    else:
        data = ds[()]
    out = set_value(f, dst_path, data, format_function)
    for key, value in ds.attrs.items():
        if key in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"):
            continue
        out.attrs[key] = np.bytes_(value) if isinstance(value, str) else value
    return out


def projection_attrs(epsg):
    """``build_projection_dataset_attrs_dict`` (UTM, polar stereographic, 4326)."""
    from osgeo import osr

    osr.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    proj_name = srs.GetAttrValue("PROJECTION")
    attrs = dict(
        epsg_code=np.int64(epsg),
        spatial_ref=np.bytes_(srs.ExportToWkt()),
        grid_mapping_name=np.bytes_((proj_name or "latitude_longitude").lower()),
        semi_major_axis=np.float64(6378137.0),
        inverse_flattening=np.float64(298.257223563),
        ellipsoid=np.bytes_("WGS84"),
    )
    if epsg == 4326:
        attrs["longitude_of_prime_meridian"] = np.float64(0.0)
        return attrs
    attrs["false_easting"] = np.float64(srs.GetProjParm(osr.SRS_PP_FALSE_EASTING))
    attrs["false_northing"] = np.float64(srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING))
    attrs["longitude_of_projection_origin"] = np.float64(
        srs.GetProjParm(osr.SRS_PP_LONGITUDE_OF_ORIGIN))
    if epsg in (3413, 3031):
        north = epsg == 3413
        attrs["latitude_of_projection_origin"] = np.float64(90.0 if north else -90.0)
        attrs["standard_parallel"] = np.float64(70.0 if north else -71.0)
        attrs["straight_vertical_longitude_from_pole"] = np.float64(-45.0 if north else 0.0)
        return attrs
    attrs["latitude_of_projection_origin"] = np.float64(
        srs.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN))
    if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
        attrs["utm_zone_number"] = np.int64(epsg % 100)
        attrs["longitude_of_central_meridian"] = np.float64(
            srs.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN))
        attrs["scale_factor_at_central_meridian"] = np.float64(
            srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR))
        return attrs
    raise NotImplementedError(f"EPSG {epsg} is not a NISAR L2 projection")


def set_geo_info(f, root, grid, heights=None, cube=False, save_spacing=True):
    """``set_get_geo_info``: coordinates, spacings and the projection dataset.

    Returns ``(yds, xds)``, or ``(zds, yds, xds)`` with ``heights``.
    """
    group = f.require_group(root)
    units = "meters" if grid.epsg != 4326 else None
    x_units = units or "degree_east"
    y_units = units or "degree_north"
    x_std = "projection_x_coordinate" if units else "longitude"
    y_std = "projection_y_coordinate" if units else "latitude"

    if not cube and save_spacing:
        for axis, spacing, unit, what in (("x", grid.spacing_x, x_units, "pixels"),
                                          ("y", grid.spacing_y, y_units, "lines")):
            name = f"{axis}CoordinateSpacing"
            if name not in group:
                ds = group.create_dataset(name, data=np.float64(spacing))
                ds.attrs["description"] = np.bytes_(
                    f"Nominal spacing in {unit} between consecutive {what}")
                ds.attrs["units"] = np.bytes_(unit)
                ds.attrs["long_name"] = np.bytes_(f"{axis.upper()} coordinates spacing")

    datasets = []
    for axis, vec, unit, std in (("x", grid.x_coordinates, x_units, x_std),
                                 ("y", grid.y_coordinates, y_units, y_std)):
        name = f"{axis}Coordinates"
        if name in group:
            ds = group[name]
        else:
            ds = group.create_dataset(name, data=vec.astype(np.float64))
            ds.attrs["standard_name"] = np.bytes_(std)
            ds.attrs["description"] = np.bytes_(f"{axis.upper()} coordinates in specified projection")
            ds.attrs["units"] = np.bytes_(unit)
            ds.attrs["long_name"] = np.bytes_(f"{axis.upper()} coordinates of projection")
            ds.make_scale()
        datasets.append(ds)
    xds, yds = datasets

    zds = None
    if heights is not None:
        if "heightAboveEllipsoid" in group:
            zds = group["heightAboveEllipsoid"]
        else:
            zds = group.create_dataset("heightAboveEllipsoid",
                                       data=np.asarray(heights, np.float64))
            zds.attrs["standard_name"] = np.bytes_("height_above_reference_ellipsoid")
            zds.attrs["description"] = np.bytes_(
                "Height values above WGS84 Ellipsoid corresponding to the radar grid")
            zds.attrs["units"] = np.bytes_("meters")
            zds.make_scale()

    if "projection" not in group:
        proj = group.create_dataset("projection", data=np.uint32(grid.epsg))
        for key, value in projection_attrs(grid.epsg).items():
            proj.attrs[key] = value
        proj.attrs["description"] = np.bytes_(
            "Product map grid projection: EPSG code, with additional projection "
            "information as HDF5 Attributes")
    if zds is not None:
        return zds, yds, xds
    return yds, xds


# ---------------------------------------------------------------------------
# Image datasets and direct chunk writing
# ---------------------------------------------------------------------------
def _chunks(shape, chunk=CHUNK):
    return tuple(min(chunk, s) for s in shape)


def create_image_datasets(f, frequency, grid, polarizations, compression_level=1):
    """``prep_gslc_dataset`` for one frequency: images, mask, coordinates."""
    root = f"{PRODUCT}/grids/frequency{frequency}"
    yds, xds = set_geo_info(f, root, grid)
    group = f[root]
    shape = (grid.length, grid.width)
    chunks = _chunks(shape)
    # Plain complex64, not the committed /complex64 type: ASF's images do not
    # link to it either.
    for pol in polarizations:
        ds = group.create_dataset(
            pol, shape=shape, dtype=np.complex64, chunks=chunks, compression="gzip",
            compression_opts=compression_level, shuffle=True,
            fillvalue=np.complex64(np.nan + 1j * np.nan))
        ds.attrs["description"] = np.bytes_(f"Focused SLC image ({pol})")
        ds.attrs["units"] = np.bytes_("")
        ds.attrs["grid_mapping"] = np.bytes_("projection")
        ds.attrs["long_name"] = np.bytes_(f"Geocoded single-look complex image {pol}")
        ds.dims[0].attach_scale(yds)
        ds.dims[1].attach_scale(xds)
        ds.attrs["_FillValue"] = np.complex64(np.nan + 1j * np.nan)
    mask = group.create_dataset(
        "mask", shape=shape, dtype=np.uint8, chunks=chunks, compression="gzip",
        compression_opts=compression_level, shuffle=True, fillvalue=255)
    mask.attrs["description"] = np.bytes_("GSLC mask")
    mask.attrs["grid_mapping"] = np.bytes_("projection")
    mask.attrs["long_name"] = np.bytes_("Geocoded mask of single-look complex image")
    mask.dims[0].attach_scale(yds)
    mask.dims[1].attach_scale(xds)
    mask.attrs["_FillValue"] = np.int64(255)
    mask.attrs["valid_min"] = 0
    pols = group.create_dataset("listOfPolarizations",
                                data=np.array(polarizations, dtype="S2"))
    pols.attrs["description"] = np.bytes_(
        f"List of processed polarization layers with frequency {frequency}")
    return group


class ChunkWriter:
    """Compress 2-D chunks off the h5py lock, then ``write_direct_chunk`` them.

    HDF5 applies shuffle (by element size) then deflate on write; a chunk is
    always stored at full chunk size, so edge chunks are padded with the
    dataset's fill value first. Chunks that are entirely fill are skipped --
    HDF5 returns the fill value for unallocated chunks, so they read back the
    same while costing nothing on disk.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.chunks = dataset.chunks
        self.dtype = dataset.dtype
        self.level = dataset.compression_opts if dataset.compression == "gzip" else None
        self.fill = np.array(dataset.fillvalue, dtype=self.dtype)
        self._shuffle = None
        if dataset.shuffle and self.dtype.itemsize > 1:
            import numcodecs
            self._shuffle = numcodecs.Shuffle(elementsize=self.dtype.itemsize)
        self._lock = _H5_WRITE_LOCK

    def _is_fill(self, block):
        if self.dtype.kind in "fc" and np.isnan(self.fill).all():
            # np.isnan of a complex is true when *either* part is NaN; a chunk
            # of nan+0j is not the nan+nanj fill and must be written.
            if self.dtype.kind == "c":
                return bool(np.isnan(block.real).all() and np.isnan(block.imag).all())
            return bool(np.isnan(block).all())
        return bool((block == self.fill).all())

    def write(self, row, col, block):
        """Write the chunk whose top-left pixel is ``(row, col)``."""
        cy, cx = self.chunks
        if row % cy or col % cx:
            raise ValueError("ChunkWriter.write needs chunk-aligned offsets")
        block = np.asarray(block, dtype=self.dtype)
        if self._is_fill(block):
            return False
        if block.shape != (cy, cx):
            full = np.full((cy, cx), self.fill, dtype=self.dtype)
            full[:block.shape[0], :block.shape[1]] = block
            block = full
        raw = np.ascontiguousarray(block).tobytes()
        if self._shuffle is not None:
            raw = bytes(self._shuffle.encode(raw))
        if self.level is not None:
            raw = zlib.compress(raw, self.level)
        with self._lock:
            self.dataset.id.write_direct_chunk((row, col), raw)
        return True


#: h5py is not thread-safe for concurrent writes to one file handle.
_H5_WRITE_LOCK = threading.RLock()


class ComplexStats:
    """``StatsRealImagFloat32``, mergeable across chunks.

    Min, max and the sample standard deviation follow ISCE3 exactly. The mean
    is accumulated in float64: ISCE3's is a float32 running mean, which over a
    frame's ~10^9 samples is dominated by its own rounding error.
    """

    def __init__(self):
        self.n = 0
        self.mean = np.zeros(2)
        self.ssq = np.zeros(2)          # sum of squared deviations
        self.min = np.full(2, np.nan)
        self.max = np.full(2, np.nan)
        self._lock = threading.Lock()

    def update(self, values):
        values = np.asarray(values)
        valid = values[~(np.isnan(values.real) | np.isnan(values.imag))]
        if valid.size == 0:
            return
        parts = np.stack([valid.real.astype(np.float64), valid.imag.astype(np.float64)])
        n = valid.size
        mean = parts.mean(axis=1)
        ssq = ((parts - mean[:, None]) ** 2).sum(axis=1)
        mn, mx = parts.min(axis=1), parts.max(axis=1)
        with self._lock:
            if self.n == 0:
                self.n, self.mean, self.ssq, self.min, self.max = n, mean, ssq, mn, mx
                return
            total = self.n + n
            delta = mean - self.mean
            self.mean = self.mean + delta * n / total
            self.ssq = self.ssq + ssq + delta ** 2 * self.n * n / total
            self.min = np.minimum(self.min, mn)
            self.max = np.maximum(self.max, mx)
            self.n = total

    def write(self, dataset):
        std = np.sqrt(self.ssq / (self.n - 1)) if self.n > 1 else np.zeros(2)
        for i, part in enumerate(("real", "imag")):
            dataset.attrs.create(f"min_{part}_value", data=np.float64(np.float32(self.min[i])))
            dataset.attrs.create(f"mean_{part}_value", data=np.float64(np.float32(self.mean[i])))
            dataset.attrs.create(f"max_{part}_value", data=np.float64(np.float32(self.max[i])))
            dataset.attrs.create(f"sample_stddev_{part}", data=np.float64(np.float32(std[i])))


# ---------------------------------------------------------------------------
# Metadata LUT datasets (save_hdf5_dataset)
# ---------------------------------------------------------------------------
def write_lut(f, root, name, data, grid, compute_stats=False, fill_value=None):
    """``save_hdf5_dataset`` for one geocoded LUT band.

    gzip 9, no shuffle, chunks ``min(512, dim)``; attributes ``grid_mapping``,
    ``_FillValue`` (NaN for float, NaN+NaNj for complex, none for integers) and,
    with ``compute_stats``, ISCE3's stats cast to the dataset's own type.
    """
    yds, xds = set_geo_info(f, root, grid, save_spacing=False)
    group = f[root]
    data = np.asarray(data)
    if name in group:
        del group[name]
    ds = group.create_dataset(name, data=data, chunks=_chunks(data.shape),
                              compression="gzip", compression_opts=9, shuffle=False)
    ds.dims[0].attach_scale(yds)
    ds.dims[1].attach_scale(xds)
    ds.attrs["grid_mapping"] = np.bytes_("projection")
    kind = data.dtype.type
    if fill_value is None and np.issubdtype(data.dtype, np.complexfloating):
        fill_value = kind(np.nan + 1j * np.nan)
    elif fill_value is None and np.issubdtype(data.dtype, np.floating):
        fill_value = kind(np.nan)
    if compute_stats:
        if np.issubdtype(data.dtype, np.complexfloating):
            st = ComplexStats()
            st.update(data)
            std = np.sqrt(st.ssq / (st.n - 1)) if st.n > 1 else np.zeros(2)
            for i, part in enumerate(("real", "imag")):
                for label, value in (("min", st.min[i]), ("mean", st.mean[i]),
                                     ("max", st.max[i])):
                    ds.attrs.create(f"{label}_{part}_value", data=kind(value))
                ds.attrs.create(f"sample_stddev_{part}", data=kind(std[i]))
        else:
            finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
            ds.attrs.create("min_value", data=kind(finite.min()) if finite.size else kind(np.nan))
            ds.attrs.create("mean_value", data=kind(finite.mean()) if finite.size else kind(np.nan))
            ds.attrs.create("max_value", data=kind(finite.max()) if finite.size else kind(np.nan))
            ds.attrs.create("sample_stddev", data=kind(finite.std(ddof=1)) if finite.size > 1 else kind(0))
    if fill_value is not None:
        ds.attrs.create("_FillValue", data=fill_value)
    return ds


def create_cube_datasets(f, root, grid, heights, epoch, compression_level=1):
    """``add_radar_grid_cubes_to_hdf5``'s dataset creation. Returns name -> dataset.

    ``epoch`` is the radar grid's reference epoch, for the azimuth-time units.
    """
    zds, yds, xds = set_geo_info(f, root, grid, heights=heights, cube=True)
    group = f[root]
    shape = (len(heights), grid.length, grid.width)
    chunks = (1,) + _chunks(shape[1:])
    specs = {
        "slantRange": dict(long_name="Slant range", descr="Slant range in meters", units="meters"),
        "zeroDopplerAzimuthTime": dict(long_name="Zero Doppler azimuth time",
                                       descr="Zero Doppler azimuth time in seconds since UTC epoch",
                                       units=radar.epoch_units(epoch)),
        "incidenceAngle": dict(long_name="Incidence angle", units="degrees", vmin=0.0, vmax=90.0,
                               descr="Incidence angle is defined as the angle between the LOS vector"
                                     " and the normal to the ellipsoid at the target height"),
        "losUnitVectorX": dict(long_name="LOS unit vector X", units="1", vmin=-1.0, vmax=1.0,
                               descr="East component of the line-of-sight (LOS) unit vector"),
        "losUnitVectorY": dict(long_name="LOS unit vector Y", units="1", vmin=-1.0, vmax=1.0,
                               descr="North component of the line-of-sight (LOS) unit vector"),
        "alongTrackUnitVectorX": dict(long_name="Along-track unit vector X", units="1",
                                      vmin=-1.0, vmax=1.0,
                                      descr="East component of the along-track unit vector"),
        "alongTrackUnitVectorY": dict(long_name="Along-track unit vector Y", units="1",
                                      vmin=-1.0, vmax=1.0,
                                      descr="North component of the along-track unit vector"),
        "elevationAngle": dict(long_name="Elevation angle", units="degrees", vmin=0.0, vmax=90.0,
                               descr="Elevation angle is defined as the angle between the LOS"
                                     " vector and the normal to the ellipsoid at the sensor"),
        "groundTrackVelocity": dict(long_name="Ground track velocity", units="meters / second",
                                    descr="Absolute value of the platform velocity scaled at the"
                                          " target height"),
    }
    out = {}
    for name, dtype in radar.CUBE_LAYERS.items():
        spec = specs[name]
        ds = group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks,
                                  compression="gzip", compression_opts=compression_level,
                                  shuffle=True)
        ds.dims[0].attach_scale(zds)
        ds.dims[1].attach_scale(yds)
        ds.dims[2].attach_scale(xds)
        ds.attrs["grid_mapping"] = np.bytes_("projection")
        ds.attrs["long_name"] = np.bytes_(spec["long_name"])
        ds.attrs["description"] = np.bytes_(spec["descr"])
        if "units" in spec:
            ds.attrs["units"] = np.bytes_(spec["units"])
        ds.attrs.create("_FillValue", data=np.nan)
        if "vmin" in spec:
            ds.attrs.create("valid_min", data=spec["vmin"])
            ds.attrs.create("valid_max", data=spec["vmax"])
        out[name] = ds
    return out


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------
def write_orbit(f, orbit, root=f"{PRODUCT}/metadata/orbit"):
    """``isce3::core::saveToH5(Orbit)`` + the spec's descriptions."""
    group = f.require_group(root)
    for name in ("time", "position", "velocity", "orbitType", "interpMethod"):
        if name in group:
            del group[name]
    t = group.create_dataset("time", data=orbit.time)
    t.attrs["units"] = np.bytes_(radar.epoch_units(orbit.reference_epoch))
    p = group.create_dataset("position", data=orbit.position)
    p.attrs["units"] = np.bytes_("meters")
    v = group.create_dataset("velocity", data=orbit.velocity)
    v.attrs["units"] = np.bytes_("meters / second")
    group.create_dataset("orbitType", data=np.bytes_(orbit.orbit_type or "Undefined"))
    group.create_dataset("interpMethod", data=np.bytes_(orbit.interp_method))
    return group


# ---------------------------------------------------------------------------
# Specification decoration
# ---------------------------------------------------------------------------
def decorate_from_spec(f, spec_path=SPEC_XML):
    """``check_and_decorate_product_using_specs_xml``: root attributes, then
    units / long_name / standard_name / valid range / description per dataset.
    """
    spec = ET.ElementTree(file=str(spec_path))
    for key, value in spec.find("./product/science/annotation").items():
        if key != "app":
            f.attrs[key] = np.bytes_(value)
    for entry in spec.find("./product/science/nodes"):
        path = entry.attrib["name"]
        if path not in f:
            continue
        ds = f[path]
        description = None
        for ann in entry:
            if ann.attrib.get("app") != "conformance":
                continue
            for key, value in ann.items():
                if key == "units":
                    if value in CF_UNITS:
                        ds.attrs["units"] = np.bytes_(CF_UNITS[value])
                elif key in ("long_name", "standard_name"):
                    ds.attrs[key] = np.bytes_(value)
                elif key in ("valid_min", "valid_max"):
                    ds.attrs.create(key, data=value, dtype=ds.dtype)
            if ann.text:
                description = ann.text
        if description:
            ds.attrs["description"] = np.bytes_(description)
