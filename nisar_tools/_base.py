"""Shared base for stack-like objects wrapping a lazy xarray Dataset."""

from pathlib import Path

import numpy as np
import xarray as xr

# On-disk / in-memory spatial chunk size (complex64 2048^2 ~= 32 MB).
SPATIAL_CHUNK = 2048


def wrapped_phase(da):
    """Wrapped phase (radians in ``[-pi, pi]``) of a complex DataArray.

    Stays lazy for dask-backed input; ``float32`` output. Used to derive the
    real ``phase`` field of an SLC or interferogram for ``.grd`` export.
    """
    return xr.apply_ufunc(
        lambda z: np.angle(z).astype(np.float32),
        da,
        dask="parallelized",
        output_dtypes=[np.float32],
    )


def open_stage(path):
    """Open a stage's Zarr store, restoring the CRS coordinate.

    Zarr does not distinguish coordinates from data variables, so the
    ``spatial_ref`` that :meth:`~xarray.Dataset.rio.write_crs` wrote as a
    coordinate comes back as a *variable*. rioxarray then stops recognising the
    CRS, and every field of a reopened stack reports ``rio.crs is None`` --
    which surfaces far downstream as "Provide a CRS-aware DataArray" from
    anything that reprojects, such as exporting to lon/lat.
    """
    ds = xr.open_zarr(path)
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    return ds


class RasterStackMixin:
    """Common accessors for objects backed by an ``xr.Dataset``."""

    ds: xr.Dataset

    #: Dimension along which :meth:`to_grd` writes one ``.grd`` file per slice.
    GRD_STACK_DIM = "pair"
    #: Field names :meth:`to_grd` writes when ``fields`` is not given; ``None``
    #: means every field :meth:`_grd_specs` offers.
    GRD_DEFAULT_FIELDS = None

    @property
    def epsg(self):
        return int(self.ds.attrs["epsg"])

    @property
    def direction(self):
        return self.ds.attrs.get("direction")

    @property
    def x(self):
        return self.ds["x"].values

    @property
    def y(self):
        return self.ds["y"].values

    @property
    def sizes(self):
        return dict(self.ds.sizes)

    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """Return a new, lazily cropped stack of the same type.

        Available at every stage, so a merged union grid or a swath edge can be
        trimmed away after interferograms are formed, not only before.
        """
        from . import geo  # local: geo imports rioxarray, and stages import geo

        x_min, x_max, y_min, y_max = geo.bbox_to_native(
            lon_min, lon_max, lat_min, lat_max, self.epsg
        )
        x = self.x
        y = self.y
        x_slice = slice(x_min, x_max) if x[0] <= x[-1] else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y[0] <= y[-1] else slice(y_max, y_min)
        out = self.ds.sel(x=x_slice, y=y_slice)
        out.attrs.update(self.ds.attrs)
        return type(self)(out)

    def disk_chunks(self, stack_dim):
        return {stack_dim: 1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}

    # -- export ------------------------------------------------------------
    def to_grd(self, outdir, fields=None, indices=None):
        """Export this stack's fields to GMT-readable ``.grd`` grids.

        Each field is reprojected to lon/lat and written as a single-variable
        GMT grid (see :func:`nisar_tools.geo.write_grd`). A field carrying the
        stack dimension is written **one file per slice**, named
        ``{field}_{dim}{i}.grd`` (``dim`` is ``pair`` or ``time``); a shared 2-D
        field (e.g. a ``LOSStack``'s look geometry) is written once as
        ``{field}.grd``. Complex data -- an SLC, an interferogram -- is split
        into an ``amplitude`` and a wrapped ``phase`` field.

        ``fields`` selects which fields to write by name; the default set is
        per stage (see each class's :meth:`_grd_specs`), and passing an unknown
        name raises with the available menu. ``indices`` selects which slices
        along the stack dimension to write for the stacked fields (default:
        all). ``outdir`` is created if needed. Returns the written paths.
        """
        from . import geo  # local: geo pulls in rioxarray (registers .rio)

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = []
        for stem, da in self._grd_layers(fields, indices):
            if da.rio.crs is None:  # derived amp/phase can drop the CRS coord
                da = da.rio.write_crs(f"EPSG:{self.epsg}")
            written.append(geo.write_grd(da, outdir / f"{stem}.grd"))
        return written

    def _grd_layers(self, fields, indices):
        """Yield ``(filename_stem, 2-D DataArray)`` for each field to export."""
        specs = {name: (da, stacked) for name, da, stacked in self._grd_specs()}
        if fields is None:
            fields = self.GRD_DEFAULT_FIELDS
            if fields is None:
                fields = list(specs)
        dim = self.GRD_STACK_DIM
        for name in fields:
            try:
                da, stacked = specs[name]
            except KeyError:
                raise KeyError(
                    f"{type(self).__name__}.to_grd: unknown field {name!r}; "
                    f"available: {sorted(specs)}"
                ) from None
            if not stacked:
                yield name, da
                continue
            idx = range(self.ds.sizes[dim]) if indices is None else indices
            for i in idx:
                yield f"{name}_{dim}{i}", da.isel({dim: i})

    def _grd_specs(self):
        """Return ``(field_name, DataArray, is_stacked)`` for every exportable
        field. ``is_stacked`` fields carry :attr:`GRD_STACK_DIM` and are written
        per slice. Subclasses implementing ``.grd`` export override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support .grd export"
        )
