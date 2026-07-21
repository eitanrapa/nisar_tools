"""Shared base for stack-like objects wrapping a lazy xarray Dataset."""

import xarray as xr

# On-disk / in-memory spatial chunk size (complex64 2048^2 ~= 32 MB).
SPATIAL_CHUNK = 2048


class RasterStackMixin:
    """Common accessors for objects backed by an ``xr.Dataset``."""

    ds: xr.Dataset

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
