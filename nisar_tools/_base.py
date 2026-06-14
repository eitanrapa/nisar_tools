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

    def disk_chunks(self, stack_dim):
        return {stack_dim: 1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
