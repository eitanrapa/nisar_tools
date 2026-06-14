"""Plotting helpers for wrapped and unwrapped phase.

Both functions reproject a single 2D georeferenced slice to lon/lat and render
it. They are eager by design and must only be handed one (already multilooked,
hence small) slice at a time; they never trigger a whole-stack compute.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import geo


def _to_latlon(data, epsg_code):
    if isinstance(data, xr.DataArray):
        return geo.project_to_latlon(data if data.rio.crs is not None
                                     else data.rio.write_crs(f"EPSG:{epsg_code}"))
    raise TypeError("Expected an xarray.DataArray")


def plot_wrapped_phase(igram, epsg_code=None):
    """Plot wrapped interferogram phase on a cyclic colormap, ``[-pi, pi]``."""
    phase = np.angle(igram)
    if isinstance(igram, xr.DataArray):
        phase = igram.copy(data=phase)
        if phase.rio.crs is None and epsg_code is not None:
            phase = phase.rio.write_crs(f"EPSG:{epsg_code}")
    phase_latlon = geo.project_to_latlon(phase)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    phase_latlon.plot.imshow(
        ax=ax,
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        cbar_kwargs={"label": "Phase (Radians)", "shrink": 0.8},
    )
    ax.set_title("Wrapped Phase", fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(color="gray", linestyle="--", alpha=0.5)
    return fig, ax


def plot_unwrapped_phase(unw, epsg_code=None):
    """Plot unwrapped phase on a diverging colormap."""
    phase_latlon = _to_latlon(unw, epsg_code)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    phase_latlon.plot.imshow(
        ax=ax,
        cmap="RdBu_r",
        cbar_kwargs={"label": "Phase (Radians)", "shrink": 0.8},
    )
    ax.set_title("Unwrapped Phase", fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(color="gray", linestyle="--", alpha=0.5)
    return fig, ax
