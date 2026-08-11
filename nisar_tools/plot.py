"""Plotting helpers for wrapped and unwrapped phase, and for offset fields.

Every function reprojects a single 2D georeferenced slice to lon/lat and renders
it. They are eager by design and must only be handed one (already multilooked,
hence small) slice at a time; they never trigger a whole-stack compute.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import geo

#: Colour for samples with no measurement. Mid grey: dark enough to separate from
#: a diverging colormap's near-white midpoint, unsaturated so it cannot be read as
#: a value on a red/blue ramp, and light enough not to dominate a field that is
#: half invalid (a GSLC is ~45% out of swath before anything else is dropped).
NAN_COLOR = "0.62"


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


def plot_los_displacement(los, epsg_code=None):
    """Plot LOS displacement (m) on a diverging colormap centred on zero."""
    los_latlon = _to_latlon(los, epsg_code)
    vmax = float(np.nanpercentile(np.abs(los_latlon.values), 98)) or None

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    los_latlon.plot.imshow(
        ax=ax,
        cmap="RdBu_r",
        vmin=None if vmax is None else -vmax,
        vmax=vmax,
        cbar_kwargs={"label": "LOS displacement (m, + toward sensor)", "shrink": 0.8},
    )
    ax.set_title("Line-of-sight displacement", fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(color="gray", linestyle="--", alpha=0.5)
    return fig, ax


def plot_offsets(x_offset, y_offset, correlation=None, epsg_code=None,
                 units="pixels", min_correlation=0.0, quiver=False,
                 vmax=None, centred=False, cmap="RdBu_r", nan_color=NAN_COLOR):
    """Plot a pixel-offset field as two panels, x beside y.

    Both panels share **one** symmetric diverging scale, taken from the 98th
    percentile of the two together unless ``vmax`` sets it: the components are
    meant to be compared by eye, which fails if each picks its own range.
    ``min_correlation`` blanks locations below that correlation, and
    ``quiver=True`` overlays the offsets as arrows on the second panel.

    ``centred`` only labels the panels -- pass it when the fields have already had
    their median removed (see
    :meth:`~nisar_tools.offsets.PixelOffsetStack.robust_offsets`), so the plot
    says which quantity is on screen.

    ``nan_color`` is not cosmetic on an offset field. A diverging colormap is
    near-white at its midpoint and matplotlib leaves invalid samples transparent,
    so on a white figure **a zero offset and a dropped location are the same
    pixel** -- and after :meth:`robust_offsets` most of the field is near zero
    while a third of it can be dropped. Painting the invalid samples a neutral grey
    is what separates "measured, and it did not move" from "not measured".

    Returns ``(fig, axes)``, with ``axes`` a length-2 array.
    """
    x_field = _to_latlon(x_offset, epsg_code)
    y_field = _to_latlon(y_offset, epsg_code)
    if correlation is not None and min_correlation > 0.0:
        corr = _to_latlon(correlation, epsg_code)
        x_field = x_field.where(corr >= min_correlation)
        y_field = y_field.where(corr >= min_correlation)

    if vmax is None:
        both = np.concatenate([x_field.values.ravel(), y_field.values.ravel()])
        finite = both[np.isfinite(both)]
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else None
    vmax = float(vmax) if vmax else None
    label = "Offset (m)" if units == "metres" else "Offset (pixels)"
    names = (("East", "North") if units == "metres" else ("Map x", "Map y"))
    suffix = " (median removed)" if centred else ""

    # A copy, or set_bad would mutate the globally registered colormap.
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(nan_color)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, sharey=True)
    for ax, field, name in zip(axes, (x_field, y_field), names):
        # The reprojection to lon/lat leaves its own out-of-coverage margin, which
        # imshow draws as nothing at all rather than as a bad sample, so the axes
        # have to carry the same colour for the panel to read as one surface.
        ax.set_facecolor(nan_color)
        field.plot.imshow(
            ax=ax,
            cmap=colormap,
            vmin=None if vmax is None else -vmax,
            vmax=vmax,
            cbar_kwargs={"label": label, "shrink": 0.8},
        )
        ax.set_title(f"{name} offset{suffix}", fontsize=14, pad=10)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.grid(color="white", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Latitude", fontsize=12)
    axes[1].set_ylabel("")

    if quiver:
        lon, lat = np.meshgrid(x_field["x"].values, x_field["y"].values)
        axes[1].quiver(
            lon, lat, x_field.values, y_field.values,
            angles="xy", color="black", alpha=0.6,
        )
    return fig, axes


def plot_angle(angle, epsg_code=None, title="Angle", label="Angle (deg)"):
    """Plot a per-pixel angle field (e.g. incidence) on a sequential colormap."""
    angle_latlon = _to_latlon(angle, epsg_code)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    angle_latlon.plot.imshow(
        ax=ax, cmap="viridis", cbar_kwargs={"label": label, "shrink": 0.8}
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(color="gray", linestyle="--", alpha=0.5)
    return fig, ax
