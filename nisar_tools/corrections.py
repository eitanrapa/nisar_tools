"""Geolocation timing corrections for geocoding an RSLC.

A port of ISCE3 v0.25.16's ``nisar.workflows.geocode_corrections.AzSrgCorrections``
with its two model-driven sources:

- **Solid Earth tides** (``isce3.solid_earth_tides``): pysolid's tidal
  displacement on a 0.23 deg grid, projected into radar range and azimuth by a
  ``geo2rdr`` of each point with and without the displacement. A few centimetres
  of slant range; needs the optional ``pysolid`` package.
- **Ionosphere** (``isce3.atmosphere.tec_product``): from a NISAR IMAGEN TEC
  file (``NISAR_ANC_TEC_*.json``). Range delay ``40.31 * TEC_sub / f^2 /
  cos(inc)`` at near and far range; azimuth shift from the along-track TEC
  gradient. **Metres** of slant range (2.6-3.2 m in ASF's dawn reference GSLC,
  7.7-9.7 m for an evening frame) and 0.1-0.2 ms of azimuth time.

Both only move *where* a map pixel samples the RSLC: ISCE3 flattens with the
uncorrected range, so neither enters the GSLC phase through the flattening.

The ionospheric correction is applied only when a TEC file is supplied --
exactly what ISCE3 does -- and the product records whether it was. The data
guide does not list TEC among the supporting products, but ASF distributes the
files as the ``NISAR_TEC`` collection; :func:`nisar_tools.download.download_tec`
finds and fetches the one covering an RSLC. Every P05023 GSLC and GUNW checked
was made with one.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import radar

#: Ionospheric range-delay constant (m^3 / s^2) and the TECU scale.
K_IONO = 40.31
TECU = 1e16


# ---------------------------------------------------------------------------
# Decimated radar grid
# ---------------------------------------------------------------------------
def nominal_ground_spacing(grid, orbit):
    """``get_radar_grid_nominal_ground_spacing``: (azimuth, ground range) metres."""
    pos, vel = orbit.interpolate(np.array(grid.sensing_mid))
    llh, _ = radar.rdr2geo([grid.sensing_mid], [grid.mid_range], orbit, height=0.0,
                           wavelength=grid.wavelength, look_side=grid.look_side)
    target = radar.lonlat_to_xyz(llh[0, 0], llh[0, 1], llh[0, 2])
    azimuth_spacing = (np.linalg.norm(vel) / grid.prf * np.linalg.norm(target)
                       / np.linalg.norm(pos))
    enu = radar.xyz_to_enu_matrix(llh[0, 1], llh[0, 0]) @ (target - pos)
    cos_inc = -(enu[2] / np.linalg.norm(enu))
    sin_inc = np.sqrt(1.0 - cos_inc ** 2)
    return float(azimuth_spacing), float(grid.range_pixel_spacing / sin_inc)


def decimated_radar_grid(grid, orbit, posting=5000.0):
    """``_get_decimated_radar_grid``: the same corners at ~``posting`` metres."""
    az_spacing, gr_spacing = nominal_ground_spacing(grid, orbit)
    length = max(int(az_spacing / posting * grid.length), 2)
    width = max(int(gr_spacing / posting * grid.width), 2)
    return grid.resize_keep_start_stop(length, width)


# ---------------------------------------------------------------------------
# Solid Earth tides
# ---------------------------------------------------------------------------
def _enu_vectors(lon_deg, lat_deg):
    """``get_enu_vec_from_lat_lon``: unit E, N, U in ECEF, each ``(3, ...)``."""
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    vec_u = np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])
    vec_n = np.array([-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)])
    vec_e = np.cross(vec_n, vec_u, axis=0)
    return vec_e, vec_n, vec_u


def solid_earth_tides(grid, lon_deg, lat_deg, hgt, orbit):
    """``isce3.solid_earth_tides.solid_earth_tides``: ``(set_rg [m], set_az [s])``.

    ``lon_deg``/``lat_deg``/``hgt`` locate the decimated radar grid's pixels on
    the ground (degrees, metres).
    """
    try:
        import pysolid
    except ImportError as exc:
        raise ImportError(
            "solid Earth tide corrections need the optional 'pysolid' package "
            "(pip install pysolid), or pass solid_earth_tides=False"
        ) from exc
    from scipy.interpolate import RegularGridInterpolator

    step = 0.23
    x_min, x_max, y_min, y_max = radar.geo_bounding_box(grid, orbit, 4326, margin=step)
    width = int(np.ceil((x_max - x_min) / step))
    length = int(np.ceil((y_max - y_min) / step))
    attrs = {"LENGTH": length, "WIDTH": width, "X_FIRST": x_min, "Y_FIRST": y_max,
             "X_STEP": step, "Y_STEP": -step}
    mid = grid.ref_epoch + np.timedelta64(int(round(grid.sensing_mid * 1e6)), "us")
    when = datetime.fromisoformat(np.datetime_as_string(mid, unit="us"))
    set_e, set_n, set_u = pysolid.calc_solid_earth_tides_grid(when, attrs, display=False,
                                                              verbose=False)
    lat_vec = attrs["Y_FIRST"] + np.arange(length) * attrs["Y_STEP"]
    lon_vec = attrs["X_FIRST"] + np.arange(width) * attrs["X_STEP"]
    pts_src = (np.flipud(lat_vec), lon_vec)
    pts_dst = (np.ravel(lat_deg), np.ravel(lon_deg))
    shape = np.shape(lat_deg)
    rdr = [
        RegularGridInterpolator(pts_src, np.flipud(c), method="linear",
                                bounds_error=False, fill_value=0)(pts_dst).reshape(shape)
        for c in (set_e, set_n, set_u)
    ]

    vec_e, vec_n, vec_u = _enu_vectors(np.ravel(lon_deg), np.ravel(lat_deg))
    llh0 = np.stack([np.deg2rad(np.ravel(lon_deg)), np.deg2rad(np.ravel(lat_deg)),
                     np.ravel(hgt)], axis=1)
    xyz0 = radar.lonlat_to_xyz(llh0[:, 0], llh0[:, 1], llh0[:, 2])
    xyz1 = (xyz0 + vec_e.T * rdr[0].ravel()[:, None] + vec_n.T * rdr[1].ravel()[:, None]
            + vec_u.T * rdr[2].ravel()[:, None])
    kw = dict(orbit=orbit, doppler=None, wavelength=grid.wavelength,
              look_side=grid.look_side, threshold=1e-9, maxiter=25)
    t0, r0, _ = radar.geo2rdr_xyz(xyz0, **kw)
    t1, r1, _ = radar.geo2rdr_xyz(xyz1, **kw)
    return (r1 - r0).reshape(shape), (t1 - t0).reshape(shape)


# ---------------------------------------------------------------------------
# Ionosphere from an IMAGEN TEC file
# ---------------------------------------------------------------------------
def _tec_times(tec, grid, orbit, margin, staggered):
    """``_get_tec_time``: TEC sample times (s since epoch) and the kept mask."""
    utc = np.array([np.datetime64(t.replace("Z", ""), "ns") for t in tec["utc"]])
    t = np.array([radar.seconds_since(grid.ref_epoch, u) for u in utc])
    stagger = 0.0
    if staggered:
        spacing = (t[-1] - t[0]) / (len(t) - 1)
        stagger = spacing / 2.0
        margin = max(margin, stagger)
    start = max(orbit.start_time - stagger, grid.sensing_start - margin)
    stop = min(orbit.end_time + stagger, grid.sensing_stop + margin)
    keep = (t >= start) & (t <= stop)
    kept = t[keep]
    if kept.size < 2:
        raise ValueError("TEC file does not cover the radar grid")
    spacing = (kept[-1] - kept[0]) / (kept.size - 1)
    lo, hi = grid.sensing_start, grid.sensing_stop
    if staggered:
        lo, hi = lo - spacing / 2, hi + spacing / 2
    if not (kept[0] < lo and hi < kept[-1]):
        raise ValueError("cropped TEC grid does not contain the radar grid")
    return t, keep


def _incidence(t, r, grid, orbit):
    """``compute_incidence_angle`` on the zero-height ellipsoid, radians."""
    sat, _ = orbit.interpolate(np.array(t))
    llh, _ = radar.rdr2geo([t], [r], orbit, height=0.0, wavelength=grid.wavelength,
                           look_side=grid.look_side)
    target = radar.lonlat_to_xyz(llh[0, 0], llh[0, 1], llh[0, 2])
    enu = radar.xyz_to_enu_matrix(llh[0, 1], llh[0, 0]) @ (target - sat)
    return float(np.arccos(abs(enu[2]) / np.linalg.norm(enu)))


def tec_luts(json_path, center_frequency, orbit, grid, margin=40.0):
    """``tec_lut2d_from_json_{az,srg}``: ``(azimuth LUT [s], range LUT [m])``."""
    with open(json_path) as fh:
        tec = json.load(fh)
    rg_vec = [grid.starting_range, grid.end_range]

    # Slant range: delay at near and far range for each TEC sample.
    t_all, keep = _tec_times(tec, grid, orbit, margin, staggered=False)
    t_keep = t_all[keep]
    delta = []
    for key, rg in zip(("Nr", "Fr"), rg_vec):
        sub = (np.array(tec[f"totTec{key}"]) - np.array(tec[f"topTec{key}"]))[keep]
        inc = np.array([_incidence(t, rg, grid, orbit) for t in t_keep])
        delta.append(K_IONO * sub * TECU / center_frequency ** 2 / np.cos(inc))
    srg = radar.LUT2d.from_coords(rg_vec, t_keep, np.vstack(delta).T)

    # Azimuth: from the along-track TEC gradient, on a half-sample-staggered grid.
    t_all, keep = _tec_times(tec, grid, orbit, margin, staggered=True)
    t_keep = t_all[keep]
    sub = np.vstack([(np.array(tec[f"totTec{k}"]) - np.array(tec[f"topTec{k}"]))[keep]
                     for k in ("Nr", "Fr")]).T
    spacing = (t_keep[-1] - t_keep[0]) / (t_keep.size - 1)
    t_stag = t_keep[1:] - spacing / 2
    gradient = np.diff(sub, axis=0) / spacing
    _, vel = orbit.interpolate(t_stag)
    speed = np.linalg.norm(vel, axis=1)
    fm_rate = np.outer(-2.0 * speed ** 2 * (center_frequency / radar.SPEED_OF_LIGHT),
                       1.0 / np.array(rg_vec))
    delay = (-2.0 * K_IONO / (radar.SPEED_OF_LIGHT * fm_rate * center_frequency)
             * gradient * TECU)
    az = radar.LUT2d.from_coords(rg_vec, t_stag, delay)
    return az, srg


# ---------------------------------------------------------------------------
# The combined corrections (AzSrgCorrections)
# ---------------------------------------------------------------------------
@dataclass
class TimingCorrections:
    """Correction LUTs on the decimated radar grid, plus what to record.

    ``azimuth`` (seconds) and ``slant_range`` (metres) are ``None`` when no
    correction of that kind was computed -- ISCE3's empty LUT.
    """

    grid: radar.RadarGrid
    azimuth: radar.LUT2d = None
    slant_range: radar.LUT2d = None
    arrays: dict = field(default_factory=dict)
    tec_file: str = None
    solid_earth_tides: bool = False

    @property
    def az_vec(self):
        return self.grid.sensing_start + np.arange(self.grid.length) / self.grid.prf

    @property
    def rg_vec(self):
        return self.grid.starting_range + np.arange(self.grid.width) * self.grid.range_pixel_spacing


def compute_timing_corrections(radar_grid, orbit, center_frequency, dem=None,
                               tec_file=None, solid_earth_tides=True):
    """``AzSrgCorrections`` for one frequency's radar grid.

    ``dem`` is a :class:`~nisar_tools.dem.DEMInterpolator` for the SET
    ground positions (``None`` = ellipsoid). Returns :class:`TimingCorrections`.
    """
    scaled = decimated_radar_grid(radar_grid, orbit)
    out = TimingCorrections(grid=scaled, tec_file=str(tec_file) if tec_file else None,
                            solid_earth_tides=bool(solid_earth_tides))
    if not solid_earth_tides and tec_file is None:
        return out

    def _lut(data):
        return radar.LUT2d(scaled.starting_range, scaled.sensing_start,
                           scaled.range_pixel_spacing, 1.0 / scaled.prf, data)

    set_rg = None
    if solid_earth_tides:
        tt, rr = np.meshgrid(out.az_vec, out.rg_vec, indexing="ij")
        dem_fn = None if dem is None else dem.coarse_sampler(
            radar.geo_bounding_box(scaled, orbit, 4326, margin=0.1))
        llh, _ = radar.rdr2geo(tt.ravel(), rr.ravel(), orbit, height=0.0,
                               wavelength=scaled.wavelength, look_side=scaled.look_side,
                               threshold=1e-7, maxiter=25, extraiter=10, dem=dem_fn)
        lon = np.degrees(llh[:, 0]).reshape(tt.shape)
        lat = np.degrees(llh[:, 1]).reshape(tt.shape)
        hgt = llh[:, 2].reshape(tt.shape)
        set_rg, _ = solid_earth_tides_fn(scaled, lon, lat, hgt, orbit)

    if tec_file is not None:
        az_tec, rg_tec = tec_luts(tec_file, center_frequency, orbit, radar_grid)
        if set_rg is None:
            out.azimuth, out.slant_range = az_tec, rg_tec
            out.arrays["azimuthIonosphere"] = az_tec.data
            out.arrays["slantRangeIonosphere"] = rg_tec.data
            return out
        tt, rr = np.meshgrid(out.az_vec, out.rg_vec, indexing="ij")
        tec_az = az_tec.eval(tt, rr)
        tec_rg = rg_tec.eval(tt, rr)
        out.arrays["azimuthIonosphere"] = tec_az
        out.arrays["slantRangeIonosphere"] = tec_rg
        out.arrays["slantRangeSolidEarthTides"] = set_rg
        out.azimuth = _lut(tec_az)
        out.slant_range = _lut(set_rg + tec_rg)
        return out

    out.arrays["slantRangeSolidEarthTides"] = set_rg
    out.slant_range = _lut(set_rg)
    return out


# The public name stays importable; the module-level alias avoids the
# parameter of the same name shadowing it inside compute_timing_corrections.
solid_earth_tides_fn = solid_earth_tides
