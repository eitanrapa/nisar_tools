"""Populate a GSLC's metadata from its RSLC: ``GslcWriter.populate_metadata``.

Split from :mod:`nisar_tools.gslc_geocode` because it is long and has no
numerics of its own beyond calling :func:`~nisar_tools.gslc_geocode.geocode_lut`.
Method-by-method port of ISCE3 v0.25.16's ``GslcWriter``,
``BaseL2WriterSingleInput`` and ``BaseWriterSingleInput``; the comments name
the method each block reproduces and flag the ISCE3 behaviours kept on purpose.
"""

import numpy as np

from . import gslc_product as gp, radar

P = gp.PRODUCT
R = gp.RSLC_PRODUCT
I = gp.IDENT

LUT_1D_AZ_DATASETS = ["referenceTerrainHeight"]
LUT_1D_RG_DATASETS = gp.CROSSTALK_DATASETS
LUT_1D_EXPANSION_POINTS = 21
LUT_1D_EXPANSION_MARGIN = 5


def populate_metadata(f, rslc, freq_pols, grids, lut_grid, orbit, dem, *,
                      granule_id, processing_datetime, orbit_file, tec_file, flatten,
                      solid_earth_tides, dem_path, dem_description, bounds, snap, epsg,
                      software_version, run_config_text):
    src = rslc._file
    ctx = dict(f=f, src=src, rslc=rslc, freq_pols=freq_pols, grids=grids,
               lut_grid=lut_grid, orbit=orbit, dem=dem)
    _identification_common(f, src, granule_id, processing_datetime)
    _identification_l2(f, src, freq_pols)
    _data_parameters(f, src, freq_pols)
    _ceos(f, grids, freq_pols)
    _calibration_information(ctx)
    _source_data(ctx)
    _processing_information_l2_common(ctx, orbit_file, tec_file, dem_path,
                                      dem_description, software_version)
    _processing_information_gslc(ctx, tec_file, flatten, run_config_text)
    gp.write_orbit(f, orbit)
    _attitude(f, src)


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
def _identification_common(f, src, granule_id, processing_datetime):
    """``populate_identification_common``."""
    gp.set_value(f, f"{I}/productDoi", "(NOT SPECIFIED)")
    gp.copy_from(src, f, f"{I}/absoluteOrbitNumber", format_function=np.uint32)
    gp.copy_from(src, f, f"{I}/trackNumber", format_function=np.uint32)
    gp.copy_from(src, f, f"{I}/frameNumber", format_function=np.uint16)
    gp.copy_from(src, f, f"{I}/missionId")
    gp.set_value(f, f"{I}/processingCenter", "(NOT SPECIFIED)")
    gp.set_value(f, f"{I}/productType", "GSLC")
    gp.set_value(f, f"{I}/granuleId", granule_id or "(NOT SPECIFIED)")
    gp.set_value(f, f"{I}/productVersion", "0.1.0")
    gp.set_value(f, f"{I}/productSpecificationVersion", gp.SPEC_VERSION)
    gp.copy_from(src, f, f"{I}/lookDirection", format_function=str.title)
    gp.copy_from(src, f, f"{I}/orbitPassDirection", format_function=str.title)
    gp.copy_from(src, f, f"{I}/plannedDatatakeId")
    gp.copy_from(src, f, f"{I}/plannedObservationId")
    gp.copy_from(src, f, f"{I}/isUrgentObservation")
    gp.copy_from(src, f, f"{I}/diagnosticModeFlag", skip_if_not_present=True)
    gp.set_value(f, f"{I}/processingDateTime", processing_datetime.strftime("%Y-%m-%dT%H:%M:%S"))
    gp.copy_from(src, f, f"{I}/radarBand", default="L")
    gp.copy_from(src, f, f"{I}/instrumentName", skip_if_not_present=True)
    # processing_type is unset outside the SDS, which ISCE3 maps to "Custom".
    gp.set_value(f, f"{I}/processingType", "Custom", format_function=str.title)
    gp.copy_from(src, f, f"{I}/isDithered", default=False)
    gp.copy_from(src, f, f"{I}/isMixedMode", default=False)
    gp.copy_from(src, f, f"{I}/isFullFrame", skip_if_not_present=True)
    gp.copy_from(src, f, f"{I}/isJointObservation", skip_if_not_present=True)
    gp.copy_from(src, f, f"{I}/hasInputDataException", skip_if_not_present=True)
    # The SDS passes its CRID; this product inherits the source RSLC's, which
    # also appears in the granule name.
    gp.copy_from(src, f, f"{I}/compositeReleaseId", default="A10000")


def _identification_l2(f, src, freq_pols):
    """``populate_identification_l2_specific``."""
    gp.copy_from(src, f, f"{I}/zeroDopplerStartTime")
    gp.copy_from(src, f, f"{I}/zeroDopplerEndTime")
    gp.set_value(f, f"{I}/productLevel", "L2")
    gp.set_value(f, f"{I}/isGeocoded", True)
    gp.copy_from(src, f, f"{I}/boundingPolygon")
    gp.set_value(f, f"{I}/listOfFrequencies", list(freq_pols))
    gp.copy_from(src, f, f"{I}/listOfObservationModes", default=["(NOT SPECIFIED)"])
    gp.copy_from(src, f, f"{I}/platformName", default="(NOT SPECIFIED)")
    gp.set_value(f, f"{I}/staticLayersDataAccess", "(NOT SPECIFIED)")


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def _data_parameters(f, src, freq_pols):
    """``GslcWriter.populate_data_parameters`` (the exception mask is written
    by the block loop, on the first frequency's grid)."""
    for freq in freq_pols:
        rin = f"{R}/swaths/frequency{freq}"
        out = f"{P}/grids/frequency{freq}"
        gp.copy_from(src, f, f"{out}/numberOfSubSwaths", f"{rin}/numberOfSubSwaths",
                     skip_if_not_present=True)
        gp.copy_from(src, f, f"{out}/rangeBandwidth", f"{rin}/processedRangeBandwidth")
        gp.copy_from(src, f, f"{out}/azimuthBandwidth", f"{rin}/processedAzimuthBandwidth")
        gp.copy_from(src, f, f"{out}/centerFrequency", f"{rin}/processedCenterFrequency")
        gp.copy_from(src, f, f"{out}/slantRangeSpacing", f"{rin}/slantRangeSpacing")
        gp.copy_from(src, f, f"{out}/zeroDopplerTimeSpacing", f"{R}/swaths/zeroDopplerTimeSpacing")


# ---------------------------------------------------------------------------
# ceosAnalysisReadyData
# ---------------------------------------------------------------------------
def _ceos(f, grids, freq_pols):
    """``populate_ceos_analysis_ready_data_parameters{_l2_common,}``."""
    from osgeo import ogr, osr

    ogr.UseExceptions()
    osr.UseExceptions()
    base = f"{P}/metadata/ceosAnalysisReadyData"
    gp.set_value(f, f"{base}/ceosAnalysisReadyDataDocumentIdentifier",
                 "https://ceos.org/ard/files/PFS/SAR/v1.0/CEOS-ARD_PFS_Synthetic_Aperture_Radar_v1.0.pdf")
    start_x, end_x, start_y, end_y = np.inf, -np.inf, -np.inf, np.inf
    epsg = None
    for freq in freq_pols:
        g = grids[freq]
        epsg = g.epsg if epsg is None else epsg
        start_x, end_x = min(start_x, g.start_x), max(end_x, g.end_x)
        start_y, end_y = max(start_y, g.start_y), min(end_y, g.end_y)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((start_x, start_y), (start_x, end_y), (end_x, end_y), (end_x, start_y),
                 (start_x, start_y)):
        ring.AddPoint(float(x), float(y))
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    poly.AssignSpatialReference(srs)
    ds = gp.set_value(f, f"{base}/boundingBox", poly.ExportToWkt())
    ds.attrs["epsg"] = np.int64(epsg)
    for xy in ("x", "y"):
        for kind in ("bias", "standardDeviation"):
            d = gp.set_value(f, f"{base}/geometricAccuracy/{kind}/{xy}", np.float32(np.nan))
            d.attrs["epsg"] = np.int64(epsg)
    gp.set_value(f, f"{base}/ceosAnalysisReadyDataProductType",
                 "Geocoded Single-Look Complex (GSLC)")


# ---------------------------------------------------------------------------
# calibrationInformation
# ---------------------------------------------------------------------------
def _lut_radar_grid(ctx, group_path, names, frequency):
    """``geocode_metadata_group``'s input radar grid for a LUT group.

    Returns ``(grid, arrays)`` or ``None`` when the group is absent. 1-D range
    vectors (crosstalk) are expanded along azimuth over the RSLC's time span
    with 21 lines plus 5 either side; ``referenceTerrainHeight`` takes the 2-D
    path whenever its group also holds ``slantRange`` -- and then comes out all
    NaN, which is what ISCE3 writes (reproduced by returning NaN arrays).
    """
    src, rslc = ctx["src"], ctx["rslc"]
    if group_path not in src:
        return None
    group = src[group_path]
    slc_grid = rslc.radar_grid(frequency)
    one_d_rg = all(n in LUT_1D_RG_DATASETS for n in names)
    noise = "noiseEquivalentBackscatter" in group_path
    arrays = {}
    if not one_d_rg and not noise:
        zdt = group["zeroDopplerTime"] if "zeroDopplerTime" in group else None
        if zdt is None:
            return None
        t = zdt[()]
        prf = 1.0 / np.average(np.diff(t))
        sensing_start, lines = float(t[0]), t.size
        epoch = radar.parse_epoch(zdt.attrs["units"])
    elif noise:
        sensing_start, lines, prf, epoch = _noise_az_parameters(ctx, group, names)
    else:
        start, end = slc_grid.sensing_start, slc_grid.sensing_stop
        lines = LUT_1D_EXPANSION_POINTS
        pri = (end - start) / (lines - 1)
        prf = 1.0 / pri
        sensing_start = start - LUT_1D_EXPANSION_MARGIN * pri
        lines += 2 * LUT_1D_EXPANSION_MARGIN
        epoch = slc_grid.ref_epoch

    one_d_az = all(n in LUT_1D_AZ_DATASETS for n in names) and "slantRange" not in group
    if not one_d_az:
        sr = group["slantRange"][()]
        spacing = np.average(np.diff(sr))
        starting_range, samples = float(sr[0]), sr.size
    else:
        start, end = slc_grid.starting_range, slc_grid.end_range
        samples = LUT_1D_EXPANSION_POINTS
        spacing = (end - start) / (samples - 1)
        starting_range = start - LUT_1D_EXPANSION_MARGIN * spacing
        samples += 2 * LUT_1D_EXPANSION_MARGIN
    if epoch != ctx["orbit"].reference_epoch:
        sensing_start += radar.seconds_since(ctx["orbit"].reference_epoch, epoch)
    grid = radar.RadarGrid(sensing_start=sensing_start, wavelength=slc_grid.wavelength,
                           prf=prf, starting_range=starting_range,
                           range_pixel_spacing=float(spacing),
                           look_side=slc_grid.look_side, length=int(lines),
                           width=int(samples), ref_epoch=ctx["orbit"].reference_epoch)

    for name in names:
        if name not in group:
            continue
        ds = group[name]
        data = ds[()]
        if noise:
            data = _resample_noise(group, name, grid)
        elif one_d_rg:
            data = np.repeat(data[None, :], lines, axis=0)
        elif data.ndim == 1:
            data = None                      # ISCE3's all-NaN outcome
        arrays[name] = (data, ds.dtype)
    return grid, arrays


def _noise_az_parameters(ctx, group, names):
    """``get_az_parameters_for_noise_equivalent_backscatter_luts``."""
    zdt = group["zeroDopplerTime"]
    az = zdt[()]
    start, end = float(az[0]), float(az[-1])
    mid = 0.5 * (start + end)
    lut_grid = ctx["lut_grid"]
    min_spacing = min(abs(lut_grid.spacing_x), abs(lut_grid.spacing_y))
    epoch = radar.parse_epoch(zdt.attrs["units"])
    t_orbit = mid + radar.seconds_since(ctx["orbit"].reference_epoch, epoch)
    _, vel = ctx["orbit"].interpolate(np.array(t_orbit))
    max_pri = min_spacing / np.linalg.norm(vel)
    lines = int(np.ceil(2 * (end - start) / max_pri)) + 1
    pri = (end - start) / (lines - 1)
    return start, lines, 1.0 / pri, epoch


def _resample_noise(group, pol, grid):
    """``getResampledNoiseEquivalentBackscatter``: nearest azimuth record,
    linear along range, onto the grid's lines (float64)."""
    power = group[pol][()]
    sr = group["slantRange"][()]
    az = group["zeroDopplerTime"][()]
    times = grid.sensing_start + np.arange(grid.length) / grid.prf
    # the grid was built in the noise LUT's own epoch before re-referencing;
    # nearest-record selection only needs relative times
    times = times - times[0] + az[0]
    nearest = np.argmin(np.abs(times[None, :] - az[:, None]), axis=0)
    out = np.zeros((grid.length, sr.size), np.float64)
    for i in range(az.size):
        rows = np.flatnonzero(nearest == i)
        if rows.size:
            out[rows, :] = np.interp(sr, sr, power[i, :])
    return out


def _geocode_group(ctx, out_group, in_group, names, frequency=None, compute_stats=False,
                   method=None, out_names=None):
    """``geocode_lut``: geocode ``names`` from ``in_group`` onto the LUT grid."""
    from .gslc_geocode import geocode_lut

    frequency = frequency or sorted(ctx["freq_pols"])[0]
    got = _lut_radar_grid(ctx, in_group, names, frequency)
    if got is None:
        return False
    grid, arrays = got
    lut_grid = ctx["lut_grid"]
    for i, name in enumerate(names):
        if name not in arrays:
            continue
        data, dtype = arrays[name]
        out_name = (out_names or names)[i]
        if data is None:
            result = np.full((lut_grid.length, lut_grid.width), np.nan, dtype=dtype)
        else:
            if data.dtype == np.uint8:
                data = data.astype(np.float32)
            m = _lut_method(grid.length, grid.width, method)
            result = geocode_lut(data, grid, lut_grid, ctx["dem"], ctx["orbit"], m)
            result = result.astype(data.dtype, copy=False)
        gp.write_lut(ctx["f"], out_group, out_name, result, lut_grid,
                     compute_stats=compute_stats)
    return True


def _lut_method(lines, samples, requested=None):
    from .gslc_geocode import _lut_method as choose
    return choose(lines, samples, requested)


def _calibration_information(ctx):
    """``populate_calibration_information`` + the GSLC-specific RTC LUTs."""
    f, src, freq_pols = ctx["f"], ctx["src"], ctx["freq_pols"]
    all_pols = [p for pols in freq_pols.values() for p in pols]
    if all(p in gp.LEXICOGRAPHIC_BASE_POLS for p in all_pols):
        basis = gp.LEXICOGRAPHIC_BASE_POLS
    elif all(p in gp.COMPACT_POLS for p in all_pols):
        basis = gp.COMPACT_POLS
    else:
        raise KeyError(f"unsupported polarimetric channels: {freq_pols}")
    for freq, pols in freq_pols.items():
        cal = f"{P}/metadata/calibrationInformation/frequency{freq}"
        cal_in = f"{R}/metadata/calibrationInformation/frequency{freq}"
        for parameter in ("commonDelay", "faradayRotation"):
            gp.copy_from(src, f, f"{cal}/{parameter}", f"{cal_in}/{parameter}", default=np.nan)
        for pol in pols:
            gp.copy_from(src, f, f"{cal}/{pol}/rfiLikelihood", f"{cal_in}/{pol}/rfiLikelihood",
                         default=np.nan)
        for pol in basis:
            for parameter in ("differentialDelay", "differentialPhase", "scaleFactor",
                              "scaleFactorSlope"):
                gp.copy_from(src, f, f"{cal}/{pol}/{parameter}", f"{cal_in}/{pol}/{parameter}",
                             default=np.nan)

    base_in = f"{R}/metadata/calibrationInformation"
    base_out = f"{P}/metadata/calibrationInformation"
    _geocode_group(ctx, f"{base_out}/crosstalk", f"{base_in}/crosstalk",
                   list(gp.CROSSTALK_DATASETS), compute_stats=True)
    for freq, pols in freq_pols.items():
        _geocode_group(ctx, f"{base_out}/frequency{freq}/elevationAntennaPattern",
                       f"{base_in}/frequency{freq}/elevationAntennaPattern",
                       list(pols), frequency=freq, compute_stats=False)
    for freq, pols in freq_pols.items():
        _geocode_group(ctx, f"{base_out}/frequency{freq}/noiseEquivalentBackscatter",
                       f"{base_in}/frequency{freq}/noiseEquivalentBackscatter",
                       list(pols), frequency=freq, compute_stats=True)
    _geocode_group(ctx, f"{base_out}/geometry", f"{base_in}/geometry",
                   ["beta0", "sigma0", "gamma0"], frequency=sorted(freq_pols)[0])


# ---------------------------------------------------------------------------
# sourceData
# ---------------------------------------------------------------------------
def _near_far_incidence(grid, orbit):
    """``get_near_and_far_range_incidence_angles``: zero Doppler, zero height."""
    t = grid.sensing_time(grid.length // 2)
    out = []
    for r in (grid.starting_range, grid.end_range):
        sat, _ = orbit.interpolate(np.array(t))
        llh, _ = radar.rdr2geo([t], [r], orbit, height=0.0, wavelength=grid.wavelength,
                               look_side=grid.look_side)
        target = radar.lonlat_to_xyz(llh[0, 0], llh[0, 1], llh[0, 2])
        enu = radar.xyz_to_enu_matrix(llh[0, 1], llh[0, 0]) @ (target - sat)
        out.append(np.degrees(np.arccos(abs(enu[2]) / np.linalg.norm(enu))))
    return out


def _source_data(ctx):
    """``populate_source_data``."""
    f, src, rslc, orbit = ctx["f"], ctx["src"], ctx["rslc"], ctx["orbit"]
    sd = f"{P}/metadata/sourceData"
    gp.copy_from(src, f, f"{sd}/productDoi", f"{I}/productDoi", skip_if_not_present=True)
    gp.copy_from(src, f, f"{sd}/productVersion", f"{I}/productVersion", skip_if_not_present=True)
    gp.copy_from(src, f, f"{sd}/lookDirection", f"{I}/lookDirection", format_function=str.title)
    gp.copy_from(src, f, f"{sd}/productLevel", f"{I}/productLevel", default="L1")
    gp.set_value(f, f"{sd}/productGeometry", "Slant range")
    gp.copy_from(src, f, f"{sd}/processingDateTime", f"{I}/processingDateTime",
                 skip_if_not_present=True)
    gp.copy_from(src, f, f"{sd}/processingCenter", f"{I}/processingCenter",
                 default="(NOT SPECIFIED)")
    gp.copy_from(src, f, f"{sd}/productType", f"{I}/productType", default="(NOT SPECIFIED)")
    gp.copy_from(src, f, f"{sd}/granuleId", f"{I}/granuleId", default="(NOT SPECIFIED)")
    pin = f"{R}/metadata/processingInformation"
    for name in ("parameters/runConfigurationContents", "parameters/referenceTerrainHeight",
                 "parameters/zeroDopplerTime", "parameters/slantRange",
                 "algorithms/rfiDetection", "algorithms/rfiMitigation",
                 "algorithms/rangeCompression", "algorithms/elevationAntennaPatternCorrection",
                 "algorithms/rangeSpreadingLossCorrection", "algorithms/dopplerCentroidEstimation",
                 "algorithms/azimuthPresumming", "algorithms/azimuthCompression",
                 "algorithms/softwareVersion"):
        gp.copy_from(src, f, f"{sd}/processingInformation/{name}", f"{pin}/{name}",
                     skip_if_not_present=True)
    gp.copy_from(src, f, f"{sd}/swaths/zeroDopplerStartTime", f"{I}/zeroDopplerStartTime")
    gp.copy_from(src, f, f"{sd}/swaths/zeroDopplerEndTime", f"{I}/zeroDopplerEndTime")
    gp.copy_from(src, f, f"{sd}/swaths/zeroDopplerTimeSpacing", f"{R}/swaths/zeroDopplerTimeSpacing")

    for i, freq in enumerate(ctx["freq_pols"]):
        grid = rslc.radar_grid(freq)
        out = f"{sd}/swaths/frequency{freq}"
        rin = f"{R}/swaths/frequency{freq}"
        near, far = _near_far_incidence(grid, orbit)
        gp.set_value(f, f"{out}/nearRangeIncidenceAngle", np.float64(near))
        gp.set_value(f, f"{out}/farRangeIncidenceAngle", np.float64(far))
        gp.copy_from(src, f, f"{out}/listOfPolarizations", f"{rin}/listOfPolarizations")
        if i == 0:
            gp.set_value(f, f"{sd}/swaths/numberOfAzimuthLines", np.uint64(grid.length))
        az_bw = float(src[f"{rin}/processedAzimuthBandwidth"][()])
        at_spacing = float(src[f"{rin}/sceneCenterAlongTrackSpacing"][()])
        dt = float(src[f"{R}/swaths/zeroDopplerTimeSpacing"][()])
        gp.set_value(f, f"{out}/sceneCenterAlongTrackResolution",
                     np.float64(at_spacing / dt / az_bw))
        rg_bw = float(src[f"{rin}/processedRangeBandwidth"][()])
        gp.set_value(f, f"{out}/rangeResolution", np.float64(radar.SPEED_OF_LIGHT / (2 * rg_bw)))
        for name, src_name in (("sceneCenterAlongTrackSpacing", None),
                               ("sceneCenterGroundRangeSpacing", None),
                               ("acquiredRangeBandwidth", None),
                               ("processedRangeBandwidth", None),
                               ("processedAzimuthBandwidth", None),
                               ("centerFrequency", "processedCenterFrequency")):
            gp.copy_from(src, f, f"{out}/{name}", f"{rin}/{src_name or name}")
        gp.set_value(f, f"{out}/slantRangeStart", np.float64(grid.starting_range))
        gp.copy_from(src, f, f"{out}/slantRangeSpacing", f"{rin}/slantRangeSpacing")
        gp.set_value(f, f"{out}/numberOfRangeSamples", np.uint64(grid.width))
        pfreq = f"{pin}/parameters/frequency{freq}"
        opar = f"{sd}/processingInformation/parameters/frequency{freq}"
        gp.copy_from(src, f, f"{opar}/dopplerCentroid", f"{pfreq}/dopplerCentroid",
                     skip_if_not_present=True)
        for axis in ("zeroDopplerTime", "slantRange"):
            if f"{pfreq}/{axis}" in src:
                gp.copy_from(src, f, f"{opar}/{axis}", f"{pfreq}/{axis}")
            else:
                gp.copy_from(src, f, f"{opar}/{axis}", f"{pin}/parameters/{axis}")


# ---------------------------------------------------------------------------
# processingInformation
# ---------------------------------------------------------------------------
def _input_runconfig_value(src, field):
    """``get_value_from_input_runconfig``: a ``key: value`` line of the RSLC's."""
    path = f"{R}/metadata/processingInformation/parameters/runConfigurationContents"
    if path not in src:
        return None
    text = src[path].asstr()[()] if src[path].shape == () else str(src[path][()])
    for line in text.split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == field:
            return value.strip()
    return None


def _processing_information_l2_common(ctx, orbit_file, tec_file, dem_path,
                                      dem_description, software_version):
    """``populate_processing_information_l2_common``."""
    f, src, freq_pols = ctx["f"], ctx["src"], ctx["freq_pols"]
    params = f"{P}/metadata/processingInformation/parameters"
    inputs = f"{P}/metadata/processingInformation/inputs"
    # ISCE3 tests the *path string* for "disabled" and membership in the output
    # (where algorithms/rfiMitigation never is), so this is always False; ASF's
    # products say False even when the RSLC applied RFI mitigation.
    rfi_path = f"{P}/metadata/processingInformation/algorithms/rfiMitigation"
    gp.set_value(f, f"{params}/rfiMitigationApplied",
                 bool(rfi_path in f and "disabled" not in rfi_path.lower()))
    dry = _input_runconfig_value(src, "dry_troposphere_model")
    gp.set_value(f, f"{params}/dryTroposphericGeolocationCorrectionApplied",
                 "(NOT SPECIFIED)" if dry is None else str(dry != "nodelay"))
    gp.set_value(f, f"{params}/wetTroposphericGeolocationCorrectionApplied", False)
    gp.set_value(f, f"{P}/metadata/processingInformation/algorithms/softwareVersion",
                 software_version)
    gp.set_value(f, f"{inputs}/l1SlcGranules", [str(ctx["rslc"].path).split("/")[-1]])
    gp.set_value(f, f"{inputs}/orbitFiles", [str(orbit_file) if orbit_file else "(NOT SPECIFIED)"])
    gp.set_value(f, f"{inputs}/tecFiles", [str(tec_file) if tec_file else "(NOT SPECIFIED)"])
    gp.set_value(f, f"{inputs}/configFiles", ["(NOT SPECIFIED)"])
    gp.set_value(f, f"{inputs}/demSource", dem_description or "(NOT SPECIFIED)")
    _geocode_group(ctx, params, f"{R}/metadata/processingInformation/parameters",
                   ["referenceTerrainHeight"])
    for freq in freq_pols:
        _geocode_group(ctx, f"{params}/frequency{freq}",
                       f"{R}/metadata/processingInformation/parameters/frequency{freq}",
                       ["dopplerCentroid"], frequency=freq)


def _processing_information_gslc(ctx, tec_file, flatten, run_config_text):
    """``GslcWriter.populate_processing_information``."""
    f, src = ctx["f"], ctx["src"]
    params = f"{P}/metadata/processingInformation/parameters"
    gp.copy_from(src, f, f"{params}/azimuthChirpWeighting",
                 f"{R}/metadata/processingInformation/parameters/azimuthChirpWeighting",
                 skip_if_not_present=True)
    gp.copy_from(src, f, f"{params}/rangeChirpWeighting",
                 f"{R}/metadata/processingInformation/parameters/rangeChirpWeighting",
                 skip_if_not_present=True)
    gp.set_value(f, f"{params}/rangeIonosphericGeolocationCorrectionApplied", tec_file is not None)
    gp.set_value(f, f"{params}/azimuthIonosphericGeolocationCorrectionApplied", tec_file is not None)
    gp.set_value(f, f"{params}/ellipsoidalFlatteningApplied", bool(flatten))
    gp.set_value(f, f"{params}/topographicFlatteningApplied", bool(flatten))
    gp.set_value(f, f"{params}/runConfigurationContents", run_config_text)
    gp.set_value(f, f"{P}/metadata/processingInformation/algorithms/demInterpolation", "biquintic")
    gp.set_value(f, f"{P}/metadata/processingInformation/algorithms/geocoding", "Sinc interpolation")


def _attitude(f, src):
    """``populate_attitude``: the RSLC's attitude group, minus angularVelocity."""
    in_path = f"{R}/metadata/attitude"
    if in_path not in src:
        return
    out = f.require_group(f"{P}/metadata/attitude")
    for name, obj in src[in_path].items():
        if name == "angularVelocity":
            continue
        src.copy(obj, out, name=name)


def describe_run(**kwargs):
    """A YAML-ish record of the parameters, for ``runConfigurationContents``."""
    lines = ["# written by nisar_tools.gslc_geocode.geocode_rslc", "runconfig:"]
    for key, value in kwargs.items():
        lines.append(f"    {key}: {value}")
    return "\n".join(lines) + "\n"

