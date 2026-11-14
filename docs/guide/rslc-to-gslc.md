# Making GSLCs from RSLCs

ASF does not ship a GSLC for every RSLC. `geocode_rslc` makes the missing one. It is a port of the ISCE3 (v0.25.16) GSLC
workflow that produces ASF's products, and it writes the NISAR L2 GSLC layout (specification 1.5.0), so `GSLC` and
everything downstream read its output like any granule. Geocoding ASF's own source RSLC reproduces ASF's GSLC bit for
bit on almost every pixel; the rest differ by one step of the 10-bit mantissa truncation.

```python
from nisar_tools import download_tec, geocode_rslc
from nisar_tools.dem import stage_nisar_dem

rslc = "data/NISAR_L1_PR_RSLC_....h5"
dem = stage_nisar_dem("data/dem.tif", (lon_min, lon_max, lat_min, lat_max))
tec = download_tec("data/", rslc=rslc)
gslc = geocode_rslc(rslc, "data/", dem, tec_file=tec, like="data/NISAR_L2_PR_GSLC_<same frame>.h5")
```

Given a directory, the output is named after the RSLC with `L1_PR_RSLC` replaced by `L2_OD_GSLC` (`OD`, on demand, so
it cannot be mistaken for a product of the science data system; see `default_gslc_name`). `RSLC(path).to_gslc(output,
dem, **kwargs)` is the same call.

## What to get right

- **Use the NISAR DEM v1.2, as `stage_nisar_dem` copies it.** A GSLC's phase is flattened by the range to the DEM
  surface, so a height difference changes the phase by about 0.4 rad per centimetre at L-band. A GSLC flattened against
  another DEM, or against a resampled copy of this one, is not interferometrically compatible with ASF's.
  `stage_nisar_dem` needs the `gdal` extra and Earthdata credentials in `~/.netrc`.
- **Pass the TEC file** to reproduce ASF's geolocation. The ionospheric correction moves where each map pixel samples
  the RSLC by metres of slant range. Without it the output is still a valid GSLC, records that the correction was not
  applied, and sits up to a pixel or more from ASF's geolocation.
- **Pass `like=` a GSLC or GUNW of the same track and frame** to land on ASF's grid for that frame. ASF puts every L2
  product of a frame on one box, so the output is then pixel-aligned with ASF's GSLCs of other dates. Without it the
  box is the swath footprint with its corners snapped outward to `snap` metres (240 by default): the same lattice, but a
  different extent. `product_bounds(path)` returns a product's box.
- **Solid Earth tides** need the `gslc` extra (`pysolid`); pass `solid_earth_tides=False` to skip them.
- **Budget disk and time.** A full frame with both sub-bands and polarizations is tens of gigabytes. `workers=` sets
  the number of threads and `block_size` the output block edge.
- Sub-bands (`frequencies=`), polarizations, projection (`epsg=`), posting, an explicit box (`bounds=`) and an external
  orbit file (`orbit_file=`) can all be overridden; see the `geocode_rslc` reference.

## Registering a secondary

Two GSLCs geocoded independently are registered only as well as their geolocation agrees. ASF's per-date ionospheric
geolocation corrections, computed from a global TEC model, can disagree between two dates by about a resolution cell,
which costs coherence. Because NISAR keeps the Doppler carrier in its SLCs, an along-track misregistration `dt` also
biases the interferometric phase by \(-2\pi f_\mathrm{dc}\,dt\), about 1.7 cm of LOS per metre at L-band.

`geocode_rslc(registration=...)` fixes both: it geocodes the secondary at its geometric position plus measured offsets,
while the phase is still flattened by the geometric range. That is the GSLC counterpart of the "rubbersheeting" that
ISCE3 applies when it makes a GUNW.

```python
from nisar_tools import RegistrationField, geocode_rslc

# Offsets measured between the two RSLCs, interpolated onto a regular map grid x, y
# (output projection): along_track and slant_range in metres, finite everywhere.
field = RegistrationField.from_offsets(
    x, y, along_track, slant_range,
    along_track_spacing=..., range_spacing=...,    # metres per line and per sample
)
secondary = geocode_rslc("data/<date 2 RSLC>.h5", "data/", dem,
                         like="data/<date 1 GSLC>.h5", registration=field)
```

- `RegistrationField(x, y, lines, samples)` takes the shifts in radar lines and samples directly. It is evaluated
  bilinearly, and beyond its grid it takes the nearest edge node. Fill gaps before building one.
- The offsets usually come from [`roff.dense_offsets`](offsets.md#nisars-roff-from-two-rslcs) between the two RSLCs,
  or from a GUNW's `pixelOffsets` layer.
- The registration is added after the timing corrections and before interpolation. When the field already holds
  the reference date's own corrections, pass `tec_file=None` and `solid_earth_tides=False` so they are not applied
  twice.
- Mixing a corrected GSLC with an uncorrected one in one pair is safe: the corrections only move where pixels are
  sampled, and both are flattened with the uncorrected range.

!!! warning "The carrier term"
    The Doppler carrier is re-ramped at the shifted position, which is right for geolocation, timing and motion. For the
    part of the field that follows the ionosphere's along-track shift, it leaves \(-2\pi f_\mathrm{dc}\,dt\) in the
    phase, as it does in the GUNW. This term scales with frequency, so no split-spectrum screen removes it.

The [InSAR pipeline](../insar_pipeline.md#1-registration) page describes how registration fits into a coseismic
workflow.

## ISCE3's defects, and which to keep

The port reproduces ISCE3 v0.25.16, defects included, so its GSLCs pair with ASF's. `isce3_compatible` chooses:

| `isce3_compatible` | Pixels | Masks and metadata |
| --- | --- | --- |
| `"pixels"` (default) | ISCE3's | fixed |
| `True` | ISCE3's | ISCE3's |
| `False` | fixed | fixed |

The metadata fixes: the mask tests every row of the interpolation chip (ISCE3 tests its corners and misses a line whose
valid interval narrows inside it); `validSamplesSubSwath` lines that drop out of the swath for one line are repaired;
timing corrections are clamped at their table's edge instead of skipped; band B's `inputDataExceptionMask` is written
on band B's grid; and the cube's ground-track velocity uses its own `rdr2geo` tolerance.

The pixel fixes change the phase, so both dates of a pair must use them, and never with an ASF GSLC: a centred DEM
spline window (ISCE3's is one node off, and never reads the first row or column) and the Doppler table read at
zero-Doppler coordinates (ISCE3 writes it over beam-centre times, about 2 s earlier, and labels them zero-Doppler).
`GSLC.pixel_convention` reads which a GSLC has, and `GSLCStack.from_gslcs` refuses a mix.

One departure is always there: the port reads the RSLC's last line and range sample, which ISCE3 drops, so along the
far-range edge a ribbon about a pixel wide is finite here and NaN in ASF's products, with mask 0 in both.

## Correcting an RSLC's own errors

Two errors of delivered RSLCs are corrected where a band samples its RSLC, like a registration, with the flattening left
on the geometric range (see [Checking an RSLC before use](checking-rslcs.md)):

```python
from nisar_tools import geocode_rslc
from nisar_tools.l0b import timing_report
from nisar_tools.qa import band_offset

offset = band_offset(rslc)         # band B's range against A, per data window and range bin
report = timing_report(rslc)       # the L0B's pulse times, read remotely
gslc = geocode_rslc(rslc, "data/", dem, range_shift={"B": offset.shift}, azimuth_shift=report)
```

`range_shift` aligns band B's ground with A's, which split-spectrum ionosphere screens need; `azimuth_shift` moves
spans tagged a pulse early or late back by a pulse, sets the mask to 0 where a slip's edge mixes good and bad echoes,
and restores the start of a datatake that ISCE3 left out of `validSamplesSubSwath`. Do not pass `azimuth_shift` together with a `registration` measured on the same RSLC, which already holds the
shift.
