# The InSAR pipeline

This page describes how the package's pieces are combined to process a NISAR pair that spans an earthquake: what each
step does, why, and how the result differs from NISAR's ISCE3 production (PR) products, the GUNWs. For a pair with no
rupture in it, most of these steps reduce to the [Getting started](getting-started.md) flow.

## In short

1. **Registration.** Each pair is formed from the first date's ASF GSLC and a second-date GSLC made by the package,
   geocoded onto the same grid with a registration field built from pixel offsets. Two ASF GSLCs as downloaded can be
   misregistered by about a resolution cell, by their own per-date ionospheric geolocation corrections.
2. **The rupture.** A narrow strip along the mapped trace is masked before unwrapping, so the unwrapper never crosses
   the rupture. A block that the cut isolates is tied by offsets, or given its own offset in the inversion.
3. **The ionosphere.** Only the main band is unwrapped. The side band enters through a residual that contains no
   deformation, summed on the same map cells as the main band, so the screen does not copy the deformation.
4. **Partial focus.** The main band is partially focused next to the near-range swath edge. Those samples are kept in
   the data and left out only of the ionospheric estimate.
5. **Few masks.** Water only: no coherence mask, no outlier removal, no cycle repair near the fault, and no smoothing
   before the data are sampled for an inversion. Each of these removes or alters data exactly where the near-fault
   signal is.

## Inputs

- **PR GSLCs of both dates**, with both sub-bands: band A, the main band (40 MHz centred at
  \(f_A = 1239\) MHz in the common mode), and band B, the side band (5 MHz at \(f_B = 1293.5\) MHz). ASF grids every
  L2 product of a frame on one box. Each GSLC carries its own per-date ionospheric (TEC) and solid Earth tide
  geolocation correction.
- **PR RSLCs of both dates**: the offsets are measured between them, and the second date's is geocoded.
- **The NISAR DEM**, staged with `dem.stage_nisar_dem`.
- **A mapped trace of the rupture**, for example drawn on the step that the pixel offsets show.

## The pipeline, step by step

### 1. Registration

**The problem.** ASF's GSLCs each apply their own ionospheric and solid Earth tide geolocation correction. Between two
dates these can differ by about a resolution cell in both range and azimuth, which costs a large part of the pair's
coherence. Because NISAR keeps the Doppler carrier in its SLCs, an along-track misregistration \(\Delta t\) also biases
the phase, by \(-2\pi f_\mathrm{dc}\,\Delta t\).

**What is done.** The first date's ASF GSLC is kept. The second date's RSLC is geocoded with `geocode_rslc` (the port of
ISCE3's GSLC workflow) onto the first date's grid (`like=`), with `tec_file=None`, `solid_earth_tides=False` and a
`RegistrationField` that holds the first date's own TEC and tide shifts plus the measured offsets between the two dates.

- **The offsets** come from the package's ROFF (`roff.dense_offsets`, a numpy port of ISCE3's offsets workflow, whose
  correlator is otherwise GPU-only), densely near the rupture and on a coarse lattice elsewhere. Far from the rupture a
  GUNW's `pixelOffsets` can supply the frame-scale part, joined to the ROFF by a taper. Those layers are not measured
  offsets but ISCE3's fitted surface (below), so there the registration follows only that surface.
- **Band B** takes the same shifts in metres, converted to its own pixels, with its own TEC shift (an ionospheric
  geolocation shift scales as \((f_A/f_B)^2\); a tide shift does not).
- **Mixing a corrected GSLC with an uncorrected one is safe**: the corrections only move where pixels are sampled, and
  both ISCE3 and the port flatten the phase with the uncorrected range.
- **The carrier term.** When the field follows an along-track shift, the Doppler carrier is re-ramped at the shifted
  position, as ISCE3's resampler also does. That is right for motion and geolocation. For the part of the shift caused
  by the ionosphere it leaves \(-2\pi f_\mathrm{dc}\,s/V_g\) in the phase (\(s\) the shift on the ground, \(V_g\) the
  ground-track velocity), about 17 mm of LOS per metre of shift. It scales with frequency, so no split-spectrum screen
  removes it; the screen's along-track gradient predicts it.

**The GUNW** coregisters the two RSLCs in radar geometry and geocodes the result. A "rubbersheet" refines the geometry;
in the solid-earth PR products this is one quadratic per frame and offset component, fitted to dense offsets with
outlier removal, and the GUNW's `pixelOffsets` layers are that quadratic. A GUNW pair is therefore registered at the
scale of the frame but cannot follow a coseismic offset step. ISCE3 also has a dense branch that applies a filtered
offset field. The other difference is where the pair is formed: here on the 5 m GSLC grid.

### 2. Interferograms

- `GSLCStack.form_interferograms(pairs="sequential", looks=10, convolution="Gaussian", sigma=...)`: reference times
  conjugate secondary at full resolution, multilooked 10 × 10 with Gaussian weights onto a 50 m lattice anchored to
  absolute coordinates. The coherence comes from the same looks.
- **Frames are merged before forming** (`GSLCStack.merge`), so there is one interferogram across a frame boundary, one
  unwrapping, and one integer.
- **The GUNW** is posted at 80 m and processes each frame separately, so each frame carries its own integer and ramp.

### 3. Masks and regions

- **Water** is masked with the full-resolution coastline (`mask_water(resolution="f")`). Nothing else is masked; in
  particular there is no coherence mask.
- **The fault cut** (coseismic pairs only): the interferogram is set to NaN within ±200 m of the mapped trace. Near a
  rupture the phase is decorrelated and its gradient steep, and an unwrapper that integrates across it lays an
  arbitrary number of cycles between the two sides. With the cut, each side is unwrapped on its own. The cut has to
  reach the edge of the data, or the unwrapper goes around its end. The package has no function for this step; it is a
  distance mask on the interferogram's grid.
- **Regions.** `remove_unconnected_regions(min_size=...)` drops the specks, then a region rule keeps the largest region,
  the blocks the cut isolates and any large islands worth keeping, and drops every other region. Every region kept
  besides the largest carries an arbitrary integer.
- **The GUNW** masks water, invalid and partially focused samples, and flags anomalies; it does not cut the rupture.

### 4. Filtering and unwrapping

- The adaptive Goldstein filter (`filter_goldstein(alpha="adaptive")`) on the 50 m interferogram, then SNAPHU in smooth
  mode (`unwrap`). SNAPHU's deformation mode does not make the integer across a rupture reliable, which is why the cut
  is used instead.
- **The GUNW** is unwrapped by ISCE3, frame by frame, without a cut and across the rupture.

### 5. Tying cut-off blocks

A block that the cut isolates is unwrapped with an arbitrary integer. Two independent readings tie it to the main region.

- **The range offsets.** The phase predicted by the ROFF slant-range offsets minus the unwrapped phase,
  \(m = (4\pi/\lambda)\,\Delta r - \phi\), contains no deformation, because the offsets and the phase see the same
  ground motion. What remains is the block's whole cycles plus twice the ionosphere, which advances the phase and
  delays the group by equal amounts. Smooth surfaces fitted to \(m\) on each side of the trace, a few kilometres away
  where the phase is reliable, are extrapolated to the trace, and the nearest integer is applied.
- **The side band** (`split_spectrum.block_cycles`). A block whose main-band integer is off by \(e\) cycles moves its raw
  dispersive estimate by \(3.21\,e\) rad (see step 6), while the ionosphere itself is continuous across the gap. A
  quadratic over both blocks plus a step, fitted in windows along the trace and reweighted against local cycle errors,
  reads \(e\); the median over the windows is the answer.
- Correcting the offset tie with the screen does not help: the screen depends on the block's integer in nearly the same
  way as the tie, so the two together cannot separate the integer from the ionosphere. The screen's continuity across
  the gap is the independent check.
- Because a tie can still be a cycle off, each region keeps its own offset in the slip inversion, whatever the tie.
- **Patch checks.** Where another unwrapping of the same pair disagrees by whole cycles (one without the cut, or one
  with the offsets' phase taken out before SNAPHU), `split_spectrum.cycle_patches` lists the patches, and
  `patch_cycles` and `patch_verdict` read which unwrapping the side band supports in each.
- **The GUNW** has no cut and nothing to tie; its unwrapper chooses the integer across the rupture.

### 6. The split-spectrum ionosphere

The physics is the same for any split spectrum. The phase of band X is

\[
\phi_X = \frac{f_X}{f_A}\,\phi_\mathrm{nd} + \frac{f_A}{f_X}\,\phi_\mathrm{io} + 2\pi n_X ,
\]

with \(\phi_\mathrm{nd}\) everything non-dispersive (deformation, topography, troposphere) and \(\phi_\mathrm{io}\) the
ionosphere at \(f_A\). What differs between estimators is how the two bands are combined (`split_spectrum`).

- **Cells.** Both bands' interferograms are summed over identical 240 m map cells, straight from the registered GSLCs
  (`gslc_cells`). Pixels the product mask marks as partially focused are left out of the cells only in band A's
  near-range strip, roughly the 10 km next to the near-range swath edge (`subswath_mask="near_range"`). The partially
  focused start of a datatake is kept, since dropping it would make the screen extrapolate over tens of kilometres.
  With `phase_model=`, the unwrapped band A is taken out of every pixel of both bands before the sums, scaled to each
  band's frequency, so that steep fringes inside a cell neither decorrelate it nor weight the two bands differently.
- **Band A from the fault-aware unwrapping.** \(\phi_A\) is the unwrapped band-A phase on the cells, from the main region
  and tied blocks only: an untied island's arbitrary integer would shift its screen by 3.21 rad per cycle.
- **The residual.** Band B is never unwrapped. It enters through

    \[
    r = \operatorname{wrap}\!\left(\phi_B - \frac{f_B}{f_A}\,\phi_A\right)
      = \operatorname{wrap}\!\left(k\,\phi_\mathrm{io} - 0.044 \cdot 2\pi n_A\right),
    \qquad k = \frac{f_A}{f_B} - \frac{f_B}{f_A} = -0.0861 .
    \]

    The non-dispersive term, deformation included, cancels exactly, so \(r\) is small and smooth. It is unwrapped
    (`unwrap_smooth_phase`), and the screen is \(\phi_\mathrm{io} = r/k\). The unwrapping low-passes \(e^{ir}\), fills
    gaps with wider Gaussians and runs SNAPHU twice: over the whole grid, and over the data's reach alone, a couple of
    kilometres past the cells. Each connected piece of the reach takes the whole-grid solution's integer, so the fill
    of a large gap cannot cut across connected data.

- **Which cells.** Coherence above 0.2 in both bands, at least half the pixels valid, and more than 2 km from the trace.
- **The filter.** ISCE3's ionosphere low-pass filter (`IonosphereFilter.low_pass_filter`), ported in metres: a
  Gaussian of 2.0 × 2.6 km, a guide, outliers beyond 3.5 median absolute deviations removed, inverse-variance weights.
  One change: masked cells are filled with a local plane (10 km wide, at most 12 km from the data) instead of ISCE3's
  zero-order guide, which lags a gradient at a one-sided edge, such as a coast along the fault together with the 2 km
  exclusion. A ladder of plane widths and a first-order final step are available as options.
- **Applied.** The screen is interpolated onto the 50 m grid and subtracted from \(\phi_A\).

What this buys, and what it still depends on:

- **No deformation in the screen**, because \(r\) contains none.
- **No differential errors from band B**, because band B has no integer of its own to get wrong.
- **No cycle corrections**, so nothing can mistake deformation for a cycle error.
- **It depends on band A's integers.** One cycle of \(\phi_A\) on a region moves its screen by
  \(2\pi f_B/(f_A + f_B) \approx 3.21\) rad, about 62 mm of LOS.
- **And on \(r\)'s integers.** One cycle of \(r\) is \(2\pi/|k| \approx 73\) rad of screen, about 1.4 m of LOS. \(r\)
  is smooth, but the fill of a large gap averages data whose \(r\) differs by up to π and can leave a phase singularity
  there, from which a whole-grid unwrapping must cut to the grid's edge, possibly across the data. Unwrapping over the
  data's reach prevents that within any connected piece of data; pieces further apart than the reach are still tied
  through the fill.

**The GUNW's split spectrum** (ISCE3's `main_diff_ms_band`, \(\phi_\mathrm{io} = 0.511\,\phi_A + 11.61\,(\phi_A -
\phi_B)\)) differs in five ways:

1. Both bands are unwrapped, and the band difference, which still carries 4.4% of the deformation, is unwrapped on its
   own. Both are bridged across gaps by the median difference at the nearest boundary pixels.
2. Two cycle-correction passes run. First each unwrapped field is rounded by whole cycles against its own low-pass;
   then band A is rounded against the first pass's filtered non-dispersive and dispersive estimate, and the band
   difference against its prediction. Near a rupture these passes read the deformation's curvature as cycle errors, at
   3.21 rad of screen per false main-band cycle.
3. Band A is decimated to band B's columns by stride, which leaves a constant offset of about 9 m in range.
4. The two bands' footprints are very different (about 40 m against 325 m of slant range).
5. The filter fills gaps with ISCE3's zero-order guide.

The result can be a sign-flipped copy of part of the deformation near a rupture. Away from ruptures the two estimators
agree closely. The GUNW's `unwrappedPhase` is not corrected for the ionosphere: its screen is the separate
`ionospherePhaseScreen` layer.

### 7. Line of sight

- `to_los` converts to \(d = -(\lambda/4\pi)\,\phi\), positive toward the sensor, with look vectors from the products'
  geometry cubes (see [Line of sight and export](guide/los-and-export.md)). The 50 m LOS stacks go to the slip
  inversion, which samples them without any smoothing beforehand.
- The ROFF offsets can enter as data too (`roff.offset_los_stacks`: slant range on the LOS vector, along-track on the
  along-track vector).

## How it differs from the ISCE3 PR products

| | ISCE3 PR GUNW | This pipeline |
|---|---|---|
| Pair | RSLCs coregistered in radar geometry; geometry plus one quadratic per frame fitted to dense offsets | the first date's ASF GSLC and a second-date GSLC geocoded onto its grid, with a registration field from the package's ROFF |
| Posting | 80 m | 50 m (10 × 10 Gaussian looks on 5 m GSLC pixels) |
| Frames | each processed separately | merged before forming, one unwrapping |
| Partially focused samples | dropped (near-range strip, start of a datatake) | kept in the data; left out of the screen's cells only in band A's near-range strip |
| Rupture | unwrapped across | ±200 m cut; an isolated block tied by offsets or the side band, and given its own offset |
| Masks | water, validity, anomalies; coherence and component layers | water only |
| Unwrapping | ISCE3, per frame | SNAPHU (smooth) on the Goldstein-filtered 50 m interferogram |
| Ionosphere | both bands unwrapped, band difference unwrapped, both bridged, two cycle-correction passes, band A decimated, zero-order fill | band A only; a deformation-free residual on identical cells; no cycle corrections; 2 km fault zone filled; plane fill |
| Ionosphere in the phase | a separate layer, not applied | subtracted |

## Limitations

- **Tie ambiguity.** The offset tie and the screen carry the block's integer in nearly the same way, so only the
  screen's continuity across the gap (`block_cycles`) separates the integer from the ionosphere. That reading is
  biased when the ionosphere carries waves of 10 to 20 km along the boundary, or when many cells beside the gap carry
  local cycle errors. A wrong cycle shifts the block's corrected LOS by a constant far from the trace, which the
  block's own offset in the inversion absorbs; near the trace, the filter spreads the 3.21 rad step of the raw
  estimate into a spurious near-trace step.
- **Near-trace integers.** Within a few kilometres of a rupture, the main band's own integers can be locally wrong, and
  the screen depends on them. The wide plane fill and the 2 km exclusion keep the screen robust to them; narrower
  fills follow them. The patch checks can decide only where the cells are coherent, so most disagreements within
  about 2 km of the trace stay undecided.
- **Partially focused data** at the start of a datatake are kept, and cannot be checked against a GUNW there, since the
  GUNW drops them.
- **The filter's own uncertainty** (ISCE3's output sigma) is larger than most small-scale residuals, so it cannot say
  whether a feature is noise.
- **The carrier term** (step 1) stays in the phase wherever the registration follows an ionospheric along-track shift.
  Its long-wavelength part is absorbed by the ramps an inversion solves for; within kilometres of the trace, where the
  field follows the ROFF, its size has not been established.
