# Ionosphere

The ionosphere is dispersive: its contribution to the interferometric phase scales as \(1/f\), while everything else
(topography, deformation, orbit errors, the troposphere) scales as \(f\). NISAR records each acquisition in two sub-bands,
a main band A and a side band B, so one pair of acquisitions separates the two. With \(\phi_\mathrm{nd}\) the
non-dispersive phase and \(\phi_\mathrm{io}\) the ionospheric phase, both at band A's carrier, the phase of band X is

\[
\phi_X = \frac{f_X}{f_A}\,\phi_\mathrm{nd} + \frac{f_A}{f_X}\,\phi_\mathrm{io} + 2\pi n_X ,
\]

where \(n_X\) is the band's unknown integer. The package has two estimators built on this. They differ in how the two
phases are obtained, which matters most next to a rupture.

## Dual-frequency correction from two unwrapped bands

`nisar_tools.ionosphere` solves the two equations directly. At NISAR's carriers (1.239 and 1.2935 GHz) the ionospheric
phase at band A is \(\phi_\mathrm{io} = 12.12\,\phi_A - 11.61\,\phi_B\).

```python
from nisar_tools import GSLC, GSLCStack

unw = {}
for band, looks in (("A", 16), ("B", 2)):        # 80 m in x for both bands
    stack = GSLCStack.from_gslcs([GSLC(p, frequency=band) for p in paths], bbox=bbox)
    unw[band] = stack.form_interferograms(looks=looks).unwrap(ws, name=f"unw_{band}")

corrected = unw["A"].correct_ionosphere(unw["B"])  # subtracts the screen, keeps it as phase_screen
```

The result is band A's non-dispersive phase, still at band A's wavelength, so `to_los` needs no adjustment; the screen
is kept as the `phase_screen` layer. To take the screen out before unwrapping instead, which leaves the unwrapper fewer
fringes, estimate it and rotate the wrapped interferogram by it:

```python
from nisar_tools import ionosphere

screen = ionosphere.dual_frequency_screen(unw["A"], unw["B"], min_coherence=0.3)
unw_a = igrams_a.correct_ionosphere(screen).unwrap(ws, name="unw_A_corrected")
```

The screen is interpolated onto whatever grid it is applied to, so it can be estimated from a coarsely multilooked pair
and applied to a finer one.

Things to know:

- **Both inputs must be unwrapped.** Both coefficients are large, so an error of one cycle in either band moves the
  screen by 73 to 76 rad. Nor does the complex product of the two bands help: the dispersive phase also equals
  \(0.511\,\phi_A + 11.61\,(\phi_A - \phi_B)\), which needs band A's absolute phase, not only the difference. A screen
  can be *applied* to wrapped phase, but not estimated from it.
- **The low-pass is not optional.** The combination amplifies independent phase noise by
  \(\sqrt{12.12^2 + 11.61^2} \approx 16.8\), so an unsmoothed screen makes the data worse. `scale` sets the Gaussian's
  sigma in pixels of the corrected stack; the default is 2 km of ground. Each band is smoothed on its own grid before
  the two are combined, because band B is posted 8 times coarser in x and resampling its raw phase would put every
  interpolation error through the large coefficients. It warns when the scale is below twice the coarser posting.
- **`remove_offset=True`** removes each pair's median from the screen, whose absolute level is arbitrary.
- **It is not a deramp.** It does not remove orbital ramps or the troposphere, which are not dispersive. A deramp, on
  the other hand, is a fit to the very field it cleans, so where deformation spans the scene it removes part of it; the
  dual-frequency screen does not, because the deformation is not dispersive. A deramp always leaves less variance; the
  screen is a measurement.
- A GUNW already carries NASA's screen as `phase_screen`; `correct_ionosphere(..., replace=True)` replaces it.

## Split-spectrum screen without the deformation

Near a rupture, unwrapping both bands is where the deformation gets into the screen: every error in the side band's
own unwrapping, and every whole-cycle "correction" made against a smoothed non-dispersive field, moves the screen.
`nisar_tools.split_spectrum` therefore never unwraps band B. It takes band A's phase from the caller's own unwrapping
(which can be cut along the fault and tied, see [the pipeline](../insar_pipeline.md)), and with that unwrapped
\(\phi_A\) unwraps only the residual

\[
r = \phi_B - \frac{f_B}{f_A}\,\phi_A = \left(\frac{f_A}{f_B} - \frac{f_B}{f_A}\right)\phi_\mathrm{io} = -0.0861\,\phi_\mathrm{io}
\pmod{2\pi},
\]

which holds no deformation at all. Both bands are summed over the same map cells, so their footprints match exactly.

It works on a GSLC pair on one grid (for example a secondary registered with
[`geocode_rslc(like=, registration=)`](rslc-to-gslc.md#registering-a-secondary)) and on band A's unwrapped stack, `unw`:

```python
import numpy as np
from nisar_tools import UnwrappedStack, ionosphere, split_spectrum as ss

ref, sec = "data/<date 1 GSLC>.h5", "data/<date 2 GSLC>.h5"      # one grid, both sub-bands
bounds = (unw.x.min(), unw.y.min(), unw.x.max(), unw.y.max())
cells_a = ss.gslc_cells(ref, sec, "A", cell=240.0, bounds=bounds, subswath_mask="near_range")
cells_b = ss.gslc_cells(ref, sec, "B", cell=240.0, bounds=bounds, subswath_mask="near_range")

# Band A's unwrapped phase on the same cells: this is where a fault-aware unwrapping enters.
phase_a, weight = ss.cell_average(unw.ds["unw"].isel(pair=0).values, unw.x, unw.y, cells_a)
covered = weight / (240.0 / abs(unw.x[1] - unw.x[0])) ** 2 > 0.5

# Leave out the cells within 2 km of the mapped trace (trace_x, trace_y in map metres).
near = ss.fault_distance(cells_a, trace_x, trace_y) < 2000.0

out = ss.split_spectrum_screen(cells_a, cells_b, phase_a, valid=np.isfinite(phase_a) & covered,
                               exclude=near, fill="plane", fill_m=10000.0, fill_reach_m=12000.0)

# Onto the interferogram's grid, and out of band A's phase.
screen = ionosphere.interp_to_grid(out["screen"], unw.y, unw.x)
ds = unw.ds.copy()
ds["unw"] = (unw.ds["unw"] - screen).astype("float32")
corrected = UnwrappedStack(ds).persist(ws, "unwrapped_iono", ionosphere="split spectrum")
```

- `gslc_cells` sums one band's `reference * conj(secondary)` over square map cells (240 m by default) that both bands'
  grids share. `subswath_mask="near_range"` leaves out the partially focused samples next to the near-range swath edge,
  which are finite but carry phase errors that would become a strip in the screen. `phase_model=` (band A's unwrapped
  phase on a regular grid) takes the phase out of every pixel of both bands before the sums, so that steep fringes
  inside a cell neither decorrelate it nor weight the two bands differently; `phase_a` is then optional.
- `split_spectrum_screen` also requires coherence above `min_coherence` (0.2) in both bands and at least half of each
  cell's pixels valid. The filter is ISCE3's ionosphere low-pass (outlier rejection from a guide, gap filling,
  inverse-variance-weighted truncated Gaussian), sized in metres. `fill="plane"` fills gaps with a local plane instead
  of ISCE3's zero-order guide, which lags the gradient at a one-sided edge such as a coast along a fault; `fill_m` is
  the plane's width and `fill_reach_m` how far it may extrapolate. `order=1` makes the final smoothing a weighted local
  plane as well.
- The output is a dataset on the cells: `screen` (radians at band A's carrier), ISCE3's `sigma`, the raw `dispersive`
  estimate, `mask`, `residual` and the cell coherences. Pixels beyond the outermost cell centres come out NaN in
  `interp_to_grid`, so make the cells cover the interferogram.
- A region unwrapped on its own, such as a block cut off along a rupture, can be tied to the main region by the side
  band itself: `ss.block_cycles(out, block, reference)` reads the whole cycles that make the block's raw estimate
  continue the reference's across the gap. Where two unwrappings of the same pair disagree, `ss.cycle_patches`,
  `ss.patch_cycles` and `ss.patch_verdict` read which one the side band supports, patch by patch.

The screen still depends on band A's integers: a region whose band-A integer is off by one cycle moves its screen by
\(2\pi f_B/(f_A + f_B) \approx 3.21\) rad. The [pipeline](../insar_pipeline.md#6-the-split-spectrum-ionosphere) page
explains the design and its limits.
