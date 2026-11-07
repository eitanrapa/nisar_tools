# Merging frames

A track is delivered as a series of frames, and an area of interest rarely sits inside one. Frames can be joined at two
stages.

## Before interferograms

```python
stack = (
    GSLCStack.from_gslcs(gslcs_frame_1, bbox=bbox)
    .merge(GSLCStack.from_gslcs(gslcs_frame_2, bbox=bbox))
    .persist(ws, "slc_stack", files=[paths_frame_1, paths_frame_2], bbox=bbox)
)
```

`GSLCStack.merge` gives one continuous complex stack on the union of the two grids, with `self` taking precedence in
the overlap. The acquisitions are paired by nearest time within `time_tolerance` seconds (600), since the frames of
one pass share a track but not an exact start time. This is the simpler route when both frames have the same
acquisitions, and it gives one interferogram, so one unwrapping and one integer across the frame boundary.

## After unwrapping

```python
unw = unw_frame_1.merge(unw_frame_2)
```

`UnwrappedStack.merge` also removes the 2π step between frames unwrapped separately: across the seam their phases
agree up to a whole number of cycles, which is read from the overlap (the rounded median of the difference) and added
to the second frame. It is the only route when the frames come from different tracks or dates. SNAPHU's component
labels of the second frame are shifted clear of the first's. If SNAPHU split the overlap into components at different
cycle offsets, the median aligns the dominant one; fix the rest with `add_cycles(conncomp=)`.

After a merge, pass one granule per frame to `to_los`, in the merge order: each granule's geometry cube covers only its
own frame.

```python
los = unw.to_los([granule_frame_1, granule_frame_2])
```

## Grids

- Both frames must lie on the same lattice. Interferograms formed with `align_looks=True` (the default) do: the
  multilook blocks are anchored to the absolute native grid, not to where each crop started. Without it, two frames
  cropped differently come out a fraction of a multilooked pixel apart.
- `UnwrappedStack.merge` resamples an off-lattice frame, with a warning; re-forming both interferograms with
  `align_looks=True` avoids the resampling. A different pixel spacing (different `looks`) raises.
- `GSLCStack.merge` refuses an off-lattice pair: interpolating a complex SLC aliases its fringes, so a misaligned pair
  must be re-cropped, not resampled.
- **Across a UTM zone boundary**, the second frame is warped onto the first frame's grid before the merge. Label
  layers (`conncomp`, `subswath_mask`) are always resampled by nearest neighbour; `resampling="nearest"` keeps exact
  values for the others too. The warp leaves rotated wedges of no data, which `crop` can trim.

## Different dates

Two frames from different dates are different interferograms, not one interferogram in two pieces. Pass
`time_tolerance=None` to pair them by position, and choose with `tie` how the overlap is reconciled:

| `tie` | Removes | Right when |
|---|---|---|
| `"cycles"` | a whole number of 2π | one interferogram unwrapped separately per frame, which can differ by nothing else |
| `"offset"` | the real-valued median difference | different date pairs, which measure different deformation, so the difference is not a whole number of cycles |
| `"none"` | nothing | the two are already on a common datum |
| `"auto"` | `"cycles"` when the pairs matched by time, `"offset"` otherwise | the default |

!!! warning "Keep scenes apart for an inversion"
    A mosaic of two different-date scenes is one raster with one arbitrary constant and one ramp. In a slip inversion
    the second scene's independent constant and ramp are then absorbed as spurious slip, while the fit statistics stay
    good. Merge for a continuous map or an export; sample each scene separately and combine the observations for an
    inversion.
