# Workspaces

A `Workspace` is a directory of Zarr stores, one per persisted stage. Each store records the parameters that produced
it, hashed, so a rerun can tell a finished stage from one that has to be recomputed.

```text
work/
├── workspace.json          # created timestamp
├── slc_stack.zarr          # one store per persisted stage
├── igrams.zarr
├── unwrapped.zarr
└── unwrapped.done.json     # per-pair progress, unwrapping only
```

```python
from nisar_tools import Workspace

ws = Workspace("work")                  # creates the directory if needed
ws = Workspace("work", create=False)    # open an existing one; never writes
```

## Persisting and resuming

Every stage method returns a new, lazy object; nothing is computed until `persist`:

```python
igrams = stack.form_interferograms(looks=10).persist(ws, "igrams")
```

`persist(ws, name, overwrite=False, **params)` computes the stage, writes it to `name` and returns the stage reopened
from disk. Reopening cuts the link to whatever the stage was computed from, such as open HDF5 granules. Then:

- If the store exists and was made with the same parameters, `persist` returns it without computing anything. This is
  how a rerun of a script resumes.
- If it exists with different parameters, `persist` raises `WorkspaceError` rather than replacing the results. Pass
  `overwrite=True` to rebuild it, or use another stage name.
- A store left incomplete by an interrupted write is recomputed, with a warning. Stores are written beside their
  target and renamed into place only once complete.
- Persisting a stage over the store it is being read from raises.

The recorded parameters are the stage's own settings (looks, filter strength, masks applied, and so on) plus any
keyword arguments you pass. Pass the inputs that define a stage, so that a rerun with other inputs raises instead of
reusing the old store:

```python
stack = GSLCStack.from_gslcs(gslcs, bbox=bbox).persist(ws, "slc_stack", files=paths, bbox=bbox)
```

!!! warning "Edits made outside the stage methods are not recorded"
    A stage changed by editing its `ds` directly (for example, a mask you apply yourself) records nothing about that
    change. Persisted under the name of an existing store, it returns the old store without a word. Give it a new name,
    pass the change as a keyword parameter, or pass `overwrite=True`.

## Which steps write

Most steps are lazy. Unwrapping is the exception: SNAPHU needs a whole raster, so `igrams.unwrap(ws, name=...)`
creates its store first, then unwraps and writes one pair at a time and marks each one done. When it returns, the store
exists. An interrupted unwrap resumes at the first unfinished pair.

The unwrapping store is checked the same way. It is keyed on the unwrapping settings, and it also records the
interferogram it was made from: its looks, pairs, masks, filtering, corrections and grid. With the same settings and
the same interferogram, `unwrap` resumes, or returns, the store that exists. With other settings, or an interferogram
processed differently, it raises `WorkspaceError`: pass `overwrite=True` to rebuild it, or use another `name`. A store
written before the interferogram was recorded is reused, with a warning. As for any stage, an edit made to the
interferogram's `ds` directly is not recorded.

The rule is that a method taking the workspace as its first positional argument writes, and one taking it as a keyword
does not. `mask_water(mask_cache=ws)` is lazy: the workspace caches only the coastline mask, keyed on the grid, not the
masked data.

## Reloading finished stages

Stores open lazily, so reopening even a very large stage costs nothing until something is computed:

```python
from nisar_tools import GSLCStack, InterferogramStack, UnwrappedStack

stack = GSLCStack.from_zarr(ws.path("slc_stack"))
igrams = InterferogramStack.from_zarr(ws.path("igrams"))
ds = ws.load("igrams")                    # the xarray.Dataset itself
ws.exists("unwrapped"), ws.pairs_done("unwrapped")
```

## Clearing

```python
ws.clear("igrams")        # delete one stage and its progress file
```

Nothing cascades: after re-forming interferograms, clear or rename the stages built from them too. To start over,
delete the directory.
