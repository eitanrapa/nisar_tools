"""Zarr-backed workspace: stage persistence, parameter-hashed resume.

A :class:`Workspace` owns a directory of Zarr stores, one per pipeline stage.
Each store records the parameters that produced it, hashed, so a re-run can
cheaply decide whether a finished stage can be reused or must be recomputed.

The design is deliberately simple: no dependency graph, no automatic cascade
invalidation. A stage is reusable iff its store exists and its parameter hash
matches. The only sub-stage granularity is per-pair "done" markers for the
expensive, interruptible unwrap stage.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

# zarr v2 format is required for complex64 (the v3 spec has no complex dtype).
_ZARR_KWARGS = {}
try:  # zarr-python 3.x accepts zarr_format; 2.x does not need it.
    import zarr

    if int(zarr.__version__.split(".")[0]) >= 3:
        _ZARR_KWARGS = {"zarr_format": 2}
except Exception:  # pragma: no cover - zarr always present in practice
    pass


def _default_compressor():
    """Codec for stage data variables.

    Zarr v2's default is ``Blosc(lz4, clevel=5, shuffle)``, which on SAR data
    earns almost nothing — measured 1.07:1 on a complex64 stack — while costing
    about 2.5x the write time. Dropping to ``clevel=1`` measured roughly twice
    the throughput at the same ratio. ``shuffle`` is kept because the float32
    stages (unwrapped phase, coherence) do compress, even though complex64
    noise does not.
    """
    try:
        import numcodecs
    except ImportError:  # pragma: no cover - numcodecs ships with zarr
        return "default"
    return numcodecs.Blosc("lz4", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)


def _encoding_for(ds):
    """Per-data-variable compressor encoding for ``to_zarr``.

    Coordinates are left alone (they are tiny), and the codec is deliberately
    absent from the params hash: it does not change the data, so switching it
    must not invalidate a finished stage. Stores written with the old default
    stay readable — Zarr records the codec per array.
    """
    compressor = _default_compressor()
    if compressor == "default":
        return None
    return {name: {"compressor": compressor} for name in ds.data_vars}


class WorkspaceError(RuntimeError):
    """Raised when a stage exists but was produced with different parameters."""


def hash_params(params):
    """Deterministic short hash of a JSON-serializable parameter dict."""
    canonical = json.dumps(params, sort_keys=True, default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_default(obj):
    # Best-effort serialization for numpy scalars, Paths, tuples of paths, etc.
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# Attributes that lead from a dask task down to the array it reads. Covers the
# xarray lazy-indexing wrappers (``array``), xarray's ZarrArrayWrapper
# (``_array``) and dask's task objects (``func``/``args``/``kwargs``).
_SOURCE_ATTRS = ("array", "_array", "func", "args", "kwargs")

# Scalars and buffers that can never contain a store reference.
_LEAF_TYPES = (str, bytes, int, float, bool, complex, type(None), slice)


def _resolve_store_path(store):
    """Filesystem path behind a Zarr store, unwrapping wrapper stores.

    ``open_zarr(consolidated=True)`` hands back a ConsolidatedMetadataStore
    wrapping the real one, so follow the chain a few links.
    """
    for _ in range(6):
        for attr in ("path", "dir_path", "root"):
            value = getattr(store, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, (str, Path)) and str(value):
                return Path(value)
        nxt = getattr(store, "store", None)
        if nxt is None:
            nxt = getattr(store, "map", None)
        if nxt is None or nxt is store:
            return None
        store = nxt
    return None


def _zarr_source_paths(ds, per_layer=8, max_nodes=20000):
    """Resolved paths of the Zarr stores ``ds`` would still read from lazily.

    A dataset opened from Zarr keeps that store alive inside its lazy data — in
    a dask graph when chunked, in xarray's indexing wrappers when not — so
    deleting the store before the write computes silently corrupts the result.

    Only *lazy* data counts. ``ds.encoding["source"]`` survives
    :meth:`~xarray.Dataset.compute`, so it would flag an in-memory dataset that
    is perfectly safe to write back; conversely xarray drops encoding whenever a
    dataset is rebuilt from its variables, which :meth:`GSLCStack.merge` and
    interferogram formation both do. The data itself is the reliable signal.

    Dask graphs are sampled a few tasks per layer rather than walked in full: a
    layer's tasks all share the same store object, so sampling finds it for a
    fraction of the work on a large stack.

    Best effort by design. Any failure returns what was found so far, because a
    missed source only restores the old (silently lossy) behaviour, whereas a
    false positive would refuse a write that is perfectly safe.
    """
    import numpy as np
    import zarr

    paths = set()
    seen = set()
    visited = 0

    def _walk(roots):
        nonlocal visited
        pending = list(roots)
        while pending and visited < max_nodes:
            obj = pending.pop()
            visited += 1
            if id(obj) in seen:
                continue
            seen.add(id(obj))

            if isinstance(obj, zarr.Array):
                resolved = _resolve_store_path(obj.store)
                if resolved is not None:
                    paths.add(resolved)
                continue
            # Real buffers are leaves. Note that xarray's lazy wrappers also
            # expose ``shape``, so only concrete arrays may be skipped here.
            if isinstance(obj, (np.ndarray, np.generic)) or isinstance(obj, _LEAF_TYPES):
                continue
            if isinstance(obj, (list, tuple, set, frozenset)):
                pending.extend(obj)
                continue
            if isinstance(obj, dict):
                pending.extend(obj.values())
                continue
            for attr in _SOURCE_ATTRS:
                try:
                    child = getattr(obj, attr, None)
                except Exception:
                    continue
                if child is not None:
                    pending.append(child)

    for var in ds.variables.values():
        data = getattr(var, "_data", None)
        if data is None or isinstance(data, (np.ndarray, np.generic)):
            continue  # already in memory: writing it reads nothing from disk

        graph = getattr(data, "dask", None)
        if graph is None:
            # Lazy but not chunked (e.g. open_zarr(chunks=None)); the wrapper
            # chain hangs directly off the variable's data.
            _walk([data])
            continue

        layers = getattr(graph, "layers", None)
        buckets = list(layers.values()) if layers else [graph]
        for layer in buckets:
            roots = []
            try:
                for i, value in enumerate(layer.values()):
                    if i >= per_layer:
                        break
                    roots.append(value)
            except Exception:
                continue
            _walk(roots)

    return {p.resolve() for p in paths}


def _check_not_reading_target(ds, target, name):
    """Refuse to delete a store that ``ds`` is still lazily reading from.

    :meth:`Workspace.store` removes the target directory *before* ``to_zarr``
    computes, so if the dataset's graph reads from that same store the delete
    races the reads. That does not raise on its own — it yields a
    correctly-shaped array with a varying fraction of pixels silently turned to
    NaN — so it is caught up front instead.
    """
    try:
        sources = _zarr_source_paths(ds)
    except Exception:  # detection must never break a legitimate write
        return
    if target.resolve() in sources:
        raise WorkspaceError(
            f"Stage '{name}' cannot be overwritten from data that reads it: "
            f"{target} is still an input to this dataset, and overwriting "
            "deletes it before the write computes, which silently corrupts "
            "the result. Persist to a new stage name instead (and clear the "
            "old one afterwards if you no longer need it)."
        )


class Workspace:
    """A directory of per-stage Zarr stores."""

    def __init__(self, workdir, create=True):
        self.workdir = Path(workdir)
        if create:
            self.workdir.mkdir(parents=True, exist_ok=True)
            meta = self.workdir / "workspace.json"
            if not meta.exists():
                meta.write_text(
                    json.dumps(
                        {
                            "created": datetime.now(timezone.utc).isoformat(),
                            "tool": "nisar_tools",
                        },
                        indent=2,
                    )
                )

    # -- paths -------------------------------------------------------------
    def path(self, name):
        """Filesystem path of the Zarr store for stage ``name``."""
        return self.workdir / f"{name}.zarr"

    def _done_path(self, name):
        return self.workdir / f"{name}.done.json"

    # -- existence / resume ------------------------------------------------
    def exists(self, name):
        return self.path(name).exists()

    def stored_params_hash(self, name):
        if not self.exists(name):
            return None
        ds = xr.open_zarr(self.path(name))
        try:
            return ds.attrs.get("params_hash")
        finally:
            ds.close()

    def has(self, name, params):
        """True if stage ``name`` exists and was built with ``params``."""
        if not self.exists(name):
            return False
        return self.stored_params_hash(name) == hash_params(params)

    # -- whole-dataset store / load ---------------------------------------
    def store(self, name, ds, params, overwrite=False):
        """Write ``ds`` to the stage store and return the reopened lazy dataset.

        Reopening from Zarr (rather than returning ``ds``) severs any upstream
        dependency, e.g. open HDF5 file handles, and gives downstream stages
        well-chunked local reads.
        """
        target = self.path(name)
        if target.exists():
            if not overwrite and self.stored_params_hash(name) == hash_params(params):
                return self.load(name)
            if not overwrite:
                raise WorkspaceError(
                    f"Stage '{name}' already exists with different parameters. "
                    f"Pass overwrite=True or use a fresh workspace."
                )
            _check_not_reading_target(ds, target, name)
            shutil.rmtree(target)

        ds = ds.copy()
        ds.attrs["params"] = json.dumps(params, default=_json_default)
        ds.attrs["params_hash"] = hash_params(params)
        encoding = _encoding_for(ds)
        ds.to_zarr(
            target, mode="w", consolidated=True,
            **({} if encoding is None else {"encoding": encoding}),
            **_ZARR_KWARGS,
        )
        return self.load(name)

    def load(self, name):
        """Open stage ``name`` lazily, with its CRS coordinate restored.

        Goes through :func:`~nisar_tools._base.open_stage` rather than
        ``xr.open_zarr`` because ``store`` returns ``load``'s result -- so this
        is the one place every persisted stage is reopened, including the one
        handed straight back from ``persist``.
        """
        from ._base import open_stage

        if not self.exists(name):
            raise WorkspaceError(f"Stage '{name}' does not exist in {self.workdir}.")
        return open_stage(self.path(name))

    def clear(self, name):
        """Delete a stage store and any done-markers."""
        target = self.path(name)
        if target.exists():
            shutil.rmtree(target)
        done = self._done_path(name)
        if done.exists():
            done.unlink()

    # -- region-written stores (for the per-pair unwrap loop) --------------
    def init_store(self, name, template, params, overwrite=False, source=None):
        """Create a metadata-only store so slices can be written by region.

        ``template`` is a lazy dataset with the final shape, dims, chunks and
        dtypes; only its metadata is written here (``compute=False``).

        ``source`` is the dataset that will be *read* while this store is being
        filled in, if any. It is checked against the target for the same
        read-what-you-delete hazard :meth:`store` guards.
        """
        target = self.path(name)
        if target.exists():
            if not overwrite and self.stored_params_hash(name) == hash_params(params):
                return
            if source is not None:
                _check_not_reading_target(source, target, name)
            shutil.rmtree(target)
            self._done_path(name).unlink(missing_ok=True)

        template = template.copy()
        template.attrs["params"] = json.dumps(params, default=_json_default)
        template.attrs["params_hash"] = hash_params(params)
        encoding = _encoding_for(template)
        template.to_zarr(
            target, mode="w", compute=False, consolidated=False,
            **({} if encoding is None else {"encoding": encoding}),
            **_ZARR_KWARGS,
        )
        # Consolidate the freshly written metadata so later existence/param
        # checks (and resumes) open it without a fallback warning. Region data
        # writes don't change shapes, so this stays valid until the final
        # re-consolidation after the last pair.
        self.consolidate(name)
        self._done_path(name).write_text(json.dumps({"pairs_done": []}))

    def write_region(self, name, ds, region):
        """Write a slice of ``ds`` into an initialised store at ``region``."""
        ds.to_zarr(
            self.path(name), region=region, consolidated=False, **_ZARR_KWARGS
        )

    def consolidate(self, name):
        """Consolidate metadata of a region-written store for fast reopening."""
        import zarr

        zarr.consolidate_metadata(str(self.path(name)))

    def mark_pair_done(self, name, pair):
        done = self._done_path(name)
        state = json.loads(done.read_text()) if done.exists() else {"pairs_done": []}
        if pair not in state["pairs_done"]:
            state["pairs_done"].append(pair)
        done.write_text(json.dumps(state))

    def pairs_done(self, name):
        done = self._done_path(name)
        if not done.exists():
            return set()
        return set(json.loads(done.read_text()).get("pairs_done", []))
