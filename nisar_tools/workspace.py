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
            shutil.rmtree(target)

        ds = ds.copy()
        ds.attrs["params"] = json.dumps(params, default=_json_default)
        ds.attrs["params_hash"] = hash_params(params)
        ds.to_zarr(target, mode="w", consolidated=True, **_ZARR_KWARGS)
        return self.load(name)

    def load(self, name):
        """Open stage ``name`` lazily."""
        if not self.exists(name):
            raise WorkspaceError(f"Stage '{name}' does not exist in {self.workdir}.")
        return xr.open_zarr(self.path(name))

    def clear(self, name):
        """Delete a stage store and any done-markers."""
        target = self.path(name)
        if target.exists():
            shutil.rmtree(target)
        done = self._done_path(name)
        if done.exists():
            done.unlink()

    # -- region-written stores (for the per-pair unwrap loop) --------------
    def init_store(self, name, template, params, overwrite=False):
        """Create a metadata-only store so slices can be written by region.

        ``template`` is a lazy dataset with the final shape, dims, chunks and
        dtypes; only its metadata is written here (``compute=False``).
        """
        target = self.path(name)
        if target.exists():
            if not overwrite and self.stored_params_hash(name) == hash_params(params):
                return
            shutil.rmtree(target)
            self._done_path(name).unlink(missing_ok=True)

        template = template.copy()
        template.attrs["params"] = json.dumps(params, default=_json_default)
        template.attrs["params_hash"] = hash_params(params)
        template.to_zarr(
            target, mode="w", compute=False, consolidated=False, **_ZARR_KWARGS
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
