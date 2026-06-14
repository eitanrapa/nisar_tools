"""nisar_tools: object-oriented, out-of-core NISAR GSLC InSAR processing.

The pipeline is built around lazy ``xarray`` datasets backed by ``dask`` and
persisted between stages to a Zarr :class:`~nisar_tools.workspace.Workspace`,
so a full stack of acquisitions never needs to fit in memory at once.

Typical flow::

    from nisar_tools import GSLC, GSLCStack, Workspace

    ws = Workspace("workdir/")
    stack = GSLCStack.from_gslcs([GSLC(p) for p in paths], bbox=bbox)
    stack = stack.persist(ws, "slc_stack", files=paths, bbox=bbox)
    igrams = stack.form_interferograms(looks=5).persist(ws, "igrams")
    unw = igrams.unwrap(ws, nlooks=5, spacing_az=..., spacing_rg=...)
"""

from .gslc import GSLC
from .stack import GSLCStack
from .interferogram import InterferogramStack, make_pairs
from .unwrap import UnwrappedStack
from .workspace import Workspace, WorkspaceError

__version__ = "0.1.0"

__all__ = [
    "GSLC",
    "GSLCStack",
    "InterferogramStack",
    "UnwrappedStack",
    "Workspace",
    "WorkspaceError",
    "make_pairs",
    "__version__",
]
