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
    igrams = igrams.filter_goldstein(alpha=0.5)   # adaptive phase filter
    unw = igrams.unwrap(ws, nproc=8)
"""

from . import download
from . import slip
from .download import download_dem, download_gslcs, login
from .gslc import GSLC
from .stack import GSLCStack
from .interferogram import InterferogramStack, make_pairs
from .los import LOSStack
# Everything needed to set up an inversion, at either end of the three choices
# (vertical/curved geometry, homogeneous/layered medium, element/nodal slip).
# The rest of the subpackage -- the samplers, diagnostics and plotting -- stays on
# `nisar_tools.slip`, which is also importable as a whole.
from .slip import (
    EdgrnTables,
    FaultMesh,
    FaultSegment,
    FaultSurface,
    FaultTrace,
    HalfSpaceTDE,
    LayeredPointSource,
    Observations,
    SlipInversion,
    SlipModel,
    VelocityModel,
    run_edgrn,
    write_edgrn_input,
)
from .unwrap import UnwrappedStack
from .workspace import Workspace, WorkspaceError

__version__ = "0.1.0"

__all__ = [
    "GSLC",
    "GSLCStack",
    "EdgrnTables",
    "FaultMesh",
    "FaultSegment",
    "FaultSurface",
    "FaultTrace",
    "HalfSpaceTDE",
    "InterferogramStack",
    "LOSStack",
    "LayeredPointSource",
    "Observations",
    "SlipInversion",
    "SlipModel",
    "UnwrappedStack",
    "VelocityModel",
    "Workspace",
    "WorkspaceError",
    "download",
    "download_dem",
    "download_gslcs",
    "login",
    "make_pairs",
    "run_edgrn",
    "slip",
    "write_edgrn_input",
    "__version__",
]
