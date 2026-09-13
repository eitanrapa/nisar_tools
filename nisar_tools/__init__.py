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

    offsets = stack.pixel_offsets(step=64)        # amplitude offset tracking

Making a GSLC from an RSLC (a port of ISCE3's GSLC workflow)::

    from nisar_tools import RSLC, download_tec, geocode_rslc

    tec = download_tec("data/", rslc=rslc_path)
    geocode_rslc(rslc_path, "data/", dem_path, tec_file=tec, like=same_frame_gunw)
"""

from . import download
from . import ionosphere
from .download import download_dem, download_gslcs, download_tec, login, search_tec
from .gslc import GSLC
from .gslc_geocode import default_gslc_name, geocode_rslc, product_bounds
from .rslc import RSLC
from .stack import GSLCStack
from .interferogram import InterferogramStack, make_pairs
from .los import LOSStack
from .offsets import PixelOffsetStack
from .unwrap import UnwrappedStack
from .workspace import Workspace, WorkspaceError

__version__ = "0.1.0"

__all__ = [
    "GSLC",
    "GSLCStack",
    "InterferogramStack",
    "LOSStack",
    "PixelOffsetStack",
    "RSLC",
    "UnwrappedStack",
    "Workspace",
    "WorkspaceError",
    "default_gslc_name",
    "download",
    "download_dem",
    "download_gslcs",
    "download_tec",
    "geocode_rslc",
    "ionosphere",
    "login",
    "make_pairs",
    "product_bounds",
    "search_tec",
    "__version__",
]
