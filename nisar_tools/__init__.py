"""nisar_tools: out-of-core NISAR InSAR processing with xarray and dask.

Every stage is a lazy `xarray` dataset backed by `dask` and is persisted to a
Zarr `Workspace`, so a full stack of acquisitions never has to fit in memory
and an interrupted run resumes where it left off.

The main entry points, in pipeline order:

- `download`: GSLC granules, DEMs and NISAR TEC files from NASA Earthdata
  (`download_gslcs`, `download_dem`, `download_tec`, `login`).
- `GSLC` and `RSLC`: one granule each, read lazily. `geocode_rslc` makes a
  NISAR-format GSLC from an RSLC (a port of ISCE3's GSLC workflow), and a
  `RegistrationField` coregisters it to a reference GSLC.
- `GSLCStack`: co-registered GSLCs on one grid: crop, merge adjacent frames,
  mask water and swath edges.
- `InterferogramStack`: multilooked interferograms and coherence, the Goldstein
  filter, water, edge and island masking.
- `UnwrappedStack`: SNAPHU unwrapping (or a NASA GUNW, read as it is),
  deramping, outlier rejection.
- `ionosphere` and `nisar_tools.split_spectrum`: the dual-frequency
  (split-spectrum) ionosphere correction.
- `roff`: NISAR's dense ROFF/GOFF pixel offsets from two RSLCs, ported from ISCE3.
- `LOSStack`: line-of-sight displacement in metres, positive toward the
  sensor, with the per-pixel look geometry.

A typical run:

```python
from nisar_tools import GSLC, GSLCStack, Workspace

ws = Workspace("workdir/")
stack = GSLCStack.from_gslcs([GSLC(p) for p in paths], bbox=bbox)
stack = stack.persist(ws, "slc_stack", files=paths, bbox=bbox)
igrams = stack.form_interferograms(looks=5).persist(ws, "igrams")
igrams = igrams.filter_goldstein(alpha=0.5)
unw = igrams.unwrap(ws, nproc=8)
los = unw.to_los(gslc=paths[0], dem=dem_path)
```

Making a GSLC from an RSLC:

```python
from nisar_tools import download_tec, geocode_rslc

tec = download_tec("data/", rslc=rslc_path)
geocode_rslc(rslc_path, "data/", dem_path, tec_file=tec, like=same_frame_gunw)
```
"""

from . import download
from . import ionosphere
from . import roff
from .download import download_dem, download_gslcs, download_tec, login, search_tec
from .gslc import GSLC
from .gslc_geocode import RegistrationField, default_gslc_name, geocode_rslc, product_bounds
from .rslc import RSLC
from .roff import dense_offsets, geocode_offsets, open_roff
from .stack import GSLCStack
from .interferogram import InterferogramStack, make_pairs
from .los import LOSStack
from .unwrap import UnwrappedStack
from .workspace import Workspace, WorkspaceError

__version__ = "0.1.0"

__all__ = [
    "GSLC",
    "GSLCStack",
    "InterferogramStack",
    "LOSStack",
    "RSLC",
    "RegistrationField",
    "UnwrappedStack",
    "Workspace",
    "WorkspaceError",
    "default_gslc_name",
    "dense_offsets",
    "download",
    "download_dem",
    "download_gslcs",
    "download_tec",
    "geocode_offsets",
    "geocode_rslc",
    "ionosphere",
    "login",
    "make_pairs",
    "open_roff",
    "product_bounds",
    "roff",
    "search_tec",
    "__version__",
]
