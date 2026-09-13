"""Download NISAR inputs: GSLC granules and DEMs.

Three sources, all behind optional, lazily-imported dependencies so the package
still imports without them:

- **earthaccess** (NASA Earthdata) -- the primary path for GSLC granules
  (``short_name`` ``NISAR_L2_GSLC_BETA_V1``), searchable by granule name,
  bounding box, and time range.
- **direct ASF** -- the original bulk-download script (kept as
  :mod:`nisar_tools._asf_bulk`), which pulls GSLCs straight from the NISAR bucket
  by granule name with no search and only the standard library. This is the "as
  we did before" method and the most robust fallback where earthaccess can't be
  installed (its latest release needs Python >= 3.12).
- **sardem** -- builds/stitches a DEM for a bounding box; the default source is
  the official NISAR reference DEM (``data_source="NISAR"``), also on Earthdata.

All of these need Earthdata Login. :func:`login` wraps earthaccess auth (netrc,
``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` env vars, or interactive); the
direct ASF path prompts for credentials; sardem's ``NISAR``/``3DEP`` sources read
``~/.netrc``.

Bounding boxes follow this package's convention -- ``(lon_min, lon_max, lat_min,
lat_max)``, the same order :meth:`nisar_tools.GSLC.crop` takes -- and are
converted internally to the ``(west, south, east, north)`` order earthaccess and
sardem expect.

Example::

    from nisar_tools import download

    download.login()
    bbox = (-120.5, -119.5, 34.0, 35.0)          # lon_min, lon_max, lat_min, lat_max
    gslcs = download.download_gslcs("data/gslc", bbox=bbox, temporal=("2025-11", "2025-12"))
    dem = download.download_dem("data/dem.tif", bbox)

    # Or the direct-by-name method, no earthaccess needed:
    download.download_gslcs("data/gslc", granules=["NISAR_L2_..._001"], method="asf")
"""

import importlib
import signal
import threading
from pathlib import Path
from urllib.parse import urlparse

# NISAR GSLC product short name on NASA CMR / Earthdata.
GSLC_SHORT_NAME = "NISAR_L2_GSLC_BETA_V1"

# Direct-download URL template for a GSLC granule on the NISAR ASF bucket.
_GSLC_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/{short_name}/{name}/{name}.h5"

_DOWNLOAD_METHODS = ("earthaccess", "asf")


def _require(module_name, extra="download"):
    """Import an optional dependency, or raise a helpful install hint."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised via _require test
        top = module_name.split(".")[0]
        raise ImportError(
            f"{top!r} is required for this download but is not installed. "
            f"Install it with `pip install nisar_tools[{extra}]` "
            f"(or `pip install {top}`). Note: earthaccess's latest release needs "
            f"Python >= 3.12; on 3.11 pin an older earthaccess or use method='asf'."
        ) from exc


def _as_list(value):
    """Normalise ``None`` / scalar / iterable into a list."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _bbox_wsen(bbox):
    """``(lon_min, lon_max, lat_min, lat_max)`` -> ``(west, south, east, north)``."""
    lon_min, lon_max, lat_min, lat_max = bbox
    west, east = sorted((float(lon_min), float(lon_max)))
    south, north = sorted((float(lat_min), float(lat_max)))
    return (west, south, east, north)


def _strip_ext(name, ext=".h5"):
    name = str(name)
    return name[: -len(ext)] if name.endswith(ext) else name


# -- Earthdata authentication ------------------------------------------------
def login(strategy="interactive", persist=True, **kwargs):
    """Log in to NASA Earthdata via earthaccess. Returns the auth object.

    ``strategy`` is passed through: ``"interactive"`` (prompt, and with
    ``persist=True`` save to ``~/.netrc``), ``"netrc"``, or ``"environment"``
    (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``).
    """
    earthaccess = _require("earthaccess")
    return earthaccess.login(strategy=strategy, persist=persist, **kwargs)


# -- earthaccess search ------------------------------------------------------
def search(short_name, *, granules=None, bbox=None, temporal=None, count=None, **kwargs):
    """Search Earthdata (CMR) for granules of ``short_name``. Returns a list.

    Filter by explicit ``granules`` (names, matched with earthaccess'
    ``granule_name``; one CMR query per name), and/or ``bbox`` (this package's
    ``(lon_min, lon_max, lat_min, lat_max)``) and ``temporal`` (``(start, end)``
    date strings). Extra keyword args pass straight through to
    ``earthaccess.search_data``.
    """
    earthaccess = _require("earthaccess")

    params = {"short_name": short_name}
    if bbox is not None:
        params["bounding_box"] = _bbox_wsen(bbox)
    if temporal is not None:
        params["temporal"] = tuple(temporal)
    if count is not None:
        params["count"] = count
    params.update(kwargs)

    granules = _as_list(granules)
    if not granules:
        return list(earthaccess.search_data(**params))

    # Explicit granule names take precedence over any wildcard name filter.
    params.pop("granule_name", None)
    results = []
    for name in granules:
        results.extend(earthaccess.search_data(granule_name=str(name), **params))
    return results


def _download_results(results, dest):
    earthaccess = _require("earthaccess")
    paths = earthaccess.download(results, str(dest))
    return [Path(p) for p in paths]


# -- GSLC granules -----------------------------------------------------------
def gslc_url(name, short_name=GSLC_SHORT_NAME):
    """Direct NISAR-bucket URL for a GSLC granule (accepts ``name`` or ``name.h5``)."""
    return _GSLC_URL.format(short_name=short_name, name=_strip_ext(name))


def download_gslcs(dest, *, granules=None, bbox=None, temporal=None,
                   method="earthaccess", short_name=GSLC_SHORT_NAME, count=None,
                   **kwargs):
    """Download NISAR GSLC granules to ``dest``. Returns the local file paths.

    ``method="earthaccess"`` searches CMR by ``granules`` / ``bbox`` / ``temporal``
    then downloads. ``method="asf"`` is the direct-by-name path: it needs explicit
    ``granules`` and only the standard library (no earthaccess).
    """
    if method not in _DOWNLOAD_METHODS:
        raise ValueError(f"method must be one of {_DOWNLOAD_METHODS}")
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    if method == "asf":
        names = _as_list(granules)
        if not names:
            raise ValueError("method='asf' requires explicit granule names")
        urls = [gslc_url(n, short_name=short_name) for n in names]
        return download_urls(dest, urls)

    results = search(
        short_name, granules=granules, bbox=bbox, temporal=temporal,
        count=count, **kwargs,
    )
    return _download_results(results, dest)


# -- DEM via sardem ----------------------------------------------------------
def download_dem(dest, bbox, *, data_source="NISAR", output_format="GTiff",
                 output_type="float32", xrate=1, yrate=1, cache_dir=None):
    """Download/stitch a DEM covering ``bbox`` with sardem. Returns ``dest``.

    ``bbox`` is this package's ``(lon_min, lon_max, lat_min, lat_max)``.
    ``data_source`` defaults to ``"NISAR"`` (the official NISAR reference DEM,
    which needs Earthdata credentials in ``~/.netrc``); other sardem sources
    include ``"COP"`` (Copernicus, no login) and ``"3DEP"`` (US only).
    ``xrate``/``yrate`` upsample the DEM in x/y.
    """
    dem = _require("sardem.dem")
    west, south, east, north = _bbox_wsen(bbox)
    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dem.main(
        output_name=str(dest),
        bbox=[west, south, east, north],  # sardem: [left, bottom, right, top]
        data_source=data_source,
        output_format=output_format,
        output_type=output_type,
        xrate=xrate,
        yrate=yrate,
        cache_dir=cache_dir,
    )
    return dest


# -- NISAR ionosphere TEC (GSLC timing corrections) ---------------------------
#: CMR concept id of ASF's ``NISAR_TEC`` collection ("NISAR Total Electron
#: Content"): daily IMAGEN JSON files, ``NISAR_ANC_TEC_<made>_<start>_<end>_s015``,
#: each spanning 23:00 the day before to 23:59:52 UTC.
TEC_COLLECTION = "C3622219145-ASF"
_CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _utc(value):
    """A timezone-naive UTC ``datetime`` from a datetime, numpy datetime64 or ISO string."""
    import datetime as dt

    import numpy as np

    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, np.datetime64):
        return dt.datetime.fromisoformat(str(value.astype("datetime64[us]")))
    text = str(value).strip().replace("Z", "")
    return dt.datetime.fromisoformat(text)


def _acquisition_span(rslc):
    """First and last zero-Doppler time of an RSLC (path or :class:`~nisar_tools.rslc.RSLC`)."""
    import numpy as np

    from .rslc import RSLC

    own = not isinstance(rslc, RSLC)
    r = RSLC(rslc) if own else rslc
    try:
        epoch = r.ref_epoch
        t = r.zero_doppler_time
        to_ns = lambda s: epoch + np.timedelta64(int(round(float(s) * 1e9)), "ns")  # noqa: E731
        return _utc(to_ns(t[0])), _utc(to_ns(t[-1]))
    finally:
        if own:
            r.close()


def search_tec(start, end=None, *, margin=60.0, collection=TEC_COLLECTION):
    """NISAR TEC files covering ``[start - margin, end + margin]``, best first.

    ``start``/``end`` are UTC datetimes, ``datetime64`` or ISO strings
    (``end`` defaults to ``start``); ``margin`` is seconds. Returns dicts with
    ``name``, ``url``, ``start``, ``end`` and ``produced``, newest production
    first. A day's file overlaps the next by an hour, so an acquisition after
    23:00 UTC is covered twice; either works. Uses CMR's public search API, so
    no login is needed to search.
    """
    import datetime as dt
    import json
    import urllib.parse
    import urllib.request

    start = _utc(start)
    end = _utc(end) if end is not None else start
    lo = start - dt.timedelta(seconds=margin)
    hi = end + dt.timedelta(seconds=margin)
    params = {"collection_concept_id": collection, "page_size": 50,
              "temporal[]": f"{lo.isoformat()}Z,{hi.isoformat()}Z"}
    url = _CMR_GRANULES + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)["feed"]["entry"]
    found = []
    for e in entries:
        name = e["title"]
        t0, t1 = _utc(e["time_start"]), _utc(e["time_end"])
        if not (t0 <= lo and hi <= t1):
            continue
        links = [link["href"] for link in e.get("links", [])
                 if link.get("href", "").endswith(f"/{name}.json") and "PRIVATE" not in link["href"]]
        if not links:
            continue
        made = name.split("_")[3] if name.count("_") >= 5 else ""
        found.append(dict(name=name, url=links[0], start=t0, end=t1, produced=made))
    return sorted(found, key=lambda d: d["produced"], reverse=True)


def download_tec(dest, start=None, end=None, *, rslc=None, margin=60.0, overwrite=False):
    """Download the NISAR TEC file for an acquisition. Returns its path.

    Give the acquisition either as ``start``/``end`` (see :func:`search_tec`) or
    as ``rslc=`` (a path or :class:`~nisar_tools.rslc.RSLC`), whose zero-Doppler
    span is used. ``dest`` is a directory (the file keeps its own name) or a
    ``.json`` path. The download authenticates with Earthdata Login through
    ``~/.netrc``, the way ``requests`` re-applies netrc credentials across the
    URS redirect. Pass the result to ``geocode_rslc(tec_file=...)``.
    """
    requests = _require("requests")
    if rslc is not None:
        if start is not None:
            raise ValueError("pass either rslc= or start/end, not both")
        start, end = _acquisition_span(rslc)
    if start is None:
        raise ValueError("download_tec needs start (and end) or rslc=")
    hits = search_tec(start, end, margin=margin)
    if not hits:
        raise FileNotFoundError(f"no NISAR TEC file in CMR covers {start} .. {end or start}")
    best = hits[0]
    dest = Path(dest).expanduser()
    path = dest if dest.suffix == ".json" else dest / f"{best['name']}.json"
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.get(best["url"], timeout=300)
        response.raise_for_status()
        if response.content[:1] not in (b"{", b"["):
            raise IOError(f"{best['url']} did not return JSON (Earthdata login in ~/.netrc?)")
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


# -- direct ASF bulk download (the "current method") -------------------------
def download_urls(dest, urls):
    """Download fully-formed Earthdata URLs with the ASF bulk downloader.

    The low-level path behind ``method="asf"``: it authenticates with Earthdata
    Login (prompting if needed) and streams each URL into ``dest``, following the
    URS re-auth redirect and waiting on the burst-extraction service. Returns the
    expected local paths (one per URL, named by the URL's basename).
    """
    from . import _asf_bulk

    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    urls = _as_list(urls)
    for url in urls:
        if not isinstance(url, str):
            raise ValueError("urls must be strings")

    # SIGINT trapping only works from the main thread (e.g. not inside a dask
    # worker), so guard it.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _asf_bulk.signal_handler)

    downloader = _asf_bulk.bulk_downloader(download_dir=str(dest), files=urls)
    downloader.download_files()
    downloader.print_summary()
    return [dest / Path(urlparse(u).path).name for u in urls]


def download_files(download_dir, files):
    """Backwards-compatible shim: download GSLCs by name via the direct method.

    Equivalent to ``download_gslcs(download_dir, granules=files, method="asf")``.
    """
    return download_gslcs(download_dir, granules=files, method="asf")
