"""Downloads from NASA Earthdata servers with Earthdata Login, using only the standard library.

The direct path behind `nisar_tools.download.download_gslcs(method="asf")`. A protected file
answers a first request with a redirect to Earthdata Login (URS), which asks for HTTP basic
credentials, sets a session cookie and redirects back to the file. `urllib` follows that exchange
with a cookie jar and a password manager holding the credentials for Earthdata Login alone, so
they are sent to no other host.
"""

import getpass
import http.cookiejar
import io
import netrc
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

#: Earthdata Login: the only address the credentials are sent to.
URS_URL = "https://urs.earthdata.nasa.gov"

#: A server preparing a file on request (ASF's burst extraction) answers 202 until it is ready.
ACCEPTED_WAIT = 5.0
ACCEPTED_TRIES = 120

_CHUNK = 1 << 20


def credentials():
    """Earthdata Login `(username, password)`.

    From the netrc file (`$NETRC`, else `~/.netrc`, which `nisar_tools.download.login` writes),
    else `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, else asked for at the terminal.
    """
    host = urlparse(URS_URL).hostname
    try:
        entry = netrc.netrc(os.environ.get("NETRC") or None).authenticators(host)
    except FileNotFoundError:
        entry = None
    if entry and entry[0] and entry[2]:
        return entry[0], entry[2]
    username, password = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password
    if not sys.stdin.isatty():
        raise RuntimeError(f"no Earthdata Login credentials: add {host} to ~/.netrc (download.login() does), "
                           "or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
    return input("Earthdata Login username: "), getpass.getpass("Earthdata Login password: ")


def opener(username, password):
    """A `urllib` opener that keeps cookies and answers Earthdata Login's challenge."""
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, URS_URL, username, password)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(manager),
                                       urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def download(urls, dest, username=None, password=None, progress=True):
    """Download `urls` into `dest`, each under its URL's basename; returns the paths.

    A file already in `dest` at the server's size is skipped. A download is written to
    `<name>.part` and renamed once complete, so a file under its final name is always whole.
    Every URL is tried, and any failures are raised together at the end.
    """
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    if username is None or password is None:
        username, password = credentials()
    open_url = opener(username, password).open
    paths, failed = [], []
    for k, url in enumerate(urls, 1):
        path = dest / Path(urlparse(url).path).name
        paths.append(path)
        label = f"[{k}/{len(urls)}] {path.name}"
        try:
            _fetch(open_url, url, path, label, progress)
        except OSError as err:              # HTTPError, URLError and TimeoutError among them
            failed.append(f"{url}: {_reason(err)}")
            print(f"{label}: failed, {_reason(err)}", file=sys.stderr)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(urls)} downloads failed:\n  " + "\n  ".join(failed))
    return paths


def _fetch(open_url, url, path, label, progress):
    with _open(open_url, url) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        if path.exists() and path.stat().st_size == total:
            print(f"{label}: already complete")
            return
        part = path.with_name(path.name + ".part")
        done, shown, start = 0, -1, time.monotonic()
        with open(part, "wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                done += len(chunk)
                if progress and total and (step := 10 * done // total) != shown:
                    shown = step
                    print(f"\r{label}: {10 * step}%", end="", flush=True)
    if total is not None and done != total:
        raise OSError(f"the connection ended after {done} of {total} bytes")
    os.replace(part, path)
    rate = done / 2**20 / max(time.monotonic() - start, 1e-3)
    print(f"\r{label}: {done / 2**20:.1f} MiB at {rate:.1f} MiB/s")


class RangeFile(io.RawIOBase):
    """A read-only remote file over HTTP Range requests, behind Earthdata Login.

    Opens with `h5py.File(RangeFile(url), "r")`, so the metadata of a large
    granule (an L0B is 10-90 GB) can be read without downloading it. The login
    redirects are followed once, then every read asks the signed URL for whole
    `block`-byte blocks and keeps up to `max_blocks` of them.
    """

    def __init__(self, url, username=None, password=None, block=1 << 21, max_blocks=2048):
        if username is None or password is None:
            username, password = credentials()
        self._open_url = opener(username, password).open
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with self._open_url(request) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        self.block, self.max_blocks = int(block), int(max_blocks)
        self.requests = 0
        self._pos = 0
        self._cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = base + offset
        return self._pos

    def _block(self, k):
        if k not in self._cache:
            first, last = k * self.block, min((k + 1) * self.block, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={first}-{last}"})
            with self._open_url(request) as response:
                self._cache[k] = response.read()
            self.requests += 1
            if len(self._cache) > self.max_blocks:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[k]

    def readinto(self, buffer):
        n = max(0, min(len(buffer), self.size - self._pos))
        view, done = memoryview(buffer).cast("B"), 0
        while done < n:
            k, offset = divmod(self._pos + done, self.block)
            data = self._block(k)
            m = min(n - done, len(data) - offset)
            view[done:done + m] = data[offset:offset + m]
            done += m
        self._pos += n
        return n


def _open(open_url, url):
    """Open `url`, waiting while the server answers 202 (still preparing the file)."""
    for _ in range(ACCEPTED_TRIES):
        response = open_url(urllib.request.Request(url))
        if response.status != 202:
            return response
        response.close()
        time.sleep(ACCEPTED_WAIT)
    raise TimeoutError(f"still being prepared after {ACCEPTED_TRIES * ACCEPTED_WAIT:.0f} s")


def _reason(err):
    if isinstance(err, urllib.error.HTTPError):
        hint = {401: "; check the Earthdata Login credentials",
                403: "; the account may first have to approve the data provider's application in "
                     "Earthdata Login"}.get(err.code, "")
        return f"HTTP {err.code} {err.reason}{hint}"
    if isinstance(err, urllib.error.URLError):
        return str(err.reason)
    return str(err)
