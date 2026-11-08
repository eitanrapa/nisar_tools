"""The direct Earthdata download path (`nisar_tools._earthdata`), against a local stand-in for Earthdata Login.

The server plays both parts. A data request without the session cookie is redirected to
`/urs/authorize`, which wants HTTP basic credentials; with them it redirects to `/login`, which
sets the cookie and redirects back to the file.
"""

import base64
import http.server
import os
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from nisar_tools import _earthdata

USER, PASSWORD = "someone", "secret"
FILES = {"/data/a.bin": os.urandom(3_000_000), "/data/b.bin": b"b" * 1234, "/data/slow.bin": b"ready"}


class _Server:
    def __init__(self):
        self.log = []                            # (path, whether it carried an Authorization header)
        self.accepted = {"/data/slow.bin": 2}    # answer 202 this many times before the file
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _empty(self, code, **headers):
                self.send_response(code)
                for key, value in headers.items():
                    self.send_header(key.replace("_", "-"), value)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                outer.log.append((url.path, "Authorization" in self.headers))
                if url.path == "/urs/authorize":
                    good = "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
                    if self.headers.get("Authorization") != good:
                        return self._empty(401, WWW_Authenticate='Basic realm="Earthdata Login"')
                    return self._empty(302, Location=f"/login?back={query['back'][0]}")
                if url.path == "/login":
                    return self._empty(302, Set_Cookie="session=ok; Path=/", Location=query["back"][0])
                if "session=ok" not in (self.headers.get("Cookie") or ""):
                    return self._empty(302, Location=f"/urs/authorize?back={url.path}")
                if outer.accepted.get(url.path, 0) > 0:
                    outer.accepted[url.path] -= 1
                    return self._empty(202)
                if url.path not in FILES:
                    return self._empty(404)
                body = FILES[url.path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):      # a client that only wanted the size
                    pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server(monkeypatch):
    s = _Server()
    monkeypatch.setattr(_earthdata, "URS_URL", s.url + "/urs")
    monkeypatch.setattr(_earthdata, "ACCEPTED_WAIT", 0.0)
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


def _get(server, tmp_path, *names):
    return _earthdata.download([f"{server.url}/data/{n}" for n in names], tmp_path, USER, PASSWORD)


def test_downloads_through_the_login_exchange(server, tmp_path):
    out = _get(server, tmp_path, "a.bin", "b.bin")
    assert [p.name for p in out] == ["a.bin", "b.bin"]
    assert out[0].read_bytes() == FILES["/data/a.bin"] and out[1].read_bytes() == FILES["/data/b.bin"]
    assert not list(tmp_path.glob("*.part"))
    # The credentials went to the login service, and only there.
    sent = {path for path, auth in server.log if auth}
    assert sent == {"/urs/authorize"}


def test_a_complete_file_is_kept_and_a_truncated_one_replaced(server, tmp_path, capsys):
    a, b = _get(server, tmp_path, "a.bin", "b.bin")
    os.utime(a, (1, 1))
    b.write_bytes(b"b" * 100)                    # an interrupted earlier download
    _get(server, tmp_path, "a.bin", "b.bin")
    assert a.stat().st_mtime == 1 and a.read_bytes() == FILES["/data/a.bin"]
    assert b.read_bytes() == FILES["/data/b.bin"]
    assert "a.bin: already complete" in capsys.readouterr().out


def test_waits_while_the_server_prepares_the_file(server, tmp_path):
    (out,) = _get(server, tmp_path, "slow.bin")
    assert out.read_bytes() == b"ready" and server.accepted["/data/slow.bin"] == 0


def test_a_failure_is_raised_after_the_other_files(server, tmp_path):
    with pytest.raises(RuntimeError, match=r"1 of 2 downloads failed:\n.*missing\.bin: HTTP 404"):
        _get(server, tmp_path, "missing.bin", "b.bin")
    assert (tmp_path / "b.bin").read_bytes() == FILES["/data/b.bin"]
    assert not (tmp_path / "missing.bin").exists()


def test_wrong_credentials_say_so(server, tmp_path):
    with pytest.raises(RuntimeError, match="HTTP 401.*check the Earthdata Login credentials"):
        _earthdata.download([f"{server.url}/data/b.bin"], tmp_path, USER, "wrong")


def test_credentials_come_from_netrc_then_the_environment(tmp_path, monkeypatch):
    rc = tmp_path / "netrc"
    rc.write_text("machine urs.earthdata.nasa.gov login from_netrc password p1\n")
    rc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(rc))
    monkeypatch.setenv("EARTHDATA_USERNAME", "from_env")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p2")
    assert _earthdata.credentials() == ("from_netrc", "p1")
    monkeypatch.setenv("NETRC", str(tmp_path / "none"))
    assert _earthdata.credentials() == ("from_env", "p2")
    monkeypatch.delenv("EARTHDATA_PASSWORD")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError, match="no Earthdata Login credentials"):
        _earthdata.credentials()
