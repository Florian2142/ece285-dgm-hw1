import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dgm.data import download
from dgm.data.chest_xray import archive_split_root, in_split_root

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class FlakyRangeServer(BaseHTTPRequestHandler):
    """Serves PAYLOAD with HTTP Range support; the first response breaks off after a third."""

    broken_once = False

    def do_GET(self):
        requested = self.headers.get("Range")
        start = int(requested.split("=")[1].split("-")[0]) if requested else 0
        body = PAYLOAD[start:]
        self.send_response(206 if requested else 200)
        self.send_header("Content-Length", str(len(body)))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        if not FlakyRangeServer.broken_once:
            FlakyRangeServer.broken_once = True
            self.wfile.write(body[: len(body) // 3])
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server_url():
    FlakyRangeServer.broken_once = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyRangeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/archive.zip"
    server.shutdown()


def test_download_resumes_after_a_broken_connection(tmp_path, server_url, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    dest = download.download_file(server_url, tmp_path / "archive.zip", retries=3)
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "archive.zip.part").exists()
    # An existing file is returned without any request.
    assert download.download_file("http://unused.invalid/", dest) == dest


def test_extract_zip_filters_members(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/train/x.txt", "x")
        zf.writestr("data/other/y.txt", "y")
        zf.writestr("__MACOSX/data/train/._x.txt", "junk")
    download.extract_zip(archive, tmp_path / "out", keep=lambda name: "/train/" in name)
    assert (tmp_path / "out" / "data" / "train" / "x.txt").exists()
    assert not (tmp_path / "out" / "data" / "other").exists()
    assert not (tmp_path / "out" / "__MACOSX").exists()


def test_archive_split_root_picks_one_copy():
    names = [
        "chest_xray/train/NORMAL/a.jpeg",
        "chest_xray/test/NORMAL/b.jpeg",
        "chest_xray/chest_xray/train/NORMAL/a.jpeg",
        "chest_xray/chest_xray/test/NORMAL/b.jpeg",
        "chest_xray/__MACOSX/chest_xray/train/._a.jpeg",
    ]
    root = archive_split_root(names)
    assert root == "chest_xray"
    kept = [name for name in names if in_split_root(name, root)]
    assert kept == ["chest_xray/train/NORMAL/a.jpeg", "chest_xray/test/NORMAL/b.jpeg"]
    with pytest.raises(ValueError):
        archive_split_root(["readme.txt"])
