"""Download and unpack dataset archives (standard library + tqdm progress bars)."""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


def download_file(
    url: str, dest: Path, *, desc: str | None = None, retries: int = 8, chunk_size: int = 1 << 20
) -> Path:
    """Stream ``url`` to ``dest``; skip if ``dest`` exists.

    Data go to ``dest.part`` first. After a network error the download resumes where it
    stopped (HTTP Range request), up to ``retries`` times, so a flaky connection does not
    restart a multi-GB download from zero.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "dgm-course-repo"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as response:
                if done and response.status != 206:  # the server ignored the range: start over
                    done = 0
                remaining = int(response.headers.get("Content-Length", 0))
                total = done + remaining if remaining else None
                with (
                    tmp.open("ab" if done else "wb") as out,
                    tqdm(
                        total=total,
                        initial=done,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc,
                    ) as bar,
                ):
                    while chunk := response.read(chunk_size):
                        out.write(chunk)
                        bar.update(len(chunk))
            if total is not None and tmp.stat().st_size != total:
                raise ConnectionError(
                    f"connection closed early ({tmp.stat().st_size} of {total} bytes)"
                )
            tmp.replace(dest)
            return dest
        except NETWORK_ERRORS as err:
            if attempt == retries:
                raise
            wait = min(60, 2**attempt)
            print(
                f"download interrupted ({err}); resuming in {wait}s (attempt {attempt + 1}/{retries})"
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def extract_zip(archive: Path, dest: Path, keep: Callable[[str], bool] | None = None) -> Path:
    """Extract ``archive`` into ``dest``; skip ``__MACOSX`` folders and names ``keep`` rejects."""
    with zipfile.ZipFile(archive) as zf:
        members = [
            m
            for m in zf.infolist()
            if "__MACOSX" not in Path(m.filename).parts and (keep is None or keep(m.filename))
        ]
        for member in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            zf.extract(member, dest)
    return dest
