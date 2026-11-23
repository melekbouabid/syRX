"""On-disk HTTP cache with ETag / Last-Modified revalidation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    key: str
    url: str
    status: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    stored_at: float
    validated_at: float
    body_path: Path

    def is_fresh(self, ttl: float, now: float) -> bool:
        return ttl > 0 and (now - self.validated_at) < ttl

    def read(self) -> bytes:
        return self.body_path.read_bytes()

    def conditional_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write through a temporary file so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


class HttpCache:
    """Stores response bodies next to a small JSON record of their validators.

    Callers must pass URLs with secret parameters already removed: the URL is both hashed
    into the key and written to disk.
    """

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = root
        self._clock = clock

    @staticmethod
    def make_key(method: str, url: str, body: bytes | None = None) -> str:
        digest = hashlib.sha256()
        digest.update(method.upper().encode())
        digest.update(b"\n")
        digest.update(url.encode())
        if body:
            digest.update(b"\n")
            digest.update(body)
        return digest.hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        folder = self.root / key[:2]
        return folder / f"{key}.json", folder / f"{key}.body"

    def now(self) -> float:
        return self._clock()

    def get(self, key: str) -> CacheEntry | None:
        meta_path, body_path = self._paths(key)
        if not (meta_path.is_file() and body_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return CacheEntry(body_path=body_path, **meta)
        except (ValueError, TypeError):
            return None

    def put(
        self, key: str, *, url: str, status: int, headers: Mapping[str, str], body: bytes
    ) -> CacheEntry:
        meta_path, body_path = self._paths(key)
        now = self._clock()
        entry = CacheEntry(
            key=key,
            url=url,
            status=status,
            content_type=headers.get("content-type"),
            etag=headers.get("etag"),
            last_modified=headers.get("last-modified"),
            stored_at=now,
            validated_at=now,
            body_path=body_path,
        )
        atomic_write_bytes(body_path, body)
        self._write_meta(meta_path, entry)
        return entry

    def touch(self, entry: CacheEntry) -> None:
        """Mark an entry as revalidated (the server answered 304)."""
        entry.validated_at = self._clock()
        self._write_meta(self._paths(entry.key)[0], entry)

    @staticmethod
    def _write_meta(meta_path: Path, entry: CacheEntry) -> None:
        meta = asdict(entry)
        meta.pop("body_path")
        atomic_write_bytes(meta_path, json.dumps(meta).encode("utf-8"))
