"""Connector interface: every source implements discover, fetch, parse and validate."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from synergyrx_data.config import Settings
from synergyrx_data.connectors.schemas import Record
from synergyrx_data.http.cache import atomic_write_bytes
from synergyrx_data.http.client import Kind, PoliteClient
from synergyrx_data.logging_utils import get_logger
from synergyrx_data.manifest import Manifest, utc_now_iso
from synergyrx_data.preflight import ensure_disk_space
from synergyrx_data.progress import ProgressReporter
from synergyrx_data.provenance import PROVENANCE_COLUMNS, Provenance
from synergyrx_data.state import StateStore

# Bump when a parser changes shape, so cached interim tables are rebuilt.
PARSER_VERSION = "1"


class ManualStepRequired(Exception):
    """The source cannot be fetched automatically; a person has to place a file."""


@dataclass(frozen=True)
class Link:
    """A URL listed in the project brief, probed by `synergyrx sources --check`."""

    url: str
    label: str
    kind: Kind = "html"
    origin: str = "verify"  # "checked" or "verify", as marked in the brief


@dataclass(frozen=True)
class SourceInfo:
    name: str
    title: str
    category: str  # synergy | cell_line | chemistry | literature | interaction
    access: str  # api | bulk | package | manual
    homepage: str
    license: str
    citation: str
    hosts: tuple[str, ...]
    links: tuple[Link, ...] = ()
    license_url: str | None = None
    terms_url: str | None = None
    stage: int = 1
    optional: bool = False
    default_enabled: bool = True
    redistribution: str = "Do not redistribute raw files; cite the source."
    notes: str = ""


@dataclass(frozen=True)
class PlannedAction:
    """One thing `fetch` would do. Printed by `--dry-run`; building it touches nothing."""

    source: str
    action: str  # download | sample | api | manual | package | skip
    target: str
    dest: str | None = None
    est_bytes: int | None = None
    note: str = ""


@dataclass
class Artifact:
    """A raw file on disk plus what is needed to cite it."""

    source: str
    path: Path
    url: str
    version: str
    sha256: str
    size: int
    retrieved_at: str
    license: str
    terms_url: str | None = None
    is_sample: bool = False
    status: str = "downloaded"  # downloaded | unchanged | manual
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    table: str
    rows: int
    valid: int
    invalid: int
    examples: list[str] = field(default_factory=list)


@dataclass
class Validated:
    valid: pd.DataFrame
    rejects: pd.DataFrame
    result: ValidationResult


@dataclass
class Context:
    settings: Settings
    manifest: Manifest
    state: StateStore
    client: PoliteClient | None = None
    progress: ProgressReporter = field(default_factory=lambda: ProgressReporter(False))


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts: missing values become None, array cells become lists."""
    objects = frame.astype(object).where(frame.notna(), None)
    for column in objects.columns:
        non_null = objects[column].dropna()
        if len(non_null) and isinstance(non_null.iloc[0], np.ndarray):
            objects[column] = objects[column].map(
                lambda v: v.tolist() if isinstance(v, np.ndarray) else v
            )
    records: list[dict[str, Any]] = objects.to_dict("records")
    return records


def validate_frame(
    frame: pd.DataFrame,
    model: type[Record],
    *,
    chunk_size: int = 50_000,
    max_examples: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """Validate every row against `model`. Returns (boolean mask of valid rows, examples)."""
    adapter = TypeAdapter(list[model])  # type: ignore[valid-type]
    mask = np.ones(len(frame), dtype=bool)
    examples: list[str] = []
    for start in range(0, len(frame), chunk_size):
        records = to_records(frame.iloc[start : start + chunk_size])
        try:
            adapter.validate_python(records)
        except ValidationError as exc:
            for error in exc.errors(include_url=False, include_context=False, include_input=False):
                row = start + int(error["loc"][0])
                if mask[row] and len(examples) < max_examples:
                    where = ".".join(str(part) for part in error["loc"][1:]) or "row"
                    examples.append(f"row {row}: {where}: {error['msg']}")
                mask[row] = False
    return mask, examples


def sample_name(filename: str, n: int) -> str:
    """`table.csv` -> `table.sample100.csv`, so a sample can never pass for the full file."""
    path = Path(filename)
    return f"{path.stem}.sample{n}{path.suffix}"


class Connector(ABC):
    info: ClassVar[SourceInfo]
    default_version: ClassVar[str] = "current"
    default_options: ClassVar[dict[str, Any]] = {}
    schemas: ClassVar[dict[str, type[Record]]] = {}

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        config = ctx.settings.sources.get(self.info.name)
        self.enabled = config.enabled if config else self.info.default_enabled
        self.version = config.version if config and config.version else self.default_version
        self.options: dict[str, Any] = {
            **self.default_options,
            **(config.options if config else {}),
        }
        self.log = get_logger(f"connector.{self.info.name}")
        self.stats: dict[str, Any] = {}  # counts worth reporting (rows skipped, ...)
        self.problems: list[str] = []  # non-fatal issues: the source "needs attention"

    # ----------------------------------------------------------- interface

    @abstractmethod
    async def discover(self) -> list[PlannedAction]:
        """List what `fetch` would do. Must not touch the network or the disk."""

    @abstractmethod
    async def fetch(self) -> list[Artifact]:
        """Retrieve raw files into `data/raw`, skipping whatever is already current."""

    @abstractmethod
    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        """Turn raw files into tidy tables that carry provenance columns."""

    def validate(self, tables: dict[str, pd.DataFrame]) -> dict[str, Validated]:
        """Check every row against the table's schema; rejects are kept, never dropped."""
        validated: dict[str, Validated] = {}
        for table, frame in tables.items():
            model = self.schemas.get(table)
            missing = [c for c in PROVENANCE_COLUMNS if c not in frame.columns]
            if missing:
                raise ValueError(f"{self.name}.{table} lacks provenance columns: {missing}")
            if model is None or frame.empty:
                mask = np.ones(len(frame), dtype=bool)
                examples: list[str] = []
            else:
                mask, examples = validate_frame(frame, model)
            result = ValidationResult(
                table, len(frame), int(mask.sum()), int((~mask).sum()), examples
            )
            validated[table] = Validated(frame[mask], frame[~mask], result)
        return validated

    # ------------------------------------------------------------- helpers

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def client(self) -> PoliteClient:
        if self.ctx.client is None:
            raise RuntimeError(f"{self.name}: no HTTP client in this context (offline run)")
        return self.ctx.client

    @property
    def sample(self) -> int | None:
        return self.settings.sample

    @property
    def raw_dir(self) -> Path:
        return self.settings.raw_dir / self.name / self.version

    @property
    def interim_dir(self) -> Path:
        return self.settings.interim_dir / self.name

    @property
    def manual_dir(self) -> Path:
        return self.settings.manual_dir / self.name

    def rel(self, path: Path) -> str:
        return self.settings.relative(path)

    def fingerprint(self, artifacts: list[Artifact]) -> str:
        """Identifies the exact inputs of a parse, so an unchanged source is not re-parsed."""
        digest = hashlib.sha256(f"{PARSER_VERSION}|{self.version}|{self.sample}".encode())
        for artifact in sorted(artifacts, key=lambda a: a.path.as_posix()):
            if artifact.extra.get("auxiliary"):
                continue  # listings and logs do not change what gets parsed
            digest.update(f"|{artifact.path.name}:{artifact.sha256}".encode())
        return digest.hexdigest()

    def provenance(self, artifact: Artifact) -> Provenance:
        return Provenance(
            source=self.name,
            source_url=artifact.url,
            retrieved_at=artifact.retrieved_at,
            source_version=artifact.version,
            license=artifact.license,
        )

    def record_raw(
        self,
        path: Path,
        url: str,
        *,
        is_sample: bool = False,
        status: str = "downloaded",
        layer: str = "raw",
        refreshed: bool = False,
        **extra: Any,
    ) -> Artifact:
        """Hash a raw file, write it to the manifest and return its artifact.

        `refreshed=True` stamps the current time even if the bytes did not change, which is
        how unversioned files record that they were re-checked.
        """
        stamp = {"retrieved_at": utc_now_iso()} if refreshed else {}
        previous = self.ctx.manifest.get(path)
        entry = self.ctx.manifest.record_file(
            path,
            source=self.name,
            layer=layer,  # type: ignore[arg-type]
            url=url,
            version=self.version,
            license=self.info.license,
            terms_url=self.info.terms_url,
            is_sample=is_sample,
            etag=extra.pop("etag", None),
            last_modified=extra.pop("last_modified", None),
            **stamp,
        )
        if previous is not None and previous.sha256 == entry.sha256 and not refreshed:
            status = "unchanged" if status == "downloaded" else status
        return Artifact(
            source=self.name,
            path=path,
            url=url,
            version=self.version,
            sha256=entry.sha256,
            size=entry.size_bytes,
            retrieved_at=entry.retrieved_at,
            license=self.info.license,
            terms_url=self.info.terms_url,
            is_sample=is_sample,
            status=status,
            extra=extra,
        )

    def current_artifact(
        self, path: Path, url: str, *, is_sample: bool = False, max_age_days: float | None = None
    ) -> Artifact | None:
        """The artifact for `path` if it is already on disk and matches the manifest.

        Version-pinned files never expire. Unversioned ones pass `max_age_days` and are
        fetched again once the copy is older than that.
        """
        if self.settings.refresh:
            return None
        if not self.ctx.manifest.is_current(path, url=url, version=self.version):
            return None
        entry = self.ctx.manifest.get(path)
        assert entry is not None
        if max_age_days is not None:
            fetched = datetime.fromisoformat(entry.retrieved_at.replace("Z", "+00:00"))
            if datetime.now(UTC) - fetched > timedelta(days=max_age_days):
                return None
        return Artifact(
            source=self.name,
            path=path,
            url=url,
            version=self.version,
            sha256=entry.sha256,
            size=entry.size_bytes,
            retrieved_at=entry.retrieved_at,
            license=self.info.license,
            terms_url=self.info.terms_url,
            is_sample=is_sample,
            status="unchanged",
        )

    async def fetch_file(
        self,
        url: str,
        filename: str,
        *,
        expected_size: int | None = None,
        expected_md5: str | None = None,
        sample_rows: int | None = None,
        header_lines: int = 1,
        kind: Kind = "bulk",
        max_age_days: float | None = None,
    ) -> Artifact:
        """Download one file, or only its first `sample_rows` rows in sample mode. Idempotent."""
        is_sample = sample_rows is not None
        dest = self.raw_dir / (sample_name(filename, sample_rows) if sample_rows else filename)
        current = self.current_artifact(dest, url, is_sample=is_sample, max_age_days=max_age_days)
        if current is not None:
            return current
        if sample_rows is not None:
            lines = sample_rows + header_lines
            data = await self.client.head_lines(url, lines, source=self.name, kind=kind)
            atomic_write_bytes(dest, data)
            return self.record_raw(dest, url, is_sample=True, refreshed=True)
        if expected_size:
            ensure_disk_space(dest, expected_size)
        result = await self.client.download(
            url,
            dest,
            source=self.name,
            kind=kind,
            expected_size=expected_size,
            expected_md5=expected_md5,
            on_progress=self.ctx.progress.download(f"{self.name}: {filename}", expected_size),
        )
        return self.record_raw(
            dest, url, refreshed=True, etag=result.etag, last_modified=result.last_modified
        )

    def manual_artifact(self, filename: str, url: str) -> Artifact | None:
        """A file a person placed under `data/manual/<source>/`, if present."""
        path = self.manual_dir / filename
        if not path.is_file():
            return None
        return self.record_raw(path, url, status="manual", layer="manual")

    def plan_file(
        self, url: str, filename: str, est_bytes: int | None, *, sample_rows: int | None = None
    ) -> PlannedAction:
        name = sample_name(filename, sample_rows) if sample_rows else filename
        return PlannedAction(
            source=self.name,
            action="sample" if sample_rows else "download",
            target=url,
            dest=self.rel(self.raw_dir / name),
            est_bytes=None if sample_rows else est_bytes,
            note=f"first {sample_rows} rows only" if sample_rows else "",
        )
