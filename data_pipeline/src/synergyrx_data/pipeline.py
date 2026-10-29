"""Orchestration: what the CLI commands actually do.

`fetch` runs the connectors stage by stage (synergy and cell lines, then identifiers, then
literature). A failing source never stops the others: its state becomes "needs attention"
and the run carries on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from synergyrx_data.config import Settings
from synergyrx_data.connectors import (
    REGISTRY,
    allowed_hosts,
    get_connector,
    is_enabled,
    source_names,
)
from synergyrx_data.connectors.base import (
    Artifact,
    Connector,
    Context,
    ManualStepRequired,
    PlannedAction,
    ValidationResult,
    validate_frame,
)
from synergyrx_data.http.client import PoliteClient
from synergyrx_data.http.errors import HttpError
from synergyrx_data.logging_utils import get_logger, log_event
from synergyrx_data.manifest import Manifest, hash_file, utc_now_iso
from synergyrx_data.preflight import PreflightError
from synergyrx_data.progress import ProgressReporter
from synergyrx_data.provenance import read_parquet, write_parquet
from synergyrx_data.quality.drift import DriftTracker
from synergyrx_data.state import DISABLED, MANUAL_STEP, NEEDS_ATTENTION, OK, StateStore

log = get_logger("pipeline")


@dataclass
class SourceRun:
    source: str
    outcome: str  # ok | unchanged | needs_attention | manual_step_required | disabled
    message: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    tables: dict[str, int] = field(default_factory=dict)
    rejects: dict[str, int] = field(default_factory=dict)
    validation: list[ValidationResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    drift: list[dict[str, str]] = field(default_factory=list)
    rebuilt: bool = False  # the interim tables were parsed again in this run


def make_context(settings: Settings, client: PoliteClient | None = None) -> Context:
    return Context(
        settings=settings,
        manifest=Manifest(settings),
        state=StateStore(settings.state_dir / "connectors.json", read_only=settings.dry_run),
        client=client,
        progress=ProgressReporter(settings.logging.progress and not settings.dry_run),
    )


def select_sources(settings: Settings, requested: list[str] | None) -> list[str]:
    """Explicitly requested sources (even if disabled), or every enabled one, in run order."""
    if requested:
        unknown = [name for name in requested if name not in REGISTRY]
        if unknown:
            raise KeyError(
                f"unknown source(s): {', '.join(unknown)}; known: {', '.join(source_names())}"
            )
        return [name for name in source_names() if name in requested]
    return [name for name in source_names() if is_enabled(name, settings)]


async def plan(settings: Settings, requested: list[str] | None) -> list[PlannedAction]:
    """What `fetch` would do. Offline and read-only: this is the whole of `--dry-run`."""
    ctx = make_context(settings)
    actions: list[PlannedAction] = []
    for name in select_sources(settings, requested):
        actions.extend(await get_connector(name, ctx).discover())
    return actions


def _interim_is_current(connector: Connector, fingerprint: str) -> dict[str, int] | None:
    """Row counts of the interim tables if they were built from exactly these inputs."""
    entries = [
        e
        for e in connector.ctx.manifest.for_source(connector.name, "interim")
        if not e.path.endswith(".rejects.parquet")
    ]
    if not entries or connector.settings.refresh:
        return None
    for entry in entries:
        path = connector.settings.resolve(entry.path)
        if entry.input_fingerprint != fingerprint or not path.is_file():
            return None
        if hash_file(path)[0] != entry.sha256:
            return None
    return {e.path.rsplit("/", 1)[-1].removesuffix(".parquet"): e.rows or 0 for e in entries}


def _write_interim(connector: Connector, artifacts: list[Artifact], run: SourceRun) -> None:
    """parse -> validate -> write valid rows and rejects to `data/interim/<source>/`."""
    fingerprint = connector.fingerprint(artifacts)
    if (current := _interim_is_current(connector, fingerprint)) is not None:
        run.tables = current
        return
    run.rebuilt = True
    tables = connector.parse(artifacts)
    validated = connector.validate(tables)
    inputs = [connector.rel(a.path) for a in artifacts]
    is_sample = connector.sample is not None
    tracker = DriftTracker(connector.settings.state_dir / "schemas")
    for table, outcome in validated.items():
        # A new release that renames or drops a column shows up here, not three steps later.
        findings = tracker.check(
            f"interim/{connector.name}/{table}",
            outcome.valid,
            connector.version,
            is_sample=is_sample,
        )
        run.drift += [f.to_dict() for f in findings if f.kind != "first_snapshot"]
        path = write_parquet(outcome.valid, connector.interim_dir / f"{table}.parquet")
        connector.ctx.manifest.record_file(
            path,
            source=connector.name,
            layer="interim",
            version=connector.version,
            license=connector.info.license,
            terms_url=connector.info.terms_url,
            is_sample=is_sample,
            rows=len(outcome.valid),
            derived_from=inputs,
            input_fingerprint=fingerprint,
        )
        rejects_path = connector.interim_dir / f"{table}.rejects.parquet"
        if len(outcome.rejects):
            write_parquet(outcome.rejects, rejects_path)
            run.rejects[table] = len(outcome.rejects)
        elif rejects_path.is_file():
            rejects_path.unlink()
        run.tables[table] = len(outcome.valid)
        run.validation.append(outcome.result)


async def _run_connector(connector: Connector) -> SourceRun:
    run = SourceRun(connector.name, OK)
    try:
        run.artifacts = await connector.fetch()
        if run.artifacts:
            _write_interim(connector, run.artifacts, run)
        if connector.problems:
            run.outcome, run.message = NEEDS_ATTENTION, "; ".join(connector.problems)
        elif not run.rebuilt and all(a.status == "unchanged" for a in run.artifacts):
            run.outcome = "unchanged"
    except ManualStepRequired as exc:
        run.outcome, run.message = MANUAL_STEP, str(exc)
    except (HttpError, httpx.HTTPError, PreflightError) as exc:
        run.outcome, run.message = NEEDS_ATTENTION, f"{type(exc).__name__}: {exc}"
        log_event(log, logging.ERROR, "source.failed", source=connector.name, error=str(exc))
    except Exception as exc:
        # A corrupt archive, a changed file layout or a bug in one connector must not stop
        # the other sources. The traceback goes to the log; the source needs attention.
        run.outcome, run.message = NEEDS_ATTENTION, f"{type(exc).__name__}: {exc}"
        log.exception("source.crashed", extra={"source": connector.name})
    run.stats = dict(connector.stats)
    state = OK if run.outcome in (OK, "unchanged") else run.outcome
    connector.ctx.state.update(
        connector.name,
        state=state,
        message=run.message,
        last_run_at=utc_now_iso(),
        is_sample=connector.sample is not None,
        tables=run.tables,
        rejects=run.rejects,
        stats=run.stats,
        drift=run.drift,
    )
    connector.ctx.manifest.save()
    log_event(
        log,
        logging.INFO,
        "source.done",
        source=connector.name,
        outcome=run.outcome,
        tables=run.tables,
        rejects=run.rejects,
    )
    return run


async def fetch(
    settings: Settings,
    requested: list[str] | None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[SourceRun]:
    """Fetch, parse and validate the selected sources, one stage after another."""
    names = select_sources(settings, requested)
    runs: list[SourceRun] = []
    async with PoliteClient(settings, allowed_hosts=allowed_hosts(), transport=transport) as client:
        ctx = make_context(settings, client)
        with ctx.progress:
            for name in names:
                runs.append(await _run_connector(get_connector(name, ctx)))
        log_event(
            log,
            logging.INFO,
            "fetch.done",
            requests=dict(client.stats.requests),
            cache_hits=client.stats.cache_hits,
            retries=client.stats.retries,
            bytes=client.stats.bytes,
        )
    ctx.state.update(
        "_last_fetch",
        at=utc_now_iso(),
        requests=dict(client.stats.requests),
        cache_hits=client.stats.cache_hits,
    )
    return runs


def mark_disabled(settings: Settings) -> None:
    """Record sources that are switched off, so `status` can say why they have no data."""
    ctx = make_context(settings)
    for name in source_names():
        if not is_enabled(name, settings) and ctx.state.get(name).get("state") in (
            None,
            "never_run",
        ):
            ctx.state.update(name, state=DISABLED, message="disabled in configuration")


def validate(settings: Settings, requested: list[str] | None) -> list[dict[str, Any]]:
    """Re-check what is on disk: raw file hashes against the manifest, interim rows against
    their schemas."""
    manifest = Manifest(settings)
    ctx = make_context(settings)
    names = requested or source_names()
    rows: list[dict[str, Any]] = []
    for name in names:
        connector = get_connector(name, ctx)
        for entry in manifest.for_source(name):
            if entry.layer not in ("raw", "manual"):
                continue
            path = settings.resolve(entry.path)
            ok = path.is_file() and hash_file(path)[0] == entry.sha256
            rows.append(
                {
                    "source": name,
                    "check": "raw file hash",
                    "target": entry.path,
                    "ok": ok,
                    "detail": "matches manifest" if ok else "missing or modified",
                }
            )
        for table, model in connector.schemas.items():
            frame: pd.DataFrame | None = read_parquet(connector.interim_dir / f"{table}.parquet")
            if frame is None:
                continue
            mask, examples = validate_frame(frame, model)
            invalid = int((~mask).sum())
            rows.append(
                {
                    "source": name,
                    "check": f"schema {model.__name__}",
                    "target": connector.rel(connector.interim_dir / f"{table}.parquet"),
                    "ok": invalid == 0,
                    "detail": f"{len(frame)} rows, {invalid} invalid"
                    + (f" (e.g. {examples[0]})" if examples else ""),
                }
            )
    return rows
