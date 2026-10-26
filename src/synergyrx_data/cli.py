"""Command line interface: `synergyrx <command>`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from synergyrx_data import __version__, build, pipeline
from synergyrx_data.config import ConfigError, Settings, load_settings
from synergyrx_data.connectors import REGISTRY, is_enabled, source_names
from synergyrx_data.logging_utils import setup_logging
from synergyrx_data.manifest import Manifest
from synergyrx_data.preflight import run_preflight
from synergyrx_data.state import MANUAL_STEP, NEEDS_ATTENTION, StateStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Polite, reproducible data collection for SynergyRx AI.",
)
# Piped or redirected output is not wrapped at 80 columns: tables stay one row per line.
console = Console(width=None if sys.stdout.isatty() else 200)

SourceOption = Annotated[
    list[str] | None,
    typer.Option("--source", "-s", help="Source to run (repeatable). Default: all enabled."),
]


def _settings(ctx: typer.Context, **updates: Any) -> Settings:
    options = ctx.obj or {}
    overrides = list(options.get("overrides", []))
    if options.get("data_dir"):
        overrides.append(f"data_dir={options['data_dir']}")
    if options.get("log_level"):
        overrides.append(f"logging.level={options['log_level']}")
    if options.get("text_logs"):
        overrides.append("logging.json_logs=false")
    try:
        settings = load_settings(options.get("config"), overrides)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(2) from exc
    settings = settings.model_copy(update={k: v for k, v in updates.items() if v is not None})
    log_file = settings.logging.file
    setup_logging(
        settings.logging.level,
        json_logs=settings.logging.json_logs,
        # A dry run must not create or modify anything, log files included.
        file=settings.resolve(log_file) if log_file and not settings.dry_run else None,
        secrets=settings.secrets.values_to_redact(),
    )
    return settings


def _table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    table = Table(title=title, title_justify="left", show_lines=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*["" if v is None else str(v) for v in row])
    console.print(table)


def _size(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return ""


def _print_build(title: str, result: build.BuildResult) -> None:
    _table(
        title,
        ["output", "rows"],
        [[path, f"{rows:,}" if rows >= 0 else ""] for path, rows in result.outputs.items()],
    )
    for key, value in result.stats.items():
        console.print(
            f"  {key.replace('_', ' ')}: {value:,}"
            if isinstance(value, int)
            else f"  {key.replace('_', ' ')}: {value}"
        )
    for finding in result.drift:
        if finding.severity == "warning":
            console.print(
                f"[yellow]schema drift[/yellow] {finding.table}: {finding.kind} {finding.detail}"
            )
    for warning in result.warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")


def _run_build(title: str, step: Any, settings: Settings) -> build.BuildResult:
    try:
        result: build.BuildResult = step(settings)
    except build.BuildError as exc:
        console.print(f"[red]{title} cannot run:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_build(title, result)
    return result


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config.yaml.")
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a setting, e.g. --set http.cache_ttl_seconds=0."),
    ] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
    log_level: Annotated[str | None, typer.Option(help="DEBUG, INFO, WARNING or ERROR.")] = None,
    text_logs: Annotated[bool, typer.Option(help="Human-readable logs instead of JSON.")] = False,
    version: Annotated[bool, typer.Option("--version", help="Print the version.")] = False,
) -> None:
    if version:
        console.print(f"synergyrx-data {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()
    ctx.obj = {
        "config": config,
        "overrides": set_ or [],
        "data_dir": data_dir,
        "log_level": log_level,
        "text_logs": text_logs,
    }


@app.command()
def sources(
    ctx: typer.Context,
    check: Annotated[
        bool, typer.Option("--check", help="Probe every link and rewrite docs/SOURCES.md.")
    ] = False,
) -> None:
    """List the sources, their licence and state. With --check, test every link."""
    settings = _settings(ctx)
    states = build.source_states(settings)
    _table(
        "Sources",
        ["source", "category", "access", "enabled", "state", "licence"],
        [
            [
                name,
                REGISTRY[name].info.category,
                REGISTRY[name].info.access,
                "yes" if is_enabled(name, settings) else "no",
                states.loc[states["source"] == name, "state"].iloc[0],
                REGISTRY[name].info.license,
            ]
            for name in source_names()
        ],
    )
    if check:
        from synergyrx_data.quality import linkcheck

        rows = asyncio.run(linkcheck.check_links(settings))
        path = linkcheck.write_sources_doc(settings, rows)
        _table(
            "Link check",
            ["source", "link", "HTTP", "note"],
            [[r["source"], r["label"], r["status"] or "-", r["note"]] for r in rows],
        )
        failed = [r for r in rows if r["status"] is None or r["status"] >= 400]
        console.print(f"{len(rows)} links checked, {len(failed)} need attention. Wrote {path}.")


@app.command()
def fetch(
    ctx: typer.Context,
    source: SourceOption = None,
    sample: Annotated[
        int | None, typer.Option(min=1, help="Fetch only about N records per source.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan; request and write nothing.")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore caches and fetch everything again.")
    ] = False,
) -> None:
    """Download, parse and validate the selected sources into data/raw and data/interim."""
    settings = _settings(ctx, sample=sample, dry_run=dry_run, refresh=refresh)
    try:
        if dry_run:
            actions = asyncio.run(pipeline.plan(settings, source))
            _table(
                "Dry run: what `fetch` would do",
                ["source", "action", "target", "destination", "size", "note"],
                [
                    [a.source, a.action, a.target, a.dest, _size(a.est_bytes), a.note]
                    for a in actions
                ],
            )
            total = sum(a.est_bytes or 0 for a in actions)
            console.print(
                f"{len(actions)} planned actions, about {_size(total) or '0 B'} to download. "
                "Dry run: nothing was requested, written or cached."
            )
            return
        runs = asyncio.run(pipeline.fetch(settings, source))
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(2) from exc
    _table(
        "Fetch" + (f" (sample of {sample})" if sample else ""),
        ["source", "outcome", "tables (rows)", "rejected", "note"],
        [
            [
                run.source,
                run.outcome,
                ", ".join(f"{t} ({n:,})" for t, n in run.tables.items()),
                ", ".join(f"{t} ({n:,})" for t, n in run.rejects.items()),
                run.message,
            ]
            for run in runs
        ],
    )
    if any(run.outcome == MANUAL_STEP for run in runs):
        console.print("Some sources need a manual step (see the note column).")
    if any(run.outcome == NEEDS_ATTENTION for run in runs):
        raise typer.Exit(1)


@app.command()
def validate(ctx: typer.Context, source: SourceOption = None) -> None:
    """Re-check raw file hashes against the manifest and interim rows against their schemas."""
    settings = _settings(ctx)
    rows = pipeline.validate(settings, source)
    _table(
        "Validation",
        ["source", "check", "target", "ok", "detail"],
        [
            [r["source"], r["check"], r["target"], "yes" if r["ok"] else "NO", r["detail"]]
            for r in rows
        ],
    )
    if not rows:
        console.print("Nothing to validate yet: run `synergyrx fetch` first.")
    if any(not r["ok"] for r in rows):
        raise typer.Exit(1)


@app.command()
def clean(ctx: typer.Context) -> None:
    """Build the processed tables (data/parquet): drugs, cell lines, pairs, expression."""
    _run_build("Clean", build.clean, _settings(ctx))


@app.command()
def literature(ctx: typer.Context) -> None:
    """Build the abstract store, candidate sentences and the annotation sample."""
    _run_build("Literature", build.literature, _settings(ctx))


@app.command()
def report(ctx: typer.Context) -> None:
    """Write the quality report, coverage and conflict tables and LICENSES.md."""
    _run_build("Report", build.report, _settings(ctx))


@app.command()
def export(ctx: typer.Context) -> None:
    """Write the frontend JSON files into data/json/."""
    _run_build("Export", build.export, _settings(ctx))


@app.command()
def preflight(ctx: typer.Context, source: SourceOption = None) -> None:
    """Check disk space, permissions, identity and tooling before a run."""
    settings = _settings(ctx, dry_run=True)
    actions = asyncio.run(pipeline.plan(settings, source))
    planned = sum(a.est_bytes or 0 for a in actions)
    checks = run_preflight(settings, planned)
    _table(
        "Preflight",
        ["check", "result", "required", "detail"],
        [
            [c.name, "ok" if c.ok else "MISSING", "yes" if c.required else "no", c.detail]
            for c in checks
        ],
    )
    if any(not c.ok and c.required for c in checks):
        raise typer.Exit(1)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show the state of every connector and what is on disk."""
    settings = _settings(ctx)
    states = build.source_states(settings)
    _table(
        "Connector state",
        [str(c) for c in states.columns],
        [list(row) for row in states.itertuples(index=False)],
    )
    manifest = Manifest(settings)
    layers: dict[str, list[int]] = {}
    for entry in manifest.entries.values():
        bucket = layers.setdefault(entry.layer, [0, 0])
        bucket[0] += 1
        bucket[1] += entry.size_bytes
    _table(
        "Files in manifest.json",
        ["layer", "files", "size"],
        [[layer, n, _size(size)] for layer, (n, size) in sorted(layers.items())],
    )
    if samples := build.sample_sources(settings, manifest):
        console.print(f"[yellow]Sample data present for:[/yellow] {', '.join(samples)}")


@app.command()
def summary(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Print a final report of everything collected."""
    settings = _settings(ctx)
    from synergyrx_data.provenance import read_parquet
    from synergyrx_data.quality.reports import coverage_table

    state = StateStore(settings.state_dir / "connectors.json", read_only=True)
    manifest = Manifest(settings)
    states = build.source_states(settings)
    payload: dict[str, Any] = {
        "version": __version__,
        "sample_sources": build.sample_sources(settings, manifest),
        "sources": states.to_dict("records"),
        "clean": state.get("_clean").get("stats"),
        "literature": state.get("_literature").get("stats"),
        "outputs": {
            path: entry.rows
            for path, entry in sorted(manifest.entries.items())
            if entry.layer in ("processed", "export", "report")
        },
    }
    drugs = read_parquet(settings.parquet_root / "drugs.parquet")
    cells = read_parquet(settings.parquet_root / "cell_lines.parquet")
    pairs = read_parquet(settings.parquet_root / "synergy_pairs.parquet")
    coverage = None
    if drugs is not None and cells is not None and pairs is not None:
        coverage = coverage_table(pairs, drugs, cells)
        payload["coverage"] = coverage.to_dict("records")
    if as_json:
        console.print_json(json.dumps(payload, default=str))
        return
    if payload["sample_sources"]:
        console.print(
            f"[yellow]SAMPLE DATA[/yellow] for: {', '.join(payload['sample_sources'])}. "
            "Counts below are not the full datasets."
        )
    _table(
        "Sources",
        ["source", "enabled", "state", "last run", "rows", "message"],
        [
            [
                r["source"],
                "yes" if r["enabled"] else "no",
                r["state"],
                r["last_run"],
                f"{int(r['rows']):,}" if r["rows"] == r["rows"] and r["rows"] else "",
                r["message"],
            ]
            for r in payload["sources"]
        ],
    )
    if coverage is not None and len(coverage):
        shown = [
            "source",
            "records",
            "distinct_drugs",
            "distinct_cell_lines",
            "distinct_pairs",
            "distinct_triplets",
            "drugs_with_valid_smiles",
            "cell_lines_mapped_to_depmap",
        ]
        _table(
            "Coverage",
            [c.replace("_", " ") for c in shown],
            [
                [f"{v:,}" if isinstance(v, int) else v for v in row]
                for row in coverage[shown].itertuples(index=False)
            ],
        )
        total = coverage[coverage["source"] == "ALL SOURCES"].iloc[0]
        console.print(f"Distinct drugs across all sources: {int(total['distinct_drugs']):,}")
    else:
        console.print("No processed tables yet: run `synergyrx fetch` then `synergyrx clean`.")
    for title, key in (("Cleaning", "clean"), ("Literature", "literature")):
        if payload[key]:
            _table(
                title,
                ["measure", "value"],
                [
                    [k.replace("_", " "), f"{v:,}" if isinstance(v, int) else v]
                    for k, v in payload[key].items()
                ],
            )
    _table(
        "Outputs",
        ["file", "rows"],
        [
            [path, f"{rows:,}" if rows is not None else ""]
            for path, rows in payload["outputs"].items()
        ],
    )


@app.command()
def update(
    ctx: typer.Context,
    sample: Annotated[int | None, typer.Option(min=1)] = None,
    skip_literature: Annotated[bool, typer.Option(help="Stop after the synergy tables.")] = False,
) -> None:
    """Incremental end-to-end run for a scheduler: fetch, clean, literature, report, export.

    Safe to run repeatedly: unchanged files are skipped and PubMed is only asked for records
    entered since the previous run.
    """
    settings = _settings(ctx, sample=sample)
    pipeline.mark_disabled(settings)
    literature_sources = {"pubmed", "pubmed_baseline", "europepmc", "pubtator"}
    enabled = pipeline.select_sources(settings, None)
    first = [name for name in enabled if name not in literature_sources]
    second = [name for name in enabled if name in literature_sources]
    runs = asyncio.run(pipeline.fetch(settings, first)) if first else []
    _run_build("Clean", build.clean, settings)
    if second and not skip_literature:
        runs += asyncio.run(pipeline.fetch(settings, second))
        try:
            _print_build("Literature", build.literature(settings))
        except build.BuildError as exc:
            console.print(f"[yellow]literature skipped:[/yellow] {exc}")
    _run_build("Report", build.report, settings)
    _run_build("Export", build.export, settings)
    attention = [run.source for run in runs if run.outcome == NEEDS_ATTENTION]
    manual = [run.source for run in runs if run.outcome == MANUAL_STEP]
    if manual:
        console.print(f"Manual step required for: {', '.join(manual)} (see `synergyrx status`).")
    if attention:
        console.print(f"[red]Needs attention:[/red] {', '.join(attention)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
