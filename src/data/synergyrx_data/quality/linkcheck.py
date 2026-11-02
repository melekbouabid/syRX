"""Link check and generation of `docs/data_pipeline/sources.md`.

`synergyrx sources --check` probes every URL from the project brief plus the data endpoint
each connector really uses, then rewrites the document with the HTTP status it got. Pages
are only requested if robots.txt allows it; access walls are reported, not worked around.
"""

from __future__ import annotations

import json
from typing import Any

from src.data.synergyrx_data.config import Settings
from src.data.synergyrx_data.connectors import (
    REGISTRY,
    all_links,
    allowed_hosts,
    is_enabled,
    source_names,
)
from src.data.synergyrx_data.http.cache import atomic_write_bytes
from src.data.synergyrx_data.http.client import Kind, PoliteClient
from src.data.synergyrx_data.manifest import utc_now_iso

# (source, label, url, kind): the endpoint each connector downloads from or calls.
ENDPOINTS: list[tuple[str, str, str, Kind]] = [
    (
        "pubmed",
        "E-utilities EInfo",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json",
        "api",
    ),
    (
        "ddinter",
        "DDInter download file (bulk)",
        "https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_V.csv",
        "bulk",
    ),
]

# What each provider documents, and how it was confirmed when this pipeline was built.
RATE_LIMITS: list[tuple[str, str, str, str]] = [
    (
        "eutils.ncbi.nlm.nih.gov",
        "`http.ncbi`: 3 req/s, 10 req/s with an API key",
        "3 requests per second without an API key, 10 with one; identify with `tool` and `email`",
        "NCBI E-utilities documentation (NBK25497). The Bookshelf page answered automated "
        "requests with a reCAPTCHA browser check, so the text could not be re-read by the "
        "pipeline: confirm in a browser.",
    ),
    (
        "www.ncbi.nlm.nih.gov (NCBI documentation pages)",
        "2 req/s",
        "-",
        "Only the documentation links are probed there; kept conservative.",
    ),
    (
        "rxnav.nlm.nih.gov",
        "10 req/s",
        "No more than 20 requests per second per IP address",
        "RxNav API terms of service; used by the Tujya ingredient step (module 2 corpus).",
    ),
    (
        "api.fda.gov",
        "3 req/s",
        "240 requests per minute per IP address without an API key",
        "openFDA authentication documentation; used by the Tujya label step (module 2 corpus).",
    ),
    (
        "every other host",
        "`http.default_limit`: 1 req/s, one connection",
        "-",
        "Conservative default.",
    ),
]

FINDINGS = """## What was found while building the connectors

Measured on 2026-10-13 with the pipeline itself. The SynergyRx synergy sources (DrugComb,
TDC, O'Neil, NCI-ALMANAC, DepMap, GDSC, PubChem, ChEMBL, Europe PMC, PubTator, DrugBank,
PubMed baseline) were removed later (decision D85); their findings are in the git history.

- **DDInter**: licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). Disabled, and it
  refuses to download until `terms_acknowledged: true` is set; the Tujya corpus step sets
  both for this academic project.
- **Documentation pages** (NCBI Bookshelf) sit behind a browser check or a CAPTCHA. They
  are reported as such. The APIs themselves answer normally.
"""


def _is_failure(status: int | None) -> bool:
    return status is None or status >= 400


async def check_links(settings: Settings) -> list[dict[str, Any]]:
    """Probe every brief link and data endpoint once, politely. Returns one row per URL."""
    rows: list[dict[str, Any]] = []
    targets: list[tuple[str, str, str, Kind, str]] = [
        (source, link.label, link.url, link.kind, link.origin or "added")
        for source, link in all_links()
    ] + [(source, label, url, kind, "endpoint") for source, label, url, kind in ENDPOINTS]
    async with PoliteClient(settings, allowed_hosts=allowed_hosts()) as client:
        for source, label, url, kind, origin in targets:
            result = await client.probe(url, kind=kind)
            rows.append(
                {
                    "source": source,
                    "label": label,
                    "url": url,
                    "kind": kind,
                    "origin": origin,
                    "status": result.status,
                    "final_url": result.final_url if result.final_url != url else None,
                    "content_type": result.content_type,
                    "note": result.note,
                }
            )
    payload = {"checked_at": utc_now_iso(), "links": rows}
    atomic_write_bytes(
        settings.state_dir / "link_status.json", json.dumps(payload, indent=1).encode("utf-8")
    )
    return rows


def _status_cell(row: dict[str, Any]) -> str:
    if row["status"] is None:
        if row["note"].startswith("not requested"):
            return "unreachable" if "could not be fetched" in row["note"] else "not requested"
        return "no answer"
    return str(row["status"])


def render_sources_doc(settings: Settings, rows: list[dict[str, Any]], checked_at: str) -> str:
    lines = [
        "# Sources",
        "",
        "Generated by `synergyrx sources --check`. Re-run it to refresh the HTTP statuses; the",
        "rest comes from the connector registry (`src/data/synergyrx_data/connectors/`).",
        "",
        f"Last link check: **{checked_at}**.",
        "",
        "## Links from the project brief",
        "",
        "`checked` and `verify` are the marks the brief gave each link. `added` links were not",
        "in the brief; the pipeline depends on them.",
        "",
        "| Source | Link | Mark | HTTP | Note | Licence | Terms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in (r for r in rows if r["origin"] != "endpoint"):
        info = REGISTRY[row["source"]].info
        note = row["note"] or ("redirects to " + row["final_url"] if row["final_url"] else "")
        lines.append(
            f"| {row['source']} | [{row['label']}]({row['url']}) | {row['origin']} | "
            f"{_status_cell(row)} | {note} | {info.license} | {info.terms_url or ''} |"
        )
    lines += [
        "",
        "## Data endpoints the connectors use",
        "",
        "| Source | Endpoint | Kind | HTTP | Note |",
        "|---|---|---|---|---|",
    ]
    for row in (r for r in rows if r["origin"] == "endpoint"):
        lines.append(
            f"| {row['source']} | [{row['label']}]({row['url']}) | {row['kind']} | "
            f"{_status_cell(row)} | {row['note']} |"
        )
    failed = [r for r in rows if _is_failure(r["status"])]
    lines += ["", "## Links that need attention", ""]
    if failed:
        for row in failed:
            lines.append(
                f"- **{row['source']}**: {row['url']} - {_status_cell(row)}"
                + (f" ({row['note']})" if row["note"] else "")
            )
    else:
        lines.append("None: every link answered.")
    walled = [r for r in rows if "access wall" in (r["note"] or "")]
    if walled:
        lines += ["", "Pages that answered with a browser check or login wall (not bypassed):", ""]
        lines += [f"- {row['url']} ({row['note']})" for row in walled]
    lines += [
        "",
        "## Sources, access and licences",
        "",
        "| Source | What | Access | Enabled | Licence | Terms | Redistribution |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in source_names():
        info = REGISTRY[name].info
        lines.append(
            f"| {name} | {info.title} | {info.access} | "
            f"{'yes' if is_enabled(name, settings) else 'no'} | {info.license} | "
            f"{info.terms_url or ''} | {info.redistribution} |"
        )
    lines += [
        "",
        "## Rate limits",
        "",
        f"All limits are applied at {settings.http.safety_factor:.0%} of the configured value "
        "(`http.safety_factor`), and a robots.txt `Crawl-delay` always wins when it is slower.",
        "",
        "| Host | Configured | What the provider documents | How it was confirmed |",
        "|---|---|---|---|",
    ]
    lines += [f"| {host} | {ours} | {theirs} | {how} |" for host, ours, theirs, how in RATE_LIMITS]
    lines += ["", FINDINGS]
    return "\n".join(lines).rstrip() + "\n"


def write_sources_doc(settings: Settings, rows: list[dict[str, Any]]) -> str:
    path = settings.resolve(settings.sources_doc_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_sources_doc(settings, rows, utc_now_iso()), encoding="utf-8")
    return settings.relative(path)
