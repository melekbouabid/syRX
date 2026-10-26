"""Link check and generation of `docs/SOURCES.md`.

`synergyrx sources --check` probes every URL from the project brief plus the data endpoint
each connector really uses, then rewrites the document with the HTTP status it got. Pages
are only requested if robots.txt allows it; access walls are reported, not worked around.
"""

from __future__ import annotations

import json
from typing import Any

from synergyrx_data.config import Settings
from synergyrx_data.connectors import REGISTRY, all_links, allowed_hosts, is_enabled, source_names
from synergyrx_data.http.cache import atomic_write_bytes
from synergyrx_data.http.client import Kind, PoliteClient
from synergyrx_data.manifest import utc_now_iso

# (source, label, url, kind): the endpoint each connector downloads from or calls.
ENDPOINTS: list[tuple[str, str, str, Kind]] = [
    (
        "drugcomb",
        "Zenodo file (bulk)",
        "https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content",
        "bulk",
    ),
    (
        "tdc",
        "Harvard Dataverse file used by PyTDC",
        "https://dataverse.harvard.edu/api/access/datafile/4215720",
        "api",
    ),
    ("oneil", "labels.csv (bulk)", "https://bioinf.jku.at/software/DeepSynergy/labels.csv", "bulk"),
    (
        "nci_almanac",
        "ComboDrugGrowth_Nov2017.zip (bulk)",
        "https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip",
        "bulk",
    ),
    (
        "depmap",
        "Figshare API, DepMap 24Q4 article",
        "https://api.figshare.com/v2/articles/27993248",
        "api",
    ),
    (
        "gdsc",
        "Cell Model Passports model list (bulk)",
        "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz",
        "bulk",
    ),
    (
        "pubmed",
        "E-utilities EInfo",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json",
        "api",
    ),
    (
        "pubmed_baseline",
        "Baseline README (bulk)",
        "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt",
        "bulk",
    ),
    (
        "pubchem",
        "PUG-REST property call",
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/InChIKey/JSON",
        "api",
    ),
    ("chembl", "ChEMBL API status", "https://www.ebi.ac.uk/chembl/api/data/status.json", "api"),
    (
        "europepmc",
        "Europe PMC search API",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:31600000&format=json"
        "&pageSize=1",
        "api",
    ),
    (
        "pubtator",
        "PubTator 3 export API",
        "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
        "?pmids=31600000",
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
        "pubchem.ncbi.nlm.nih.gov",
        "5 req/s",
        "No more than 5 requests per second and 400 per minute; dynamic throttling headers",
        "PubChem programmatic access documentation. The page needs JavaScript and could not be "
        "re-read by the pipeline: confirm in a browser.",
    ),
    (
        "www.ncbi.nlm.nih.gov (PubTator 3 API)",
        "2 req/s",
        "No more than three requests per second",
        "PubTator 3 API page. Needs JavaScript; could not be re-read by the pipeline.",
    ),
    (
        "zenodo.org",
        "1 request per 10 s (robots.txt Crawl-delay)",
        "Guests: 60 requests per minute, 2000 per hour. robots.txt: Crawl-delay 10; `/api` "
        "disallowed except `/api/records/*/files`",
        "Read from developers.zenodo.org and zenodo.org/robots.txt on 2026-10-13.",
    ),
    (
        "api.figshare.com / ndownloader.figshare.com",
        "1 req/s",
        'No automatic rate limit; "do not make more than one request per second"',
        "Read from docs.figshare.com on 2026-10-13.",
    ),
    (
        "www.ebi.ac.uk (ChEMBL, Europe PMC)",
        "2 req/s",
        "No numeric limit found in the ChEMBL web-service documentation",
        "ChEMBL documentation read on 2026-10-13 mentions no rate limit; kept conservative.",
    ),
    (
        "dataverse.harvard.edu",
        "1 req/s",
        "No numeric limit found",
        "Dataverse guide page for rate limits was not found (404); kept conservative.",
    ),
    (
        "wiki.nci.nih.gov",
        "1 request per 15 s (robots.txt Crawl-delay)",
        "robots.txt: Crawl-delay 15, everything disallowed except wiki pages",
        "Read from wiki.nci.nih.gov/robots.txt on 2026-10-13.",
    ),
    (
        "every other host",
        "`http.default_limit`: 1 req/s, one connection",
        "-",
        "Conservative default.",
    ),
]

FINDINGS = """## What was found while building the connectors

Measured on 2026-10-13 with the pipeline itself.

- **DrugComb**: the Zenodo record holds one file, `summary_table_v1.4.csv` (193 MB, CC BY 4.0,
  MD5 pinned in the connector). It has **no study and no tissue column**, so a row cannot be
  traced back to the screen it came from. Of its 1,175,220 rows, 709,187 are single-drug
  screens (second drug `NA`, synergy 0) and are skipped; 466,033 are combinations, covering
  4,150 drug names and 112 cell lines. The portal `drugcomb.org` did not answer at all
  (connection timeout); the data does not depend on it.
- **TDC**: PyTDC 1.1.15 pins `numpy<2`, `pandas<3` and `rdkit<2024.3.1`, so it does not
  install on Python 3.13+. The connector downloads the same Harvard Dataverse file PyTDC
  uses and reads it with a restricted loader when PyTDC is absent. TDC's `drugcomb` set
  (297,098 rows, 129 drug names, 59 cell lines) is a copy of the NCI-60 part of DrugComb:
  on the 286,674 combinations they share, the scores are identical.
- **O'Neil / DeepSynergy**: 23,052 rows, 38 drugs, 39 cell lines, Loewe only. The same
  experiments are inside DrugComb with scores recomputed by a different pipeline: on the
  22,737 shared combinations the two Loewe values correlate at only 0.42.
- **NCI-ALMANAC**: `dtp.cancer.gov` serves `Disallow: /` to every automated client, and the
  data files are attachments on `wiki.nci.nih.gov`, whose robots.txt disallows `/download/`.
  The pipeline requests neither; a person downloads the zip in a browser and places it under
  `data/manual/nci_almanac/`. The ALMANAC screen is also present inside DrugComb and TDC.
- **DepMap**: the portal API answers automated requests with a "verifying you are a person"
  page and asks not to be scraped. The connector uses the official Figshare+ release
  instead (documented API, MD5 checksums, CC BY 4.0), pinned to 24Q4 (2,105 models).
- **GDSC**: `cancerrxgene.org` answers **410 Gone**; GDSC moved into the Sanger DepMap. The
  optional connector reads the Cell Model Passports model list.
- **PubMed baseline**: `ftp.ncbi.nlm.nih.gov` serves `Disallow: /` in robots.txt, although
  NLM documents the server as its bulk download site. The optional connector therefore does
  not fetch from it and reads files placed under `data/manual/pubmed_baseline/`.
- **DDInter**: licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). Disabled, and it
  refuses to download until `terms_acknowledged: true` is set.
- **DrugBank**: never requested by a connector. The site answers the link check with a
  Cloudflare challenge (403). The connector only reads a file a licensed user placed.
- **Documentation pages** (NCBI Bookshelf, PubChem docs, Europe PMC `RestfulWebService`,
  the Figshare+ record page) sit behind a browser check, a CAPTCHA or JavaScript. They are
  reported as such. The APIs themselves answer normally.
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
        "rest comes from the connector registry (`src/synergyrx_data/connectors/`).",
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
