"""PubMed annual baseline files (optional, very large, disabled by default).

The whole baseline is more than a thousand files and tens of gigabytes. Nothing is
downloaded unless file numbers are listed explicitly in the configuration. File names are
predictable (`pubmed26n0001.xml.gz`), so the directory listing is never parsed, and each
file is verified against the `.md5` NLM publishes next to it.
"""

from __future__ import annotations

import gzip
import re
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import (
    Artifact,
    Connector,
    Link,
    ManualStepRequired,
    PlannedAction,
    SourceInfo,
)
from synergyrx_data.connectors.schemas import ArticleRecord, Record
from synergyrx_data.http.errors import HttpError, RobotsDisallowedError
from synergyrx_data.literature.abstracts import articles_frame, iter_pubmed_articles

_MD5 = re.compile(r"\b([0-9a-fA-F]{32})\b")


class PubMedBaselineConnector(Connector):
    info = SourceInfo(
        name="pubmed_baseline",
        title="PubMed annual baseline files",
        category="literature",
        access="bulk",
        homepage="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
        license=(
            "NLM data terms; abstracts may be under publisher copyright (research use, "
            "do not redistribute abstracts)"
        ),
        terms_url="https://www.nlm.nih.gov/databases/download/terms_and_conditions.html",
        citation="National Library of Medicine. PubMed annual baseline. Bethesda (MD): NLM.",
        hosts=("ftp.ncbi.nlm.nih.gov",),
        links=(
            Link(
                "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
                "PubMed baseline files",
                "html",
                "verify",
            ),
        ),
        stage=4,
        optional=True,
        default_enabled=False,
        notes=(
            "List the file numbers to fetch in `options.files`; the default is none. The "
            "server answers `Disallow: /` in robots.txt, so files are placed manually."
        ),
    )
    default_version = "2026"
    default_options: ClassVar[dict[str, Any]] = {
        "base_url": "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
        "year_prefix": 26,
        "files": [],
        "approx_file_bytes": 60_000_000,
        "only_with_abstract": True,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"articles": ArticleRecord}

    def _names(self) -> list[str]:
        prefix = int(self.options["year_prefix"])
        return [f"pubmed{prefix:02d}n{int(n):04d}.xml.gz" for n in self.options["files"]]

    async def discover(self) -> list[PlannedAction]:
        names = self._names()
        if not names:
            return [
                PlannedAction(
                    self.name,
                    "skip",
                    self.options["base_url"],
                    None,
                    None,
                    "no file numbers listed in sources.pubmed_baseline.options.files",
                )
            ]
        return [
            self.plan_file(self.options["base_url"] + name, name, self.options["approx_file_bytes"])
            for name in names
        ]

    async def fetch(self) -> list[Artifact]:
        artifacts = []
        for name in self._names():
            url = self.options["base_url"] + name
            if (manual := self.manual_artifact(name, url)) is not None:
                artifacts.append(manual)
                continue
            if (current := self.current_artifact(self.raw_dir / name, url)) is not None:
                artifacts.append(current)
                continue
            try:
                checksum = await self.client.fetch(
                    url + ".md5", source=self.name, kind="bulk", expect=("text",)
                )
            except RobotsDisallowedError as exc:
                # ftp.ncbi.nlm.nih.gov serves `Disallow: /` to every automated client.
                raise ManualStepRequired(
                    "robots.txt of ftp.ncbi.nlm.nih.gov disallows automated downloads. "
                    f"Download {name} yourself from {self.options['base_url']} and save it as "
                    f"{self.rel(self.manual_dir / name)}, then run the command again."
                ) from exc
            match = _MD5.search(checksum.text())
            if match is None:
                raise HttpError(f"no MD5 found in {name}.md5", url=url + ".md5")
            artifacts.append(await self.fetch_file(url, name, expected_md5=match.group(1)))
        return artifacts

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        records: list[dict[str, Any]] = []
        for artifact in artifacts:
            provenance = {
                "source": self.name,
                "retrieved_at": artifact.retrieved_at,
                "source_version": artifact.version,
                "license": artifact.license,
            }
            kept = 0
            with gzip.open(artifact.path, "rb") as handle:
                for record in iter_pubmed_articles(handle):
                    if self.options["only_with_abstract"] and not record["has_abstract"]:
                        continue
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/"
                    records.append({**record, **provenance, "source_url": url})
                    kept += 1
                    if self.sample and kept >= self.sample:
                        break
        return {"articles": articles_frame(records)} if records else {}
