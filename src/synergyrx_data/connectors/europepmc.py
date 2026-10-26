"""Europe PMC REST API: open-access status, missing abstracts, and a fallback search.

Normal mode ("enrich") looks up the PMIDs PubMed returned. If the PubMed connector is in a
failed state, or `mode: search` is set, the same drug queries are run against Europe PMC
instead so the literature step still has articles to work with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._lookup_store import LookupStore
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.pubchem import chunked
from synergyrx_data.connectors.schemas import EuropePmcRecord, Record
from synergyrx_data.http.errors import HttpError
from synergyrx_data.literature.queries import build_queries, load_query_inputs
from synergyrx_data.manifest import utc_now_iso
from synergyrx_data.provenance import read_parquet
from synergyrx_data.state import NEEDS_ATTENTION

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePmcConnector(Connector):
    info = SourceInfo(
        name="europepmc",
        title="Europe PMC REST API",
        category="literature",
        access="api",
        homepage="https://europepmc.org/",
        license=(
            "Europe PMC terms: metadata is free to reuse; abstracts and full text keep "
            "their publisher or open-access licence"
        ),
        terms_url="https://europepmc.org/Copyright",
        citation=(
            "Rosonovski S, et al. Europe PMC in 2023. Nucleic Acids Res. "
            "2024;52(D1):D1668-76. doi:10.1093/nar/gkad1085"
        ),
        hosts=("www.ebi.ac.uk", "europepmc.org"),
        links=(
            Link(
                "https://europepmc.org/RestfulWebService", "Europe PMC REST API", "html", "verify"
            ),
        ),
        stage=5,
        notes="Runs after PubMed. Adds `is_open_access` and fills abstracts PubMed lacks.",
    )
    default_options: ClassVar[dict[str, Any]] = {
        "mode": "auto",  # auto | enrich | search
        "batch_size": 25,
        "max_queries": 25,
        "max_records_per_query": 200,
        "lookup_ttl_days": 30,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"articles": EuropePmcRecord}

    @property
    def store_path(self) -> Path:
        return self.raw_dir / "articles.jsonl"

    def _mode(self) -> str:
        mode = str(self.options["mode"])
        if mode != "auto":
            return mode
        pubmed_failed = self.ctx.state.get("pubmed").get("state") == NEEDS_ATTENTION
        return "search" if pubmed_failed else "enrich"

    def _pmids(self) -> list[str]:
        articles = read_parquet(self.settings.interim_dir / "pubmed" / "articles.parquet", ["pmid"])
        pmids = sorted(set(articles["pmid"].astype(str))) if articles is not None else []
        return pmids[: self.sample] if self.sample else pmids

    async def discover(self) -> list[PlannedAction]:
        store = LookupStore(self.store_path, ("pmid",))
        if self._mode() == "search":
            note = "fallback search: PubMed is unavailable or mode=search"
        else:
            pending = [p for p in self._pmids() if p not in store]
            note = f"{len(pending)} PMIDs to enrich, {len(store)} already answered"
        return [PlannedAction(self.name, "api", SEARCH, None, None, note)]

    @staticmethod
    def _record(
        pmid: str, hit: dict[str, Any] | None, query_id: str | None = None
    ) -> dict[str, Any]:
        hit = hit or {}
        journal = (hit.get("journalInfo") or {}).get("journal") or {}
        year = hit.get("pubYear")
        return {
            "pmid": pmid,
            "found": bool(hit),
            "pmcid": hit.get("pmcid"),
            "doi": (hit.get("doi") or "").lower() or None,
            "title": hit.get("title"),
            "abstract": hit.get("abstractText"),
            "journal": journal.get("isoabbreviation") or journal.get("title"),
            "year": int(year) if year and str(year).isdigit() else None,
            "language": hit.get("language"),
            "publication_types": (hit.get("pubTypeList") or {}).get("pubType", []),
            "is_open_access": hit.get("isOpenAccess") == "Y",
            "in_epmc": hit.get("inEPMC") == "Y",
            "has_pdf": hit.get("hasPDF") == "Y",
            "article_license": hit.get("license"),
            "found_by_query": query_id,
            "retrieved_at": utc_now_iso(),
        }

    async def _search(self, query: str, cursor: str = "*") -> dict[str, Any]:
        result = await self.client.fetch(
            SEARCH,
            source=self.name,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": "100",
                "cursorMark": cursor,
            },
            ttl=self.options["lookup_ttl_days"] * 86_400,
        )
        payload: dict[str, Any] = result.json()
        return payload

    async def _enrich(self, store: LookupStore) -> None:
        pending = [p for p in self._pmids() if p not in store]
        advance = self.ctx.progress.counter("europepmc: PMIDs", len(pending))
        for batch in chunked(pending, self.options["batch_size"]):
            query = "(" + " OR ".join(f"EXT_ID:{p}" for p in batch) + ") AND SRC:MED"
            payload = await self._search(query)
            hits = {h.get("pmid"): h for h in payload.get("resultList", {}).get("result", [])}
            for pmid in batch:
                store.add(self._record(pmid, hits.get(pmid)))
            advance(len(batch))

    async def _fallback_search(self, store: LookupStore) -> None:
        drugs, pairs = load_query_inputs(self.settings)
        limit = min(self.options["max_queries"], 3) if self.sample else self.options["max_queries"]
        cap = int(self.options["max_records_per_query"])
        cap = min(cap, self.sample) if self.sample else cap
        for query in build_queries(drugs, pairs, max_queries=limit):
            cursor, fetched = "*", 0
            while fetched < cap:
                payload = await self._search(query.europepmc, cursor)
                hits = payload.get("resultList", {}).get("result", [])
                for hit in hits[: cap - fetched]:
                    if pmid := hit.get("pmid"):
                        store.add(self._record(pmid, hit, query.query_id))
                fetched += len(hits)
                next_cursor = payload.get("nextCursorMark")
                if not hits or not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

    async def fetch(self) -> list[Artifact]:
        store = LookupStore(self.store_path, ("pmid",))
        mode = self._mode()
        self.stats["mode"] = mode
        try:
            await (self._fallback_search(store) if mode == "search" else self._enrich(store))
        except HttpError as exc:
            self.problems.append(f"Europe PMC lookups incomplete: {exc}")
        store.save()
        self.stats["total_answers"] = len(store)
        if not self.store_path.is_file():
            return []
        return [self.record_raw(self.store_path, SEARCH)]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        if not artifacts:
            return {}
        frame = pd.DataFrame(LookupStore(artifacts[0].path, ("pmid",)).rows())
        if frame.empty:
            return {}
        found = frame[frame["found"]].drop(columns=["found"]).reset_index(drop=True)
        for column in (
            "pmid",
            "pmcid",
            "doi",
            "title",
            "abstract",
            "journal",
            "language",
            "article_license",
            "found_by_query",
            "retrieved_at",
        ):
            found[column] = found[column].astype("string")
        found["year"] = pd.to_numeric(found["year"], errors="coerce").astype("Int64")
        articles = found.assign(
            source=self.name,
            source_url="https://europepmc.org/abstract/MED/" + found["pmid"].astype(str),
            source_version=self.version,
            license=self.info.license,
        )
        return {"articles": articles}
