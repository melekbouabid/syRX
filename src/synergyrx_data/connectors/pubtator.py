"""PubTator 3 entity annotations for collected PMIDs (optional, disabled by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._lookup_store import LookupStore
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.pubchem import chunked
from synergyrx_data.connectors.schemas import AnnotationRecord, Record
from synergyrx_data.http.errors import HttpError
from synergyrx_data.manifest import utc_now_iso
from synergyrx_data.provenance import read_parquet

EXPORT = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"


class PubTatorConnector(Connector):
    info = SourceInfo(
        name="pubtator",
        title="PubTator 3 annotations",
        category="literature",
        access="api",
        homepage="https://www.ncbi.nlm.nih.gov/research/pubtator3/",
        license="Public domain (NCBI); annotations are machine-generated",
        terms_url="https://www.ncbi.nlm.nih.gov/home/about/policies/",
        citation=(
            "Wei C-H, et al. PubTator 3.0: an AI-powered literature resource for unlocking "
            "biomedical knowledge. Nucleic Acids Res. 2024;52(W1):W540-6. "
            "doi:10.1093/nar/gkae235"
        ),
        hosts=("www.ncbi.nlm.nih.gov",),
        links=(
            Link(
                "https://www.ncbi.nlm.nih.gov/research/pubtator3/", "PubTator 3", "html", "verify"
            ),
        ),
        stage=5,
        optional=True,
        default_enabled=False,
        notes="Chemical / gene / disease mentions, useful to cross-check the offline matcher.",
    )
    default_options: ClassVar[dict[str, Any]] = {"batch_size": 100, "lookup_ttl_days": 30}
    schemas: ClassVar[dict[str, type[Record]]] = {"annotations": AnnotationRecord}

    @property
    def store_path(self) -> Path:
        return self.raw_dir / "annotations.jsonl"

    def _pending(self, store: LookupStore) -> list[str]:
        articles = read_parquet(self.settings.interim_dir / "pubmed" / "articles.parquet", ["pmid"])
        pmids = sorted(set(articles["pmid"].astype(str))) if articles is not None else []
        pmids = pmids[: self.sample] if self.sample else pmids
        return [p for p in pmids if p not in store]

    async def discover(self) -> list[PlannedAction]:
        store = LookupStore(self.store_path, ("pmid",))
        note = f"{len(self._pending(store))} PMIDs to annotate, {len(store)} already answered"
        return [PlannedAction(self.name, "api", EXPORT, None, None, note)]

    @staticmethod
    def _annotations(document: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for passage in document.get("passages", []):
            section = (passage.get("infons") or {}).get("type")
            for annotation in passage.get("annotations", []):
                infons = annotation.get("infons") or {}
                location = (annotation.get("locations") or [{}])[0]
                if not annotation.get("text") or not infons.get("type"):
                    continue
                rows.append(
                    {
                        "entity_type": infons["type"],
                        "text": annotation["text"],
                        "identifier": infons.get("identifier"),
                        "section": section,
                        "start": int(location.get("offset", 0)),
                        "length": int(location.get("length", 0)),
                    }
                )
        return rows

    async def fetch(self) -> list[Artifact]:
        store = LookupStore(self.store_path, ("pmid",))
        try:
            for batch in chunked(self._pending(store), self.options["batch_size"]):
                result = await self.client.fetch(
                    EXPORT,
                    source=self.name,
                    params={"pmids": ",".join(batch)},
                    ttl=self.options["lookup_ttl_days"] * 86_400,
                )
                documents = {str(d.get("id")): d for d in result.json().get("PubTator3", [])}
                now = utc_now_iso()
                for pmid in batch:
                    document = documents.get(pmid)
                    store.add(
                        {
                            "pmid": pmid,
                            "annotations": self._annotations(document) if document else [],
                            "retrieved_at": now,
                        }
                    )
        except HttpError as exc:
            self.problems.append(f"PubTator lookups incomplete: {exc}")
        store.save()
        if not self.store_path.is_file():
            return []
        return [self.record_raw(self.store_path, EXPORT)]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        if not artifacts:
            return {}
        rows = []
        for record in LookupStore(artifacts[0].path, ("pmid",)).rows():
            for annotation in record["annotations"]:
                rows.append(
                    {
                        "pmid": record["pmid"],
                        **annotation,
                        "source": self.name,
                        "source_url": f"{EXPORT}?pmids={record['pmid']}",
                        "retrieved_at": record["retrieved_at"],
                        "source_version": self.version,
                        "license": self.info.license,
                    }
                )
        return {"annotations": pd.DataFrame(rows)} if rows else {}
