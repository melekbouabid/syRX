"""PubChem PUG-REST: drug name -> CID -> SMILES, InChIKey, synonyms."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._lookup_store import LookupStore
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import PubChemRecord, Record
from synergyrx_data.http.errors import CircuitOpenError, HttpError, NotFoundError
from synergyrx_data.manifest import utc_now_iso
from synergyrx_data.provenance import read_parquet

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
# PubChem renamed CanonicalSMILES / IsomericSMILES to ConnectivitySMILES / SMILES in 2025.
PROPERTIES = "SMILES,ConnectivitySMILES,InChIKey,Title"


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class PubChemConnector(Connector):
    info = SourceInfo(
        name="pubchem",
        title="PubChem PUG-REST",
        category="chemistry",
        access="api",
        homepage="https://pubchem.ncbi.nlm.nih.gov/",
        license="Public domain (NCBI); depositor-supplied records may carry their own terms",
        terms_url="https://www.ncbi.nlm.nih.gov/home/about/policies/",
        citation=(
            "Kim S, et al. PubChem 2025 update. Nucleic Acids Res. 2025;53(D1):D1516-25. "
            "doi:10.1093/nar/gkae1059"
        ),
        hosts=("pubchem.ncbi.nlm.nih.gov",),
        links=(
            Link(
                "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
                "PubChem PUG-REST",
                "html",
                "verify",
            ),
        ),
        stage=2,
        notes="Resolves the drug names found in the synergy tables. Runs after stage 1.",
    )
    default_options: ClassVar[dict[str, Any]] = {
        "property_batch": 100,
        "synonym_batch": 25,
        "max_synonyms": 50,
        "lookup_ttl_days": 30,
        "concurrency": 8,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"compounds": PubChemRecord}

    @property
    def store_path(self) -> Path:
        return self.raw_dir / "name_lookups.jsonl"

    def candidate_names(self) -> list[str]:
        """Distinct drug names in the interim synergy tables, most-measured first.

        The order matters for partial runs: with `--sample N`, or if a run is interrupted,
        the drugs that carry the most data are the ones that get resolved.
        """
        counts: dict[str, int] = {}
        for table in sorted(self.settings.interim_dir.glob("*/synergy.parquet")):
            frame = read_parquet(table, ["drug_a_name", "drug_b_name"])
            if frame is None:
                continue
            stacked = pd.concat([frame["drug_a_name"], frame["drug_b_name"]]).dropna()
            for name, n in stacked.astype(str).str.strip().value_counts().items():
                if name:
                    counts[str(name)] = counts.get(str(name), 0) + int(n)
        ordered = sorted(counts, key=lambda name: (-counts[name], name))
        return ordered[: self.sample] if self.sample else ordered

    def pending_names(self, store: LookupStore) -> list[str]:
        return [name for name in self.candidate_names() if name not in store]

    async def discover(self) -> list[PlannedAction]:
        store = LookupStore(self.store_path, ("query_name",))
        pending = self.pending_names(store)
        note = f"{len(pending)} drug names to resolve, {len(store)} already answered"
        return [PlannedAction(self.name, "api", f"{BASE}/name/cids/JSON", None, None, note)]

    async def _cids(self, name: str) -> list[int]:
        try:
            result = await self.client.fetch(
                f"{BASE}/name/cids/JSON",
                source=self.name,
                method="POST",
                data={"name": name},
                ttl=self.options["lookup_ttl_days"] * 86_400,
            )
        except NotFoundError:
            return []
        return [int(c) for c in result.json().get("IdentifierList", {}).get("CID", [])]

    async def _properties(self, cids: list[int]) -> dict[int, dict[str, Any]]:
        ids = ",".join(str(c) for c in cids)
        result = await self.client.fetch(
            f"{BASE}/cid/{ids}/property/{PROPERTIES}/JSON",
            source=self.name,
            ttl=self.options["lookup_ttl_days"] * 86_400,
        )
        rows = result.json().get("PropertyTable", {}).get("Properties", [])
        return {int(row["CID"]): row for row in rows}

    async def _synonyms(self, cids: list[int]) -> dict[int, list[str]]:
        ids = ",".join(str(c) for c in cids)
        try:
            result = await self.client.fetch(
                f"{BASE}/cid/{ids}/synonyms/JSON",
                source=self.name,
                ttl=self.options["lookup_ttl_days"] * 86_400,
            )
        except NotFoundError:
            return {}
        rows = result.json().get("InformationList", {}).get("Information", [])
        limit = self.options["max_synonyms"]
        return {int(row["CID"]): list(row.get("Synonym", []))[:limit] for row in rows}

    async def fetch(self) -> list[Artifact]:
        store = LookupStore(self.store_path, ("query_name",))
        pending = self.pending_names(store)
        advance = self.ctx.progress.counter("pubchem: names", len(pending))
        found: dict[str, list[int]] = {}
        failed = 0
        stop = False

        async def lookup(name: str) -> None:
            nonlocal failed, stop
            if stop:
                return
            try:
                found[name] = await self._cids(name)
            except CircuitOpenError:
                stop = True
            except HttpError as exc:
                failed += 1
                self.log.warning("pubchem.lookup_failed", extra={"query": name, "error": str(exc)})
            advance(1)

        for batch in chunked(pending, self.options["concurrency"]):
            await asyncio.gather(*(lookup(name) for name in batch))
            if stop:
                break

        cids = sorted({ids[0] for ids in found.values() if ids})
        properties: dict[int, dict[str, Any]] = {}
        synonyms: dict[int, list[str]] = {}
        try:
            for batch in chunked(cids, self.options["property_batch"]):
                properties.update(await self._properties(batch))
            for batch in chunked(cids, self.options["synonym_batch"]):
                synonyms.update(await self._synonyms(batch))
        except HttpError as exc:
            self.problems.append(f"property / synonym lookups incomplete: {exc}")

        now = utc_now_iso()
        for name, ids in found.items():
            cid = ids[0] if ids else None
            if cid is not None and cid not in properties:
                continue  # no structure yet: leave the name pending for the next run
            props = properties.get(cid, {}) if cid is not None else {}
            store.add(
                {
                    "query_name": name,
                    "status": "not_found" if not ids else "found" if len(ids) == 1 else "ambiguous",
                    "n_cids": len(ids),
                    "candidate_cids": ids[:20],
                    "pubchem_cid": cid,
                    "smiles": props.get("SMILES") or props.get("IsomericSMILES"),
                    "connectivity_smiles": props.get("ConnectivitySMILES")
                    or props.get("CanonicalSMILES"),
                    "inchikey": props.get("InChIKey"),
                    "title": props.get("Title"),
                    "synonyms": synonyms.get(cid, []) if cid is not None else [],
                    "retrieved_at": now,
                }
            )
        store.save()
        if failed or stop:
            self.problems.append(
                f"{failed} name lookups failed"
                + ("; stopped early because the circuit breaker opened" if stop else "")
            )
        self.stats.update(resolved_this_run=len(found), total_answers=len(store))
        if not self.store_path.is_file():
            return []
        return [self.record_raw(self.store_path, f"{BASE}/name/cids/JSON")]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        if not artifacts:
            return {}
        rows = LookupStore(artifacts[0].path, ("query_name",)).rows()
        frame = pd.DataFrame(rows)
        if frame.empty:
            return {}
        cid = frame["pubchem_cid"].astype("Int64")
        compounds = pd.DataFrame(
            {
                "query_name": frame["query_name"].astype("string"),
                "status": frame["status"].astype("string"),
                "n_cids": frame["n_cids"].astype("int64"),
                "pubchem_cid": cid,
                "smiles": frame["smiles"].astype("string"),
                "connectivity_smiles": frame["connectivity_smiles"].astype("string"),
                "inchikey": frame["inchikey"].astype("string"),
                "title": frame["title"].astype("string"),
                "synonyms": frame["synonyms"],
                "source": self.name,
                "source_url": [
                    f"https://pubchem.ncbi.nlm.nih.gov/compound/{c}"
                    if pd.notna(c)
                    else f"{BASE}/name/cids/JSON"
                    for c in cid
                ],
                "retrieved_at": frame["retrieved_at"].astype("string"),
                "source_version": self.version,
                "license": self.info.license,
            }
        )
        return {"compounds": compounds}
