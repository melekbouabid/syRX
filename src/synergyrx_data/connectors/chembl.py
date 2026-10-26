"""ChEMBL web services: molecule lookup by InChIKey, then by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._lookup_store import LookupStore
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.pubchem import PubChemConnector, chunked
from synergyrx_data.connectors.schemas import ChemblRecord, Record
from synergyrx_data.http.errors import HttpError
from synergyrx_data.manifest import utc_now_iso
from synergyrx_data.provenance import read_parquet

BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
FIELDS = "molecule_chembl_id,pref_name,max_phase,molecule_structures"


class ChemblConnector(Connector):
    info = SourceInfo(
        name="chembl",
        title="ChEMBL web services",
        category="chemistry",
        access="api",
        homepage="https://www.ebi.ac.uk/chembl/",
        license="CC-BY-SA-3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        terms_url="https://www.ebi.ac.uk/about/terms-of-use/",
        citation=(
            "Zdrazil B, et al. The ChEMBL Database in 2023. Nucleic Acids Res. "
            "2024;52(D1):D1180-92. doi:10.1093/nar/gkad1004"
        ),
        hosts=("www.ebi.ac.uk",),
        links=(
            Link(
                "https://www.ebi.ac.uk/chembl/api/data/docs",
                "ChEMBL web services",
                "html",
                "verify",
            ),
        ),
        stage=3,
        redistribution="Share-alike: derived tables that include ChEMBL IDs must credit ChEMBL.",
        notes="Looks up the InChIKeys PubChem returned; names PubChem missed are tried by name.",
    )
    default_options: ClassVar[dict[str, Any]] = {
        "batch_size": 20,
        "name_fallback": True,
        "max_name_lookups": 500,
        "lookup_ttl_days": 30,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"molecules": ChemblRecord}

    @property
    def store_path(self) -> Path:
        return self.raw_dir / "molecule_lookups.jsonl"

    def _queries(self) -> tuple[list[str], list[str]]:
        """(InChIKeys to look up, names PubChem could not resolve)."""
        compounds = read_parquet(self.settings.interim_dir / "pubchem" / "compounds.parquet")
        if compounds is None:
            return [], []
        # Same priority as PubChem: the most-measured drugs first, so a partial or
        # `--sample` run covers the drugs that carry the most data.
        priority = PubChemConnector(self.ctx).candidate_names()
        by_name = dict(zip(compounds["query_name"].astype(str), compounds["inchikey"], strict=True))
        ranked = [by_name[n] for n in priority if isinstance(by_name.get(n), str)]
        rest = sorted(set(compounds["inchikey"].dropna().astype(str)) - set(ranked))
        keys = list(dict.fromkeys(ranked)) + ([] if self.sample else rest)
        names: list[str] = []
        if self.options["name_fallback"]:
            missed = compounds.loc[compounds["status"] == "not_found", "query_name"]
            names = sorted(set(missed.astype(str)))[: self.options["max_name_lookups"]]
        if self.sample:
            keys, names = keys[: self.sample], names[: self.sample]
        return keys, names

    def _pending(self, store: LookupStore) -> tuple[list[str], list[str]]:
        keys, names = self._queries()
        return (
            [k for k in keys if ("inchikey", k) not in store],
            [n for n in names if ("name", n) not in store],
        )

    async def discover(self) -> list[PlannedAction]:
        keys, names = self._pending(LookupStore(self.store_path, ("query_type", "query")))
        note = f"{len(keys)} InChIKeys and {len(names)} names to look up"
        return [PlannedAction(self.name, "api", BASE, None, None, note)]

    async def _get(self, **params: str) -> list[dict[str, Any]]:
        result = await self.client.fetch(
            BASE,
            source=self.name,
            params={**params, "only": FIELDS, "limit": "1000"},
            ttl=self.options["lookup_ttl_days"] * 86_400,
        )
        return list(result.json().get("molecules", []))

    @staticmethod
    def _record(query_type: str, query: str, molecule: dict[str, Any] | None) -> dict[str, Any]:
        structures = (molecule or {}).get("molecule_structures") or {}
        phase = (molecule or {}).get("max_phase")
        return {
            "query_type": query_type,
            "query": query,
            "chembl_id": (molecule or {}).get("molecule_chembl_id"),
            "pref_name": (molecule or {}).get("pref_name"),
            "max_phase": float(phase) if phase not in (None, "") else None,
            "inchikey": structures.get("standard_inchi_key"),
            "retrieved_at": utc_now_iso(),
        }

    async def fetch(self) -> list[Artifact]:
        store = LookupStore(self.store_path, ("query_type", "query"))
        keys, names = self._pending(store)
        try:
            for batch in chunked(keys, self.options["batch_size"]):
                molecules = await self._get(
                    molecule_structures__standard_inchi_key__in=",".join(batch)
                )
                by_key = {
                    (m.get("molecule_structures") or {}).get("standard_inchi_key"): m
                    for m in molecules
                }
                for key in batch:
                    store.add(self._record("inchikey", key, by_key.get(key)))
            for name in names:
                molecules = await self._get(pref_name__iexact=name)
                if not molecules:
                    molecules = await self._get(molecule_synonyms__molecule_synonym__iexact=name)
                # Several hits for one name are ambiguous: record a miss rather than guess.
                store.add(self._record("name", name, molecules[0] if len(molecules) == 1 else None))
        except HttpError as exc:
            self.problems.append(f"ChEMBL lookups incomplete: {exc}")
        store.save()
        self.stats.update(total_answers=len(store))
        if not self.store_path.is_file():
            return []
        return [self.record_raw(self.store_path, BASE)]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        if not artifacts:
            return {}
        frame = pd.DataFrame(LookupStore(artifacts[0].path, ("query_type", "query")).rows())
        if frame.empty:
            return {}
        hits = frame[frame["chembl_id"].notna()].reset_index(drop=True)
        self.stats.update(hits=len(hits), misses=len(frame) - len(hits))
        molecules = pd.DataFrame(
            {
                "query": hits["query"].astype("string"),
                "query_type": hits["query_type"].astype("string"),
                "chembl_id": hits["chembl_id"].astype("string"),
                "pref_name": hits["pref_name"].astype("string"),
                "max_phase": pd.to_numeric(hits["max_phase"], errors="coerce"),
                "inchikey": hits["inchikey"].astype("string"),
                "source": self.name,
                "source_url": "https://www.ebi.ac.uk/chembl/compound_report_card/"
                + hits["chembl_id"].astype(str)
                + "/",
                "retrieved_at": hits["retrieved_at"].astype("string"),
                "source_version": self.version,
                "license": self.info.license,
            }
        )
        return {"molecules": molecules}
