"""PubMed through NCBI E-utilities: ESearch on the history server, EFetch XML in batches.

Each query keeps the last date it was searched up to. The next run only asks for records
entered since then (minus a small overlap), and articles are deduplicated by PMID, so
scheduled runs are incremental.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import ArticleRecord, Record
from synergyrx_data.http.cache import atomic_write_bytes
from synergyrx_data.http.errors import HttpError
from synergyrx_data.literature.abstracts import articles_frame, deduplicate, parse_pubmed_xml
from synergyrx_data.literature.queries import (
    Query,
    append_query_log,
    build_ingredient_queries,
    build_queries,
    load_query_inputs,
    read_ingredients,
)
from synergyrx_data.manifest import utc_now_iso

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATE_FORMAT = "%Y/%m/%d"
# The history server serves at most 10,000 records per PubMed search.
HISTORY_LIMIT = 10_000


class PubMedConnector(Connector):
    info = SourceInfo(
        name="pubmed",
        title="PubMed (NCBI E-utilities)",
        category="literature",
        access="api",
        homepage="https://pubmed.ncbi.nlm.nih.gov/",
        license=(
            "NLM data terms; abstracts may be under publisher copyright (research use, "
            "do not redistribute abstracts)"
        ),
        terms_url="https://www.nlm.nih.gov/databases/download/terms_and_conditions.html",
        citation=(
            "Sayers E. A General Introduction to the E-utilities. In: Entrez Programming "
            "Utilities Help. Bethesda (MD): NCBI; 2010-. NBK25497"
        ),
        hosts=(
            "eutils.ncbi.nlm.nih.gov",
            "www.ncbi.nlm.nih.gov",
            "dataguide.nlm.nih.gov",
            "dtd.nlm.nih.gov",
            "www.nlm.nih.gov",
        ),
        links=(
            Link(
                "https://www.ncbi.nlm.nih.gov/books/NBK25501/",
                "NCBI E-utilities documentation",
                "html",
                "checked",
            ),
            Link(
                "https://dataguide.nlm.nih.gov/eutilities/how_eutilities_works.html",
                "E-utilities principles",
                "html",
                "checked",
            ),
            Link(
                "https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/190101/index.html",
                "Downloading PubMed data (bulk)",
                "html",
                "checked",
            ),
        ),
        stage=4,
        notes="Queries are built from the processed drug table, so run `clean` first.",
    )
    default_options: ClassVar[dict[str, Any]] = {
        "mode": "pairs",  # pairs | drugs | ingredients (Tujya: interactions, adverse effects)
        "ingredients_file": "config/pill_classes.csv",  # mode `ingredients` only
        "ingredient_aliases": {},  # extra search terms per ingredient, mode `ingredients`
        "sort": None,  # ESearch sort order, e.g. "relevance"; None keeps PubMed's default
        "max_queries": 25,
        "max_records_per_query": 200,
        "batch_size": 200,
        "start_date": "2000/01/01",
        "date_type": "edat",
        "overlap_days": 2,
        "max_terms_per_drug": 6,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"articles": ArticleRecord}

    # ------------------------------------------------------------- planning

    @property
    def state_path(self) -> Path:
        """Where each query's last searched date is kept.

        Sample runs keep their own file. A sample only fetches a few records per query, so
        if it advanced the real state, the next full run would start from today and never
        fetch the older records the sample skipped.
        """
        name = "pubmed_queries.sample.json" if self.sample else "pubmed_queries.json"
        return self.settings.state_dir / name

    @property
    def query_log_path(self) -> Path:
        return self.interim_dir / "query_log.jsonl"

    def queries(self) -> list[Query]:
        limit = min(self.options["max_queries"], 3) if self.sample else self.options["max_queries"]
        if self.options["mode"] == "ingredients":
            names = read_ingredients(self.settings.resolve(Path(self.options["ingredients_file"])))
            return build_ingredient_queries(
                names,
                self.options["ingredient_aliases"],
                max_queries=limit,
                max_terms=self.options["max_terms_per_drug"],
            )
        drugs, pairs = load_query_inputs(self.settings)
        return build_queries(
            drugs,
            pairs,
            mode=self.options["mode"],
            max_queries=limit,
            max_terms=self.options["max_terms_per_drug"],
        )

    def _records_cap(self) -> int:
        cap = int(self.options["max_records_per_query"])
        return min(cap, self.sample, HISTORY_LIMIT) if self.sample else min(cap, HISTORY_LIMIT)

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if self.state_path.is_file():
            state: dict[str, dict[str, Any]] = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            return state
        return {}

    def _window(self, previous: dict[str, Any] | None, today: date) -> tuple[str, str] | None:
        """Date range to search, or None if the query was already run through today."""
        maxdate = today.strftime(DATE_FORMAT)
        if previous is None or self.settings.refresh:
            return self.options["start_date"], maxdate
        if previous["last_maxdate"] == maxdate:
            return None
        last = datetime.strptime(previous["last_maxdate"], DATE_FORMAT).date()
        since = last - timedelta(days=int(self.options["overlap_days"]))
        return since.strftime(DATE_FORMAT), maxdate

    async def discover(self) -> list[PlannedAction]:
        queries = self.queries()
        if not queries:
            return [
                PlannedAction(
                    self.name,
                    "skip",
                    f"{EUTILS}/esearch.fcgi",
                    None,
                    None,
                    "no queries yet: run `fetch` for the synergy sources and `clean` first",
                )
            ]
        state = self._load_state()
        today = datetime.now(UTC).date()
        plans = []
        for query in queries:
            window = self._window(state.get(query.query_id), today)
            note = (
                "already searched through today"
                if window is None
                else f"{' + '.join(query.drug_names)}: entered {window[0]} to {window[1]}, "
                f"up to {self._records_cap()} records"
            )
            plans.append(
                PlannedAction(
                    self.name,
                    "skip" if window is None else "api",
                    f"{EUTILS}/esearch.fcgi + efetch.fcgi",
                    self.rel(self.raw_dir / query.query_id),
                    None,
                    note,
                )
            )
        return plans

    # -------------------------------------------------------------- network

    def _identity(self) -> dict[str, str]:
        """`tool`, `email` and `api_key`, as NCBI asks every E-utilities client to send."""
        secrets = self.settings.secrets
        params = {"tool": secrets.ncbi_tool}
        if secrets.contact_email:
            params["email"] = secrets.contact_email
        if secrets.api_key:
            params["api_key"] = secrets.api_key
        return params

    async def _esearch(self, query: Query, mindate: str, maxdate: str) -> dict[str, Any]:
        result = await self.client.fetch(
            f"{EUTILS}/esearch.fcgi",
            source=self.name,
            method="POST",
            data={
                "db": "pubmed",
                "term": query.pubmed,
                "usehistory": "y",
                "retmax": "0",
                "retmode": "json",
                "datetype": self.options["date_type"],
                "mindate": mindate,
                "maxdate": maxdate,
                **({"sort": self.options["sort"]} if self.options["sort"] else {}),
                **self._identity(),
            },
            use_cache=False,  # a cached WebEnv would have expired on the history server
        )
        payload = result.json().get("esearchresult", {})
        if "ERROR" in payload:
            raise HttpError(f"ESearch error: {payload['ERROR']}")
        return dict(payload)

    async def _efetch(self, webenv: str, query_key: str, retstart: int, retmax: int) -> bytes:
        result = await self.client.fetch(
            f"{EUTILS}/efetch.fcgi",
            source=self.name,
            params={
                "db": "pubmed",
                "query_key": query_key,
                "WebEnv": webenv,
                "retstart": str(retstart),
                "retmax": str(retmax),
                "retmode": "xml",
                **self._identity(),
            },
            expect=("xml",),
            use_cache=False,
        )
        return result.content

    async def _run_query(self, query: Query, mindate: str, maxdate: str) -> dict[str, Any]:
        search = await self._esearch(query, mindate, maxdate)
        count = int(search.get("count", 0))
        wanted = min(count, self._records_cap())
        folder = self.raw_dir / query.query_id
        atomic_write_bytes(
            folder / "query.json", json.dumps(query.to_dict(), indent=2).encode("utf-8")
        )
        window = f"{mindate}_{maxdate}".replace("/", "")
        batch = int(self.options["batch_size"])
        for retstart in range(0, wanted, batch):
            xml = await self._efetch(
                search["webenv"], search["querykey"], retstart, min(batch, wanted - retstart)
            )
            path = folder / window / f"batch_{retstart:06d}.xml"
            atomic_write_bytes(path, xml)
            self._fetched_now.add(path)
            self.record_raw(path, f"{EUTILS}/efetch.fcgi?db=pubmed&query_id={query.query_id}")
        record = {
            "logged_at": utc_now_iso(),
            "query_id": query.query_id,
            "kind": query.kind,
            "drug_names": list(query.drug_names),
            "pair_key": query.pair_key,
            "query": query.pubmed,
            "date_type": self.options["date_type"],
            "mindate": mindate,
            "maxdate": maxdate,
            "count": count,
            "fetched": wanted,
            "truncated": wanted < count,
            "query_translation": search.get("querytranslation"),
        }
        append_query_log(self.query_log_path, record)
        return record

    async def fetch(self) -> list[Artifact]:
        queries = self.queries()
        self._fetched_now: set[Path] = set()
        if not queries:
            self.problems.append("no queries: the processed drug and pair tables are empty")
        state = self._load_state()
        today = datetime.now(UTC).date()
        advance = self.ctx.progress.counter("pubmed: queries", len(queries))
        truncated = 0
        for query in queries:
            window = self._window(state.get(query.query_id), today)
            if window is not None:
                try:
                    record = await self._run_query(query, *window)
                except HttpError as exc:
                    self.problems.append(f"query {query.query_id} failed: {exc}")
                    break
                if record["truncated"]:
                    truncated += 1
                state[query.query_id] = {
                    "last_maxdate": record["maxdate"],
                    "last_count": record["count"],
                    "last_run_at": record["logged_at"],
                }
                atomic_write_bytes(
                    self.state_path, json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
                )
            advance(1)
        # Truncated = PubMed has more records for the window than `max_records_per_query`.
        self.stats.update(queries=len(queries), queries_truncated=truncated)
        return self._all_artifacts()

    # -------------------------------------------------------------- parsing

    def _all_artifacts(self) -> list[Artifact]:
        """Every XML batch ever fetched: parsing always rebuilds from the whole raw layer."""
        artifacts = []
        for entry in self.ctx.manifest.for_source(self.name, "raw"):
            path = self.settings.resolve(entry.path)
            if path.suffix == ".xml" and path.is_file():
                artifacts.append(
                    Artifact(
                        source=self.name,
                        path=path,
                        url=entry.url or EUTILS,
                        version=entry.version or self.version,
                        sha256=entry.sha256,
                        size=entry.size_bytes,
                        retrieved_at=entry.retrieved_at,
                        license=self.info.license,
                        terms_url=self.info.terms_url,
                        status="downloaded" if path in self._fetched_now else "unchanged",
                    )
                )
        return sorted(artifacts, key=lambda a: a.path.as_posix())

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        records: list[dict[str, Any]] = []
        hits: list[dict[str, Any]] = []
        for artifact in artifacts:
            query_id = artifact.path.parent.parent.name
            for record in parse_pubmed_xml(artifact.path.read_bytes()):
                url = f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/"
                provenance = {
                    "source": self.name,
                    "source_url": url,
                    "retrieved_at": artifact.retrieved_at,
                    "source_version": artifact.version,
                    "license": artifact.license,
                }
                records.append({**record, **provenance})
                hits.append({"query_id": query_id, "pmid": record["pmid"], **provenance})
        articles = deduplicate(articles_frame(records))
        self.stats.update(records_parsed=len(records), distinct_pmids=len(articles))
        tables = {"articles": articles}
        if hits:
            query_hits = pd.DataFrame(hits).drop_duplicates(["query_id", "pmid"], keep="last")
            definitions = self._query_definitions()
            query_hits["kind"] = query_hits["query_id"].map(
                lambda q: definitions.get(q, {}).get("kind")
            )
            query_hits["pair_key"] = query_hits["query_id"].map(
                lambda q: definitions.get(q, {}).get("pair_key")
            )
            tables["query_hits"] = query_hits.reset_index(drop=True)
        return tables

    def _query_definitions(self) -> dict[str, dict[str, Any]]:
        definitions = {}
        for path in self.raw_dir.glob("*/query.json"):
            definitions[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        return definitions
