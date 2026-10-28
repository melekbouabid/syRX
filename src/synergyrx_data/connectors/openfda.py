"""openFDA drug labels (SPL) for a list of active ingredients, one label per ingredient.

For each ingredient the connector asks openFDA for the most recent labels whose generic or
substance name contains it, then keeps one (see `choose_label`): single-ingredient products
first, prescription before OTC, then the label with the most of the indexed sections, then
the most recent `effective_time`. The chosen label is stored as raw JSON with the list of
candidates it was chosen from, so the choice can be audited and repeated.

Dosage sections are never kept (`EXCLUDED_SECTIONS`): Tujya must not be able to quote a dose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import LabelSectionRecord, Record
from synergyrx_data.http.cache import atomic_write_bytes
from synergyrx_data.http.errors import HttpError, NotFoundError
from synergyrx_data.literature.queries import read_ingredients

API = "https://api.fda.gov/drug/label.json"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
PRESCRIPTION = "HUMAN PRESCRIPTION DRUG"

# Sections kept for retrieval, in label order. Old-format labels put interactions under
# `precautions`; both formats are covered.
INDEXED_SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "precautions",
    "general_precautions",
    "drug_interactions",
    "drug_and_or_laboratory_test_interactions",
    "adverse_reactions",
    "use_in_specific_populations",
    "pregnancy",
    "lactation",
    "nursing_mothers",
    "pediatric_use",
    "geriatric_use",
    "information_for_patients",
    "mechanism_of_action",
)
# Never stored: doses and strengths (the project never gives doses), and long sections that
# are mostly numbers (pharmacokinetics, trials) or packaging.
EXCLUDED_SECTIONS = (
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "overdosage",
    "how_supplied",
    "package_label_principal_display_panel",
    "clinical_pharmacology",
    "pharmacokinetics",
    "clinical_studies",
)
_SPACES = re.compile(r"\s+")
_UNSAFE = re.compile(r'["\\]')


def _openfda(label: dict[str, Any], key: str) -> list[str]:
    return [str(v) for v in label.get("openfda", {}).get(key, [])]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def matches(label: dict[str, Any], terms: list[str]) -> bool:
    """True if one of the label's generic or substance names contains one of the terms."""
    names = _openfda(label, "generic_name") + _openfda(label, "substance_name")
    return any(_words(term) <= _words(name) for term in terms for name in names)


def is_single_ingredient(label: dict[str, Any]) -> bool:
    substances = _openfda(label, "substance_name")
    return len(substances) == 1


def section_count(label: dict[str, Any]) -> int:
    return sum(1 for section in INDEXED_SECTIONS if label.get(section))


def choose_label(candidates: list[dict[str, Any]], terms: list[str]) -> dict[str, Any] | None:
    """The documented selection rule. Returns None if no candidate is about the ingredient."""
    relevant = [label for label in candidates if matches(label, terms)]
    if not relevant:
        return None

    def rank(label: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            int(is_single_ingredient(label)),
            int(PRESCRIPTION in _openfda(label, "product_type")),
            section_count(label),
            str(label.get("effective_time", "")),
        )

    return max(relevant, key=rank)


def clean_text(value: Any) -> str:
    parts = value if isinstance(value, list) else [value]
    return _SPACES.sub(" ", " ".join(str(p) for p in parts if p)).strip()


def label_sections(label: dict[str, Any]) -> list[tuple[str, str]]:
    """(section, text) for every indexed section the label has; dosage is never returned."""
    sections = []
    for section in INDEXED_SECTIONS:
        text = clean_text(label.get(section) or "")
        if text:
            sections.append((section, text))
    return sections


class OpenFdaLabelConnector(Connector):
    info = SourceInfo(
        name="openfda_label",
        title="openFDA drug labels (Structured Product Labeling)",
        category="literature",
        access="api",
        homepage="https://open.fda.gov/apis/drug/label/",
        license="CC0 1.0 (openFDA terms; labels are US public domain)",
        license_url="https://open.fda.gov/license/",
        terms_url="https://open.fda.gov/terms/",
        citation=(
            "U.S. Food and Drug Administration. openFDA drug labeling API "
            "(https://open.fda.gov/apis/drug/label/). Labels from DailyMed (NLM)."
        ),
        hosts=("api.fda.gov", "open.fda.gov"),
        links=(
            Link("https://open.fda.gov/apis/drug/label/", "openFDA drug label API", "html"),
            Link("https://open.fda.gov/terms/", "openFDA terms of service", "html"),
        ),
        stage=4,
        optional=True,
        default_enabled=False,
        redistribution="Public domain (CC0); openFDA asks not to imply FDA endorsement.",
        notes=(
            "Tujya RAG corpus. One label per ingredient; dosage sections are never stored. "
            "Without an API key openFDA allows 240 requests per minute and 1,000 per day."
        ),
    )
    default_options: ClassVar[dict[str, Any]] = {
        "ingredients_file": "config/pill_classes.csv",
        "candidates": 25,
        # Names openFDA uses that differ from the RxNorm base ingredient.
        "search_aliases": {
            "valproate": ["valproic acid", "divalproex"],
            "vitamin b12": ["cyanocobalamin"],
            "vitamin b6": ["pyridoxine"],
        },
        "max_age_days": 30,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"label_sections": LabelSectionRecord}

    def ingredients(self) -> list[str]:
        names = read_ingredients(self.settings.resolve(Path(self.options["ingredients_file"])))
        return names[: self.sample] if self.sample else names

    def terms(self, ingredient: str) -> list[str]:
        aliases = self.options["search_aliases"].get(ingredient, [])
        return [ingredient, *aliases]

    def _label_path(self, ingredient: str) -> Path:
        return self.raw_dir / "labels" / f"{re.sub(r'[^a-z0-9]+', '_', ingredient)}.json"

    async def discover(self) -> list[PlannedAction]:
        names = self.ingredients()
        if not names:
            return [PlannedAction(self.name, "skip", API, None, None, "no ingredient list")]
        return [
            PlannedAction(
                self.name,
                "api",
                API,
                self.rel(self._label_path(name)),
                None,
                f"{name}: up to {self.options['candidates']} candidates, keep one",
            )
            for name in names
        ]

    async def _search(self, terms: list[str], prescription: bool) -> list[dict[str, Any]]:
        fields = " OR ".join(
            f'openfda.{field}:"{_UNSAFE.sub(" ", term)}"'
            for term in terms
            for field in ("generic_name", "substance_name")
        )
        search = f"({fields})"
        if prescription:
            search += f' AND openfda.product_type:"{PRESCRIPTION}"'
        try:
            result = await self.client.fetch(
                API,
                source=self.name,
                params={
                    "search": search,
                    "sort": "effective_time:desc",
                    "limit": str(self.options["candidates"]),
                },
            )
        except NotFoundError:
            return []  # openFDA answers 404 when nothing matches
        results: list[dict[str, Any]] = result.json().get("results", [])
        return results

    async def fetch_one(self, ingredient: str) -> Artifact | None:
        path = self._label_path(ingredient)
        url = f"{API}?ingredient={ingredient}"
        current = self.current_artifact(path, url, max_age_days=self.options["max_age_days"])
        if current is not None:
            return current
        terms = self.terms(ingredient)
        candidates = await self._search(terms, prescription=True)
        label = choose_label(candidates, terms)
        if label is None:  # some ingredients are only sold over the counter
            candidates = await self._search(terms, prescription=False)
            label = choose_label(candidates, terms)
        if label is None:
            self.problems.append(f"no openFDA label for {ingredient}")
            return None
        record = {
            "ingredient": ingredient,
            "search_terms": terms,
            "candidates": [
                {
                    "id": c.get("id"),
                    "effective_time": c.get("effective_time"),
                    "product_type": _openfda(c, "product_type"),
                    "substance_name": _openfda(c, "substance_name"),
                    "sections": section_count(c),
                }
                for c in candidates
            ],
            "label": {k: v for k, v in label.items() if k not in EXCLUDED_SECTIONS},
        }
        atomic_write_bytes(path, json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8"))
        return self.record_raw(path, url, refreshed=True)

    async def fetch(self) -> list[Artifact]:
        names = self.ingredients()
        if not names:
            self.problems.append("no ingredients: config/pill_classes.csv has none")
        advance = self.ctx.progress.counter("openfda: labels", len(names))
        artifacts = []
        for name in names:
            try:
                artifact = await self.fetch_one(name)
            except HttpError as exc:
                self.problems.append(f"{name}: {exc}")
                break
            if artifact is not None:
                artifacts.append(artifact)
            advance(1)
        self.stats.update(ingredients=len(names), labels=len(artifacts))
        return artifacts

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        rows = []
        for artifact in artifacts:
            record = json.loads(artifact.path.read_text(encoding="utf-8"))
            label = record["label"]
            set_id = label.get("set_id")
            base = {
                "ingredient": record["ingredient"],
                "label_id": label.get("id"),
                "set_id": set_id,
                "label_version": str(label.get("version") or ""),
                "effective_time": str(label.get("effective_time") or ""),
                "product_type": "; ".join(_openfda(label, "product_type")),
                "brand_name": "; ".join(_openfda(label, "brand_name")),
                "generic_name": "; ".join(_openfda(label, "generic_name")),
                "manufacturer": "; ".join(_openfda(label, "manufacturer_name")),
                "substances": "; ".join(_openfda(label, "substance_name")),
                "label_url": f"{DAILYMED}{set_id}" if set_id else None,
            }
            provenance = self.provenance(artifact)
            for section, text in label_sections(label):
                rows.append(
                    {
                        **base,
                        "section": section,
                        "text": text,
                        "source": provenance.source,
                        "source_url": f"{API}?search=id:{label.get('id')}",
                        "retrieved_at": provenance.retrieved_at,
                        "source_version": base["effective_time"] or provenance.source_version,
                        "license": provenance.license,
                    }
                )
        frame = pd.DataFrame(rows)
        self.stats.update(sections=len(frame))
        return {"label_sections": frame} if rows else {}
