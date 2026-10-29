"""Offline build steps: clean, literature, report and export.

Everything here reads `data/interim` and writes the generated files, grouped by format:
`data/parquet` (processed tables), `data/csv`, `data/json` and `data/reports` (documents).
None of it touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from synergyrx_data import __version__
from synergyrx_data.cleaning.cell_lines import (
    build_cell_line_table,
    collect_cell_line_names,
    load_aliases,
)
from synergyrx_data.cleaning.drugs import build_drug_table, collect_mentions, load_overrides
from synergyrx_data.cleaning.pairs import build_synergy_pairs
from synergyrx_data.cleaning.scores import find_conflicts, source_agreement
from synergyrx_data.config import Settings
from synergyrx_data.connectors import REGISTRY, SYNERGY_SOURCES, is_enabled, source_names
from synergyrx_data.export.frontend import export_frontend
from synergyrx_data.literature.abstracts import (
    ARTICLE_COLUMNS,
    articles_frame,
    build_abstract_store,
)
from synergyrx_data.literature.annotation_export import export_annotation_sample, stratified_sample
from synergyrx_data.literature.mentions import SynonymIndex, find_candidate_sentences
from synergyrx_data.manifest import Manifest, utc_now_iso
from synergyrx_data.provenance import (
    PROVENANCE_COLUMNS,
    combine_provenance,
    read_parquet,
    write_parquet,
)
from synergyrx_data.quality.drift import DriftFinding, DriftTracker
from synergyrx_data.quality.licenses import write_licenses
from synergyrx_data.quality.reports import (
    build_quality_report,
    coverage_table,
    drug_depth,
    drug_overlap,
    frame_table,
    missingness_table,
)
from synergyrx_data.state import StateStore


class BuildError(RuntimeError):
    """A build step cannot run because its inputs are missing."""


@dataclass
class BuildResult:
    outputs: dict[str, int] = field(default_factory=dict)  # project-relative path -> rows
    stats: dict[str, Any] = field(default_factory=dict)
    drift: list[DriftFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _Writer:
    """Writes outputs, records them in the manifest and checks them for schema drift."""

    def __init__(self, settings: Settings, source: str) -> None:
        self.settings = settings
        self.source = source
        self.manifest = Manifest(settings)
        self.tracker = DriftTracker(settings.state_dir / "schemas")
        self.result = BuildResult()
        self.is_sample = bool(sample_sources(settings, self.manifest))

    def parquet(self, frame: pd.DataFrame, path: Path, *, derived_from: list[str]) -> None:
        write_parquet(frame, path)
        self._record(path, "processed", len(frame), derived_from)
        tag = f"synergyrx-data {__version__}"
        self.result.drift += self.tracker.check(
            f"processed/{path.stem}", frame, tag, is_sample=self.is_sample
        )

    def csv(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        self._record(path, "report", len(frame), [])

    def file(self, path: Path, layer: str, rows: int | None = None) -> None:
        self._record(path, layer, rows, [])

    def _record(self, path: Path, layer: str, rows: int | None, derived_from: list[str]) -> None:
        self.manifest.record_file(
            path,
            source=self.source,
            layer=layer,  # type: ignore[arg-type]
            is_sample=self.is_sample,
            rows=rows,
            derived_from=derived_from,
            version=f"synergyrx-data {__version__}",
        )
        self.result.outputs[self.settings.relative(path)] = rows if rows is not None else -1

    def finish(self) -> BuildResult:
        self.manifest.save()
        return self.result


def sample_sources(settings: Settings, manifest: Manifest | None = None) -> list[str]:
    """Sources whose interim tables come from a `--sample` run."""
    manifest = manifest or Manifest(settings)
    return sorted(
        {e.source for e in manifest.entries.values() if e.layer == "interim" and e.is_sample}
    )


def load_synergy_frames(settings: Settings) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in SYNERGY_SOURCES:
        frame = read_parquet(settings.interim_dir / name / "synergy.parquet")
        if frame is not None and len(frame):
            frames[name] = frame
    return frames


def _interim(settings: Settings, source: str, table: str) -> pd.DataFrame | None:
    return read_parquet(settings.interim_dir / source / f"{table}.parquet")


def _disagreements(conflicts: pd.DataFrame) -> pd.DataFrame:
    """Rows of the duplicate table whose records really disagree.

    Most duplicates are exact copies (TDC's set is a copy of part of DrugComb): hundreds of
    thousands of rows with a range of zero. They are counted in the quality report and
    flagged in `synergy_pairs` (`n_records > 1`); the CSV keeps the ones worth reading.
    """
    return conflicts.loc[conflicts["is_conflict"].astype(bool)].reset_index(drop=True)


# -------------------------------------------------------------------- clean


def clean(settings: Settings) -> BuildResult:
    """Build drugs, cell lines, synergy pairs and expression from the interim tables."""
    frames = load_synergy_frames(settings)
    if not frames:
        raise BuildError("no synergy tables in data/interim: run `synergyrx fetch` first")
    writer = _Writer(settings, "synergyrx-data:clean")
    options = settings.cleaning
    inputs = [settings.relative(settings.interim_dir / n / "synergy.parquet") for n in frames]

    resolution = build_drug_table(
        collect_mentions(frames),
        pubchem=_interim(settings, "pubchem", "compounds"),
        chembl=_interim(settings, "chembl", "molecules"),
        overrides=load_overrides(settings.resolve(options.manual_overrides)),
        review_below=options.review_confidence_below,
        max_synonyms=options.max_synonyms,
    )
    depmap_models = _interim(settings, "depmap", "models")
    if depmap_models is None:
        writer.result.warnings.append(
            "DepMap model table not found: cell lines are left unmapped (fetch `depmap`)"
        )
    cells = build_cell_line_table(
        collect_cell_line_names(frames),
        depmap_models,
        load_aliases(settings.resolve(options.cell_line_aliases)),
    )
    provenance = combine_provenance(
        [f[PROVENANCE_COLUMNS].drop_duplicates() for f in frames.values()]
        + (
            [depmap_models[PROVENANCE_COLUMNS].drop_duplicates()]
            if depmap_models is not None
            else []
        )
    )
    cell_lines = cells.cell_lines.assign(
        source=cells.cell_lines["sources"].map("; ".join)
        + cells.cell_lines["mapped"].map({True: "; depmap", False: ""}),
        **{c: provenance[c] for c in PROVENANCE_COLUMNS if c != "source"},
    )
    pairs, pair_stats = build_synergy_pairs(
        frames,
        resolution.name_map,
        cells.name_map,
        resolution.drugs,
        cell_lines,
        outlier_abs=options.score_outlier_abs,
    )
    conflicts = find_conflicts(pairs, options.conflict_abs_tolerance)
    flagged = (
        set(
            zip(
                conflicts.loc[conflicts["is_conflict"], "pair_key"],
                conflicts.loc[conflicts["is_conflict"], "cell_line_id"],
                strict=True,
            )
        )
        if len(conflicts)
        else set()
    )
    pairs["has_conflict"] = [
        key in flagged for key in zip(pairs["pair_key"], pairs["cell_line_id"], strict=True)
    ]

    processed = settings.parquet_root
    writer.parquet(resolution.drugs, processed / "drugs.parquet", derived_from=inputs)
    writer.parquet(cell_lines, processed / "cell_lines.parquet", derived_from=inputs)
    writer.parquet(pairs, processed / "synergy_pairs.parquet", derived_from=inputs)
    write_parquet(resolution.name_map, settings.interim_dir / "_maps" / "drug_names.parquet")
    write_parquet(cells.name_map, settings.interim_dir / "_maps" / "cell_line_names.parquet")

    expression = _interim(settings, "depmap", "expression")
    if expression is not None:
        in_use = set(pairs["depmap_id"].dropna())
        subset = expression[expression["depmap_id"].isin(in_use)].reset_index(drop=True)
        top = options.expression_top_variable_genes
        genes = [c for c in subset.columns if c not in ("depmap_id", *PROVENANCE_COLUMNS)]
        if top and len(genes) > top and len(subset) > 1:
            keep = subset[genes].var().nlargest(top).index.tolist()
            subset = subset[["depmap_id", *keep, *PROVENANCE_COLUMNS]]
        writer.parquet(
            subset,
            processed / "expression.parquet",
            derived_from=[
                settings.relative(settings.interim_dir / "depmap" / "expression.parquet")
            ],
        )
        pair_stats["cell_lines_with_expression"] = len(subset)
        pair_stats["cell_lines_without_expression"] = len(in_use) - len(subset)
    else:
        writer.result.warnings.append("no expression matrix in data/interim/depmap")

    csv_folder = settings.csv_root
    writer.csv(resolution.review_queue, csv_folder / "drug_review_queue.csv")
    writer.csv(cells.review_queue, csv_folder / "cell_line_review_queue.csv")
    writer.csv(_disagreements(conflicts), csv_folder / "conflicts.csv")
    drugs = resolution.drugs
    writer.result.stats = {
        **pair_stats,
        "distinct_drugs": len(drugs),
        "drugs_invalid_smiles": int((~drugs["smiles_valid"]).sum()),
        "drugs_needing_review": int(drugs["needs_review"].sum()),
        "cell_lines": len(cell_lines),
        "cell_lines_unmapped": int((~cell_lines["mapped"]).sum()),
        "multi_record_triplets": len(conflicts),
        "conflicts": int(conflicts["is_conflict"].sum()) if len(conflicts) else 0,
    }
    state = StateStore(settings.state_dir / "connectors.json")
    state.update("_clean", at=utc_now_iso(), stats=writer.result.stats)
    if writer.is_sample:
        writer.result.warnings.append(
            "built from SAMPLE data ("
            + ", ".join(sample_sources(settings))
            + "): not suitable for training or reporting"
        )
    return writer.finish()


# --------------------------------------------------------------- literature


def _collect_articles(settings: Settings) -> pd.DataFrame:
    """PubMed articles, plus the baseline and Europe PMC search results when present."""
    parts = []
    for source in ("pubmed", "pubmed_baseline"):
        frame = _interim(settings, source, "articles")
        if frame is not None and len(frame):
            parts.append(frame)
    known = set().union(*[set(p["pmid"]) for p in parts]) if parts else set()
    epmc = _interim(settings, "europepmc", "articles")
    if epmc is not None and len(epmc):
        # Articles Europe PMC found that PubMed did not return (fallback search).
        extra = epmc[~epmc["pmid"].isin(known)].copy()
        if len(extra):
            extra["has_abstract"] = extra["abstract"].notna()
            extra["article_type"] = "research"
            for column in ARTICLE_COLUMNS:
                if column not in extra.columns:
                    extra[column] = None
            records = extra[[*ARTICLE_COLUMNS, *PROVENANCE_COLUMNS]].to_dict("records")
            for record in records:
                for column in ("publication_types", "mesh_terms"):
                    value = record[column]
                    record[column] = list(value) if value is not None else []
            parts.append(articles_frame(records))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def literature(settings: Settings) -> BuildResult:
    """Abstract store, candidate sentences and the annotation sample."""
    articles = _collect_articles(settings)
    if articles.empty:
        raise BuildError("no articles in data/interim: run `synergyrx fetch --source pubmed` first")
    drugs = read_parquet(settings.parquet_root / "drugs.parquet")
    if drugs is None:
        raise BuildError("the drugs table was not found: run `synergyrx clean` first")
    writer = _Writer(settings, "synergyrx-data:literature")
    options = settings.literature
    abstracts = build_abstract_store(articles, _interim(settings, "europepmc", "articles"))
    index = SynonymIndex.from_drugs(drugs, options.min_synonym_length, options.max_ngram)
    candidates, stats = find_candidate_sentences(
        abstracts, index, min_drugs=options.min_drugs_per_sentence, max_ngram=options.max_ngram
    )
    inputs = [settings.relative(settings.interim_dir / "pubmed" / "articles.parquet")]
    writer.parquet(abstracts, settings.parquet_root / "abstracts.parquet", derived_from=inputs)
    writer.parquet(
        candidates, settings.parquet_root / "candidate_sentences.parquet", derived_from=inputs
    )
    sample = stratified_sample(
        candidates,
        size=options.annotation_sample_size,
        max_per_pmid=options.annotation_max_per_pmid,
        seed=options.annotation_seed,
    )
    paths = export_annotation_sample(
        sample,
        csv_dir=settings.csv_root,
        json_dir=settings.json_root,
        guide_dir=settings.reports_root,
    )
    writer.file(paths["csv"], "export", len(sample))
    writer.file(paths["jsonl"], "export", len(sample))
    writer.file(paths["guide"], "report")
    if len(sample) < 200:
        writer.result.warnings.append(
            f"only {len(sample)} candidate sentences available for the annotation sample "
            "(target 200-500): fetch more PubMed queries"
        )
    writer.result.stats = {
        **stats,
        "articles": len(abstracts),
        "articles_with_abstract": int(abstracts["has_abstract"].sum()),
        "retracted": int(abstracts["is_retracted"].sum()),
        "non_english": int((abstracts["language"].fillna("eng") != "eng").sum()),
        "synonym_terms": len(index.terms),
        "ambiguous_terms_dropped": index.dropped_ambiguous,
        "annotation_sample": len(sample),
    }
    StateStore(settings.state_dir / "connectors.json").update(
        "_literature", at=utc_now_iso(), stats=writer.result.stats
    )
    return writer.finish()


# ------------------------------------------------------------------ export


def _processed(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = [
        read_parquet(settings.parquet_root / f"{name}.parquet")
        for name in ("drugs", "cell_lines", "synergy_pairs")
    ]
    if any(table is None for table in tables):
        raise BuildError("processed tables not found: run `synergyrx clean` first")
    drugs, cell_lines, pairs = tables
    assert drugs is not None and cell_lines is not None and pairs is not None
    return drugs, cell_lines, pairs


def export(settings: Settings) -> BuildResult:
    """Write the frontend JSON files."""
    drugs, cell_lines, pairs = _processed(settings)
    writer = _Writer(settings, "synergyrx-data:export")
    abstracts = read_parquet(settings.parquet_root / "abstracts.parquet")
    for path in export_frontend(settings, drugs, cell_lines, pairs, abstracts).values():
        writer.file(path, "export")
    return writer.finish()


# ------------------------------------------------------------------ report


def source_states(settings: Settings) -> pd.DataFrame:
    state = StateStore(settings.state_dir / "connectors.json", read_only=True)
    rows = []
    for name in source_names():
        entry = state.get(name)
        current = entry.get("state", "never_run")
        if current == "never_run" and not is_enabled(name, settings):
            current = "disabled"
        rows.append(
            {
                "source": name,
                "enabled": is_enabled(name, settings),
                "state": current + (" (sample)" if entry.get("is_sample") else ""),
                "last_run": (entry.get("last_run_at") or "-")[:16],
                "rows": sum((entry.get("tables") or {}).values()) or None,
                "rejected_rows": sum((entry.get("rejects") or {}).values()) or None,
                "access": REGISTRY[name].info.access,
                "message": (entry.get("message") or "")[:400],
            }
        )
    return pd.DataFrame(rows)


def load_link_status(settings: Settings) -> tuple[pd.DataFrame, str]:
    path = settings.state_dir / "link_status.json"
    if not path.is_file():
        return pd.DataFrame(), "never"
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["links"])
    if len(frame):
        frame = frame[["source", "label", "status", "note", "url"]]
    return frame, payload.get("checked_at", "unknown")


def _literature_context(settings: Settings) -> dict[str, Any] | None:
    abstracts = read_parquet(settings.parquet_root / "abstracts.parquet")
    if abstracts is None:
        return None
    candidates = read_parquet(settings.parquet_root / "candidate_sentences.parquet")
    types = (
        abstracts["article_type"]
        .value_counts()
        .rename_axis("article_type")
        .reset_index(name="articles")
    )
    languages = abstracts["language"].fillna("unknown").value_counts().head(8)
    n_candidates = 0 if candidates is None else len(candidates)
    facts = {
        "Articles (distinct PMIDs)": f"{len(abstracts):,}",
        "With an abstract": f"{int(abstracts['has_abstract'].sum()):,}",
        "Retracted": f"{int(abstracts['is_retracted'].sum()):,}",
        "Languages": ", ".join(f"{k} {v}" for k, v in languages.items()),
        "Open access (Europe PMC)": f"{int(abstracts['is_open_access'].fillna(False).sum()):,}",
        "Candidate sentences (two or more drugs)": f"{n_candidates:,}",
    }
    if candidates is not None and len(candidates):
        facts["Candidate sentences with a synergy cue word"] = (
            f"{int(candidates['has_synergy_cue'].sum()):,}"
        )
    return {"facts": facts, "tables": [frame_table(types)]}


def report(settings: Settings) -> BuildResult:
    """Coverage, conflicts, the quality report (Markdown + HTML) and the licence registry."""
    drugs, cell_lines, pairs = _processed(settings)
    writer = _Writer(settings, "synergyrx-data:report")
    reports = settings.reports_root  # the documents: Markdown and HTML
    csv_folder = settings.csv_root  # the tables behind them
    options = settings.cleaning
    coverage = coverage_table(pairs, drugs, cell_lines)
    conflicts = find_conflicts(pairs, options.conflict_abs_tolerance)
    agreement = source_agreement(pairs)
    state = StateStore(settings.state_dir / "connectors.json", read_only=True)
    tracker = DriftTracker(settings.state_dir / "schemas", read_only=True)
    drift = [
        finding.to_dict() for table in tracker.tables() for finding in tracker.last_change(table)
    ]
    links, checked_at = load_link_status(settings)
    doc = build_quality_report(
        {
            "generated_at": utc_now_iso(),
            "version": __version__,
            "csv_path": settings.relative(csv_folder),
            "sample_sources": sample_sources(settings, writer.manifest),
            "pairs": pairs,
            "drugs": drugs,
            "cell_lines": cell_lines,
            "coverage": coverage,
            "conflicts": conflicts,
            "agreement": agreement,
            "drug_overlap": drug_overlap(pairs),
            "drug_depth": drug_depth(pairs, drugs),
            "pair_stats": {
                k: v
                for k, v in (state.get("_clean").get("stats") or {}).items()
                if isinstance(v, int)
            },
            "conflict_tolerance": options.conflict_abs_tolerance,
            "missingness": {
                "drugs": missingness_table(drugs),
                "cell_lines": missingness_table(cell_lines),
                "synergy_pairs": missingness_table(pairs),
            },
            "drift": drift,
            "literature": _literature_context(settings),
            "source_states": source_states(settings),
            "links": links,
            "links_checked_at": checked_at,
        }
    )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_quality.md").write_text(doc.to_markdown(), encoding="utf-8")
    (reports / "data_quality.html").write_text(doc.to_html(), encoding="utf-8")
    writer.file(reports / "data_quality.md", "report")
    writer.file(reports / "data_quality.html", "report")
    writer.csv(coverage, csv_folder / "coverage.csv")
    writer.csv(_disagreements(conflicts), csv_folder / "conflicts.csv")
    writer.csv(agreement, csv_folder / "source_agreement.csv")
    result = writer.finish()
    result.outputs[write_licenses(settings, writer.manifest)] = -1
    result.stats = {"distinct_drugs": len(drugs), "cell_lines": len(cell_lines)}
    return result
