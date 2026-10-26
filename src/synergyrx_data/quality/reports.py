"""Coverage, missingness and the auto-generated data quality report (Markdown and HTML)."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from synergyrx_data.cleaning.scores import SCORE_COLUMNS, SCORES

PAIR_STATS = (
    "input_rows",
    "self_pairs_dropped",
    "unmapped_rows_dropped",
    "replicates_averaged",
    "output_rows",
    "cell_lines_with_expression",
    "cell_lines_without_expression",
)


@dataclass
class Table:
    headers: list[str]
    rows: list[list[Any]]


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


@dataclass
class ReportDoc:
    title: str
    intro: list[str]
    sections: list[Section]

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", *[f"{p}\n" for p in self.intro]]
        for section in self.sections:
            lines += [f"## {section.title}", ""]
            for paragraph in section.paragraphs:
                lines += [paragraph, ""]
            for bullet in section.bullets:
                lines.append(f"- {bullet}")
            if section.bullets:
                lines.append("")
            for table in section.tables:
                lines.append("| " + " | ".join(table.headers) + " |")
                lines.append("|" + "|".join("---" for _ in table.headers) + "|")
                for row in table.rows:
                    lines.append(
                        "| " + " | ".join(_cell(v).replace("|", "\\|") for v in row) + " |"
                    )
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_html(self) -> str:
        esc = html.escape
        parts = [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{esc(self.title)}</title>",
            "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
            "padding:0 1rem;color:#1b1f24}table{border-collapse:collapse;margin:1rem 0;"
            "display:block;overflow-x:auto}th,td{border:1px solid #d0d7de;padding:.3rem .6rem;"
            "text-align:left;white-space:nowrap}th{background:#f6f8fa}"
            "h2{margin-top:2.2rem;border-bottom:1px solid #d0d7de;padding-bottom:.2rem}"
            "code{background:#f6f8fa;padding:.1rem .3rem;border-radius:3px}</style></head><body>",
            f"<h1>{esc(self.title)}</h1>",
            *[f"<p>{_inline(p)}</p>" for p in self.intro],
        ]
        for section in self.sections:
            parts.append(f"<h2>{esc(section.title)}</h2>")
            parts += [f"<p>{_inline(p)}</p>" for p in section.paragraphs]
            if section.bullets:
                parts.append(
                    "<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in section.bullets) + "</ul>"
                )
            for table in section.tables:
                head = "".join(f"<th>{esc(h)}</th>" for h in table.headers)
                body = "".join(
                    "<tr>" + "".join(f"<td>{esc(_cell(v))}</td>" for v in row) + "</tr>"
                    for row in table.rows
                )
                parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        parts.append("</body></html>")
        return "\n".join(parts) + "\n"


def _inline(text: str) -> str:
    """Escape, then render the two Markdown marks the report uses: **bold** and `code`."""
    out = html.escape(text)
    for mark, tag in (("**", "strong"), ("`", "code")):
        pieces = out.split(mark)
        if len(pieces) % 2 == 1:
            out = "".join(
                piece if i % 2 == 0 else f"<{tag}>{piece}</{tag}>" for i, piece in enumerate(pieces)
            )
    return out


def _cell(value: Any) -> str:
    if value is None or (not isinstance(value, list | tuple | str) and pd.isna(value)):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() and abs(value) >= 100 else f"{value:.3g}"
    if isinstance(value, list | tuple):
        return "; ".join(str(v) for v in value)
    return str(value)


def frame_table(frame: pd.DataFrame, max_rows: int | None = None) -> Table:
    shown = frame.head(max_rows) if max_rows else frame
    rows = [[v.item() if hasattr(v, "item") else v for v in row] for row in shown.to_numpy()]
    return Table([str(c) for c in frame.columns], rows)


# ------------------------------------------------------------------ tables


def coverage_table(
    pairs: pd.DataFrame, drugs: pd.DataFrame, cell_lines: pd.DataFrame
) -> pd.DataFrame:
    """Per source and overall: distinct drugs, cell lines, pairs and score availability."""
    if pairs.empty:
        return pd.DataFrame()
    valid = set(drugs.loc[drugs["smiles_valid"], "drug_id"])
    mapped = set(cell_lines.loc[cell_lines["mapped"], "cell_line_id"])
    rows = []
    for label, part in [*pairs.groupby("source", sort=True), ("ALL SOURCES", pairs)]:
        drug_ids = set(part["drug_a_id"]) | set(part["drug_b_id"])
        cells = set(part["cell_line_id"])
        row: dict[str, Any] = {
            "source": label,
            "records": len(part),
            "distinct_drugs": len(drug_ids),
            "distinct_cell_lines": len(cells),
            "distinct_pairs": part["pair_key"].nunique(),
            "distinct_triplets": len(part[["pair_key", "cell_line_id"]].drop_duplicates()),
            "drugs_with_valid_smiles": len(drug_ids & valid),
            "cell_lines_mapped_to_depmap": len(cells & mapped),
        }
        for score in SCORE_COLUMNS:
            row[f"{score}_pct"] = round(100 * float(part[score].notna().mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of nulls per column, most incomplete first."""
    if frame.empty:
        return pd.DataFrame(columns=["column", "nulls", "null_pct"])
    nulls = frame.isna().sum()
    table = pd.DataFrame(
        {
            "column": nulls.index.astype(str),
            "nulls": nulls.to_numpy(),
            "null_pct": (100 * nulls / len(frame)).round(1).to_numpy(),
        }
    )
    return table.sort_values(["null_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def drug_overlap(pairs: pd.DataFrame) -> pd.DataFrame:
    """How many drugs each pair of sources has in common."""
    by_source = {
        source: set(part["drug_a_id"]) | set(part["drug_b_id"])
        for source, part in pairs.groupby("source", sort=True)
    }
    names = list(by_source)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = by_source[left] & by_source[right]
            rows.append(
                {
                    "source_a": left,
                    "source_b": right,
                    "drugs_a": len(by_source[left]),
                    "drugs_b": len(by_source[right]),
                    "shared_drugs": len(shared),
                }
            )
    return pd.DataFrame(rows)


def drug_depth(pairs: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    """How many drugs have at least N distinct partners, and how many of those have a structure.

    A drug with a single partner on a single cell line cannot support a leave-drug-out fold
    on its own, so the usable number of drugs is smaller than the headline count.
    """
    if pairs.empty:
        return pd.DataFrame()
    unique = pairs[["pair_key", "drug_a_id", "drug_b_id", "cell_line_id"]].drop_duplicates()
    sides = pd.concat(
        [
            unique.rename(columns={"drug_a_id": "drug", "drug_b_id": "partner"}),
            unique.rename(columns={"drug_b_id": "drug", "drug_a_id": "partner"}),
        ]
    )
    per_drug = sides.groupby("drug").agg(
        partners=("partner", "nunique"), cell_lines=("cell_line_id", "nunique")
    )
    valid = set(drugs.loc[drugs["smiles_valid"], "drug_id"])
    per_drug["valid"] = per_drug.index.isin(valid)
    rows = []
    for partners, cells in ((1, 1), (2, 1), (5, 1), (10, 1), (10, 10), (20, 20), (50, 30)):
        mask = (per_drug["partners"] >= partners) & (per_drug["cell_lines"] >= cells)
        rows.append(
            {
                "at_least_partners": partners,
                "at_least_cell_lines": cells,
                "drugs": int(mask.sum()),
                "of_which_with_valid_smiles": int((mask & per_drug["valid"]).sum()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report


def build_quality_report(ctx: dict[str, Any]) -> ReportDoc:
    """Assemble the report from pre-computed tables (see `build.report`)."""
    pairs: pd.DataFrame = ctx["pairs"]
    drugs: pd.DataFrame = ctx["drugs"]
    cells: pd.DataFrame = ctx["cell_lines"]
    coverage: pd.DataFrame = ctx["coverage"]
    conflicts: pd.DataFrame = ctx["conflicts"]
    csv_path: str = ctx["csv_path"]  # where the CSVs referred to below were written
    sections: list[Section] = []
    intro = [f"Generated {ctx['generated_at']} by synergyrx-data {ctx['version']}."]
    if ctx["sample_sources"]:
        intro.append(
            "**SAMPLE DATA.** These sources were fetched with `--sample` and are incomplete: "
            f"{', '.join(ctx['sample_sources'])}. Do not train or report results on this build."
        )

    total = coverage[coverage["source"] == "ALL SOURCES"].iloc[0] if len(coverage) else None
    n_conflicts = int(conflicts["is_conflict"].sum()) if len(conflicts) else 0
    if total is not None:
        sections.append(
            Section(
                "Summary",
                bullets=[
                    f"**{int(total['distinct_drugs']):,} distinct drugs**, "
                    f"{int(total['distinct_cell_lines']):,} cell lines, "
                    f"{int(total['distinct_pairs']):,} drug pairs, "
                    f"{int(total['distinct_triplets']):,} pair x cell-line combinations.",
                    f"{len(pairs):,} records in `synergy_pairs.parquet` "
                    "(one per pair, cell line, source and study).",
                    f"{int((~drugs['smiles_valid']).sum()):,} drugs have no valid structure "
                    f"(flagged, not removed); {int(drugs['needs_review'].sum()):,} drugs are in "
                    "the manual-review queue.",
                    f"{int((~cells['mapped']).sum()):,} of {len(cells):,} cell lines are not "
                    "mapped to a DepMap identifier.",
                    f"{len(conflicts):,} pair x cell-line combinations are reported by more than "
                    f"one source or study; {n_conflicts:,} "
                    "of them disagree.",
                ],
            )
        )
        sections.append(
            Section(
                "Coverage per source",
                paragraphs=[
                    "A drug is counted once per standardised structure (InChIKey), or once per "
                    "name when no structure is known. **The number of distinct drugs caps the "
                    "unseen-drug validation protocol**: with D drugs, leave-drug-out "
                    "cross-validation has at most D held-out units, and every pair involving a "
                    "held-out drug leaves the training set.",
                    "Score columns show the percentage of records that carry each score.",
                ],
                tables=[frame_table(coverage)],
            )
        )
    depth: pd.DataFrame = ctx["drug_depth"]
    if len(depth):
        sections.append(
            Section(
                "How much data each drug has",
                paragraphs=[
                    "Drugs that have at least this many distinct partner drugs and were measured "
                    "on at least this many cell lines. Most drugs in DrugComb were screened "
                    "against one or two partners only: the number of drugs that can carry an "
                    "unseen-drug evaluation is the lower rows of this table, not the headline "
                    "count, and only drugs with a valid structure can be featurised at all."
                ],
                tables=[frame_table(depth)],
            )
        )
    overlap: pd.DataFrame = ctx["drug_overlap"]
    if len(overlap):
        sections.append(
            Section(
                "Drug overlap between sources",
                paragraphs=[
                    "Sources that share drugs also share experiments: the ALMANAC and O'Neil "
                    "screens are both inside DrugComb. Split train and test by pair or by drug "
                    "**across** sources, never within one source only."
                ],
                tables=[frame_table(overlap)],
            )
        )

    methods = drugs.groupby("resolution_method").agg(
        drugs=("drug_id", "size"),
        confidence=("resolution_confidence", "first"),
        needs_review=("needs_review", "sum"),
    )
    issues = drugs.loc[~drugs["smiles_valid"], "smiles_issue"].fillna("no SMILES").value_counts()
    sections.append(
        Section(
            "Drug identity and SMILES",
            paragraphs=[
                "How each drug got its structure. `source_smiles+pubchem` means the SMILES "
                "shipped by the dataset and the PubChem record agree on the skeleton.",
            ],
            tables=[
                frame_table(methods.reset_index()),
                Table(
                    ["invalid or missing SMILES: reason", "drugs"],
                    [[k, int(v)] for k, v in issues.items()],
                ),
            ],
            bullets=[
                f"Invalid or missing SMILES: **{int((~drugs['smiles_valid']).sum()):,}** drugs. "
                "They stay in `drugs.parquet` with `smiles_valid = false`.",
                f"Review queue: `{csv_path}/drug_review_queue.csv`.",
            ],
        )
    )
    by_method = cells.groupby("match_method").agg(
        cell_lines=("cell_line_id", "size"), mapped=("mapped", "sum")
    )
    sections.append(
        Section(
            "Cell lines",
            tables=[frame_table(by_method.reset_index())],
            bullets=[f"Unmapped names: `{csv_path}/cell_line_review_queue.csv`."],
        )
    )
    sections.append(
        Section(
            "Synergy scores",
            paragraphs=[
                "Each score has its own column and is never converted into another. "
                "Replicates within one study are averaged; `n_replicates` and `*_std` keep the "
                "count and spread."
            ],
            tables=[Table(["column", "meaning"], [[k, v] for k, v in SCORES.items()])],
            bullets=[
                f"Records with an implausibly large score (flagged, not clipped): "
                f"{int(pairs['score_outlier'].sum()):,}.",
                *[
                    f"{k.replace('_', ' ')}: {v:,}"
                    for k, v in ctx["pair_stats"].items()
                    if k in PAIR_STATS
                ],
            ],
        )
    )
    agreement: pd.DataFrame = ctx["agreement"]
    sections.append(
        Section(
            "Duplicates and conflicts",
            paragraphs=[
                "The same pair on the same cell line reported by several sources or studies. "
                "They are flagged in `synergy_pairs.parquet` (`n_records > 1`); those that "
                f"disagree are listed in `{csv_path}/conflicts.csv`. A row is a conflict "
                "when a "
                "synergy score's "
                f"range exceeds {ctx['conflict_tolerance']:g} or its sign differs.",
                "Agreement between sources on the combinations they share. A correlation of 1 "
                "with no difference means one source is a copy of the other; a low correlation "
                "on the same experiments means the two sources computed the score differently.",
            ],
            tables=[frame_table(agreement)] if len(agreement) else [],
        )
    )
    if len(conflicts):
        top = conflicts.head(10)[
            [
                "drug_a_name",
                "drug_b_name",
                "cell_line_name",
                "sources",
                "n_records",
                "max_synergy_range",
                "sign_disagreement",
            ]
        ]
        sections.append(Section("Largest disagreements", tables=[frame_table(top)]))

    sections.append(
        Section(
            "Missingness",
            paragraphs=["Columns with missing values, per processed table."],
            tables=[
                Table([name, "nulls", "null %"], frame_table(table[table["nulls"] > 0], 25).rows)
                for name, table in ctx["missingness"].items()
                if (table["nulls"] > 0).any()
            ],
        )
    )
    drift = ctx["drift"]
    sections.append(
        Section(
            "Schema drift",
            paragraphs=[
                "Changes in columns, types or row counts since the previous build of each table."
            ],
            tables=[
                Table(
                    ["table", "change", "detail", "severity"],
                    [[d["table"], d["kind"], d["detail"], d["severity"]] for d in drift],
                )
            ]
            if drift
            else [],
            bullets=[] if drift else ["No change detected."],
        )
    )
    literature = ctx.get("literature")
    if literature:
        sections.append(
            Section(
                "Literature",
                bullets=[f"{k}: {v}" for k, v in literature["facts"].items()],
                tables=literature["tables"],
            )
        )
    sections.append(
        Section(
            "Sources",
            paragraphs=["State of every connector after its last run."],
            tables=[frame_table(ctx["source_states"])],
        )
    )
    links: pd.DataFrame = ctx["links"]
    if len(links):
        sections.append(
            Section(
                "Link check",
                paragraphs=[f"Last run {ctx['links_checked_at']}. Details: `docs/SOURCES.md`."],
                tables=[frame_table(links)],
            )
        )
    return ReportDoc("SynergyRx data quality report", intro, sections)
