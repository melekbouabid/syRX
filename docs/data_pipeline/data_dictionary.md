# Data dictionary

One section per output table, one row per column. `tests/unit/test_data_dictionary.py` fails
if a column is produced that is not described here, or described but no longer produced.

The pipeline writes under the one `data/` folder: `data/raw` (downloaded files, as
received), `data/interim/<source>/` (the validated tables below), `data/cache` and
`data/state`. Rows that fail their schema are in `<table>.rejects.parquet` next to each
table. Only `manifest.json`, the index of all those files, sits beside the folder. The module
2 corpus (`python -m module2_ai_engineering.corpus`) reads these tables; the SynergyRx
processed tables (`data/parquet`, `csv`, `json`, `reports`) were removed with the synergy
sources (decision D85).

Conventions used in every table:

- **Provenance.** Every table ends with the same five columns, described once here:

| Column | Type | Meaning |
|---|---|---|
| `source` | string | Connector the row comes from |
| `source_url` | string | Where the data was retrieved from (secrets removed) |
| `retrieved_at` | string | UTC time of retrieval, ISO 8601 (`2026-10-13T18:04:11Z`) |
| `source_version` | string | Version of the source: release, record or file id |
| `license` | string | Licence of the source data, as listed in `LICENSES.md` |

- **Missing values** are nulls, never zeros or empty strings.

## data/interim/ddinter/interactions.parquet

One row per interacting pair of drugs and severity level, across the DDInter category files.

| Column | Type | Meaning |
|---|---|---|
| `drug_a_name` | string | First drug, as DDInter names it |
| `drug_b_name` | string | Second drug |
| `level` | string | Severity: `Major`, `Moderate`, `Minor` or `Unknown` |
| `drug_a_source_id` | string | DDInter id of the first drug (`DDInter1`) |
| `drug_b_source_id` | string | DDInter id of the second drug |

## data/interim/pubmed/articles.parquet

One row per article (distinct PMID).

| Column | Type | Meaning |
|---|---|---|
| `pmid` | string | PubMed identifier |
| `title` | string | Article title, inline markup removed |
| `abstract` | string | Abstract text. Structured abstracts keep their labels, one section per line |
| `journal` | string | ISO journal abbreviation, or the full title |
| `year` | int | Publication year |
| `doi` | string | DOI, lower case |
| `pmcid` | string | PubMed Central id, if any |
| `language` | string | Language code as given by PubMed (`eng`, `chi`, ...) |
| `publication_types` | list of string | PubMed publication types |
| `mesh_terms` | list of string | MeSH descriptor names |
| `article_type` | string | `research`, `review`, `meta_analysis`, `clinical_trial`, `case_report`, `commentary`, `preprint`, `retraction_notice` |
| `is_retracted` | bool | The article has been retracted |
| `is_retraction_notice` | bool | The record is itself a retraction notice |
| `has_erratum` | bool | An erratum has been published |
| `has_expression_of_concern` | bool | An expression of concern has been published |
| `has_abstract` | bool | `abstract` is not null |

## data/interim/pubmed/query_hits.parquet

Which query found which article. `query_log.jsonl`, next to it, lists every query sent.

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | Stable id of the query (hash of its kind and ingredient) |
| `pmid` | string | PubMed identifier of an article the query returned |
| `kind` | string | `ingredient` (one active ingredient of `config/pill_classes.csv`) |
| `pair_key` | string | Always null for ingredient queries; kept so older query logs read the same |

## data/interim/openfda_label/label_sections.parquet

One row per kept section of the one label chosen for each active ingredient. Dosage and
overdosage sections are never stored.

| Column | Type | Meaning |
|---|---|---|
| `ingredient` | string | Active ingredient the label was chosen for |
| `label_id` | string | openFDA label id |
| `set_id` | string | SPL set id, stable across label versions |
| `label_version` | string | SPL version number |
| `effective_time` | string | Effective date of the label (`YYYYMMDD`) |
| `product_type` | string | `HUMAN PRESCRIPTION DRUG` or `HUMAN OTC DRUG` |
| `brand_name` | string | Brand names, `; `-separated |
| `generic_name` | string | Generic names, `; `-separated |
| `manufacturer` | string | Manufacturers, `; `-separated |
| `substances` | string | Substance names, `; `-separated |
| `label_url` | string | DailyMed page of the label |
| `section` | string | openFDA section name (`drug_interactions`, `warnings`, ...) |
| `text` | string | Section text |

## manifest.json

`files` maps each project-relative path to one entry.

| Column | Type | Meaning |
|---|---|---|
| `path` | string | Project-relative path |
| `source` | string | Connector |
| `layer` | string | `raw`, `manual` or `interim` |
| `sha256` | string | SHA-256 of the file |
| `size_bytes` | int | File size |
| `retrieved_at` | string | UTC time the file was retrieved or built |
| `url` | string | Where it came from (raw files) |
| `version` | string | Source version |
| `license` | string | Licence of the source |
| `terms_url` | string | Terms of use of the source |
| `is_sample` | bool | Fetched or built with `--sample`: incomplete |
| `etag` | string | HTTP ETag at download time, if the server sent one |
| `last_modified` | string | HTTP Last-Modified at download time |
| `rows` | int | Row count, for tables |
| `derived_from` | list of string | Paths this file was built from |
| `input_fingerprint` | string | Hash of the exact inputs; an unchanged fingerprint means the table is not rebuilt |
