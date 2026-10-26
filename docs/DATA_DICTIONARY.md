# Data dictionary

One section per output file, one row per column. `tests/unit/test_data_dictionary.py` fails
if a column is produced that is not described here.

Everything is under the one `data/` folder, grouped by format: `data/parquet` (the
processed tables), `data/csv` (every CSV: coverage, conflicts, review queues, the annotation
sample), `data/json` (JSON for the frontend) and `data/reports` (the quality report as
Markdown and HTML, and the annotation guide). `data/raw` and `data/interim` are the
pipeline's working folders. Only `manifest.json`, the index of all those files, sits beside
the folder.

Conventions used in every table:

- **Identifiers.** `drug_id` is the InChIKey of the standardised structure, or
  `name:<key>` when no valid structure is known. `cell_line_id` is the DepMap id
  (`ACH-000123`), or `name:<KEY>` when the line could not be mapped.
- **Provenance.** Every table ends with the same five columns, described once here:

| Column | Type | Meaning |
|---|---|---|
| `source` | string | Connector(s) the row comes from, `; `-separated when several contributed |
| `source_url` | string | Where the data was retrieved from (secrets removed) |
| `retrieved_at` | string | UTC time of retrieval, ISO 8601 (`2026-10-13T18:04:11Z`) |
| `source_version` | string | Version of the source: release, record or file id |
| `license` | string | Licence of the source data, as recorded in `LICENSES.md` |

- **Missing values** are nulls, never zeros or empty strings.

## data/parquet/synergy_pairs.parquet

One row per drug pair, cell line, source and study. The pair is unordered: `drug_a_id` is
always the smaller id, so (A, B) and (B, A) are the same row. Replicates inside one study
are averaged. Rows from different sources are **not** merged: pick one per
(`pair_key`, `cell_line_id`) after choosing a score.

| Column | Type | Meaning |
|---|---|---|
| `pair_key` | string | `"<drug_a_id>\|<drug_b_id>"`, ids sorted. Stable key of the unordered pair |
| `drug_a_id` | string | Id of the first drug (the smaller of the two ids) |
| `drug_b_id` | string | Id of the second drug |
| `drug_a_name` | string | Preferred name of drug A, from `drugs.parquet` |
| `drug_b_name` | string | Preferred name of drug B |
| `cell_line_id` | string | Cell-line id, see `cell_lines.parquet` |
| `cell_line_name` | string | Preferred cell-line name |
| `depmap_id` | string | DepMap model id, null when the line is not mapped |
| `tissue` | string | Lineage (DepMap `OncotreeLineage`), or the source's tissue if unmapped |
| `source` | string | Connector that reported this record: `drugcomb`, `tdc`, `oneil`, `nci_almanac` |
| `study_id` | string | Study within the source (`ONEIL`, `TDC:drugcomb`, `ALMANAC:1A`). Null for DrugComb, whose bulk table has no study column |
| `synergy_bliss` | float | Bliss independence excess, mean over replicates. Above 0 means synergy |
| `synergy_loewe` | float | Loewe additivity excess, mean over replicates. Above 0 means synergy |
| `synergy_hsa` | float | Highest-single-agent excess, mean over replicates. Above 0 means synergy |
| `synergy_zip` | float | Zero-interaction-potency (ZIP) delta score, mean over replicates |
| `css` | float | DrugComb combination sensitivity score. Measures efficacy, not synergy |
| `combo_score` | float | NCI-ALMANAC ComboScore: sum over dose pairs of expected minus observed growth |
| `synergy_bliss_std` | float | Standard deviation of `synergy_bliss` across the averaged replicates (0 if one) |
| `synergy_loewe_std` | float | Same for `synergy_loewe` |
| `synergy_hsa_std` | float | Same for `synergy_hsa` |
| `synergy_zip_std` | float | Same for `synergy_zip` |
| `css_std` | float | Same for `css` |
| `combo_score_std` | float | Same for `combo_score` |
| `n_replicates` | int | Number of source rows averaged into this record |
| `score_outlier` | bool | A synergy score exceeds `cleaning.score_outlier_abs` in absolute value. Flagged, not clipped |
| `n_sources` | int | Number of distinct sources reporting this pair on this cell line |
| `n_records` | int | Number of rows in this table for this pair on this cell line |
| `has_conflict` | bool | Those rows disagree: a synergy score's range exceeds `cleaning.conflict_abs_tolerance` or its sign differs |

## data/parquet/drugs.parquet

One row per distinct drug.

| Column | Type | Meaning |
|---|---|---|
| `drug_id` | string | InChIKey of the standardised structure, or `name:<key>` if none |
| `name` | string | Preferred name: PubChem title if known, else the most frequent source name |
| `synonyms` | list of string | Other names: every source spelling first, then PubChem synonyms (capped) |
| `pubchem_cid` | int | PubChem compound id, null if not resolved or in conflict |
| `chembl_id` | string | ChEMBL molecule id, null if not found |
| `inchikey` | string | InChIKey of `smiles`, null if the structure is invalid |
| `smiles` | string | Standardised SMILES: salts and solvents removed, neutralised, canonical |
| `smiles_raw` | string | SMILES as received from the dataset or PubChem, kept for review |
| `smiles_valid` | bool | False when there is no SMILES or RDKit could not process it. Such drugs are kept, not dropped |
| `smiles_issue` | string | Why the SMILES is invalid (`missing`, `unparseable SMILES`, ...) |
| `smiles_note` | string | What standardisation changed (`salt or solvent removed`, `neutralised`, `metal complex: all fragments kept`) |
| `resolution_method` | string | How the structure was chosen: `manual_override`, `source_smiles+pubchem`, `pubchem_name`, `source_smiles`, `pubchem_name_ambiguous`, `source_smiles_conflict`, `unresolved` |
| `resolution_confidence` | float | 0 to 1, fixed per method (see docs/DECISIONS.md, D14) |
| `needs_review` | bool | In the manual-review queue (`data/csv/drug_review_queue.csv`) |
| `review_reason` | string | Why it needs review |
| `sources` | list of string | Connectors in which the drug appears |
| `source_ids` | list of string | Source-specific ids, e.g. `NSC-740` |
| `n_source_names` | int | Number of distinct spellings merged into this drug |
| `n_measurements` | int | Number of source rows mentioning the drug (both sides of a pair counted) |

## data/parquet/cell_lines.parquet

One row per distinct cell line.

| Column | Type | Meaning |
|---|---|---|
| `cell_line_id` | string | DepMap id, or `name:<KEY>` when unmapped |
| `name` | string | DepMap `CellLineName` when mapped, else the most frequent source name |
| `stripped_name` | string | Upper-case letters and digits only (DepMap `StrippedCellLineName`) |
| `depmap_id` | string | DepMap model id (`ACH-...`), null when unmapped |
| `tissue` | string | DepMap `OncotreeLineage`, or `source_tissue` when unmapped |
| `disease` | string | DepMap `OncotreePrimaryDisease` |
| `source_tissue` | string | Tissue or panel as the source wrote it |
| `source_names` | list of string | Every spelling found in the sources |
| `sources` | list of string | Connectors in which the line appears |
| `match_method` | string | `exact`, `alias` (through the alias table) or `unmapped` |
| `mapped` | bool | A DepMap id was found |
| `needs_review` | bool | Unmapped, or several DepMap models share the name |
| `review_reason` | string | Why it needs review |
| `n_measurements` | int | Number of source rows on this cell line |

## data/parquet/expression.parquet

Expression matrix for the cell lines that appear in `synergy_pairs.parquet` and have a
DepMap id. One row per cell line, one column per gene.

| Column | Type | Meaning |
|---|---|---|
| `depmap_id` | string | DepMap model id; joins to `cell_lines.depmap_id` |
| `<SYMBOL (EntrezID)>` | float32 | One column per protein-coding gene, named as in DepMap, e.g. `TP53 (7157)`. Value is log2(TPM + 1) |

## data/parquet/abstracts.parquet

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
| `abstract_source` | string | `pubmed`, or `europepmc` when PubMed had no abstract |
| `is_open_access` | bool | Open access according to Europe PMC; null if not looked up |
| `in_europepmc` | bool | Full text is in Europe PMC; null if not looked up |
| `sentences` | list of string | The abstract split into sentences |
| `n_sentences` | int | Number of sentences |
| `pubmed_url` | string | `https://pubmed.ncbi.nlm.nih.gov/<pmid>/` |

## data/parquet/candidate_sentences.parquet

One row per sentence that mentions at least two different drugs from `drugs.parquet`.
Mentions are found offline with the synonym table; these are candidates for relation
extraction, not confirmed relations.

| Column | Type | Meaning |
|---|---|---|
| `pmid` | string | Article the sentence comes from |
| `sentence_index` | int | Position of the sentence in the abstract, from 0 |
| `sentence_start` | int | Character offset of the sentence in `abstracts.abstract` |
| `sentence` | string | Sentence text |
| `n_drugs` | int | Number of distinct drugs mentioned |
| `drug_ids` | list of string | Their ids, sorted |
| `drug_names` | list of string | Their preferred names, same order as `drug_ids` |
| `mentions` | list of string | The exact strings that matched |
| `pair_keys` | list of string | `pair_key` of every pair of mentioned drugs (at most 45) |
| `has_synergy_cue` | bool | The sentence contains a cue word (synerg-, antagon-, additiv-, combination, ...) |
| `cue_terms` | list of string | The cue words found |
| `year` | int | Publication year of the article |
| `article_type` | string | As in `abstracts.parquet` |
| `language` | string | As in `abstracts.parquet` |
| `is_retracted` | bool | The article has been retracted |

## data/json/drugs.json

Array of objects, sorted by name.

| Column | Type | Meaning |
|---|---|---|
| `id` | string | `drug_id` |
| `name` | string | Preferred name |
| `smiles` | string | Standardised SMILES, null if invalid |
| `smiles_valid` | bool | Whether `smiles` is usable |
| `inchikey` | string | InChIKey, null if invalid |
| `pubchem_cid` | int | PubChem compound id |
| `chembl_id` | string | ChEMBL molecule id |
| `synonyms` | list of string | First `export.max_synonyms` synonyms |

## data/json/cell_lines.json

| Column | Type | Meaning |
|---|---|---|
| `id` | string | `cell_line_id` |
| `name` | string | Preferred name |
| `tissue` | string | Lineage |
| `disease` | string | Primary disease |
| `depmap_id` | string | DepMap model id |

## data/json/mock_predictions_seed.json

Seed for placeholder predictions in the frontend. It contains **no model output**.

| Column | Type | Meaning |
|---|---|---|
| `_note` | string | Says the file holds measured values only |
| `generated_at` | string | UTC build time |
| `seed` | int | Random seed (`export.mock_seed`) the frontend should use |
| `score_distribution` | object | Per score column: `n`, `mean`, `std`, `p05`, `p25`, `p50`, `p75`, `p95` of the measured values |
| `examples` | array | Measured records: `drug_a_id`, `drug_b_id`, `cell_line_id`, `source`, `observed` (score name to value) |

## data/json/citations.json

Citation metadata only; abstract text is never exported.

| Column | Type | Meaning |
|---|---|---|
| `pmid` | string | PubMed identifier |
| `title` | string | Article title |
| `journal` | string | Journal abbreviation |
| `year` | int | Publication year |
| `doi` | string | DOI |
| `url` | string | PubMed link |
| `is_retracted` | bool | The article has been retracted |
| `is_open_access` | bool | Open access according to Europe PMC, null if unknown |

## data/csv/coverage.csv

One row per source, plus a last row `ALL SOURCES`.

| Column | Type | Meaning |
|---|---|---|
| `source` | string | Connector name, or `ALL SOURCES` |
| `records` | int | Rows of `synergy_pairs.parquet` |
| `distinct_drugs` | int | Distinct `drug_id` on either side of a pair. This number caps the unseen-drug validation protocol |
| `distinct_cell_lines` | int | Distinct `cell_line_id` |
| `distinct_pairs` | int | Distinct `pair_key` |
| `distinct_triplets` | int | Distinct (`pair_key`, `cell_line_id`) |
| `drugs_with_valid_smiles` | int | Of those drugs, how many have a usable structure |
| `cell_lines_mapped_to_depmap` | int | Of those cell lines, how many have a DepMap id |
| `synergy_bliss_pct` | float | Percentage of records carrying a Bliss score |
| `synergy_loewe_pct` | float | Same for Loewe |
| `synergy_hsa_pct` | float | Same for HSA |
| `synergy_zip_pct` | float | Same for ZIP |
| `css_pct` | float | Same for CSS |
| `combo_score_pct` | float | Same for ComboScore |

## data/csv/conflicts.csv

One row per pair and cell line whose records from different sources or studies disagree,
largest disagreement first. Duplicates that agree are not listed here: they are flagged in
`synergy_pairs.parquet` by `n_records > 1` and counted in the quality report. For each score
`X` in `synergy_bliss`, `synergy_loewe`, `synergy_hsa`, `synergy_zip`, `css`, `combo_score`
there are four columns.

| Column | Type | Meaning |
|---|---|---|
| `pair_key` | string | Pair key |
| `cell_line_id` | string | Cell-line id |
| `drug_a_name` | string | Name of drug A |
| `drug_b_name` | string | Name of drug B |
| `cell_line_name` | string | Cell-line name |
| `n_records` | int | Number of records for this pair and cell line |
| `n_sources` | int | Number of distinct sources among them |
| `sources` | string | Those sources, `; `-separated |
| `studies` | string | Their study ids, `; `-separated |
| `X_n` | int | Records that report score `X` |
| `X_min` | float | Smallest value of `X`, null unless two or more records report it |
| `X_max` | float | Largest value of `X` |
| `X_range` | float | `X_max - X_min` |
| `max_synergy_range` | float | Largest range among the four synergy scores |
| `sign_disagreement` | bool | A synergy score is positive in one record and negative in another |
| `is_conflict` | bool | `max_synergy_range` exceeds the tolerance, or `sign_disagreement`. Always true in this file |

## data/csv/source_agreement.csv

| Column | Type | Meaning |
|---|---|---|
| `source_a` | string | First source |
| `source_b` | string | Second source |
| `score` | string | Score column compared |
| `shared_triplets` | int | Pair and cell-line combinations both sources report with this score |
| `pearson_r` | float | Correlation of the two sources' values on those |
| `median_abs_diff` | float | Median absolute difference |
| `sign_disagreement_rate` | float | Share of them where the sign differs |

## data/csv/drug_review_queue.csv

| Column | Type | Meaning |
|---|---|---|
| `drug_id` | string | Drug id |
| `name` | string | Preferred name |
| `resolution_method` | string | As in `drugs.parquet` |
| `resolution_confidence` | float | As in `drugs.parquet` |
| `review_reason` | string | What to check |
| `smiles_raw` | string | SMILES as received |
| `pubchem_cid` | int | PubChem id if any |
| `sources` | string | Connectors, `; `-separated |
| `n_measurements` | int | How much data depends on this drug (queue is sorted by it) |

## data/csv/cell_line_review_queue.csv

| Column | Type | Meaning |
|---|---|---|
| `cell_line_id` | string | Cell-line id |
| `name` | string | Preferred name |
| `source_names` | string | Spellings found, `; `-separated |
| `sources` | string | Connectors, `; `-separated |
| `review_reason` | string | What to check |
| `n_measurements` | int | How much data depends on this line |

## data/csv/sample_sentences.csv

Stratified sample for manual labelling. The same rows are written as
`data/json/sample_sentences.jsonl`; how to label is in `data/reports/annotation_guide.md`.

| Column | Type | Meaning |
|---|---|---|
| `sample_id` | string | `S0001`, `S0002`, ... |
| `stratum` | string | `cue\|nocue / 2drugs\|3plus / primary\|review` |
| `pmid` | string | Article |
| `sentence_index` | int | Position of the sentence in the abstract |
| `sentence` | string | Sentence text |
| `drug_names` | string | Drugs mentioned, `; `-separated |
| `n_drugs` | int | Number of distinct drugs |
| `has_synergy_cue` | bool | A cue word is present |
| `cue_terms` | string | Cue words, `; `-separated |
| `article_type` | string | As in `abstracts.parquet` |
| `year` | int | Publication year |
| `is_retracted` | bool | The article has been retracted |
| `pubmed_url` | string | Link to the article |
| `label` | string | **To fill in:** `synergy`, `antagonism`, `additive`, `no_relation` or `unclear` |
| `annotator` | string | **To fill in:** who labelled it |
| `comment` | string | **To fill in:** free text |

## manifest.json

`files` maps each project-relative path to one entry.

| Column | Type | Meaning |
|---|---|---|
| `path` | string | Project-relative path |
| `source` | string | Connector, or `synergyrx-data:<step>` for derived files |
| `layer` | string | `raw`, `manual`, `interim`, `processed`, `report` or `export` |
| `sha256` | string | SHA-256 of the file |
| `size_bytes` | int | File size |
| `retrieved_at` | string | UTC time the file was retrieved or built |
| `url` | string | Where it came from (raw files) |
| `version` | string | Source version, or the pipeline version for derived files |
| `license` | string | Licence of the source |
| `terms_url` | string | Terms of use of the source |
| `is_sample` | bool | Fetched or built with `--sample`: incomplete |
| `etag` | string | HTTP ETag at download time, if the server sent one |
| `last_modified` | string | HTTP Last-Modified at download time |
| `rows` | int | Row count, for tables |
| `derived_from` | list of string | Paths this file was built from |
| `input_fingerprint` | string | Hash of the exact inputs; an unchanged fingerprint means the table is not rebuilt |

## Interim tables (`data/interim/<source>/`)

Per-source tables after parsing and validation, before cleaning. Rows that fail their schema
are in `<table>.rejects.parquet` next to each table. All carry the provenance columns.

| Table | Content |
|---|---|
| `drugcomb`, `tdc`, `oneil`, `nci_almanac` / `synergy` | Source rows in a common layout: `drug_a_name`, `drug_b_name`, `cell_line_name`, `tissue`, `study_id`, `source_record_id`, `drug_a_source_id`, `drug_b_source_id`, `drug_a_smiles`, `drug_b_smiles`, and the six score columns |
| `depmap` / `models` | `depmap_id`, `cell_line_name`, `stripped_name`, `ccle_name`, `lineage`, `primary_disease`, `sanger_model_id` |
| `depmap` / `expression` | Full expression matrix, same layout as the processed one |
| `pubchem` / `compounds` | `query_name`, `status` (`found`, `ambiguous`, `not_found`), `n_cids`, `pubchem_cid`, `smiles`, `connectivity_smiles`, `inchikey`, `title`, `synonyms` |
| `chembl` / `molecules` | `query`, `query_type` (`inchikey` or `name`), `chembl_id`, `pref_name`, `max_phase`, `inchikey` |
| `pubmed` / `articles`, `query_hits` | Articles as in `abstracts.parquet` before segmentation; which query found which PMID. `query_log.jsonl` lists every query sent |
| `europepmc` / `articles` | `pmid`, `pmcid`, `doi`, `title`, `abstract`, `is_open_access`, `in_epmc`, `has_pdf`, `article_license` |
| `pubtator` / `annotations` | `pmid`, `entity_type`, `text`, `identifier`, `section`, `start`, `length` |
| `ddinter` / `interactions` | `drug_a_name`, `drug_b_name`, `level`, DDInter ids |
| `gdsc` / `models` | `sanger_model_id`, `cell_line_name`, `depmap_id`, `tissue`, `cancer_type` |
| `drugbank` / `vocabulary` | `drugbank_id`, `name`, `synonyms`, `inchikey`, `cas`, `unii` |
| `_maps` / `drug_names`, `cell_line_names` | Source spelling to `drug_id` / `cell_line_id` |
