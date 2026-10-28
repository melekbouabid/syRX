# RAG corpus and retrieval

Documents the Tujya agent answers from: what they are, how they were collected, how they
are cut into chunks, what metadata each chunk carries, and how they are searched. Built on
2026-10-28. Decisions: D59 to D62 in docs/DECISIONS.md.

```
config/pill_classes.csv (82 active ingredients of the 150 pill classes)
  -> openFDA drug labels      python -m synergyrx_data.tujya labels   -> data/rag/raw, interim
  -> PubMed abstracts         python -m synergyrx_data.tujya pubmed   -> data/rag/raw, interim
  -> DDInter pairs            data/tujya/interactions_between_classes.csv (docs/INTERACTIONS.md)
  -> chunks + metadata        dvc repro rag_corpus  (= python -m synergyrx_data.tujya corpus)
                              -> data/rag/corpus/chunks.parquet, summary.json
  -> embeddings + index       python -m app.rag.index  (backend)  -> table rag_chunks (pgvector)
```

Commands run from the repository root with the pipeline venv (`.venv-dev`); without a
`.env` add `--set http.require_contact_email=false` (as in docs/INTERACTIONS.md). The two
fetch steps reuse the pipeline's connectors (PoliteClient: rate limits, retries, HTTP cache,
manifest with SHA-256 and provenance), in their own data folder `data/rag/` with its own
manifest `data/rag/manifest.json`, so they never mix with the SynergyRx files under
`data/raw`.

## Sources and counts

| Document type | Source | Licence | Documents | Chunks |
|---|---|---|---:|---:|
| `label` | openFDA drug labels (SPL), https://open.fda.gov/apis/drug/label/ | CC0 / US public domain (openFDA terms) | 82 (one per ingredient) | 2,895 |
| `abstract` | PubMed via NCBI E-utilities | NLM terms; abstracts may be under publisher copyright (research use, not redistributed) | 1,570 | 3,416 |
| `interaction` | DDInter 2.0 (pairs between the 82 ingredients) | CC BY-NC-SA 4.0 | 2,350 | 2,350 |
| **Total** | | | **4,002** | **8,661** |

All 82 ingredients have a label and between 17 and 20 abstracts (mean 19.7). Mean chunk
length 687 characters (labels 1,164, abstracts 627).

### openFDA labels (`openfda_label` connector)

`src/synergyrx_data/connectors/openfda.py`, registered with the other connectors (off by
default, so the SynergyRx `update` run does not fetch it). For each ingredient it asks
`api.fda.gov/drug/label.json` for the 25 most recent labels (`sort=effective_time:desc`)
whose `openfda.generic_name` or `openfda.substance_name` contains the ingredient (or a
search alias: valproate -> valproic acid / divalproex, vitamin B12 -> cyanocobalamin, vitamin
B6 -> pyridoxine), prescription products first, then any product type if none matches.

**Selection rule (D60)**, the first criterion that differs decides:

1. the label is about the ingredient (one of its generic or substance names contains every
   word of the ingredient name: "CYCLOBENZAPRINE HYDROCHLORIDE" is about cyclobenzaprine);
2. single-ingredient product before combination product;
3. prescription (`HUMAN PRESCRIPTION DRUG`) before OTC;
4. the most indexed sections (a repackager label without "drug interactions" loses);
5. the most recent `effective_time`.

Result: 81 prescription labels and 1 OTC (fexofenadine, sold OTC in the US); 80 single-
ingredient; pseudoephedrine and vitamin B6 only exist in combination products
(brompheniramine + pseudoephedrine + dextromethorphan; doxylamine + pyridoxine). Effective
dates 2021-09-30 to 2026-09-11, 80 of 82 since 2024. Each raw file
`data/rag/raw/openfda_label/current/labels/<ingredient>.json` keeps the chosen label and the
list of candidates it was chosen from, so the choice can be audited. Rate: 3 requests/s
(openFDA allows 240/min without a key); answers are cached on disk.

**Sections kept** (in label order): boxed_warning, indications_and_usage, contraindications,
warnings_and_cautions, warnings, precautions, general_precautions, drug_interactions,
drug_and_or_laboratory_test_interactions, adverse_reactions, use_in_specific_populations,
pregnancy, lactation, nursing_mothers, pediatric_use, geriatric_use,
information_for_patients, mechanism_of_action. Old-format labels put interactions under
`precautions`; both formats are covered. 11.9 sections per label on average.

**Never stored (D60):** `dosage_and_administration`, `dosage_forms_and_strengths`,
`overdosage`, `how_supplied`, `package_label_principal_display_panel`, and the numeric
sections `clinical_pharmacology`, `pharmacokinetics`, `clinical_studies`. They are removed
before the raw file is written, the row schema refuses any section named "dosage" or
"overdos", and the chunker refuses them a third time. The agent therefore cannot quote a
dosing section. Limit: other sections still mention amounts (1,010 of 8,661 chunks contain
"<number> mg", e.g. "patients receiving 200 mg" in adverse reactions); the agent's safety
node removes any sentence with a dose from its answers (docs/AGENT.md).

### PubMed abstracts (existing `pubmed` connector, mode `ingredients`)

The PubMed connector gained a third query mode, `ingredients` (D61): one query per
ingredient, `<ingredient block> AND (interactions OR adverse effects) AND hasabstract AND
english[la]`, where the ingredient block is the existing one (`[tiab]`, `[mh]`, `[nm]`) and
the topic block is `"Drug Interactions"[mh] OR interaction*[tiab] OR "adverse effects"[sh]
OR "Drug-Related Side Effects and Adverse Reactions"[mh] OR "adverse effect*"[tiab] OR
"adverse event*"[tiab] OR "adverse reaction*"[tiab] OR contraindicat*[tiab]`, entered since
2000-01-01, sorted by relevance (`sort=relevance`, new option), **at most 20 records per
ingredient**. Every query reached the cap (82 truncated queries): PubMed has many more.
1,617 hits, 1,570 distinct articles (44 found by two ingredients), years 1998 to 2026
(median 2014): 805 research, 266 case reports, 248 reviews, 178 clinical trials, 65
meta-analyses, 8 commentaries; **4 retracted** articles (17 chunks), kept with
`is_retracted = true` and excluded at search time by default.

### DDInter rows

The 2,350 pairs where both ingredients belong to the pill classes (docs/INTERACTIONS.md),
one chunk each: "DDInter 2.0 lists an interaction with severity Major between fluconazole
and warfarin. DDInter's bulk files give only the two drugs and the level, not the
mechanism." "Unknown" is written as "an interaction whose severity DDInter does not grade",
never as "no interaction". The 40,508 pairs with any drug stay out of the text index (they
would drown the other documents in near-identical sentences); the `check_interaction` tool
reads all of them from the database (D62).

## Chunking (`src/synergyrx_data/tujya/corpus.py`)

- **Labels: by section.** One chunk per section; a section longer than 1,500 characters is
  split on sentence ends into pieces of at most 1,500 characters, repeating the last
  sentence of a piece at the start of the next (one-sentence overlap). A single "sentence"
  longer than the limit (flattened tables) is cut on spaces.
- **Abstracts: by paragraph.** Structured abstracts have one paragraph per label
  ("BACKGROUND:", "RESULTS:" ...), which becomes the chunk's `section`; unstructured
  abstracts are one paragraph (`section = "abstract"`); long paragraphs are split as above.
- **DDInter: one chunk per pair.**
- Each chunk gets a title that gives its context ("Fluconazole label (Diflucan): Drug
  interactions", the article title, "DDInter 2.0: fluconazole + warfarin (Major)"). The
  title and the text are what BM25 and the embeddings see.
- `chunk_id` is a hash of (document, section, position), so the same input gives the same
  ids and the indexing step can be incremental.

## Metadata schema (one row per chunk, `data/rag/corpus/chunks.parquet`)

| Column | Type | Meaning |
|---|---|---|
| `chunk_id` | string | stable id (16 hex characters) |
| `doc_id` | string | `openfda:<label id>`, `pubmed:<PMID>`, `ddinter:<id a>:<id b>` |
| `doc_type` | string | `label`, `abstract` or `interaction` |
| `ingredients` | list of strings | the ingredient(s) the document is about (lower-case RxNorm names) |
| `mentioned_ingredients` | list of strings | other class ingredients named in the chunk text |
| `section` | string | label section (e.g. `drug_interactions`), abstract paragraph label (`results`), or `interaction` |
| `chunk_index` | int | position of the chunk in its document |
| `title` | string | context title (see above) |
| `text` | string | the chunk text |
| `source_name` | string | "openFDA drug label", "PubMed (<journal>)", "DDInter 2.0" |
| `url` | string | DailyMed page of the label (`drugInfo.cfm?setid=...`), PubMed page, DDInter |
| `date` | string | label `effective_time` (YYYYMMDD), publication year, DDInter release year |
| `version` | string | label `set_id` and version, `PMID <n>`, `DDInter 2.0` |
| `license` | string | licence of the source |
| `article_type` | string or null | PubMed only: research, review, meta_analysis, clinical_trial, case_report, commentary |
| `is_retracted` | bool | PubMed only (retracted publication or "RetractionIn"); false elsewhere |
| `language` | string | `en` |
| `retrieved_at` | string | when the source was downloaded (UTC, ISO 8601) |

## Versioning (DVC)

- `data/rag/raw.dvc`, `data/rag/interim.dvc`, `data/rag/manifest.json.dvc`: the downloaded
  labels and PubMed XML, the parsed tables, and the manifest (URL, SHA-256, size, licence,
  retrieval time of every raw file), tracked with `dvc add` (they come from the network, so
  they are not a pipeline stage).
- `dvc.yaml` stage `rag_corpus`: `python -m synergyrx_data.tujya corpus`, from
  `data/rag/interim`, `data/rag/raw/pubmed`, the two DDInter / ingredient tables and the
  chunker code to `data/rag/corpus` (`dvc.lock` records the hashes). `dvc repro rag_corpus`
  rebuilds the chunks when any input changes.
- As for the images, there is no DVC remote yet: the cache is local (docs/ROADMAP.md).
- Nothing under `data/` is committed to git (some sources forbid redistribution).

## Embeddings and index (backend, `backend/app/rag/`)

- **Embeddings: bge-m3** (1,024 dimensions, multilingual) through Ollama's `/api/embed`
  (D59). Each chunk is embedded as `title + "\n" + text`.
- **Store:** table `rag_chunks` in the PostgreSQL database `tujya` (the pharma database):
  the metadata columns of the schema above, `content_hash`, and `embedding vector(1024)`
  (pgvector); ingredient lists are `text[]`. Table `rag_index_meta` keeps the model, the
  dimension, the chunk count and `indexed_at`.
- **Indexing:** `python -m app.rag.index` (backend venv; `--database-url`, `--chunks`,
  `--pull` to pull the Ollama models first). Idempotent: a chunk is embedded again only if
  its hash (text, metadata, model) changed, chunks gone from the corpus are deleted, a new
  embedding model rebuilds the table. Run on 2026-10-28: 8,661 chunks embedded in 364 s on
  the RTX 2050; second run `{"embedded": 0, "unchanged": 8661, "seconds": 0.1}`.

## Retrieval (`backend/app/rag/retrieval.py`)

| Mode | How | Time per query (8,661 chunks, laptop) |
|---|---|---:|
| `bm25` | Okapi BM25 (k1 1.2, b 0.75) in memory over title + text; lower case, accents removed, English and French stop words, plural "s" stripped | ~6 ms |
| `dense` | bge-m3 query vector, exact cosine distance in SQL (`ORDER BY embedding <=> :q`), filters in the WHERE clause | 130-190 ms (mostly the query embedding) |
| `hybrid` (default) | both lists (50 candidates each) fused by reciprocal rank fusion, score = sum 1 / (60 + rank) | ~170 ms |

Loading the retriever (all chunk texts for BM25) takes 1.1 s, once; it reloads by itself
when `indexed_at` changes. No HNSW index: at this size the exact scan is fast and exact.

**Filters** (all modes): `ingredients` (chunks about OR mentioning one of them),
`doc_types` (label, abstract, interaction), `sections`, `exclude_retracted` (default
true). In SQL: `ingredients && :names OR mentioned_ingredients && :names`.

Try it: `python scripts/search.py "warfarin bleeding" --ingredient fluconazole --doc-type
label` (prints the three modes with their ranks and timings).

Which mode is best is measured in docs/BENCHMARK.md.

Tests (CI, no model): `backend/tests/test_rag.py` (SQLite, fake hashed embeddings: index
idempotence, model change, BM25, the three modes, filters, reload) and
`backend/tests/test_rag_pgvector.py` (the same on PostgreSQL + pgvector, in its own schema;
the CI backend job now has a `pgvector/pgvector:pg16` service).

## Tests

`tests/unit/test_tujya_rag.py` (respx, no network): label selection rule, dosage never
stored, connector run (one label per ingredient, 404 handled, raw file reused), PubMed
ingredient queries and their hits, chunk splitting and overlap, chunk metadata, stable ids,
dosage refused by the chunker.
