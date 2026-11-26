# Data

Every dataset lives under `data/` and is versioned with DVC (`dvc.yaml`, `dvc.lock`), never with
git. Figures below come from the summary files the pipeline writes; each one is named next to
its table.

## Sources and licences

| Source | Used for | Licence | Redistribution |
|---|---|---|---|
| [NLM C3PI](https://data.lhncbc.nlm.nih.gov/public/Pills/) (Computational Photography Project for Pill Identification) | pill photos, detection and classification | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/), research and development use | cite NLM; images kept out of git |
| [ePillID benchmark](https://github.com/usuyama/ePillID-benchmark) (Usuyama et al., CVPR Workshops 2020) | overlap check with the literature only, never training | code MIT, images under the NLM / Pillbox terms | do not redistribute the images |
| [Italian-OTC-medicines](https://huggingface.co/datasets/mennox/Italian-OTC-medicines) (revision 29ebcd7) | OCR of medicine boxes | Apache-2.0 | allowed with attribution; kept in DVC |
| Wikimedia Commons box photos (11 files) | real-box test set of the OCR | CC0, public domain or CC BY-SA, one licence per photo (`data/raw/medicine_boxes/web_commons/labels.csv`) | attribution and share-alike |
| scikit-image samples + 3 generated images (19 files) | out-of-distribution check of the scanner | CC0 / public domain (`data/raw/off_topic/licences.csv`) | free |
| [DDInter 2.0](https://ddinter2.scbdd.com/) | drug-drug interactions and their severity | CC BY-NC-SA 4.0 | non-commercial, share-alike; not redistributed |
| [openFDA drug labels](https://open.fda.gov/apis/drug/label/) | RAG corpus (label sections) | CC0 (US public domain) | free; do not imply FDA endorsement |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) (NCBI E-utilities) | RAG corpus (abstracts) | NLM terms, abstracts may be under publisher copyright | research use, do not redistribute |
| [RxNav / RxNorm](https://lhncbc.nlm.nih.gov/RxNav/) | active ingredients of each pill (NDC -> RxCUI -> ingredients) | NLM terms | cite the source |
| [Tox21](https://moleculenet.org/) (MoleculeNet) and PubChem | bonus toxicity tool | public domain / open data | free |
| ANSM public medicines database, emc SmPC | 35 rows of the brand table `config/brand_aliases.csv`, each with its source | Licence ouverte (etalab), Datapharm terms | cited, not copied |

How to cite DDInter: Tian Y, et al. *DDInter 2.0: an enhanced drug interaction resource with
expanded data coverage, new interaction types, and improved user interface.* Nucleic Acids Res.
2025;53(D1):D1356-62. doi:10.1093/nar/gkae726.

## Pill images: C3PI v1 and v2

The C3PI archive holds 133,458 images of 4,974 products, 2.72 TB in total
(`data/processed/c3pi/bigdata/index/summary.json`). A product (NDC9 + part) becomes a class when
it has at least 20 consumer photos (`params.yaml` `bigdata.index.min_images`): 380 classes.

| Measure | Before (v1) | After (v2) |
|---|---:|---:|
| Classes | 150 | 380 |
| Images (no augmentation) | 4,206 | 16,789 |
| Consumer photos | 4,056 | 12,505 |
| Consumer photos per class (mean) | 27.0 | 32.9 |
| Consumer photos per class (min - max) | 24 - 30 | 20 - 100 |
| Augmented train images | 0 | 3,302 |
| Size (GB) | 4.77 | 37.88 |

Source: `data/processed/c3pi/bigdata/stats.json`, MLflow run `fd753192` (experiment
`tujya-data`). v2 images by role: 12,505 consumer photos, 3,040 segmented studio images, 870
challenge images and 374 reference images; the 150 classes of v1 are all in v2.

### Cleaning

The Spark job `quality` ([src/data/cleaning.py](../src/data/cleaning.py)) opens all 20,330
selected images and measures size, brightness, contrast, sharpness, SHA-1 and a 16 x 16
difference hash (`data/processed/c3pi/bigdata/quality/summary.json`):

| Reason | Images rejected |
|---|---:|
| `duplicate_file` (same SHA-1 as a kept image) | 2 |
| `too_dark` (mean grey level under 10) | 1 |
| corrupt, missing, too small, too bright, ambiguous duplicate of another class | 0 |
| **Total** | **3 of 20,330** |

10 pairs of near-identical photos of the same pill were kept and recorded, so the split keeps
each pair together.

### Split without leakage

C3PI gives no identifier of the physical pill photographed, so the unit kept together is the
**shooting session**: same package, camera, background, lighting and date. Sessions linked by a
near-identical photo or an identical file are merged (union-find), then each class's sessions
fill test (15 %) and val (15 %); the rest is train ([src/data/split.py](../src/data/split.py)).
`check_leakage` fails the job if any group, file or near-duplicate pair crosses splits.

| Split | Images | Leakage check | Result |
|---|---:|---|---:|
| train | 13,432 (+ 3,302 augmented) | files in train and val/test | 0 |
| val | 1,580 | sessions in two splits | 0 |
| test | 1,777 | near-duplicate pairs across splits | 0 |

Source: `data/processed/c3pi/bigdata/split/summary.json` (9,175 sessions). The split lists are
committed in [config/splits/c3pi_v2/](../config/splits/c3pi_v2/). Augmentation (rotation,
generated background, brightness, blur) reads **train images only** and brings 322 classes up to
40 train images each (`data/processed/c3pi/bigdata/augmented/summary.json`).

### Datasets built for training

| Dataset (DVC stage) | train | val | test | comparison split | Source |
|---|---:|---:|---:|---:|---|
| Detection, YOLO, one class `pill` (`prepare_detect_v2`) | 7,571 photos | 1,315 | 1,490 | 995 photos | `data/processed/c3pi/datasets_v2/detect/summary.json` |
| Classification, 256 px crops, 380 classes (`prepare_classify_v2`) | 15,424 crops | 1,461 | 1,713 | 1,131 crops | `data/processed/c3pi/datasets_v2/classify/summary.json` |

Boxes come from classical computer vision ([src/data/autolabel.py](../src/data/autolabel.py)):
C3PI gives no boxes; photos the auto-labeller flags as doubtful are left out. The **comparison
split** (`test_holdout.txt`) keeps only test sessions that the v1 models never saw, so v1 and v2
models are compared fairly ([module1/results.md](module1/results.md)).

### Big Data with PySpark

Six Spark jobs ([src/data/spark](../src/data/spark/)), one DVC stage each, run on the Spark
cluster of `docker-compose.yml`:

| Job | What it does |
|---|---|
| `index` | reads the 110 C3PI XML files with Spark's XML source, statistics per type and class, selection |
| `quality` | image checks and perceptual hashes (`mapInPandas`), banded self-join for near-duplicates |
| `split` | sessions, union-find of duplicates, stratified split, leakage check |
| `augment` | offline augmentation of the rare classes, from train images only |
| `epillid` | overlap with ePillID: 242 of the 380 v2 classes appear in it (`bigdata/epillid/summary.json`) |
| `rag` | RAG corpus: cleaning, exact and MinHash deduplication, chunking |

```bash
docker compose --profile spark up -d spark-master spark-worker
python -m src.data.spark.submit <job>      # or: dvc repro spark_<job>
python -m src.data.spark.log_stats         # before / after table above, logged to MLflow
```

## Medicine tables

[src/rag/extractor.py](../src/rag/extractor.py) (with the name rules of
[src/rag/metadata.py](../src/rag/metadata.py)) maps every pill class to its active ingredients
with RxNav and keeps the DDInter interactions of those ingredients
(`data/processed/pharma/`, loaded into PostgreSQL by `pharma-init`):

| Table | Rows |
|---|---:|
| Pill classes (`config/pill_classes.csv`) | 380 |
| Class -> ingredient links (`pill_ingredients.csv`) | 455 (176 distinct ingredients) |
| Interactions with any drug (`interactions_with_any_drug.csv`) | 75,653 |
| Interactions between two class ingredients (`interactions_between_classes.csv`) | 8,611 |

## RAG corpus

| Corpus | Documents | Chunks | Ingredients covered |
|---|---:|---:|---:|
| v1 (150 classes, collection `tujya_chunks`) | 4,002 | 8,661 | 82 |
| v2 (380 classes, collection `tujya_chunks_v2`, served) | 13,660 | 27,302 | 175 |

v2 by type: 171 openFDA labels (6,224 chunks), 4,878 PubMed abstracts (12,467 chunks) and 8,611
DDInter pairs (one chunk each); 7 near-duplicate abstracts removed by MinHash
(`data/processed/rag/corpus_v2/summary.json`). Chunks are at most 1,500 characters with one
sentence of overlap; dosage sections are never stored.

## Collection pipeline

[src/data/sources](../src/data/sources/) collects the external sources politely. It is
installed by `pip install -e .` and exposes `tujya-data`:

```bash
tujya-data preflight                         # disk, permissions, identity (CONTACT_EMAIL in .env)
tujya-data fetch --source pubmed             # download, parse, validate
tujya-data sources --check                   # probe every link, rewrite docs/data_sources.md
tujya-data status
```

Source access rules, checked by [tests/data/sources/unit/test_policy.py](../tests/data/sources/unit/test_policy.py):

1. Use the official API when there is one, within its documented limits; else an official bulk
   file with a pinned version and checksum. HTML is a last resort and needs a written reason.
2. Every request goes through `PoliteClient`: declared hosts only, robots.txt, rate limits,
   retries, circuit breaker, disk cache.
3. Never bypass a login, CAPTCHA or bot check, never send cookies or a browser User-Agent,
   never use proxies, never commit anything under `data/`.
4. Licence-gated or non-commercial sources (DDInter) are disabled until their terms are
   acknowledged.

Links and rate limits of each provider: [data_sources.md](data_sources.md) (generated).

## Data dictionary

One section per table the pipeline writes, one row per column.
[tests/data/sources/unit/test_data_dictionary.py](../tests/data/sources/unit/test_data_dictionary.py)
fails if a column is produced but not described here, or described but no longer produced.
Every table ends with the same five provenance columns:

| Column | Type | Meaning |
|---|---|---|
| `source` | string | Connector the row comes from |
| `source_url` | string | Where the data was retrieved from (secrets removed) |
| `retrieved_at` | string | UTC time of retrieval, ISO 8601 (`2026-10-13T18:04:11Z`) |
| `source_version` | string | Version of the source: release, record or file id |
| `license` | string | Licence of the source data |

Missing values are nulls, never zeros or empty strings. Rows that fail their schema go to
`<table>.rejects.parquet` next to each table.

## data/external/interim/ddinter/interactions.parquet

One row per interacting pair of drugs and severity level, across the DDInter category files.

| Column | Type | Meaning |
|---|---|---|
| `drug_a_name` | string | First drug, as DDInter names it |
| `drug_b_name` | string | Second drug |
| `level` | string | Severity: `Major`, `Moderate`, `Minor` or `Unknown` |
| `drug_a_source_id` | string | DDInter id of the first drug (`DDInter1`) |
| `drug_b_source_id` | string | DDInter id of the second drug |

## data/external/interim/pubmed/articles.parquet

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

## data/external/interim/pubmed/query_hits.parquet

Which query found which article. `query_log.jsonl`, next to it, lists every query sent.

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | Stable id of the query (hash of its kind and ingredient) |
| `pmid` | string | PubMed identifier of an article the query returned |
| `kind` | string | `ingredient` (one active ingredient of `config/pill_classes.csv`) |
| `pair_key` | string | Always null for ingredient queries; kept so older query logs read the same |

## data/external/interim/openfda_label/label_sections.parquet

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

`config/data_pipeline/manifest.json`: `files` maps each project-relative path to one entry.

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
