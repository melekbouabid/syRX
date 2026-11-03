# Data licences

Licences of the data the pipeline collects, from the connector registry (`SourceInfo`) and
`manifest.json`. Written by hand since the `report` step of the original pipeline was removed (decision
D85): change it together with the connector's `SourceInfo`.

This file covers the **data** the pipeline collects, not the pipeline's own code. Raw data is
never committed (`data/` is git-ignored). Before publishing a model, a dataset or a figure,
check the row of every source that contributed to it, and cite each one.

| Source | Licence | Terms | Enabled | Collected | Redistribution |
|---|---|---|---|---|---|
| **ddinter**: DDInter 2.0 drug-drug interactions | [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | [terms](https://ddinter2.scbdd.com/terms/) | no | 14 file(s), last 2026-10-27 | Non-commercial and share-alike. Personal, informational or scholarly use only; never ship it in a commercial product and do not redistribute. |
| **pubmed**: PubMed (NCBI E-utilities) | NLM data terms; abstracts may be under publisher copyright (research use, do not redistribute abstracts) | [terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html) | yes | 3 file(s), last 2026-10-13 | Do not redistribute raw files; cite the source. |
| **openfda_label**: openFDA drug labels (Structured Product Labeling) | [CC0 1.0 (openFDA terms; labels are US public domain)](https://open.fda.gov/license/) | [terms](https://open.fda.gov/terms/) | no | no | Public domain (CC0); openFDA asks not to imply FDA endorsement. |

## Image datasets (module 1, not collected by the pipeline)

Downloaded by `python -m src.data.data_cli` (C3PI) or by hand (ePillID), tracked with DVC in
`data/raw/`, never committed. Data card: [docs/module1/big_data.md](../module1/big_data.md).

| Source | Licence | Terms | Collected | Redistribution |
|---|---|---|---|---|
| **c3pi**: NLM Computational Photography Project for Pill Identification (C3PI), https://data.lhncbc.nlm.nih.gov/public/Pills/ | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) (catalog.data.gov) | Research and development use only | 2026-10-28 (v1, 4,206 files), 2026-11-03 (v2, 20,330 files) | Share-alike database licence: cite NLM, keep the licence, do not present derived data as the original. Tujya keeps the images out of git. |
| **epillid**: ePillID benchmark (Usuyama et al., CVPR Workshops 2020), https://github.com/usuyama/ePillID-benchmark, release ePillID_data_v1.0 | Code: MIT; data: derived from NIH NLM C3PI / Pill Image Recognition Challenge and NIH Pillbox images (their terms apply) | Research use, as a benchmark | 2026-11-03 (ePillID_data.zip, SHA-256 b960b3b1...11ac) | Do not redistribute the images; cite the paper and the NLM sources. Used as a benchmark only, not for training. |

## How to cite

- **ddinter**: Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded data coverage, new interaction types, and improved user interface. Nucleic Acids Res. 2025;53(D1):D1356-62. doi:10.1093/nar/gkae726
- **pubmed**: Sayers E. A General Introduction to the E-utilities. In: Entrez Programming Utilities Help. Bethesda (MD): NCBI; 2010-. NBK25497
- **openfda_label**: U.S. Food and Drug Administration. openFDA drug labeling API (https://open.fda.gov/apis/drug/label/). Labels from DailyMed (NLM).
- **c3pi**: U.S. National Library of Medicine. Computational Photography Project for Pill Identification (C3PI). https://catalog.data.gov/dataset/computational-photography-project-for-pill-identification-c3pi
- **epillid**: Usuyama N, Delgado NL, Hall AK, Lundin J. ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification. CVPR Workshops 2020. arXiv:2005.14288

## Notes that affect how the data may be used

- **ddinter**: Clinical interaction severity (Major / Moderate / Minor), not synergy.
- **pubmed**: Queries are built from the active ingredients of config/pill_classes.csv.
- **openfda_label**: Tujya RAG corpus. One label per ingredient; dosage sections are never stored. Without an API key openFDA allows 240 requests per minute and 1,000 per day.
