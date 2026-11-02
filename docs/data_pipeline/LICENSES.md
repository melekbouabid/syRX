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

## How to cite

- **ddinter**: Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded data coverage, new interaction types, and improved user interface. Nucleic Acids Res. 2025;53(D1):D1356-62. doi:10.1093/nar/gkae726
- **pubmed**: Sayers E. A General Introduction to the E-utilities. In: Entrez Programming Utilities Help. Bethesda (MD): NCBI; 2010-. NBK25497
- **openfda_label**: U.S. Food and Drug Administration. openFDA drug labeling API (https://open.fda.gov/apis/drug/label/). Labels from DailyMed (NLM).

## Notes that affect how the data may be used

- **ddinter**: Clinical interaction severity (Major / Moderate / Minor), not synergy.
- **pubmed**: Queries are built from the active ingredients of config/pill_classes.csv.
- **openfda_label**: Tujya RAG corpus. One label per ingredient; dosage sections are never stored. Without an API key openFDA allows 240 requests per minute and 1,000 per day.
