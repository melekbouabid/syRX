# Status

Five lines per phase, newest last. Everything was run on Windows 11, Python 3.14.2, on
2026-10-13. `make` is not installed on that machine, so the Makefile targets were run as
their underlying commands (`python -m pytest tests/unit`, `python -m ruff check src tests`,
`python -m mypy`).

Phase 1 was tested and linted on its own. The code of phases 2 to 4 was written together
and then tested as a whole, so their test counts below are taken from the final suite
(268 unit tests, all passing), not from separate runs after each phase.

## Phase 1: core collection (done)

1. HTTP stack, 14 connectors, manifest, state, Typer CLI and YAML config are in place.
2. Tests 120 passed with mocked HTTP, no network. `ruff check` clean, `mypy` clean (59 files).
3. Verified live: `fetch --source tdc --sample 1000` wrote a valid Parquet file with
   provenance columns; the second run made 0 requests. `--dry-run` left the tree untouched.
4. Found while verifying sources: DrugComb has no study column; PyTDC does not install on
   Python 3.14; the DepMap portal blocks automated clients; cancerrxgene.org is 410 Gone.
5. Open: NCI-ALMANAC needs a manual download (robots.txt); live runs need `CONTACT_EMAIL`.

## Phase 2: cleaning and data quality (done)

1. Pair keys, RDKit standardisation, drug identity with a review queue, DepMap mapping,
   separate score columns, conflicts, coverage, drift, licence registry, frontend export.
2. In the final suite: 21 cleaning tests, 17 build / report / export / CLI tests, 17 that
   check the data dictionary against the real columns. All pass; lint and types clean.
3. Run on the real data: 786,183 source rows became 689,081 records, 4,081 distinct drugs,
   112 cell lines (98 mapped to DepMap, 86 with expression), 304,497 combinations seen in
   more than one source, 14,462 of them in disagreement.
4. Real data exposed three bugs, now fixed and covered by tests: enantiomers merged by
   name, derived cell lines making their parent ambiguous, text columns changing type.
5. Open: 289 drugs have a structure (they carry 94 % of the measurements); the other 3,792
   need the full PubChem run (about 4,150 names, roughly 20 minutes).

## Phase 3: literature (done)

1. Europe PMC, PubTator, PubMed baseline, DDInter and DrugBank connectors; query builder
   with a query log; abstract store; sentence segmentation; offline mention detection;
   stratified annotation sample; citation export.
2. In the final suite: 16 connector tests with mocked HTTP, 19 literature tests and 3
   end-to-end tests of `update` and `sources --check`. All pass.
3. Verified live on a sample: 3 PubMed queries (6 requests) returned 174 distinct articles,
   Europe PMC enriched 60 of them, 289 candidate sentences, 262 in the annotation sample.
4. Found: `ftp.ncbi.nlm.nih.gov` disallows automated clients, so the optional baseline
   connector reads manually placed files; a sample run no longer advances PubMed's state.
5. Open: PubTator, DDInter, DrugBank, GDSC and the baseline are tested with mocks only.

## Phase 4: operations (done)

1. Live smoke tests (opt-in), Dockerfile, Makefile, pre-commit, README, data dictionary,
   cron and GitHub Actions examples, preflight, `summary`, `update`, `sources --check`.
2. Tests 268 passed, 7 live tests deselected by default. `ruff check`, `ruff format --check`
   and `mypy` clean. `pip check` clean. `pre-commit validate-config` passes.
3. `pytest tests/live -m live`: 7 passed against the real services. `sources --check`
   probed 37 links: 8 need attention, all explained in `docs/SOURCES.md`.
4. `validate` on the real data: every raw file matches its manifest hash; 0 invalid rows.
   Running a long download next to other commands exposed that `manifest.json` and the
   state file were overwritten whole; both now merge under a lock (fixed, with a test).
5. Not run here: `docker build` (Docker daemon was not running), `make` (not installed),
   pre-commit hooks (nothing is committed). The Dockerfile and Makefile are untested.

## After phase 4: one data folder (done)

1. On request, `exports/`, `reports/` and `annotation/` moved under `data/`: everything the
   pipeline produces is now in one folder. `manifest.json` and `LICENSES.md` stay beside it.
2. The three locations follow `data_dir` (and `--data-dir`); `reports_dir`, `exports_dir`
   and `annotation_dir` remain as optional overrides. See docs/DECISIONS.md, D24.
3. Existing files were moved, not rebuilt, and their `manifest.json` entries re-keyed.
4. Tests 270 passed (2 new: nothing is written outside `data/`; overrides work). Lint and
   types clean. `clean`, `literature`, `report`, `export` and `validate` re-run on the real data.
5. The web app is unaffected: it only reads `data/processed`.

## After phase 4: one folder per format (done)

1. On request, the generated files are grouped by format under `data/`: `data/parquet` (the
   processed tables), `data/csv`, `data/json` and `data/reports` (Markdown / HTML). This
   replaces `data/processed`, `data/exports` and `data/annotation` from the entry above.
2. `parquet_dir`, `csv_dir`, `json_dir` and `reports_dir` are the optional overrides now
   (the earlier `exports_dir` / `annotation_dir` keys are gone). See docs/DECISIONS.md, D24.
3. Existing files were moved and their `manifest.json` entries re-keyed. `data/raw` and
   `data/interim` are unchanged. The labelling guide is `data/reports/annotation_guide.md`.
4. Tests 270 passed, including one that checks each folder only holds its own format.
   Lint and types clean. `clean`, `literature`, `report`, `export`, `validate` re-run on
   the real data.
5. The web app's default table folder (`backend/app/config.py`) now points to `data/parquet`.
