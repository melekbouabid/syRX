# Decisions

Choices made where the brief was open or where reality differed from it. Each entry says
what was decided and why, so it can be revisited.

## Compliance

**D1. A contact email is required for live requests.** With `http.require_contact_email:
true` (the default) every request is refused until `CONTACT_EMAIL` is set in `.env`. During
the build there was no `.env` yet (the brief says to add it afterwards), so verification
runs used `--set http.require_contact_email=false` and sent the User-Agent
`synergyrx-data/0.1.0 (research data pipeline; contact not configured)`. Those runs were
kept small: single bulk files and a few dozen API calls.

**D2. robots.txt is checked for bulk downloads too, not only for HTML.** The brief requires
it before HTML fetches. Applying it to file downloads as well is stricter and is what
surfaced D4. Documented REST APIs (E-utilities, PUG-REST, ChEMBL, Europe PMC, PubTator,
Figshare, Dataverse) are governed by their API terms and rate limits instead; switch
`http.robots_for_bulk` off only with a reason. The parser follows RFC 9309 (wildcards, `$`,
longest match) because Python's `urllib.robotparser` gets Zenodo's rules wrong.

**D3. A `Crawl-delay` in robots.txt always slows the client down** (Zenodo 10 s, NCI wiki
15 s), even when the configured rate is faster. All limits also run at 90 % of their value
(`http.safety_factor`) so that clock jitter cannot push a burst over a provider's limit.

**D4. NCI-ALMANAC is a manual download.** Its files are attachments on `wiki.nci.nih.gov`
and that host's robots.txt disallows `/download/`. The connector checks robots.txt on every
run: if NCI ever allows the path it downloads by itself, until then it asks for the file
under `data/manual/nci_almanac/`. While exploring, before the robots.txt files had been
read, about seven small requests were made to that path (two text files and 64-byte ranges
of the zip), one page request to `dtp.cancer.gov` and one README request to
`ftp.ncbi.nlm.nih.gov`, both of which serve `Disallow: /`. None was repeated once the rules
were known, and the pipeline itself never requests those paths.

**D4b. The PubMed baseline is a manual download too.** `ftp.ncbi.nlm.nih.gov` serves
`Disallow: /` in robots.txt, even though NLM documents it as the bulk download site. The
two statements conflict; the stricter one is followed. The optional baseline connector
reads files placed under `data/manual/pubmed_baseline/`. If the team decides NLM's download
documentation is the applicable policy, `http.robots_for_bulk: false` restores automatic
download, for every bulk source at once, so it should be a deliberate decision.

**D5. DepMap comes from Figshare+, not the portal.** `depmap.org/portal` answers automated
clients with a "verifying you are a person" page and asks not to be scraped. That is an
access control, so it is not worked around. Every DepMap release is also published on
Figshare+ with a documented API, MD5 checksums and a CC BY 4.0 licence; release 24Q4 is
pinned (`article_id: 27993248`).

**D6. The Zenodo record metadata endpoint is not called.** Zenodo's robots.txt disallows
`/api` except `/api/records/*/files`. The file name, size and MD5 of DrugComb v1.4 are
therefore pinned in the connector, and only the allowed file URL is requested.

**D7. GDSC reads the Cell Model Passports model list.** `cancerrxgene.org` answers 410 Gone
and points to Cell Model Passports. The connector is optional and disabled; its licence
could not be verified automatically, so the registry says so instead of guessing.

**D8. DDInter needs an explicit acknowledgement; DrugBank is never requested.** DDInter is
CC BY-NC-SA 4.0, so the connector refuses to download until `terms_acknowledged: true`.
The DrugBank connector has no network code at all: it reads one user-supplied file.

**D9. Access walls are reported, never retried or bypassed.** 401 / 403, a CAPTCHA or a
login page raise `AccessControlError`, which is not retried and opens the source's circuit
breaker at once. The client sends no cookies, refuses `Authorization`, `Cookie` and
forwarding headers, has no proxy pool, and rejects a User-Agent that imitates a browser.

**D10. The client only talks to hosts a connector declares.** A request to any other host
is refused before it leaves. This is what keeps the pipeline a set of source connectors
rather than a general scraper.

**D11. No connector parses HTML.** Every source turned out to have an API or a bulk file.
HTML pages are only probed for their status by `sources --check`. The policy is enforced by
`docs/CODE_REVIEW.md`, the pull-request template and `tests/unit/test_policy.py`.

## Data

**D12. `synergy_pairs` has one row per pair, cell line, source and study.** The brief asks
for one row per drug A, drug B and cell line, with source and study columns, and also that
scores are never merged silently. Both hold only if records from different sources stay on
separate rows. Replicates *within* one study are averaged, with `n_replicates` and `*_std`
kept. `n_sources`, `n_records` and `has_conflict` make it easy to pick one row per triplet
once the project has chosen a score.

**D13. A drug is its standardised structure.** Two names are the same drug when their
desalted, neutralised structures share a full InChIKey; stereoisomers stay distinct. A drug
without a valid structure keeps a name-based id (`name:<key>`), stays in the table with
`smiles_valid = false`, and goes to the review queue. Metal complexes written as separate
fragments (carboplatin) keep all fragments, otherwise the metal would be stripped as a salt.

**D14. Confidence is a fixed scale, not a model.** 1.0 manual override or dataset SMILES
confirmed by PubChem; 0.9 single PubChem hit by name; 0.8 dataset SMILES alone; 0.6 several
PubChem hits (first taken); 0.5 dataset and PubChem disagree; 0.0 unresolved. Anything
below 0.7, or with a stated problem, is queued in `data/csv/drug_review_queue.csv`.
Decisions are fed back through `config/manual_drug_overrides.csv`.

**D14b. Names differing only in a stereo sign are different drugs.** `(+)-JQ1`, `(-)-JQ1`
and `(+/-)-JQ1` are kept apart when matching names. Found on real data: the two
blebbistatin enantiomers had been merged by a key that dropped the signs.

**D14c. Lookups go to the most-measured drugs first.** PubChem and ChEMBL are asked for
drug names in order of how many measurements depend on them, so a partial or `--sample`
run resolves the drugs that matter most.

**D14d. Cell lines are matched to DepMap in tiers.** `StrippedCellLineName` first, then the
display name, then the CCLE-name prefix, and a later tier is only used when the earlier ones
find nothing. Without this, derived lines (`A375_SKIN_CJ1`) made their parent ambiguous.
Six spellings are mapped through `config/cell_line_aliases.csv` (OVCAR3, U251, SR, 786-0,
LNCAP, MSTO); related-but-different lines (MDA-MB-435 vs 435S, EFM192B vs EFM-192A) are
deliberately left unmapped for a person to decide.

**D15. CSS and ALMANAC's ComboScore get their own columns.** They are not Bliss, Loewe, HSA
or ZIP, so mapping them onto one of those would be exactly the silent merge the brief
forbids. CSS measures efficacy, not synergy.

**D16. DrugComb has no study column.** The Zenodo table lacks `study_name` and
`tissue_name`, so `study_id` is null for DrugComb and its overlap with O'Neil, ALMANAC and
TDC is detected by pair and cell line. 709,187 of its 1,175,220 rows are single-drug
screens (second drug `NA`) and are skipped, which was checked against the raw file.

**D17. `--sample` shares the data tree but is labelled everywhere.** Sample raw files are
named `*.sampleN.*`; the manifest, the connector state, `status`, `summary` and the quality
report all flag sample data. A sample of a CSV is its first N rows, fetched with a range
request, so sampling a 193 MB file costs a few kilobytes; TDC's pickle cannot be read
partially, so it is downloaded whole and N rows are drawn with a fixed seed.

**D18. TDC is loaded through PyTDC when installed, and through a restricted loader
otherwise.** PyTDC 1.1.15 pins `numpy<2`, `pandas<3` and `rdkit<2024.3.1` and does not
install on Python 3.13+. The file is always downloaded by this pipeline's client from the
Dataverse address PyTDC uses, into the folder PyTDC reads. Without PyTDC the pickle is read
by an unpickler that only allows plain pandas / numpy constructors, because a pickle can
otherwise run arbitrary code.

**D19. PubMed updates are incremental by entry date.** Each query remembers the last date
searched; the next run asks from two days before that, and articles are deduplicated by
PMID keeping the newest copy (which carries late retraction flags). ESearch goes through
the history server and is never cached, because a cached WebEnv would have expired.

**D20. Text mining is offline and rule-based.** Sentence segmentation and drug-mention
detection use rules and the synonym table: no model download, reproducible, auditable.
Synonyms that are ordinary words ("gold", "lead") or that name two drugs are dropped.

**D21. The mock-prediction seed contains no invented value.** It holds a random seed, the
real score distributions and real measured examples; the frontend generates placeholders
from those.

## Engineering

**D22. The project root is the folder the prompt was run in**, not a nested
`synergyrx-data/` directory: the brief says to start in an empty project folder, and that
folder is the editor's workspace.

**D23. Three dependencies beyond the brief's list:** `python-dotenv` (read `.env` without
mutating the environment), `pytest-asyncio` (async tests) and `hatchling` (build backend).
PubMed XML is parsed with the standard library; the bundled expat limits entity expansion.

**D24. Everything the pipeline produces is under `data/`, grouped by format.** The brief
puts `exports/`, `reports/` and `annotation/` at the project root and the tables in
`data/processed`. On request this became one folder with one sub-folder per format:
`data/parquet` (the processed tables), `data/csv` (coverage, conflicts, source agreement, the
two review queues, the annotation sample), `data/json` (frontend files and the JSON Lines
copy of the annotation sample) and `data/reports` (the quality report in Markdown and HTML,
the annotation guide). The extension of a file tells you its folder. `data/raw` and
`data/interim` keep their per-source structure: they are working layers, the same file name
occurs in several sources, and the manifest tracks them by path. All four output folders
follow `data_dir` (also with `--data-dir`); `parquet_dir`, `csv_dir`, `json_dir` and
`reports_dir` can point one of them elsewhere. In `manifest.json` the `layer` of an entry
(`processed`, `report`, `export`) still says what a file is, whatever folder it is in.
None of it is committed: the brief requires that for `data/`, and the derived files come
from sources with redistribution limits (PubMed abstracts, unlicensed O'Neil data).
`manifest.json` and `LICENSES.md` stay at the project root: they hold no data, describe the
folder, and can be committed.

**D25. `make` is not installed on the build machine (Windows).** The Makefile targets are
one command each; the equivalent commands were run directly and are listed in the README.

## Tujya (night of 2026-10-27 to 2026-10-28)

Decisions taken while the owner was away. Each one is listed in docs/NIGHT_REPORT.md for
review.

**D26. pandas is capped below 3.1.** A fresh install pulled pandas 3.1.0 and three end-to-end
tests failed (`TypeError: data type 'list<item: string>[pyarrow]' not understood` in the
build step). With pandas 3.0.6, the version of the original environment, all 270 unit tests
pass. `pyproject.toml` now says `pandas>=2.2,<3.1`; making the build work on 3.1 is a
separate fix.

**D27. A new pipeline venv `.venv-dev`.** The committed workspace's `.venv` was created on
another machine (`C:\Users\LENOVO\...`) and cannot start. It was left untouched; `.venv-dev`
(Python 3.14, `pip install -e ".[dev]"`) is used instead and is git-ignored (`.venv-*/`).
The backend venv only lacked its dev requirements, which were installed.

**D28. Documentation language.** The README presentation, requirement checklist and
progress journal, and docs/NIGHT_REPORT.md are in French as requested; the other docs stay
in English like the rest of `docs/`.

**D29. C3PI metadata comes from the ALLXML files, not the Excel directory.** The XML has
everything the directory has plus file size and SHA-1 (exact download budget, verified
downloads), camera, background and lighting. 316 files are listed twice (always for the
same pill); each file is counted once.

**D30. A pill class is a product = NDC9 + part**, not an NDC11. The last two NDC digits only
name the package, so package variants of one pill are merged (380 classes with 20+ consumer
photos instead of 364 by NDC11).

**D31. "Consumer images" = `C3PI_Test` JPGs; "reference images" = the RxNav API JPG.** The
consumer photos are what the app will see (phone / webcam, real backgrounds). Of the
reference types, only `MC_API_RXNAV_V1.3` (~0.6 MB) is downloaded, one per package NDC,
and kept in train only. CR2 (23 MB), calibrated PNG (110 MB) and segmented PNG (3 MB) are
too large for the night's bandwidth; the segmented PNGs (alpha mask) are noted for
synthetic detection data later. WMV videos are not used.

**D32. 150 classes, at most 30 consumer photos each, two cameras left out.** The download
budget (20 GB) was not the limit; the connection was (about 1 MB/s, so ~3.5 GB per hour).
The top 200 products weigh 17.8 GB, mostly because of the Canon G10 and Nikon S3000
(~4.5 MB per photo: 28 % of the photos, 60 % of the bytes). Without those two cameras the
remaining seven (phones, webcams, compact cameras, ~1.1 MB per photo) are also closer to
what the app will receive. Result: 150 classes (the middle of the requested 100-200), 24 to
30 photos per class (mean 27), 4,206 files, 4.8 GB. The two cameras can be added back
later with `--exclude-camera` removed and a larger budget.

**D33. Split by photo, 70 / 15 / 15, stratified by class, saved in `config/splits/`.** The
split list is small, hand-checkable and must be versioned with the code that trains on it,
so it is in git (`config/splits/selected_images.csv`), while the images are in DVC.
Duplicated files (same SHA-1) always share a split; reference images are train only. The
same physical pill appears in several photos (C3PI photographed each pill many times), so
train and test share pills but never photos; ePillID makes the same choice.

**D34. Ingredients come from RxNav, by NDC.** C3PI has no RxNorm code. NDC -> RxCUI
(`ndcstatus`, which also resolves obsolete NDCs) -> base ingredients (`historystatus`).
Base ingredient (IN), not the salt, because DDInter uses base names. Strengths are dropped
(no doses in Tujya). 149/150 classes mapped; the multivitamin class has no ingredient.

**D35. DDInter is enabled by a Tujya step, not in the shared config.** Two existing tests
pin DDInter as opt-in with unacknowledged terms in `config.yaml`; changing the shared
default would have meant weakening them. `python -m synergyrx_data.tujya ddinter` runs the
same connector with the terms acknowledged (owner's decision, non-commercial use) and all
14 ATC files (the connector default lists 8; C and N were missing).

**D36. Interaction descriptions only restate DDInter.** The bulk files have no mechanism or
management text and DDInter's pages are not scraped (D9, no HTML scraping). The table's
`description` says "DDInter 2.0 lists an interaction between X and Y with severity Z.";
explanations will come from cited openFDA label passages in the RAG step.

**D37. Contact email.** No `.env` exists and the owner's address was not given for use in a
User-Agent, so RxNav and DDInter were called with `http.require_contact_email=false`, as in
D1 (about 400 RxNav requests and 14 DDInter files).

**D38. Detection boxes are made automatically from the consumer photos.** C3PI has no
boxes; public pill detection sets have unclear licences and other images. Each consumer
photo shows one or two pills on a plain-ish background, so `vision/tujya_vision/autolabel.py`
finds them with classical computer vision (robust background surface, colour distance,
coin rejection, GrabCut for pale pills). Photos with a doubtful flag are left out of both
datasets rather than labelled badly. The detector learns, and is scored on, machine-made
labels; docs/VISION.md gives the audited precision. A hand-checked subset is a later step.

**D39. Identification is trained on pill crops, not whole photos.** The pill often covers
less than 5 % of the photo; shrinking the whole photo to 224 px would leave a few pixels of
pill. The classifier sees square crops (box + 20 % margin, 224 px), which is also what it
receives in the app after detection (two-stage design).

**D40. PyTorch: CPU build first, CUDA build when the bandwidth allowed.** The CUDA wheel is
~2.5 GB; it was downloaded only after the images. The smoke run uses whatever
`torch.cuda.is_available()` reports; the hardware is logged as MLflow tags.

**D41. The registered models are the ONNX exports.** `mlflow.onnx.log_model` registers the
deployment artefact (what the API will load with ONNX Runtime); `best.pt` is kept as a run
artefact for retraining. The export is only trusted after an ONNX Runtime vs PyTorch check
on test images (top-1 agreement / box IoU logged as metrics).

**D42. Ultralytics' automatic package installs are disabled.** During the end-to-end
evaluation Ultralytics installed `onnxruntime-gpu` 1.31.1 into `vision/.venv` by itself
(about 3 minutes, a few hundred MB). The venv now has both `onnxruntime` and
`onnxruntime-gpu` (it works; CUDA and TensorRT providers are listed). It was left as it is,
because removing either package can break the other; a clean rebuild from
`vision/requirements.txt` gives the CPU runtime only. `tujya_vision/__init__.py` sets
`YOLO_AUTOINSTALL=false` so dependencies only change through requirements.txt.

**D43. The datasets are a DVC stage; Ultralytics caches are ignored.** `dvc.yaml` stage
`prepare` records the datasets built from the tracked images (`dvc.lock`); `*.cache` files
that Ultralytics writes next to the labels are in `.dvcignore` because they are rebuilt on
any machine.

**D44. Swagger lives at `/docs`, the old `/api/docs` redirects.** The course requires
Swagger at `/docs`. FastAPI now serves the UI at `/docs` and the schema at `/openapi.json`;
`/api/docs` answers 307 to `/docs` and `/api/openapi.json` returns the same schema, so old
links keep working. Both are outside `/api`, so the Vite dev proxy and the frontend's
`nginx.conf` forward them explicitly. The Swagger UI page loads its JavaScript and CSS from
FastAPI's default CDN; a demo without Internet would need them served locally.
