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

**D45. The API runs its own numpy/OpenCV pipeline around ONNX Runtime.** PyTorch and
Ultralytics would add about 2 GB to the backend image for pre- and post-processing only.
`backend/app/scanner.py` re-implements Ultralytics' letterbox and NMS and the training crop
(20 % margin, 224 px, OpenCV INTER_AREA), reads the class names from the ONNX metadata, and
runs both models with ONNX Runtime. Parity with Ultralytics was measured on the test photos
(docs/VISION.md); a change in the training code (crop margin, image size) must be mirrored
there.

**D46. `/api/scan`, `/api/pills` and `/api/interactions` need a signed-in account.** Same
rule as the catalog: an open inference route can be used to load the server, and the demo
does not need one because the first start creates the demo account (`user@gmail.com`). The
demo therefore starts on the sign-in page. A public demo route can be added later if the
defence needs it (it would need a rate limit).

**D47. Confirmation threshold 0.85, chosen on the validation photos.** Rule: the lowest
threshold at which at most 1 % of all validation photos would be accepted with a wrong pill
(the table is in docs/VISION.md). 0.85 accepts 67 % of validation photos directly (98.5 %
correct) and 70.5 % of test photos (99.3 % correct). The value is a setting
(`SYNERGYRX_SCAN_CONFIRM_THRESHOLD`). The app shows the confidence and the top-5 for every
pill anyway, confirmed or not.

**D48. The pharma loader drops and recreates its five tables.** The scan needs the generic
name, a new `pills` column; `create_all` does not alter existing tables. The tables hold
nothing but what the loader writes from versioned CSV files, so a full rebuild is the
simplest correct refresh. Reloaded into PostgreSQL on 2026-10-28: same counts (150 pills,
1,716 ingredients, 161 links, 40,508 interactions).

**D49. ONNX Runtime threads do not busy-wait.** With the default settings, the validation
run took more than 25 minutes instead of 3: after each inference ONNX Runtime's threads keep
spinning and starve OpenCV's resize threads on the shared laptop CPU (decoding one JPEG went
from 40 ms to 1.3 s). The sessions now set `session.intra_op.allow_spinning=0`;
`SYNERGYRX_ONNX_THREADS` sets the thread count (0 = one per core).

**D50. Pill names are shown as C3PI labels them, strength included.** C3PI product names
contain the strength ("AMRIX CAP 30MG"), which helps identify the product. This is the
label of the product, not advice on how much to take; Tujya still never gives doses
(posology), and the ingredient table has no strengths (docs/INTERACTIONS.md). To review: the
names could be shown without the strength if the jury reads it as a dose.

**D51. The API's models are copied from the Model Registry as single ONNX files.**
`python -m tujya_vision.publish` downloads the latest registered versions (or given ones),
merges MLflow's external weight file into one `.onnx` per model, and writes `models.json`
(version, run id, SHA-256). The API returns that manifest with every scan, so each answer
says which model versions produced it. `models/` is git-ignored; it is mounted into the
backend container (Task 5).

**D52. The demo gallery is eight hand-picked test photos.** `config/demo_gallery.csv` lists
C3PI consumer photos of the **test** split (the API refuses any other): five pills whose
ingredients have Major interactions with each other (tramadol, citalopram, fluconazole,
haloperidol, hydroxychloroquine), cyclobenzaprine, one trazodone photo whose top-1 is below
the 0.85 threshold (to show the confirmation step), and the multivitamin (no known
ingredient). They were chosen with the per-photo results of `scripts/evaluate_scan.py
--split test`, so the gallery shows the model at its best on purpose, except for the
trazodone photo; the honest numbers are the full test-split results (docs/VISION.md). The
images are served from the DVC-tracked folder, never committed to git.

**D53. The app stays in English; the safety texts are in French, word for word.** The
existing site is in English. The project rules give the safety texts in French
("Information uniquement, consultez un pharmacien ou un médecin.", "Aucune interaction
connue dans cette base de données."), so the API returns them in French and the Scanner page
shows them unchanged rather than translating them. A full French interface is a separate
choice for later.

**D54. Browser proof with Playwright, screenshots as small JPEGs.** `playwright` 1.64.1 is a
pinned frontend dev dependency (npm does not download a browser; `npx playwright install
chromium` does, once). `frontend/scripts/screenshots.mjs` signs in, uploads a test photo,
scans two demo photos, checks the interactions, opens the webcam (Chromium's fake camera)
and Swagger, and fails if a step does not show what it should. The seven screenshots in
`docs/screenshots/` are JPEG (quality 82, 33-125 kB) to keep the repository small.

**D55. The images carry code only; data, models and config are mounted read-only.** C3PI
(research use), DDInter (CC BY-NC-SA, no redistribution) and the models stay out of the
images, which can therefore be pushed to a registry (Task 6). docker-compose.yml mounts
`config/`, `data/tujya`, `data/parquet`, `data/vision/c3pi/images` and `models/` read-only;
the app's project root is `/` in the image, so the default paths of `app/config.py` match
the mounts without extra settings. A one-shot `pharma-init` service rebuilds the medicine
tables on every start (a few seconds) and the backend waits for it, so `docker compose up`
is the only command. In Docker the accounts live in PostgreSQL (database `tujya`), so
`backend/app.db` is never used or touched by the containers. Ollama is not in the compose
file yet: it comes with the Module 2 agent.

**D56. CD = images on GitHub Container Registry after a green CI, no deployment.** There is
no server to deploy to, so delivery stops at versioned images: `cd.yml` runs on
`workflow_run` of `ci` (success, push to main) and builds the commit CI tested
(`head_sha`), smoke-tests each image, then pushes `latest` and `sha-<commit>` with the
workflow's `GITHUB_TOKEN` (`packages: write`). A deployment step (a server pulling
`latest`) can be added later without changing the images. Packages stay private until
their visibility is changed by hand (docs/CD.md).

**D57. Ollama runs only as a Docker service, on the GPU.** Ollama is not installed on the
host; `docker-compose.yml` has an `ollama` service (official image pinned to
`ollama/ollama:0.40.12`, models in the named volume `ollama`, port 11434) with the NVIDIA GPU
reserved through `deploy.resources.reservations.devices` (Docker Desktop + WSL2). Checked on
2026-10-28: `docker compose exec ollama nvidia-smi` shows the RTX 2050 (4 GiB) and the Ollama
log reports `library=CUDA compute=8.6 ... total="4.0 GiB" available="3.2 GiB"`. A machine
without a GPU uses `docker compose -f docker-compose.yml -f docker/ollama/cpu.yml up -d`
(the override removes the reservation with `!reset`). `OLLAMA_MAX_LOADED_MODELS=2` keeps the
LLM and the embedding model loaded together, `OLLAMA_KEEP_ALIVE=30m` avoids reloading between
questions.

**D58. LLM: qwen2.5:3b (Q4_K_M), chosen on speed in 4 GB of VRAM and French quality.** Four
small instruction models were pulled in the container and timed on the same short French
question ("En deux phrases, explique ce qu'est une interaction médicamenteuse entre deux
principes actifs.", temperature 0, context 4,096, model already loaded, nothing else loaded):

| Model | Size in VRAM | Placement | Output tokens | Speed | Answer time |
|---|---:|---|---:|---:|---:|
| **qwen2.5:3b** | 2.2 GB | 100 % GPU | 76 | 47.4 tok/s | **1.7 s** |
| llama3.2:3b | 2.9 GB | 80 % GPU / 20 % CPU | 101 | 32.0 tok/s | 3.2 s |
| gemma3:4b | 2.9 GB | 55 % GPU / 45 % CPU | 67 | 11.8 tok/s | 12.1 s |
| qwen3:4b | 2.9 GB | 82 % GPU / 18 % CPU | 1,259 (thinking) | 14.0 tok/s | 89.9 s |

All three non-thinking models answered correctly in good French; qwen2.5:3b is the only one
that fits entirely on the GPU, is the fastest, and supports Ollama tool calling (needed by the
agent). qwen3:4b spent its time "thinking" in English (the generate API's `think: false` was
not applied to this model); llama3.2 and gemma3 spill to the CPU. Mistral 7B (4.1 GB) would
not fit at all. `SYNERGYRX_LLM_MODEL` switches the model without code changes. To review: a
3B model is weak at reasoning; answers are kept short, grounded in retrieved sources and
checked by the safety node (D63).

**D60. One openFDA label per ingredient, chosen by a fixed rule; dosage sections never
stored.** openFDA has dozens of labels per ingredient (generics, repackagers). The connector
takes the 25 most recent prescription labels (then any product type if none matches) and
keeps, in order: a label about the ingredient, single-ingredient before combination,
prescription before OTC, the most indexed sections, the most recent `effective_time`. The
most recent label alone was often a repackager copy without "drug interactions". Result:
81 prescription + 1 OTC (fexofenadine); pseudoephedrine and vitamin B6 only exist in
combination labels. The candidates are saved with each label for audit. Dosage and
administration, dosage forms and strengths, overdosage, how supplied, package label,
clinical pharmacology, pharmacokinetics and clinical studies are dropped before the raw file
is written (refused again by the schema and the chunker), so no dosing text is ever indexed.
Amounts still appear in other sections (study doses in adverse reactions): the agent's safety
node removes dose sentences from its answers.

**D61. PubMed: up to 20 relevance-sorted abstracts per ingredient, on interactions and
adverse effects.** The existing PubMed connector gained a query mode `ingredients` and an
optional `sort` (default unchanged, so SynergyRx runs are untouched): one query per
ingredient, English abstracts since 2000, `sort=relevance`, at most 20 records (1,640 at
most; 1,570 distinct articles). A larger bound would make abstracts dominate the labels in
retrieval and take longer to embed; every query hit the cap, so this is a sample of the
literature, not all of it. Retracted articles are kept and flagged, and excluded at search
time by default. The RAG sources run in their own data folder (`data/rag`, own manifest),
so they never mix with the SynergyRx PubMed records.

**D62. DDInter in the text index: only the 2,350 pairs between the class ingredients.**
DDInter rows only say "A and B interact, severity X". The 40,508 pairs with any drug would
be 80 % of the corpus in near-identical sentences and push labels and abstracts out of the
results; the `check_interaction` tool reads every pair from the database instead. The RAG
corpus is versioned with DVC: sources with `dvc add` (`data/rag/raw`, `interim`,
`manifest.json`), chunks as the `rag_corpus` stage of `dvc.yaml`.

**D59. Embeddings: bge-m3 (1,024 dimensions) served by the same Ollama container.** The
corpus is English and half of the questions are French, so the model must be
multilingual; bge-m3 (568M parameters, 100+ languages, 8,192-token context) is the
reference multilingual retrieval model. In Ollama it takes 0.7 GB of VRAM, so it stays
loaded next to qwen2.5:3b (2.2 GB) on the 4 GB GPU (`ollama ps`: both 100 % GPU), and the
backend needs neither PyTorch nor a model download (it calls `/api/embed`). Measured: the 8,661
chunks are embedded in 364 s (24 chunks/s); a French and an English sentence with the same
meaning have cosine 0.87, an unrelated English sentence 0.53. Smaller multilingual models
(multilingual-e5-small, paraphrase-multilingual) would be faster but weaker; the machine
(31.7 GB RAM, 4 GB VRAM) does not need them. Retrieval stores vectors in pgvector with exact
cosine search (no HNSW index: at 8,661 rows the exact scan takes milliseconds and is
exact); BM25 is computed in the backend over the same chunks (Okapi, k1 1.2, b 0.75, French
and English stop words) and both lists are fused by reciprocal rank fusion (k = 60).

**D63. The agent's safety rules are code, not only prompt.** A 3B model does not follow
every instruction, so the rules of docs/project_rules.md are enforced after generation by a `safety`
node (`app/rag/safety.py`), sentence by sentence: dose sentences removed (lab values such as
"mL/min" or "mg/dL" excepted), "safe"/"sûr"/"sans danger" sentences removed and replaced by
"aucune interaction connue dans cette base de données" when nothing was found, uncited timing
advice removed, out-of-range citations removed, an uncited sentence gets the source that
contains at least 60 % of its words (else the list of consulted sources is added), no source
means no answer, the disclaimer always ends the answer. A question asking for a dose is
refused before any LLM call. While streaming, unsafe sentences are held back. Limits: the
rules are pattern-based (French and English), so a dose written in another way could pass,
and a correct sentence with "sûr" in another sense ("bien sûr") is removed; the benchmark
measures what remains (docs/BENCHMARK.md).

**D64. Tool choice: the LLM plans, the graph guarantees the minimum.** The planner node
uses Ollama tool calling (qwen2.5:3b) with three tools (check_interaction, get_drug_info,
search_documents); unknown tool names are ignored and the graph adds `check_interaction`
when two or more drugs are named or confirmed, and `search_documents` always, so a weak or
failing planner never leaves the answer without sources. `identify_pills` is not offered to
the chat model (a chat question carries no image; the app sends the confirmed pills) but
is exposed by the MCP server. Each call is returned with `chosen_by` (`llm` or `rule`), so
the UI and the benchmark show which ones the model chose.

**D65. Retrieval default: dense (bge-m3) with the ingredient filter, not hybrid.** On the
83 benchmark questions with expected chunks, dense + ingredient filter has Recall@5 0.928
and MRR 0.852; hybrid (RRF) + filter 0.904 / 0.796; BM25 + filter 0.783 / 0.660. BM25 only
matches English words, so French questions (half of the set) lose with it (BM25 Recall@5
0.59 in French, 0.98 in English) and it pulls the fusion down; on English questions alone
hybrid is slightly better (MRR 0.944 vs 0.937). The setting is `SYNERGYRX_RAG_SEARCH_MODE`
(bm25, dense or hybrid); BM25 and hybrid stay available to tools and MCP clients. Not
tuned further (language-aware mode, query translation) to avoid fitting the system to its
own test set.

**D66. Benchmark built from the corpus by rules, then checked by hand.** 88 questions (44
French, 44 English): 24 interactions, 12 contraindications, 10 boxed warnings, 10
mechanisms, 10 adverse reactions, 8 warnings, 9 PubMed findings, 5 safety probes. Expected
chunks come from rules on the chosen labels (docs/BENCHMARK.md), so another relevant chunk
counts as a miss (scores are a lower bound). Recall@k is the hit rate (at least one expected
chunk in the top k), the usual RAG definition; the share of expected chunks found is
reported as Coverage@10. Every candidate was read: 14 interaction references and the 9
abstract references were rewritten by hand, references never contain a dose. Limits:
templated questions, one author for the system and the set.

**D67. Generation judged by a different local model: gemma3:4b; RAGAS on a subset.** The
generator (qwen2.5:3b) judged its own answers too strictly in a trial (faithfulness 0 for a
near-verbatim answer: "the context does not explicitly state ...") and llama3.2:3b returned
unparseable output (RAGAS NaN). gemma3:4b gave sensible scores on the same trial and is a
different model family, so it judges both the citations (one yes/no question per cited
sentence, all answers) and RAGAS 0.4.3 (faithfulness, answer relevancy with bge-m3, context
precision with the reference). RAGAS makes ~10 judge calls per question and gemma3 only
half fits on the 4 GB GPU, so RAGAS runs on a stratified subset of 8 answered questions
(one per question type, both languages), with the judge's output capped at 700 tokens
(runs on 30 then 12 questions were stopped: more than 2 hours, one call took 4 minutes). RAGAS 0.4.3 imports a module removed from
langchain-community 0.4; the benchmark script stubs that one unused import. A 4B judge is
far weaker than the GPT-4-class judges RAGAS is designed for: the scores compare runs of
this system, they are not absolute.

**D68. Two runtime fixes found by the benchmark.** (1) With `num_ctx` 8,192, qwen2.5:3b
(2.3 GB) and bge-m3 no longer fit together on the GPU and Ollama evicted one for the other
on every question (12 s per question, and one stalled request); `SYNERGYRX_LLM_NUM_CTX` is
now 4,096 (agent prompts are ~1,300 tokens) and the embedding context 2,048 (chunks are at
most ~400 tokens): 4.7 s per question on average. (2) One answer looped for more than
24,000 tokens (no length limit): answers are now capped at 450 tokens
(`SYNERGYRX_LLM_MAX_TOKENS`) with a repeat penalty of 1.1. Also added: one retry of a failed
embedding call, a citation example in the answer prompt, and the meaning-based attribution
of uncited sentences (bge-m3 cosine at least 0.65 and 0.03 above the next source; the
citation judge measures whether these citations hold).

**D69. Strictly local: the app runs only on this PC (session 4, 2026-10-28).** The owner's
rule (now in docs/project_rules.md): no public deployment, no tunnel, no published image of the app, no
service bound to 0.0.0.0 on the host; the GitHub repository stays public for the code and
the docs. Every compose port is published on 127.0.0.1, Vite listens on 127.0.0.1, and the
MCP server's HTTP transport refuses any non-loopback `--host` (argparse choices, so a typo
fails loudly instead of silently opening the server). Processes inside the containers keep
listening on 0.0.0.0, which is only the container's private interface. Replaces the
"deployment to a server" item of the roadmap: there will be none.

**D70. CD builds and smoke-tests the images but no longer publishes them (replaces D56).**
Consequence of D69: an image on ghcr.io is a way to run the app elsewhere. On every green
push to main, `cd.yml` builds the backend and frontend images of the tested commit and
smoke-tests them (ports on the runner's 127.0.0.1), with `contents: read` only. Pushing to
ghcr.io survives only as a separate `publish` job, run by hand (`workflow_dispatch` with
`push_images=true`, default false), the only job with `packages: write`; kept because the
owner asked for it as an option. The two packages published before (tujya-backend,
tujya-frontend, private, code only) could not be deleted from here: the local gh token has
no `delete:packages` scope and changing the owner's login was out of scope; the manual steps
are in docs/CD.md.

**D71. DVC remote on a local folder outside the repository.** Default remote `localstore`
= `C:\Users\msi\dvc-storage\tujya` (the owner's choice for this session; consistent with D69, nothing leaves the
PC). Its absolute path is in the committed `.dvc/config`, so a clone on another machine
must add its own remote (`dvc remote add -d --local ...`); acceptable since the project
runs only here. It is not a backup (same disk as the repository): the report asks the
owner to copy it to an external disk. Google Drive and S3 commands are documented in
docs/INFRA.md, not configured: either would put licensed data (DDInter CC BY-NC-SA, C3PI
research use) on a cloud service, which must then stay private.

**D72. Host-side clients use 127.0.0.1, not localhost.** Found while rehearsing the demo
(session 4, task 4): after D69, Docker publishes its ports on IPv4 loopback only. On this
PC `localhost` resolves to `::1` first; browsers and curl fall back to IPv4 at once, but
libpq waited 65 to 100 s per connection, so the MCP smoke test took 2 min 34 s (11 s with
127.0.0.1). Default URLs in code (Ollama, MLflow tracking) and every connection string in
the docs now use 127.0.0.1. Browser addresses (`http://localhost:8080`) are unchanged. The
alternative, also publishing on `[::1]`, would double every port line for no gain.

**D73. Tox21 data: MoleculeNet copy, licence not stated.** The bonus toxicity model uses
`tox21.csv.gz` from DeepChem's bucket (MoleculeNet; 7,831 molecules, 12 assays). No
explicit licence was found for this file or on the Tox21 challenge page; the data come from
US federal agencies (NCATS, NIEHS/NTP, EPA, FDA) and are publicly released at no cost.
Treated as public research data: cited (Huang 2016, Wu 2018), used for a non-commercial
student project, versioned with DVC and never committed. Ingredient structures come from
PubChem (public domain), committed as `config/ingredient_smiles.csv` with three lookup
aliases (valproic acid, cyanocobalamin, pyridoxine) because PubChem's best match for the
pill names was an ion or a coenzyme. To revisit if the owner finds an explicit licence.

**D74. Toxicity tool: model, split and safety design.** Morgan r2 2,048-bit fingerprints
and one LightGBM per task (simple, CPU, 77 s), scaffold split 80/10/10 so the test set
measures generalisation to new chemotypes: mean test ROC-AUC 0.725 (valid 0.766). One or
two molecules, each predicted separately, never a statement about the mixture. The wording
"prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique" is on every
output. In the chat, the real agent showed that qwen2.5:3b misreads the probabilities
(called warfarin "not identified as toxic" with SR-MMP at 0.83; called fluconazole toxic
from warfarin's source): every answer sentence about the prediction is therefore held back
and replaced by a fixed sentence generated by the code from the tool output, listing the
assays at probability >= 0.5 (a reading aid, not a validated threshold). New safety rule 6
removes "non-toxic" claims from any answer. Applicability domain: nearest-neighbour
Tanimoto < 0.3 flagged (heuristic). Version 1 of `tujya-tox21` is an identical run that
crashed after registering (console encoding); it is kept and marked FAILED rather than
deleted, version 2 is served.

**D75. Synergy bonus: one source per experiment, ZIP target, success rule fixed in advance.**
An experiment is (unordered drug pair, cell line). All 379,670 experiments are in DrugComb
v1.4; TDC is an exact copy of DrugComb (r = 1.00 on Loewe and ZIP for 286,674 shared
experiments) and O'Neil disagrees with it (r = 0.415 on Loewe, 22,737 shared, ZIP not
given): **DrugComb is the single source**, TDC and O'Neil rows are dropped. Target:
**ZIP** (DrugComb's own score, combines the Loewe and Bliss references, centred near 0;
Loewe is strongly negative-skewed in DrugComb, mean -9.0). Only experiments where both drugs
have a valid structure (287 drugs, 323,339 experiments). Split: **leave-drug-out**, a
random 15 % of the drugs is held out for the test set (every experiment with at least one
of them) and another 10 % for validation; reported separately for one and two unseen drugs.
Decision rule, written before any training: `predict_synergy` is added **only if** the
LightGBM model has a test RMSE at least 10 % below the mean baseline (training mean) and a
Pearson r of at least 0.3 on the leave-drug-out test set; otherwise the result is
documented and no tool is added.

## Restructuration en deux modules (copie locale tujia2, 2026-10-29)

Les décisions D1 à D75 sont gardées telles qu'elles ont été écrites (en anglais) ; les
suivantes, prises pendant la restructuration, sont en français.

**D76. Copie locale isolée de l'original.** La restructuration se fait dans une copie
(`C:\Users\msi\Desktop\tujia2`, robocopy sans les environnements virtuels ni les caches),
sans remote git (`git remote remove origin`), pour ne rien changer au dépôt GitHub ni au
dossier d'origine. Le projet Docker Compose s'appelle `tujia2` (conteneurs et volumes
séparés) et les images sont étiquetées `tujia2-*` : sans ce second point, un
`docker compose up --build` dans la copie aurait remplacé les images `tujya-backend:local`
et `tujya-frontend:local` de l'original, qui ne correspondent pas au même code. Mêmes ports,
liés à `127.0.0.1` : un seul des deux projets tourne à la fois. Le remote DVC est partagé
avec l'original : `dvc push` est interdit dans la copie (`dvc pull` et `dvc status` sont
permis). Tag local `v1.0-avant-restructuration` posé avant tout changement.

**D77. Le pipeline SynergyRx dans `data_pipeline/`, racine du projet inchangée.** Le code
(`src/`), ses tests, `pyproject.toml`, `config.yaml`, `manifest.json`, `LICENSES.md`, son
`Dockerfile` et son `Makefile` passent dans `data_pipeline/`, mais `data/` (DVC) et
`config/` restent à la racine, partagés avec l'application et les modules. Le
`config.yaml` livré déclare donc `project_root: ..`, désormais **résolu par rapport au
fichier** et non au dossier courant (changement de `load_settings`, testé), et range les
registres du pipeline dans son dossier (`manifest_path`, `licenses_path`,
`sources_doc_path`). On lance le pipeline depuis `data_pipeline/` ; les étapes Tujya
retrouvent sa configuration depuis la racine (étape DVC `rag_corpus`). Les docs encore
vivantes du pipeline (dictionnaire des données, vérifié par un test ; sources, réécrit par
`sources --check` ; règles d'accès, citées par un test) vont dans `data_pipeline/docs/` ;
seuls le plan et l'état de SynergyRx vont dans `docs/archive/`. Les règles ruff communes
passent dans un `ruff.toml` à la racine (seul ajout à la racine hors de la liste prévue),
que chaque paquet étend. L'image du pipeline se construit depuis la racine
(`docker build -f data_pipeline/Dockerfile .`) avec une liste blanche
`data_pipeline/Dockerfile.dockerignore`.

**D78. Incident pendant la restructuration : des tests ont écrit dans `data/` de la copie.**
Premier passage des tests du pipeline après D77 : la fixture `settings` passe
`project_root=tmp_path`, mais `load_settings` donnait la priorité à la valeur du fichier
trouvé dans le dossier courant (le nouveau `config.yaml` avec `project_root: ..`) : 47 tests
ont échoué et ont écrit des données de test dans le vrai `data/` de la copie. Réparation :
fichiers suivis par git (`config/pill_classes.csv`, `manifest.json`, `LICENSES.md`) remis
depuis HEAD ; `data/parquet` (DVC) remis depuis le cache DVC local (`dvc checkout`,
`dvc status` propre) ; les 164 fichiers non versionnés touchés (32 fichiers de travail du
pipeline SynergyRx dont l'empreinte ne correspond plus au manifeste, 132 fichiers d'état, de
cache HTTP et de test) **déplacés**, pas supprimés, dans `data/_quarantine_2026-10-29/`
(liste et raison dans son `README.txt`) : le pipeline les retéléchargera au lieu de lire des
données de test. Aucune donnée utilisée par l'application ou les modules n'est perdue (tout
est suivi par DVC ou git). Les originaux des 32 fichiers sont dans le dossier d'origine,
auquel la copie ne touche pas. Correctifs : un `project_root` explicite l'emporte sur le
fichier, et la fixture charge explicitement le `config.yaml` livré au lieu de dépendre du
dossier de lancement de pytest (tests de non-régression ; vérifié ensuite qu'aucun fichier
hors des dossiers temporaires n'est écrit).

**D79. Paquets des modules importés depuis la racine, une seule installation pour l'API.**
Les dossiers des modules sont des paquets Python (`module1_vision_mlops`,
`module2_ai_engineering`, `bonus`) : chaque import dit à quel module appartient le code
(`from module1_vision_mlops.serving.scanner import PillScanner`). Le paquet FastAPI garde
son nom `app` (`app/backend/app`, convention FastAPI, aucun import à réécrire dans le
cœur). `app/backend/pyproject.toml` ne déclare aucune dépendance (elles restent dans
`requirements.txt`, utilisé aussi par l'image) : son installation éditable
(`-e .` dans `requirements-dev.txt`, hatchling `dev-mode-dirs`) ajoute au `sys.path` du venv
du backend le dossier `app/backend` et la racine du dépôt, sans manipulation de `sys.path`
dans les scripts. L'image Docker fait de même avec `PYTHONPATH` et reprend la disposition du
dépôt sous `/tujya` (les montages compose deviennent `/tujya/config`, `/tujya/data/...`,
`/tujya/models`) ; son contexte de build est la racine, filtré par une liste blanche
`app/backend/Dockerfile.dockerignore` (contexte de 200 Ko). Les fixtures du web
(`app`, `user`, `pharma`, ...) passent dans le paquet `app/backend/tujya_testing/`, chargé
comme plugin pytest par chaque suite : le nom `tests` ne convient pas, car `python -m pytest`
met le dossier courant en tête du `sys.path` et chaque module a son propre dossier `tests`.
Les tests de chaque module se lancent depuis son dossier avec le venv du backend
(`pytest.ini` du module). Le code d'entraînement reste un projet séparé
(`module1_vision_mlops/training`, paquet `tujya_vision`, son venv avec PyTorch). Son
`prepare.py` a changé d'une ligne (racine du dépôt un niveau plus haut) : `dvc commit -f
prepare` met à jour l'empreinte dans `dvc.lock` sans relancer l'étape, `dvc status` propre.

**D80. Module 2 : ce qui a bougé, ce qui reste dans le pipeline.** `corpus/` reçoit les
étapes Tujya du pipeline (`synergyrx_data.tujya` : principes actifs RxNav, interactions
DDInter, notices openFDA, résumés PubMed, découpage en chunks), lancées par
`python -m module2_ai_engineering.corpus` ; `rag/` reçoit l'index, le stockage pgvector, la
recherche, les embeddings et le rapprochement des noms ; `agent/` le graphe LangGraph
(`agent.py` renommé `graph.py`, le paquet s'appelant déjà `agent`), les outils, les règles
de sécurité et leur câblage ; `mcp_server/` le serveur MCP (`server.py`, lancé par
`python -m module2_ai_engineering.mcp_server`) ; `benchmark/` le jeu de questions, les
résultats et les scripts `bench_*.py`, devenus des modules (`python -m
module2_ai_engineering.benchmark.bench_retrieval`, plus de `sys.path.insert`). Les
**connecteurs** openFDA et PubMed restent dans le pipeline (`data_pipeline/src/
synergyrx_data/connectors/`) : ils sont enregistrés dans son registre de sources et
configurés par son `config.yaml`, au même titre que les douze autres ; les déplacer ferait
dépendre le pipeline du module 2. Le corpus dépend donc du pipeline, jamais l'inverse ; il
est typé par le mypy du pipeline et soumis au même test de politique réseau
(`test_policy.py` couvre aussi `module2_ai_engineering/corpus`, sinon ce contrôle aurait été
perdu au déménagement). Les fixtures du pipeline deviennent le plugin
`synergyrx_data.testing` (même raison que `tujya_testing`, D79). Les tests du module 2
tournent avec le venv du backend, où le pipeline est installé en éditable. Les tests
pgvector passent dans un job CI `module2` qui a le service PostgreSQL. `make_spec.py`
écrivait dans un chemin absolu du dossier d'origine : il écrit maintenant à côté de lui
(régénéré à l'identique).

**D81. Le bonus dans `bonus/`, hors des deux modules notés.** L'outil de toxicité
(`app/toxicity.py` devient `bonus/toxicity/model.py`, avec `train_tox21.py` et
`fetch_ingredient_smiles.py`) et l'étude de synergie (`bonus/synergy/train_synergy.py`)
quittent l'API et ses scripts. La route `/api/toxicity` reste dans `app/backend` (elle
importe `bonus.toxicity.model`), comme l'outil `predict_toxicity` dans l'agent et le
serveur MCP. Les scripts deviennent des modules (`python -m bonus.toxicity.train_tox21
train`) ; le modèle MLflow journalise désormais le dossier `bonus` comme code
(`code_paths`). Les versions déjà enregistrées (`tujya-tox21` v2) et le paquet publié
`models/tox21` (fichiers texte LightGBM, aucun pickle) ne dépendent pas de ce chemin. Le
message « aucun modèle publié » donne la nouvelle commande, et le test qui le vérifie suit.
Un job CI `bonus` lance ses tests.

**D82. Rangement de la documentation.** Chaque module a son dossier `docs/` (ses documents
techniques et son rapport, Markdown + PDF) ; `docs/` à la racine garde ce qui concerne tout
le projet (architecture, API, infra, CD, démo, questions du jury, feuille de route,
décisions, captures). Noms en minuscules partout, sauf `README.md` et les rapports datés de
`docs/journal/` (`NIGHT_REPORT.md`, `SESSION_REPORT_N.md`, `JOURNAL.md`), noms donnés dans la
consigne de restructuration et cohérents entre eux. Chaque dossier de documentation range
ses sources Mermaid dans `diagrams/` et ses images dans `assets/` (le schéma d'architecture,
commun aux deux rapports, est dans `docs/`) ; `build-reports.mjs` traite les trois dossiers.
Les liens de tous les documents ont été recalculés (chaque lien résolu depuis l'ancien
emplacement du fichier, puis réécrit depuis le nouveau ; aucun lien cassé). Dans les
documents vivants, les chemins cités sont aussi mis à jour ; les **documents historiques**
(rapports de session, journal, décisions D1 à D75, archive SynergyRx) gardent leur texte
d'origine, avec une note et un renvoi vers la table de correspondance de
`docs/journal/SESSION_REPORT_5.md`. Le journal de progression quitte le README pour
`docs/journal/JOURNAL.md`. `photo.png` (source du logo) devient
`docs/assets/logo-source.png`. Le fichier non suivi `weights/yolo26n.pt` (téléchargement
automatique d'Ultralytics) rejoint les autres poids dans `module1_vision_mlops/training/`.

**D83. Langue des documents, MLflow vide dans la copie, captures inchangées.** Les README et
les documents vivants sont en français (ceux des modules, du bonus, `api`, `infra`, `cd`,
`roadmap`, `architecture`) ; le code, ses commentaires et ses messages restent en anglais.
Restent dans leur langue d'origine : les documents historiques (rapports de session,
décisions D1 à D75, archive SynergyRx) et les documents de référence du pipeline
(`data_pipeline/docs/` : `sources.md` est généré par `sources --check`,
`data_dictionary.md` est lu par un test qui compare ses sections aux colonnes écrites).
Les volumes Docker de la copie (`tujia2_*`) sont neufs, comme demandé : le serveur MLflow de
la copie fonctionne mais ne contient aucune expérience ni aucun modèle enregistré (ils sont
dans les volumes `tujya_*` de l'original, non partagés). Les modèles servis viennent du
dossier `models/` copié, donc le scan et la toxicité marchent ; le script de captures MLflow
échoue logiquement (« experiment tujya-vision not found »). Les scripts Playwright ont été
rejoués vers un dossier temporaire et les captures de `docs/screenshots/` n'ont pas été
remplacées : les nouvelles montrent les mêmes écrans, les remplacer n'aurait ajouté que du
bruit binaire.

## Conventions Git et outillage (2026-10-31)

**D84. ESLint dans un paquet à part, portée du hook de secrets.** Le frontend se construit
avec TypeScript 7, qui ne fournit plus d'API JavaScript ; typescript-eslint en a besoin pour
lire le code et n'accepte que TypeScript < 6.1 (dernière version et canary au 31/10). ESLint
est donc installé dans son propre paquet, `app/frontend/eslint/` (TypeScript 6.0 pour
l'analyse seulement), lancé par `npm run lint` depuis `app/frontend` ; la vérification des
types reste `tsc`, et le build ne change pas. Ce dossier est exclu de l'image Docker. Le hook
de secrets (`tools/hooks/check_secrets.py`) suit le TP Git avancé (fichiers Python), étendu
aux fichiers YAML, TOML et shell ; il ignore les dossiers `tests/`, dont les identifiants
factices sont voulus (même exception que les règles S105/S106 de ruff), et une ligne marquée
`pragma: allowlist secret`. Il n'est pas lancé sur le TypeScript : ses libellés de formulaire
(`password: "Password"`) seraient pris pour des mots de passe. Le contrôle des messages de
commit est refait par la CI sur les commits poussés, qu'un `git commit --no-verify` ne peut
pas contourner.

## Fichiers inutiles (2026-11-01)

**D85. Retrait de la partie SynergyRx que Tujya n'utilise pas.** Le professeur a relevé
« plusieurs fichiers et dossiers inutiles ». Un inventaire des 345 fichiers suivis (chaque
dépendance vérifiée dans les imports, `dvc.yaml`, `docker-compose.yml` et les workflows) a
séparé ce que l'application, l'entraînement, les tests, le RAG, la CI et la démonstration
utilisent de ce qui ne servait plus que l'ancien objectif (synergie de médicaments
anticancéreux). Ont été retirés, en quatre commits :
- des fichiers isolés : `config/medicines.csv` (en-tête seul), le `Makefile` et le
  `Dockerfile` du pipeline (ni `make` ni cette image ne servent : ni la CI, ni la CD, ni
  docker compose), ses exemples de planification, `docs/archive/` et le modèle de pull
  request SynergyRx ;
- le catalogue des médicaments et des lignées cellulaires de l'application (routes
  `/api/catalog/*` et `/api/public/summary`, pages Drugs et Cell lines) ; les pages
  d'accueil, de connexion et d'introduction décrivent maintenant Tujya, avec ses règles de
  sécurité et l'avertissement « information only » ;
- l'étude de synergie du bonus (résultat négatif, D75 : aucun outil `predict_synergy`) et
  le pointeur DVC `data/parquet.dvc` (le cache et le remote DVC n'ont pas été touchés) ;
- dans le pipeline, douze connecteurs (DrugComb, TDC, O'Neil, NCI-ALMANAC, DepMap, GDSC,
  PubChem, ChEMBL, Europe PMC, PubTator, DrugBank, PubMed baseline), le nettoyage, `build.py`
  (commandes `clean`, `literature`, `report`, `export`), l'export frontend, l'extraction de
  phrases et le rapport de qualité, avec leurs tests et deux fichiers de `config/`.

Le pipeline garde ce que le corpus du module 2 utilise : le client HTTP poli, le manifeste,
la provenance, la détection de dérive, la vérification des liens et trois connecteurs
(DDInter, PubMed, openFDA). PubMed ne garde que le mode `ingredients` : les modes `pairs` et
`drugs` lisaient les tables de synergie. `literature/abstracts.py` reste (le connecteur
PubMed en lit le XML) ; la structure des requêtes (`kind`, `pair_key`) ne change pas, pour
que les journaux de requêtes déjà versionnés gardent le même schéma. Les tests du code
retiré sont partis avec lui ; ceux du code gardé qui passaient par les données de synergie
(PubMed, CLI, mise à jour de bout en bout, dictionnaire des données) ont été réécrits sur
DDInter et PubMed avec les mêmes vérifications. RDKit et PyTDC ne sont plus des dépendances
du pipeline (RDKit reste dans le venv du backend, pour la toxicité). `LICENSES.md`, qui
était généré par `report`, est tenu à la main ; `docs/sources.md` a été régénéré hors ligne
depuis le dernier relevé des liens. Rien n'est perdu : tout reste dans l'historique git
(tag `v1.0-avant-restructuration`). Le nom du paquet `synergyrx_data`, la commande
`synergyrx` et le préfixe de configuration `SYNERGYRX_` ne changent pas ici (ce serait un
renommage, pas une suppression).

## Arborescence du guide des modules (2026-11-02)

**D86. Le dépôt suit l'arborescence imposée par le guide, pour les deux modules.** Le guide
du module 1 demande de respecter « scrupuleusement » `src/{data,models,utils}`,
`backend/{app,tests}`, `frontend/`, `data/{raw,processed}`, `notebooks/`, `params.yaml`,
`dvc.yaml` et un rapport à la racine, et le professeur a précisé qu'il s'applique aussi au
module 2. Les dossiers par module de D79 à D82 (`module1_vision_mlops/`,
`module2_ai_engineering/`, `app/`, `bonus/`, `data_pipeline/`) sont donc remplacés, avec
`git mv` uniquement (historique conservé), en un commit par bloc :
- `src/data` : préparation C3PI (`c3pi`, `data_cli`, `autolabel`, `prepare`), corpus RAG
  (`corpus/`) et pipeline de collecte (`synergyrx_data`, renommé `tujya_data`) ;
  `src/models` : entraînement, évaluation, publication, et le modèle Tox21 du bonus
  (`toxicity/`) ; `src/rag`, `src/agent`, `src/mcp_server`, `src/benchmark` ; `src/utils`
  attend les utilitaires des étapes suivantes (MLflow, Grad-CAM) ;
- `backend/` : l'API, dont la chaîne de scan ONNX Runtime devenue `app/inference.py` (le
  guide nomme `app/{main,schemas,inference}.py`) ; ses tests, dont ceux du scan ;
  `backend/tujya_testing` garde les fixtures partagées (plugin pytest, faux RAG) ;
- `frontend/` : `src/components` (UI, icônes, mises en page, animations) et `src/services`
  (client de l'API, session) ;
- `tests/` à la racine : tests de `src/` rangés comme lui (`data`, `models`, `rag`, `agent`,
  `mcp_server`) ; une seule configuration pytest (`pyproject.toml`) et un seul `conftest.py` ;
- réglages et manifeste du pipeline dans `config/data_pipeline/`, documentation dans
  `docs/` (`module1/`, `module2/`, `bonus/`, `data_pipeline/`, `app.md`) ;
- un seul venv Python pour l'API, `src/` et tous les tests (`.venv` à la racine,
  `requirements-dev.txt`, installation éditable du `pyproject.toml` racine) ; l'entraînement
  garde son venv (`.venv-train`, PyTorch et Ultralytics, `src/models/requirements-train.txt`).

Données : `dvc move` (mêmes empreintes, rien à envoyer au remote) range les images et
métadonnées C3PI dans `data/raw/c3pi`, Tox21 dans `data/raw/tox21` et les tables des
médicaments dans `data/processed/pharma` ; les sorties des étapes `prepare` et `rag_corpus`
vont dans `data/processed/c3pi` et `data/processed/rag/corpus` (`dvc commit -f`, sorties
inchangées). Exception : l'espace de collecte des sources RAG reste dans `data/rag`
(`raw/`, `interim/`, `manifest.json`), car son manifeste, suivi par DVC, enregistre ces
chemins ; le réécrire créerait un contenu qu'on ne peut pas envoyer au remote partagé (pas de
`dvc push`). Il sera déplacé quand la collecte sera refaite (étape 1.4). `params.yaml`
arrive avec les étapes d'entraînement (étape 2.1), le rapport LaTeX `rapport_technique.pdf`
à l'étape 4.2, et les workflows `ci-tests.yml` / `docker-build.yml` à l'étape 1.3.

Renommage : les noms hérités de SynergyRx deviennent ceux de Tujya (paquet `tujya_data`,
commande `tujya-data`, nom `tujya-data` dans le User-Agent et l'outil NCBI, préfixe de
configuration `TUJYA_*`). L'ancien préfixe `SYNERGYRX_*` est encore lu en repli, pour les
`.env` existants (testé) ; les décisions D1 à D85 gardent les anciens noms (historique).

## Workflows du guide (2026-11-03)

**D87. Deux workflows, `ci-tests.yml` et `docker-build.yml`, remplacent `ci.yml` et
`cd.yml` ; aucune image publiée.** Le guide du module 1 nomme ces deux fichiers et demande
flake8, black --check, ESLint, pytest et le build des images multi-stage avec un test de
fumée. `ci-tests.yml` reprend tous les jobs de l'ancien `ci.yml` (ruff, mypy, pytest de
`src/` avec PostgreSQL + pgvector, pytest de `backend/`, hooks et messages de commit,
ESLint et build du frontend) et ajoute flake8 et black dans un job `lint`. Ruff reste
l'outil principal ; flake8 (`.flake8`) et black (`[tool.black]` du `pyproject.toml`)
utilisent sa longueur de ligne (100). Black aurait remis en forme 7 fichiers d'une façon
que `ruff format` défaisait : ces passages ont été réécrits pour que les deux outils donnent
la même mise en page (expressions de message sorties des `assert`, en-têtes HTTP dans une
variable), sans changer le comportement ; les 4 lignes de plus de 100 caractères que ruff
tolérait (expressions régulières, `# type: ignore`) sont coupées, la regex des doses reste
identique (vérifié). Deux exceptions flake8 copient celles de ruff : `make_spec.py` (E501)
et `inference.py` (E402, variable d'OpenCV posée avant l'import de cv2). Dans pre-commit,
flake8 et black ne font que vérifier (ruff corrige). `docker-build.yml` ne dépend plus de la
CI (`workflow_run`) : il tourne sur chaque push de branche et chaque pull request, en
parallèle, pour qu'une branche soit vérifiée avant sa fusion ; il valide aussi
`docker-compose.yml` et attend que le `HEALTHCHECK` de chaque image soit `healthy`. Le
Dockerfile du backend devient multi-stage comme celui du frontend (`build` installe les
dépendances dans un venv, `runtime` ne copie que ce venv et le code). Le job manuel
`publish` (envoi sur ghcr.io à la demande, gardé par D70) est retiré : l'étape demande
qu'aucune image ne soit publiée et docs/project_rules.md interdit de publier les images de
l'application ; le workflow n'a que `contents: read`. Les images s'appellent `tujia2-*:ci`
(nom du projet Compose). Le test de fumée interroge `/api/health` : la route `GET /health`
du guide arrive avec l'API de l'étape 2.5.

## Big Data avec PySpark (2026-11-03)

**D88. Plan de données de l'étape 1.4 : C3PI v2, 380 classes candidates, 39,5 Go, ePillID
en benchmark.** Demande du professeur (plus de classes, plus d'images par classe,
augmentation). Sources fixées par le propriétaire du projet : C3PI (serveur NLM) et ePillID
seulement ; budget de 50 Go de nouvelles données au plus, au moins 30 Go libres sur C: après
téléchargement, REFERENCE_SEG abandonné en premier si nécessaire, jamais
`MC_COOKED_CALIBRATED`. Le plan, écrit ici au lieu d'être soumis (le propriétaire étant
absent), a été calculé par le job `index` sur l'index complet (133 458 fichiers,
2,7 To) : N = 20 photos « consumer » JPG par produit (NDC9 + partie), comptées sur les
9 appareils (l'étape 1.2 en excluait 2 pour tenir dans 4 Go, D32 ; le budget le permet
maintenant), donne 380 classes candidates (les 150 de v1 comprises). Sont téléchargées pour
ces classes toutes les photos « consumer » (12 508, 26,7 Go), une image RxImage par paquet
NDC11 (374, 0,2 Go) et 8 images segmentées REFERENCE_SEG par classe (3 040, 10,4 Go), plus
les 4 408 images de `MC_CHALLENGE_V1.0` (2,2 Go) : 20 330 fichiers, 39,5 Go, dont 4,8 Go
déjà présents (v1), soit 34,8 Go nouveaux. Espace libre mesuré avant : 203 Go ; le cache DVC
copie les fichiers (pas de reflink sur NTFS), donc environ 70 Go consommés au total.
Les options écartées : N = 10 (1 456 classes mais 15 photos par classe en médiane, moins
qu'en v1 : contraire à « plus d'images par classe ») ; toutes les REFERENCE_SEG (139 Go).
Seuils de qualité choisis sur les photos mesurées : luminosité moyenne < 10 ou > 245
seulement (un fond de jean donne ~15, ce n'est pas un défaut). Le jeu v1 (150 classes,
`config/pill_classes.csv`, `selected_images.csv`, étape `prepare`) reste celui des modèles
servis et de la baseline de l'étape 2.1 ; v2 vit dans `config/splits/c3pi_v2/` et
`data/processed/c3pi/bigdata/` et sera utilisé aux étapes 2.2 et 2.3. Spark tourne en
cluster autonome dans Docker (image officielle `apache/spark:4.1.1-python3`, un master, un
worker), sans Java sur l'hôte ; `pyspark==4.1.1` est figé pour la CI.

**D89. Split v2 sans fuite : la session de prise de vue est l'unité indivisible.** C3PI ne
donne aucun identifiant du comprimé physique photographié : un paquet NDC11 correspond à un
spécimen, et 347 des 380 classes de v2 n'ont qu'un paquet. Mettre
tout un spécimen dans un seul split retirerait la classe de val et de test, ce qui rend la
classification en ensemble fermé impossible. L'unité gardée ensemble est donc la session :
même paquet, même appareil, même fond, même éclairage, même date (recto et verso, prises
répétées) ; les sessions reliées par une photo quasi identique (dHash 256 bits, au plus
3 bits d'écart) ou par un fichier identique sont fusionnées (union-find). Le test mesure
donc la généralisation à une nouvelle prise de vue (appareil, fond ou lumière), pas à un
autre comprimé du même produit ; c'est écrit dans la data card. Remplissage par classe :
test puis val au plus près de 15 % chacun sans dépasser (une session trop grande est sautée),
le reste en train ; images de studio en train sauf si un quasi-doublon les lie à val/test.
Le job échoue si un groupe, un fichier ou une paire de quasi-doublons traverse les splits.
Résultat : 380 classes, 12 505 photos « consumer » en 4 891 sessions, 9 148 / 1 580 / 1 777
photos en train / val / test (73 / 13 / 14 %), chaque classe a au moins une photo en val et
en test, 0 fuite sur les trois contrôles.
Un essai sur données partielles a corrigé deux défauts avant le run final : remplissage
glouton qui donnait 22 % de test, et faux doublons entre images de studio de produits
différents (petit comprimé sur fond uni : 1 468 paires, toutes studio-studio), désormais
comptés seulement si une photo « consumer » est dans la paire.

**D90. ePillID reste un benchmark ; corpus RAG v2 dans son propre espace.** ePillID
(13 532 images 224 px, 4 902 types) est construit à partir des images du NLM Pill Image
Recognition Challenge (tirées de C3PI) et de Pillbox : ses 3 728 photos « consumer » sont
toutes des produits de l'archive C3PI, donc des recadrages de photos que v2 contient déjà
(986 d'entre elles concernent 197 des 380 classes de v2 ; 2 483 des 4 902 types ePillID sont
dans l'archive C3PI).
Les ajouter à l'entraînement dupliquerait des images et pourrait mettre la copie d'une photo
de test en train : ePillID sert seulement à comparer nos modèles à la littérature (job
`epillid` : recouvrement par NDC9 avec l'archive et avec les classes v1 / v2). Module 2 : le
même cluster traite le corpus RAG ; les sources des nouvelles classes (RxNav, openFDA, PubMed
avec 40 résumés par principe actif au lieu de 20) sont collectées par les connecteurs polis
du pipeline dans `data/rag_v2` (`python -m src.data.spark.rag_sources`, testé), pour ne pas
toucher l'espace `data/rag` du corpus servi. Le job `rag` nettoie, déduplique (PMID, texte
identique, MinHash LSH) et découpe en Parquet (`data/processed/rag/corpus_v2`) les espaces
listés dans `params.yaml` (`bigdata.rag.workspaces`) ; l'index servi reste le corpus v1
jusqu'à la phase 3. Périmètre réduit (en l'absence du propriétaire) : le client poli
refuse toute requête sans `CONTACT_EMAIL` dans `.env`, absent de cette copie du dépôt ; on
ne met pas l'adresse du propriétaire dans des requêtes à sa place. Le job a donc tourné sur
l'espace v1 seulement ; la collecte v2 attend que le propriétaire renseigne `CONTACT_EMAIL`,
puis `rag_sources ingredients|labels|pubmed`, `dvc add data/rag_v2`, ajout de `data/rag_v2`
aux paramètres et aux dépendances de `spark_rag`, `dvc repro`. Périmètre réduit : les paires DDInter du corpus v2 viennent de
`interactions_with_any_drug.csv` (au moins un principe actif v1), la table DDInter complète
n'étant plus sur le disque ; les paires entre deux nouveaux principes actifs viendront quand
elle sera reconstruite (étape 3.x).
