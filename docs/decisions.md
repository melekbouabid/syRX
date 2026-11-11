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

## MLflow conforme au lab et baseline officielle (2026-11-03)

**D91. La baseline est ré-enregistrée sans réentraînement ; les étapes d'entraînement DVC
sont gelées.** Les modèles servis depuis le 28/10 (détecteur YOLO11n, classifieur YOLO11s-cls,
jeu C3PI v1 de 150 classes) ont leurs runs MLflow sur le serveur de la première copie du
projet, pas sur celui-ci (D83) ; leurs sorties Ultralytics (poids, `results.csv`, `args.yaml`,
`summary.json`) sont restées dans `runs/`. Aucun entraînement long n'est lancé à cette étape
(les nouveaux entraînements passeront sur le GPU de Kaggle aux étapes 2.2 et 2.3) :
`src/models/register_baseline.py` les ré-enregistre au lieu de les réentraîner : il vérifie que leurs hyperparamètres sont exactement ceux de `params.yaml`
(`train.*`, sinon il refuse), crée un run suivi par modèle dans `tujya-vision` (tags
`git_commit`, `dvc_rev`, `dataset`, `seed`, `baseline=true`, `origin_*` vers le run
d'origine ; historique par époque, poids, figures, `requirements.txt`, `dvc.lock`) et
réévalue les poids maintenant sur le split de test (GPU local, tag `eval_device`). Résultats identiques à ceux du
28/10 à l'arrondi près : détecteur mAP50 0,972 et mAP@[.5:.95] 0,861 (0,8606 contre 0,8615) ;
classifieur top-1 93,7 % et top-5 98,5 % sur 588 crops (93,5 % et 98,1 % sur 589 à l'origine :
l'étape `prepare` refaite dans cette copie a produit un crop de test de moins).
Côté DVC, un stage jamais exécuté reste « changed » dans `dvc status`, même gelé (essai fait :
`frozen` masque les dépendances, pas les sorties absentes). Choix retenu : les stages
`train_detector` et `train_classifier` sont définis (commande, dépendances, paramètres,
sorties), marqués `frozen: true`, et leur sortie enregistrée par `dvc commit` est la baseline
(`models/train/<modèle>/` et `models/train/<modèle>.json`). Le `dvc.lock` décrit donc les poids
de la baseline produits avec ces hyperparamètres, mais pas par cette version du code : c'est
écrit ici et dans `dvc.yaml`, et le run MLflow le dit (`origin`). Le gel évite aussi qu'un
`dvc repro` lance par accident un entraînement de plusieurs heures ; on dégèle
(`dvc unfreeze`) quand un entraînement est voulu (Kaggle). Les stages `export_onnx` et `evaluate`, courts, ont réellement tourné
(`dvc repro export_onnx evaluate`). Rejeté : un mini-entraînement « de fumée » pour remplir
le lock (des poids sans valeur seraient devenus la sortie officielle) et laisser `dvc status`
sale jusqu'à l'étape 2.2.

**D92. Registry par alias, promotion automatique, client MLflow dans la CI.** Le guide du
module 1 parle de statut Staging / Production ; le lab MLflow utilise les alias, et les stages
sont dépréciés depuis MLflow 2.9 : la version servie porte l'alias `champion`, la deuxième
meilleure `challenger`, et chaque version reçoit un tag `status` (champion / challenger /
archived) et un tag `promotion` qui explique la décision. `promote_if_better(name, version,
metric, min_gain)` (`src/utils/mlflow_utils.py`) promeut une version seulement si elle bat le
champion d'au moins `min_gain` (strictement mieux : un score égal ne promeut pas) ; sinon elle
devient challenger si elle bat le challenger actuel, sinon archivée. Métriques de promotion
(`params.yaml`, `registry`) : mAP@[.5:.95] de test pour le détecteur (métrique du guide), top-1
de test pour le classifieur, gain minimal 0,005 pour les deux. `dvc_rev` est l'identifiant
d'objet git de `dvc.lock` (`git hash-object`) : il fixe la version de toutes les données DVC,
et `git log --find-object=<dvc_rev>` retrouve les commits qui la contiennent. Les modèles sont
journalisés avec l'API MLflow 3 (`name=`, un LoggedModel par export ONNX, métriques liées) et
`publish` copie par défaut les versions `champion` vers `models/`. Tests : `mlflow-skinny`
3.17.2 (même version que le serveur) et `alembic` rejoignent `requirements-dev.txt`, et les
tests tournent sur une base SQLite temporaire : le stockage par fichiers (`./mlruns`) est en
mode maintenance dans MLflow 3.17 et refuse de démarrer sans option.

## Comparaison des détecteurs (2026-11-04)

**D93. Les détecteurs de l'étape 2.2 sont entraînés sur C3PI v2 et tous comparés sur le
même split de test, baseline comprise.** L'étape 1.4 prévoit que les modèles soient
« ensuite entraînés et comparés sur les nouvelles données » (D88) : les cinq candidats
(YOLOv8n, YOLOv8s, YOLO11s, YOLOv10n, Faster R-CNN) sont entraînés sur v2. La détection n'a
qu'une classe (« pill ») : le jeu de détection v2 reprend les photos « consumer » du split v2
(D89), étiquetées par le même `src/data/autolabel.py` que v1, avec la même règle (les photos
douteuses sont écartées) ; les images de studio, RxImage et du challenge (comprimé seul sur
fond uni) restent pour le classifieur. Nouvelle étape DVC `prepare_detect_v2`
(`data/processed/c3pi/datasets_v2/detect`, images 640 px). Problème d'équité constaté : le
split v1 n'était pas fait par session, et 436 photos d'entraînement de la baseline (YOLO11n,
entraînée sur v1) tombent dans le test v2. Le comparer sur tout le test v2 le favoriserait.
Choix : un split de comparaison `test_holdout.txt` = les photos de test v2 dont la session
(groupe D89) ne contient aucune photo vue par la baseline en train ou en val ; tous les
modèles, baseline comprise, sont mesurés dessus par le même code
(`src/models/detection/metrics.py`, protocole COCO à 101 points, au plus 100 détections par
image, précision et rappel au seuil de meilleur F1), métrique `test_v2_mAP50_95`, qui devient
la métrique de promotion de l'étape (`params.yaml`, `compare.detector.metric`, avec le gain
minimal de `registry.detector.min_gain`). Le test v2 complet est aussi rapporté
(`test_v2_all_*`). La baseline est réévaluée sur ce split par
`python -m src.models.detection.evaluate_champion` (run `evaluate-detector-v1-c3pi-v2`, et
métriques ajoutées au run de la version, où `promote_if_better` les lit) ; l'import refuse de
promouvoir tant que le champion n'a pas cette métrique. `registry.detector.metric` (split v1)
reste celui de l'étape DVC `export_onnx`, inchangée. La latence comparée est celle de ONNX
Runtime sur le CPU du PC de développement, mesurée à l'import pour chaque modèle et pour la
baseline (même machine) ; celle du CPU de Kaggle est gardée à part (`kaggle_*`). Rejeté :
rester sur v1 (contraire à l'étape 1.4, et le split v1 a des sessions partagées entre
splits) ; comparer la baseline sur tout le test v2 (fuite) ; réentraîner la baseline sur v2
dans cette étape (c'est le rôle de la branche `experiment/detector-yolo11s` et des autres
candidats ; la baseline reste la référence servie).

**D94. Étape 2.2 en mode « en attente Kaggle » : branche commune d'outillage, une branche
par modèle, runs exportés puis importés dans MLflow.** Il n'y a ni compte ni jeton Kaggle sur
ce PC (et le GPU local, 4 Go, ne suffit pas pour cinq recherches d'hyperparamètres). Tout ce
qui ne demande pas le GPU de Kaggle est fait et testé : le code commun est sur
`feature/detector-kaggle-tooling` (jeu de détection v2, métriques, latence ONNX CPU, espaces
Optuna, exécuteur d'expérience, empaquetage du dataset Kaggle, import MLflow, réévaluation du
champion, tableau et figure de comparaison, générateur de notebooks), et chaque branche
`experiment/detector-<archi>` part de cette branche et ajoute ce qui est propre au modèle :
sa configuration Optuna `config/hpo/detector_<archi>.yaml`, son notebook
`notebooks/kaggle/detector_<archi>.ipynb` et, pour `experiment/detector-fasterrcnn`, le code
torchvision et ses tests (job CI `vision` avec PyTorch CPU). Ainsi l'historique montre
l'outillage une fois, puis une branche par expérience ; seule la branche du champion sera
fusionnée dans `main` (avec l'outillage qu'elle contient). Les notebooks clonent leur branche
(le `git_commit` du run est celui du code exécuté), lisent le dataset Kaggle privé (avec son
`MANIFEST.json` : hash DVC et `dvc_rev`) et n'écrivent que dans un dossier d'export
(`run.json`, poids, ONNX, essais) : aucun serveur MLflow n'est exposé, aucun secret n'est mis
dans un notebook (D69). `python -m src.models.detection.import_kaggle` crée un run parent par
modèle et un run enfant par essai Optuna dans `tujya-vision` (vues Compare et Parallel
Coordinates), puis enregistre l'ONNX et appelle `promote_if_better`. Recherche : Optuna (TPE
avec graine), 8 à 12 essais par modèle sur `lr0` (plage selon l'optimiseur), `imgsz`,
`batch`, l'optimiseur (SGD / AdamW) et les augmentations (mosaic, mixup, HSV, flips ; pour
Faster R-CNN, HSV et flips seulement, mosaic et mixup n'ayant pas d'équivalent torchvision),
chaque essai sur une partie du train et noté sur la validation, puis le meilleur réentraîné
sur tout le train et évalué une seule fois sur le test. Faster R-CNN est exporté au même
format ONNX que les YOLO (entrée 1x3xSxS, sortie 1x5xN), donc l'API le servirait sans
changement. Vérification locale : runs « smoke » (tag `smoke=true`, nom `smoke-...`,
quelques dizaines d'images, 2 essais, 1 époque), jamais enregistrés dans le Model Registry.
Aucune fusion dans `main`, aucun tag `model/` ni `v3.1.0-detecteurs` avant les vrais
entraînements ; les étapes DVC d'entraînement restent gelées sur `main` (D91).

## Optimisation ONNX des champions (2026-11-06)

**D95. Variantes optimisées enregistrées à côté du champion, alias `served` ; l'outillage
détecteurs est fusionné pour mesurer sur le même split.** L'étape 2.4 compare FP32, FP16 et
INT8 des deux champions et sert la meilleure variante. Deux contraintes : (1) la promotion
des détecteurs de l'étape 2.2 encore à importer (`import_kaggle` -> `promote_if_better` sur
`test_v2_mAP50_95` contre l'alias `champion`) doit rester une comparaison FP32 contre FP32 ;
(2) le détecteur champion (YOLOv8n, importé le 06/11) doit être mesuré sur le même split et
par le même code que l'étape 2.2, sans fuite. Choix : chaque variante FP16 / INT8 est un run
MLflow (`stage=optimize`) et une nouvelle version du même modèle enregistré, avec les tags
`precision` et `variant_of` (numéro de la version champion) et `status=variant` ; l'alias
`champion` ne quitte jamais le modèle FP32 ; la variante retenue reçoit l'alias `served` (la
version champion elle-même si FP32 l'emporte) et la version champion le tag
`served_precision`. `python -m src.models.publish` publie par défaut la version `served` si
elle est le champion ou une variante du champion actuel, sinon le champion FP32 (cas d'un
nouveau champion promu depuis la dernière optimisation, jusqu'à ce que
`python -m src.models.optimize` soit relancé) ; `--exact` publie l'alias tel quel
(`src/models/serving.py`). Pour mesurer, la branche `feature/detector-kaggle-tooling` (outillage
commun de l'étape 2.2 : jeu de détection v2 et `test_holdout.txt`, métriques COCO, latence
ONNX CPU en processus isolé, import Kaggle) est fusionnée dans `perf/onnx-quantization`, donc
dans `main` ; aucune branche `experiment/*` n'est fusionnée et l'étape 2.2 reste en attente des
runs Kaggle (D94). Rejeté : un modèle enregistré séparé par précision (la variante perd le
lien avec son champion dans la vue du Registry) ; mettre l'alias `champion` sur la variante
quantifiée (`promote_if_better` comparerait les candidats FP32 au score de la variante) ;
dupliquer les métriques et la latence dans un autre module (deux codes de mesure divergeraient).

**D96. INT8 du détecteur : tête YOLO gardée en float ; latence mesurée en tours entrelacés.**
Quantification statique (QDQ, int8 par canal, calibration MinMax sur 200 photos du val v2,
jamais le test) : les nœuds de la tête de détection (`/model.22/`, trouvés depuis la sortie du
graphe par `output_module_prefix`) restent en float, car le décodage des boîtes manipule des
coordonnées en pixels (0 à 640) qu'un seul facteur d'échelle int8 arrondit trop. Essai sur 200
photos de test de comparaison : tête en float mAP@[.5:.95] 0,854 ; seul le décodage en float
0,848 ; quantification dynamique 0,832 (FP32 0,876). Le classifieur est quantifié en entier
(300 crops du val v1). Latence : le PC portable de développement se bride sous charge soutenue
(YOLOv8n passe de 24 ms à 570 ms par image après une centaine de passes enchaînées) ; une
mesure longue par variante avantageait la première mesurée. Chaque variante est donc chronométrée
en 5 tours entrelacés (30 passes dans un processus neuf, 20 s de pause avant chaque mesure),
médiane des tours (`params.yaml`, `optimize`). Les 8 runs et versions d'une première exécution
mesurés avec l'ancienne méthode (200 passes d'affilée) restent dans MLflow avec le tag
`status=superseded`. Rejeté : quantifier aussi la tête (perte de localisation), exclure en plus
la première convolution (gain de 0,008 sur l'essai, non retenu pour garder une règle simple par
architecture), mesurer sur GPU (l'API sert sur CPU).

## Endpoints du guide (2026-11-06)

**D97. `/health`, `/model/metadata`, `/predict`, `/predict/visualize` : le modèle seul, sans
compte ; métadonnées lues dans `models.json` ; pas de Grad-CAM dans l'API.** Le guide du module 1
impose ces quatre routes (Swagger sur :8000/docs, 400 image invalide, 503 modèle non chargé).
Elles sont ajoutées à côté de `POST /api/scan` sans la modifier : elles renvoient seulement ce
que produisent les modèles (boîtes, top-5, confiances, latence serveur), sans tables de
médicaments ni compte, pour se tester directement depuis Swagger ou `curl`. Pas de compte : elles
ne lisent ni n'écrivent aucune donnée personnelle, gardent les limites de taille de `/api/scan`
(10 Mo, 50 mégapixels) et l'API n'écoute que sur 127.0.0.1 (règle du projet) ; `/api/scan` reste
derrière la connexion (D46). Les erreurs suivent le guide : 400 pour un type ou un contenu qui
n'est pas une image (`/api/scan` garde son 415), 413 trop grande, 503 sans modèles ; `/health`
répond toujours 200 avec `status=degraded` pour qu'une sonde voie la raison. `/model/metadata` lit
`models.json`, que `python -m src.models.publish` enrichit de l'architecture (tag `arch` du
Registry) et des métriques de la version publiée (celles d'une variante FP16 / INT8 complétées
par les scores de validation de son champion) : l'API ne contacte jamais MLflow à l'exécution.
`/predict/visualize` renvoie un PNG annoté (couleur stable par classe). Grad-CAM n'est pas servi :
il demande les gradients du modèle PyTorch et l'API n'exécute que l'ONNX, sans PyTorch ; il est
reporté à l'étape 2.6. Rejeté : protéger ces routes par la session (le guide les teste sans
compte), interroger le Registry à chaque requête (dépendance réseau et latence), une carte de
saillance par occlusion présentée comme Grad-CAM (ce n'en est pas un).

## Choix du détecteur champion (2026-11-07)

**D98. Champion des détecteurs : YOLO11s, par une règle qui ne dépend pas de l'ordre
d'import ; YOLOv10n challenger ; latence mesurée avec une charge limitée.** Les quatre derniers
exports Kaggle (YOLOv8s, YOLOv10n, YOLO11s, Faster R-CNN) arrivent après YOLOv8n, champion en
place depuis le 06/11 (version 2, `test_v2_mAP50_95` 0,8774). Règle : un candidat est promu
s'il bat ce champion en place d'au moins 0,005 (`registry.detector.min_gain`, D92-D93) ; parmi
ceux qui le font, le meilleur devient champion ; le challenger est le deuxième modèle de la
comparaison. Valeurs exactes des `run.json` : YOLO11s 0,88846 (+0,0111), YOLOv10n 0,88596
(+0,0086), YOLOv8s 0,88296 (+0,0056), Faster R-CNN 0,82177 (-0,0556) : champion YOLO11s,
challenger YOLOv10n. Problème : `promote_if_better` compare chaque import au champion du moment,
donc une suite d'imports dépend de son ordre ; simulé avec la vraie fonction sur un store
temporaire pour les 24 ordres possibles : 12 donnent YOLO11s / YOLOv10n, 4 YOLO11s / YOLOv8s
et 8 YOLOv10n / YOLO11s (YOLOv10n importé avant YOLO11s est promu, puis YOLO11s ne le bat
que de 0,0025). Choix : `import_kaggle` importe plusieurs exports du meilleur au moins bon
(métrique de promotion lue dans chaque `run.json`, `--keep-order` pour l'ordre donné). Importé
en premier, le meilleur est promu s'il bat le champion en place du gain minimal (sinon aucun
ne le fait), les suivants ne peuvent plus le battre, et le challenger devient le meilleur des
autres et de l'ancien champion : le résultat est celui de la règle pour tout ordre des
arguments (test `test_best_first_imports_give_the_rule_of_d98` avec ces valeurs). Les alias
n'ont pas été corrigés à la main : YOLO11s version 9 `champion`, YOLOv10n version 10
`challenger`, YOLOv8s (11), Faster R-CNN (12), YOLOv8n (2) et la baseline YOLO11n (1)
`archived`. L'écart est faible et doit être lu comme tel : les trois meilleurs YOLO tiennent
en 0,0055, YOLO11s ne devance YOLOv10n que de 0,0025, sur un seul split (995 photos, 1 133
boîtes produites automatiquement) et un seul entraînement par modèle (une graine) ; l'ordre
est le même sur le test v2 complet (qui contient ce split), pas sur la validation. Le mAP50
est presque égal pour tous (0,972 à 0,976) : la différence porte sur la précision des boîtes
aux IoU élevés. En contrepartie, YOLO11s coûte 2,9 fois le temps CPU de YOLOv10n (63,0 ms
contre 21,9 ms par image) et 4 fois sa taille (37,9 Mo contre 9,4 Mo, 9,5 M contre 2,3 M
paramètres). Le champion reste celui de la règle (une règle fixée avant les résultats ne se
change pas après coup) ; la variante servie (alias `served`, D95) est choisie par la règle de
l'étape 2.4 parmi les variantes FP32 / FP16 / INT8 du champion, ce qui garde la garantie « l'API
sert le champion » ; YOLOv10n, challenger, est l'option documentée si la latence devient la
contrainte (webcam continue, étape 2.6). Faster R-CNN (deux étapes, ResNet-50 FPN, l'autre
famille demandée par le professeur) détecte aussi bien (mAP50 0,972) mais localise moins bien
(mAP@[.5:.95] -0,067 contre le champion) : 480 px retenus par la recherche, 12 époques
seulement dans le temps de GPU de Kaggle, régression des boîtes sur des caractéristiques
RoIAlign 7 x 7 avec une perte smooth L1 quand les YOLO optimisent directement le recouvrement
(CIoU, distribution focal loss) ; il est 19 fois plus lent que YOLOv10n (421 ms à 480 px :
réseau de propositions puis têtes sur 1 000 propositions, 41 M paramètres). Latence : mesurée
pour les six détecteurs FP32 ensemble (`src/models/detection/compare_latency.py`, protocole
de D96 : 5 tours entrelacés, 20 s de pause, médianes) ; une première mesure sans limite de
charge (non enregistrée) a montré que le PC portable passe d'environ 60 ms à 1 000 ms par
image pour YOLO11s après environ deux secondes d'inférence continue (Faster R-CNN : 400 ms puis
2 700 ms), d'où des tours bridés pour les gros modèles. Chaque mesure est donc limitée à
1,5 s de charge (`optimize.latency_budget_s`, échauffement compris) : les petits YOLO gardent
leurs 30 passes, YOLO11s en a 16 et Faster R-CNN 2 par tour ; l'API traite une image par
requête, bien en dessous de cette charge. Les valeurs enregistrées (`onnx_cpu_latency_ms`,
moyenne de chaque tour dans `onnx_cpu_latency_round_ms`) ont au plus 15 % d'écart entre tours.
Variante servie, relancée le 07/11 (`python -m src.models.optimize --task detect`) : le FP32 de
la version 9 (alias `served`), car son FP16, de même précision, n'est que 3,9 % plus rapide
(seuil 5 %) et l'INT8 est plus lent et moins précis (docs/module1/optimisation_onnx.md).
Rejeté : corriger les alias à la main après des imports dans un ordre quelconque (le Registry
garderait des décisions contradictoires) ; promouvoir YOLOv10n pour sa vitesse (contraire à la
règle de promotion annoncée) ; garder la latence mesurée à l'import (une mesure isolée par
modèle, sur un processeur plus ou moins chaud) ; changer le nombre de fils d'ONNX Runtime pour
éviter le ralentissement (l'API garde le réglage par défaut).

## Comparaison des classifieurs (2026-11-08)

**D99. Les classifieurs de l'étape 2.3 sont entraînés sur les crops de C3PI v2 et tous comparés
sur le même split, champion v1 compris.** Comme les détecteurs (D93), les six candidats sont
entraînés sur v2 (380 classes, étape 1.4). Nouvelle étape DVC `prepare_classify_v2`
(`src/data/prepare_classify_v2.py`, sortie `data/processed/c3pi/datasets_v2/classify`) : le crop
carré (marge de 20 %, 256 px) de chaque comprimé de chaque image du split v2 (D89), c'est-à-dire
le crop que l'API donne au classifieur (`square_crop`) ; boîtes des photos « consumer » par
`autolabel.py` (photos douteuses écartées, même règle que v1 et que la détection), images de
référence RxNav, images du challenge NLM, images de studio segmentées (masque alpha, comprimé
posé sur un fond gris) et augmentation hors ligne des classes rares (étape `spark_augment`,
tirée du train seulement). La validation et le test ne contiennent que des photos « consumer »,
les conditions de l'application. Résultat : 15 424 / 1 461 / 1 713 crops (train / val / test),
au moins 10 crops de train par classe. Problème d'équité, le même qu'en D93 : 582 crops de test
v2 viennent de sessions de prise de vue dont le champion v1 (YOLO11s-cls, 150 classes) a vu des
photos en train ou en val. Split de comparaison `test_holdout.txt` : les 1 131 crops de test dont
la session ne contient aucune photo vue par v1 (`baseline_groups` de D93), dont 275 des 150
classes v1 ; 57 classes n'y ont aucun crop. Un seul code de mesure pour tous les modèles et pour
le champion (`src/models/classification/metrics.py`, à partir des probabilités) : top-1, top-5,
F1 macro (moyenne sur les classes présentes dans la vérité ou dans la prédiction), matrice de
confusion et scores par classe ; métriques `test_v2_*` (split de comparaison), `test_v2_all_*`
(test v2 complet, contient le split, donc pas une mesure indépendante) et `test_v2_v1c_*` (D100).
Métrique de promotion `test_v2_top1` (`params.yaml`, `compare.classifier.metric`), gain minimal
`registry.classifier.min_gain` (0,005) ; `registry.classifier.metric` (`test_top1`, split v1)
reste celui de l'étape DVC `export_onnx`. Rejeté : rester sur les 150 classes de v1 (contraire à
l'étape 1.4, split v1 non fait par session) ; mesurer le champion sur tout le test v2 (fuite) ;
mettre des images de studio ou augmentées dans la validation (le choix des hyperparamètres se
ferait sur d'autres images que celles de l'application).

**D100. Le champion v1 est noté sur les 380 classes, ses classes inconnues comptant comme
erreurs ; promotion sur `test_v2_top1` avec un garde-fou sur les 150 classes de v1, indépendante
de l'ordre d'import.** Le champion ne connaît que 150 classes. `python -m
src.models.classification.evaluate_champion` note le fichier ONNX enregistré (celui que sert
l'API) avec le prétraitement de l'API : ses 150 sorties sont reportées sur les identifiants v2
par la clé C3PI du produit (`class_key`, colonne `v1_class_id` de `classes.csv`), les 230 autres
classes reçoivent un score de -1 (jamais classées avant une classe connue), donc leurs crops
comptent comme des erreurs dans `test_v2_*` : c'est ce que verrait un utilisateur, le modèle
servi ne peut pas reconnaître ces comprimés (au mieux 275 / 1 131 = 24,3 % de top-1). Pour
comparer à classes égales, `test_v2_v1c_*` note tous les modèles, champion compris, sur les
crops de comparaison des 150 classes v1, la prédiction étant restreinte à ces 150 classes (même
espace d'étiquettes). Champion (version 1) mesuré le 08/11 (run `evaluate-classifier-v1-c3pi-v2`) :
`test_v2_top1` 0,1830 (207 crops justes sur 1 131, tous parmi ses classes), top-5 0,2246, F1
macro 0,1119 ; `test_v2_v1c_top1` 0,7527 (top-5 0,9236, contre 0,937 de top-1 sur le test v1 :
les nouvelles sessions sont plus difficiles) ; sur le test v2 complet, qui contient 582 crops de
sessions qu'il a vues, 0,4466 (la fuite que le split retire). Règle
de promotion : un candidat est promu s'il bat le champion en place d'au moins 0,005 sur
`test_v2_top1` ; garde-fou : un candidat dont `test_v2_v1c_top1` est plus de 0,01 en dessous de
celui de la baseline v1 (version marquée `baseline=true`, `compare.classifier.guard_max_drop`,
environ 3 des 275 crops, la tolérance de l'étape 2.4) est archivé quel que soit son score
(`promote_if_better(..., eligible=False)`, tag `promotion` « not eligible ») : un modèle à
380 classes ne doit pas reconnaître moins bien les 150 comprimés que l'application identifie
aujourd'hui. L'import refuse de promouvoir tant que le champion n'a pas `test_v2_top1` et la
référence `test_v2_v1c_top1`. Ordre d'import (D98) : plusieurs exports donnés en un appel sont
importés du meilleur au moins bon sur `test_v2_top1` ; un candidat inéligible est archivé sans
toucher aux alias ; le résultat est donc, pour tout ordre des arguments : parmi les candidats
éligibles, le meilleur devient champion s'il bat le champion en place du gain minimal, et le
challenger est le meilleur des autres et de l'ancien champion. La réévaluation ajoute les scores
`test_v2_*` au run de la version (où `promote_if_better` les lit) mais pas sa latence, mesurée
seule : la latence comparée reste celle du protocole entrelacé (D96, `compare_latency --task
classify`). Rejeté : comparer seulement sur les 150 classes de v1 (ignorerait les 230 nouvelles
classes, le but de l'étape 1.4) ; promouvoir sur `test_v2_top1` sans garde-fou (le champion
plafonne à 24,3 %, tout modèle v2 le battrait même en reconnaissant moins bien les comprimés
actuels) ; réentraîner v1 sur v2 dans cette décision (c'est la branche
`experiment/classifier-yolo11s-cls`).

**D101. Export ONNX au format du classifieur servi, liste de classes livrée dans le modèle ; la
table de classes de l'API passe à v2 avant de servir un classifieur v2.** Tous les candidats
sont exportés comme le YOLO11s-cls servi : une entrée `images` 1x3xSxS (RGB en float dans
[0, 1], le crop carré redimensionné en INTER_AREA par `backend/app/inference.py`), une sortie
`output0` 1xN de probabilités (softmax dans le graphe, comme l'export Ultralytics), opset 17 ; la
normalisation ImageNet est faite dans le réseau (`ClassifierNet`), donc l'API ne change pas. La
parité PyTorch / ONNX est mesurée sur 100 crops de test (`parity_*`). Les métadonnées ONNX
portent la liste de classes : `names` (index de sortie -> dossier `000` à `379`, ce que lit
l'API), `class_keys` (index -> clé C3PI, NDC9 et partie, stable d'une version des données à
l'autre), `label_space=c3pi-v2`, `num_classes`, `imgsz`, `arch` ; `classes.csv` est copié dans
chaque export. Problème : l'API lit `names` comme le `class_id` de `config/pill_classes.csv`
(les 150 classes de v1) ; les identifiants v2 ne sont pas ceux de v1 (7 sur 150 coïncident) et
230 classes n'y sont pas : servi tel quel, un classifieur v2 afficherait le mauvais médicament.
Choix : `python -m src.models.publish` refuse un classifieur dont les `class_keys` ne désignent
pas, pour chaque sortie, le même produit que la ligne de `config/pill_classes.csv` nommée par
`names` (`check_class_table`) ; l'export Ultralytics de v1 n'a pas de `class_keys` et passe comme
avant. Pour servir le champion de l'étape 2.3 (fin d'étape, après Kaggle), `config/pill_classes.csv`
devra décrire les 380 classes v2 avec leurs identifiants v2 (mêmes colonnes, principes actifs
par RxNorm comme pour v1) et les tables de médicaments être rechargées ; `models/` n'est pas
modifié à cette étape. Rejeté : réécrire `names` vers les identifiants v1 à la publication
(impossible pour 230 classes) ; un adaptateur dans l'API avec deux tables de classes (deux
sources de vérité) ; un export Ultralytics pour les modèles torchvision (pas de format commun).

**D102. Une recette d'entraînement pour les six classifieurs, recherchée par Optuna.** Une seule
boucle (`src/models/classification/torch_backend.py`) et une seule enveloppe
(`src/models/architecture.py`) pour : un petit CNN maison entraîné sans poids préentraînés (la
baseline, 4 blocs de convolutions, 1,3 M paramètres), ResNet-50, EfficientNetV2-S,
ConvNeXt-Tiny, ViT-B/16 (224 px, imposé par ses plongements de position ; retenu plutôt que
Swin-T : même bibliothèque, Grad-CAM sur ses jetons) et YOLO11s-cls (l'architecture servie,
réentraînée sur v2), tous préentraînés sur ImageNet-1k sauf la baseline. *Transfer learning*
puis *fine-tuning* sélectif : `head_epochs` époques avec la tête seule (colonne vertébrale gelée,
batch norm en évaluation), puis les `unfreeze` derniers blocs dégelés et entraînés à
`lr x backbone_lr_mult` (AdamW, décroissance cosinus, pas de *weight decay* sur les biais et les
normalisations) ; la baseline entraîne tout le réseau au même taux. Déséquilibre (10 à 80 crops
de train par classe) : `imbalance` = aucun, poids de classe ((n_c / moyenne)^-0,5, comptes du
train) ou *focal loss* (gamma 2, mêmes poids), choisi par la recherche. Augmentation du train
seulement : rotation quelconque puis crop redimensionné aléatoire (70 à 100 % de la surface),
RandAugment (magnitude 0 à 12), MixUp ou CutMix sur une part `mix_prob` des lots, *label
smoothing* ; pas de miroir (une empreinte inversée n'existe pas sur un vrai comprimé) ; la
validation et le test ne reçoivent que le redimensionnement de l'API (aucune fuite). Recherche
(`config/hpo/classifier_<archi>.yaml`, une par branche) : TPE avec graine (5 essais aléatoires
d'abord) sur `lr`, `weight_decay`, `backbone_lr_mult`, `unfreeze` (blocs dégelés), `imgsz` (192
ou 224 ; baseline 128 à 192 avec sa largeur et son dropout), `randaugment`, `mix_prob`,
`label_smoothing` et `imbalance` ; chaque essai sur une part stratifiée du train, noté sur
`val_top1` ; le meilleur est réentraîné sur tout le train (arrêt précoce sur la validation) et
évalué une seule fois sur le test. Mesures sur le PC qui ont changé le code de la session
précédente : `channels_last` rendait la rétropropagation des réseaux à batch norm 10 à 13 fois
plus lente (RTX 2050, torch 2.14 : ResNet-50 10 contre 113 images/s), d'où le format mémoire par
défaut ; la rotation et le crop redimensionné de torchvision coûtaient 8,7 ms par crop sur un
cœur, un seul `cv2.warpAffine` 0,3 ms (chargement à 4 processus : 133 -> 615 crops/s sans
RandAugment, 378 avec). Grad-CAM (`src/utils/gradcam.py`) : 8 images par export (moitié justes,
moitié fausses) sur la dernière carte de convolution de chaque architecture, sur la norme
d'entrée du dernier bloc pour ViT (jetons remis en grille 14 x 14) ; il n'est pas servi par l'API
(D97). Rejeté : un code d'entraînement par bibliothèque (Ultralytics pour YOLO, torchvision pour
les autres : deux recettes, comparaison inéquitable) ; une recherche par Ultralytics `tune`
(YOLO seulement).

**D103. Étape 2.3 en mode « en attente Kaggle », comme D94 ; budget de 20 h de GPU.** Pas de
compte Kaggle sur ce PC et un GPU de 4 Go : le code commun est sur
`feature/classifier-kaggle-tooling`, chaque branche `experiment/classifier-<archi>` (baseline-cnn,
resnet50, efficientnetv2s, convnext-tiny, vit-b16, yolo11s-cls) en part et ajoute sa
configuration Optuna et son notebook (`notebooks/kaggle/classifier_<archi>.ipynb`, qui clone sa
branche) ; le job CI `vision` installe PyTorch CPU et Ultralytics pour les tests des
architectures, de l'entraînement, de l'export ONNX et de Grad-CAM. Runs « smoke » locaux
(`smoke=true`, nom `smoke-...`) jamais enregistrés dans le Model Registry. Budget : débit
d'entraînement mesuré sur le PC (précision mixte, lot de 16 à 64) multiplié par 2,5 pour un T4
(bande passante mémoire et calcul FP16), et plafonné à environ 250 crops/s par le chargement des
données sur les 4 cœurs de Kaggle (débit du PC divisé par deux) : environ 1,1 min par époque
complète pour la baseline, ResNet-50 et YOLO11s-cls, 1,6 min pour ConvNeXt-Tiny, 2,2 min pour
EfficientNetV2-S et 2,5 min pour ViT-B/16. Essais, époques et part du train de chaque
configuration sont réglés pour environ 15,5 h attendues au total (installation, test, export et
Grad-CAM compris, environ 15 min par notebook), et les limites de temps (`timeout_hours` de la
recherche, `max_hours` du réentraînement) bornent le total à environ 18,7 h si les estimations sont
trop optimistes : moins que le budget de 20 h fixé, que le quota gratuit d'environ 30 h par
semaine et que la limite de 12 h par session. Calcul détaillé par notebook dans
`docs/module1/kaggle_classifieurs.md`. Aucune fusion dans `main`, aucun tag `model/` ni
`v3.2.0-classifieurs` avant les vrais entraînements. Rejeté : entraîner sur le GPU du PC (4 Go,
environ 2,5 fois plus lent qu'un T4, plusieurs jours) ; une seule branche pour les six modèles
(contraire à la remarque 5 du professeur).

## Frontend du guide et Grad-CAM (2026-11-08)

Numérotation : D99 à D103 sont réservées aux décisions de l'étape 2.3 (comparaison des
classifieurs), écrites sur sa branche d'outillage et pas encore fusionnées ; celles de
l'étape 2.6 commencent donc à D104.

**D104. Grad-CAM précalculé pour la galerie de démo, hors de l'API, avec les poids PyTorch
du classifieur servi.** Grad-CAM pondère les cartes d'une couche de convolution par la
moyenne de leur gradient : il faut le modèle PyTorch et une passe arrière, alors que l'API
n'exécute que l'ONNX, sans PyTorch dans son image (D97). Choix : le script
`src/models/gradcam_gallery.py` (venv d'entraînement, étape DVC `gradcam_gallery`) calcule
les cartes des photos de la galerie avec `src/utils/gradcam.py` (repris de l'étape 2.3 par
cherry-pick, sans le reste de cette branche) ; il reprend la chaîne ONNX de l'API
(`backend/app/inference.py` : mêmes boîtes, mêmes recadrages carrés de 224 px), charge les
poids de la version servie (`models/train/classifier`, run `6dd616a6...` vérifié contre
`models/models.json`, sinon arrêt), prend la dernière convolution de la tête Classify
(`model.10.conv`, 1 280 cartes de 7 x 7) et la classe top-1 donnée par l'API, et s'arrête si
PyTorch et ONNX Runtime ne donnent pas la même probabilité (écart mesuré 1,2e-7). Chaque carte
est fondue dans le carré recadré de son comprimé sur la photo (1 280 px au plus, mêmes
proportions, le reste assombri), pour que les boîtes du scan tombent au même endroit. Sorties
dans `data/processed/c3pi/gradcam` (DVC, jamais dans git), montées en lecture seule dans le
conteneur `backend` et servies par `GET /api/demo/images/{id}/gradcam` (404 sans carte) ;
la galerie donne `gradcam_url`. Le Scanner affiche un onglet « Grad-CAM » pour une photo de la
galerie et dit, pour toute autre photo, que la carte n'existe que pour la galerie. Rejeté :
PyTorch dans l'image de l'API (plusieurs centaines de Mo, et une seconde implémentation du modèle à
garder identique à l'ONNX) ; une carte de saillance par occlusion calculée en ONNX et
présentée comme Grad-CAM (D97) ; exporter un second ONNX qui sort aussi les activations et
leurs gradients (graphe d'entraînement à maintenir pour une seule figure) ; calculer les
cartes dans le navigateur (même problème, sans gradients).

**D105. La galerie de démo ne montre que des photos qu'aucun modèle servi n'a vues à
l'entraînement.** Contrôle de l'étape 2.6 : la galerie (D52) vérifiait le split de test v1,
celui du classifieur. Depuis l'étape 2.2, le détecteur servi (YOLO11s v9) est entraîné sur
C3PI v2, et deux des huit photos (halopéridol, hydroxychloroquine) étaient dans son jeu
d'entraînement ; deux autres (trazodone, multivitamine) n'étaient dans aucun jeu de détection v2. Règle, vérifiée par
l'API (`backend/app/gallery.py`) avec les fichiers suivis par git : une photo « consumer » du
split de test v2 (`config/splits/c3pi_v2/test.csv`, sessions entières, D89) qui n'est pas dans
le train ni la val v1 (`config/splits/selected_images.csv`). Nouvelle liste de huit photos,
toutes aussi dans `test_holdout.txt` de D93 (aucune photo de la même session vue par un modèle
v1, ce qui couvre le classifieur v1 entraîné sur le même split), avec un test qui le vérifie
quand les données DVC sont là. Rôles gardés : interactions Major (halopéridol, citalopram,
fluconazole, warfarine, nortriptyline ; fluconazole + citalopram + halopéridol donnent trois
paires Major), deux gélules AMRIX sur une photo, une photo de trazodone sous le seuil de
confirmation (81 %), le multivitamine sans principe actif connu. Les huit sont choisies parmi
les 244 candidates possibles avec les résultats de la chaîne servie, comme en D52 : la galerie
montre le modèle à son avantage, sauf la photo de trazodone ; les chiffres honnêtes restent
ceux du split de test entier. Rejeté : se fier au seul split v1 (fuite pour le détecteur) ;
exiger `test_holdout.txt` dans l'API (fichier DVC absent de l'image et de la CI).

**D106. Tailwind CSS 4 en couches CSS à côté de styles.css, sans le reset de base ; Vitest pour
les tests du site.** Le guide demande un framework CSS. Tailwind 4.3.3 (plugin Vite, version
publiée en juillet 2026) plutôt que MUI : le site a déjà son identité (couleurs du logo,
composants maison) et MUI imposerait ses propres composants et son thème. `src/index.css`
déclare les couches `theme, base, legacy, components, utilities` : thème et utilitaires de
Tailwind, `styles.css` entier dans la couche `legacy`, sans le « preflight » (reset de base)
qui changerait toutes les pages. Les utilitaires passent donc au-dessus des anciennes règles
là où un composant les emploie, et les autres pages (chat, toxicité, admin, profil,
connexion) restent identiques, vérifié par captures. Les jetons Tujya (`--surface`,
`--primary`...) deviennent des couleurs Tailwind (`bg-surface`, `text-primary`...), donc le
mode sombre suit. Le Scanner du guide (onglets, glisser-déposer, galerie, webcam, résultat,
barres) est réécrit en utilitaires et ses anciennes règles sont retirées (201 lignes). Seule
collision de nom trouvée : la classe `table` des tableaux d'administration, à qui l'utilitaire
`table` ne fait que redonner `display: table`. Tests : Vitest 5.0.5 en environnement Node (pas
de navigateur ni de jsdom) pour la logique pure (`src/lib/vision.ts`) et le rendu des
composants (`react-dom/server`) ; lancés par la CI (`npm test`).

**D107. Webcam : une requête à la fois ; latence client mesurée dans le navigateur ; couleurs
de classe de la palette d'Ultralytics.** Inférence continue : la boucle (`startLiveLoop`)
prend une image, l'envoie à `POST /api/scan` et n'attend la suivante qu'après la réponse plus
0,4 s ; elle ne crée jamais de file d'attente si le serveur ralentit (testé avec des minuteries
factices), et l'arrêt annule la requête en cours. `/api/scan` plutôt que `/predict` : mêmes
modèles, plus les noms des comprimés et la règle de confirmation. Latence : client =
`performance.now()` autour de l'appel (envoi, serveur, réception, lecture du JSON) ;
serveur = `timings.server_ms`, ajouté à `/api/scan` (toute la requête côté API) ; modèle =
ONNX Runtime seul ; la différence client - serveur est affichée comme « réseau et navigateur ».
Couleurs : la formule de teinte de D97 donnait presque le même vert au fluconazole (classe 9)
et au citalopram (120), montrés ensemble dans la démo ; les boîtes prennent maintenant la
couleur `class_id mod 20` de la palette à 20 couleurs des tracés d'Ultralytics, dans
`/predict/visualize` comme sur le canvas du site (même liste dans les deux codes, mêmes
valeurs testées des deux côtés), avec un texte noir ou blanc selon la luminance. Deux classes
distantes de 20 partagent une couleur : la couleur aide à suivre une classe d'une image à
l'autre, le nom et la probabilité restent écrits sur chaque boîte. Rejeté : une file
d'images en parallèle (latence qui grandit sans fin sur CPU) ; une couleur par confiance
seulement (le guide demande une couleur par classe).

## Étape 2.7 : boîtes de médicaments (remarque 18 du professeur)

**D108. Données des boîtes : jeu public sans compte, vérité terrain tirée des métadonnées,
photos de l'utilisateur et Roboflow en attente.** Le scanner ne connaît que des comprimés C3PI :
une boîte recevait une fausse classe (7 boîtes sur 8 testées, docs/module1/boites.md). Jeu
retenu : mennox/Italian-OTC-medicines (Hugging Face, Apache-2.0, téléchargement anonyme figé à
la révision 29ebcd7), 718 fichiers dont 5 pages HTML écartées par leur signature, 713 photos.
La vérité terrain vient des métadonnées, jamais de la photo : marque = premier mot du nom
commercial (vide pour un générique nommé d'après son principe actif) ; principes actifs = texte
« Principi attivi » entre « principio attivo » et « eccipienti » (un arôme comme « gusto Ginger »
n'est pas un principe actif), rapproché des noms DDInter avec les mêmes règles que l'API, sans la
table des marques (pour ne pas noter la voie « marque » contre elle-même) : 279 photos ont au
moins un principe actif connu. Contrôle de 20 lignes au hasard : justes, parfois incomplètes.
Images hors sujet : 16 exemples de scikit-image (CC0 ou domaine public) + 3 images générées.
Roboflow Universe et Kaggle demandent un compte : non utilisés (rien n'est contourné) ; le petit
détecteur YOLO « boîte » reste une option. Les photos de l'utilisateur (Panadol, Doliprane...)
sont le jeu de test final : dossier `data/raw/medicine_boxes/user_photos` prévu, commande
d'évaluation `--set user` prête. La validation « Panadol -> paracétamol » demande une vraie
photo de Panadol, absente du jeu public : une image de texte synthétique ne sert qu'aux tests
unitaires. L'étape reste donc EN ATTENTE PHOTOS : branche poussée, pas de merge ni de tag.

**D109. OCR : RapidOCR servi, EasyOCR comparé ; promotion sur `ingredient_recall`.** Deux
moteurs, une branche `experiment/ocr-<moteur>` chacun : RapidOCR 3.10 (PP-OCRv4 mobile en ONNX,
ONNX Runtime CPU : la même pile que les modèles de comprimés, sans PyTorch dans l'image du
backend) et EasyOCR 1.7 (CRAFT + CRNN, PyTorch, GPU, modèles italien, français et anglais ;
venv d'entraînement seulement). Tesseract n'est pas comparé : son installateur Windows demande
les droits d'administrateur ; TrOCR est exclu par le périmètre du projet (reconnaissance de
lignes seulement, sans détection). Chaque moteur est un run MLflow (expérience `tujya-boxes`)
et une version du Registry `tujya-box-ocr` (modèle pyfunc « défini par code » avec sa
configuration : les moteurs sont préentraînés, rien n'est entraîné ici). Métrique de
promotion : `ingredient_recall`, part des boîtes à principe actif connu dont au moins un vrai
principe actif est trouvé par le code de l'API (gain minimal 0,005, `params.yaml`
`compare.ocr`), avec `promote_if_better` ; la précision et le taux de principe actif faux sont
rapportés à côté (un faux est plus grave qu'un oubli). RxNav est coupé pendant l'évaluation
pour un score reproductible hors ligne. Latence : 30 boîtes, 3 tours séparés de 20 s ; le PC se
bride sous charge continue (D96) : 0,4 à 0,6 s par boîte à froid pour RapidOCR, environ 3,4 s
en régime bridé. Un moteur PyTorch champion ne serait pas servi tel quel : l'API reste en ONNX.

**D110. Rapprochement du texte : table des marques sourcée, DCI imprimée, RxNav en secours,
jamais de devinette.** `config/brand_aliases.csv` : 35 marques vendues en France ou au
Royaume-Uni (Panadol, Doliprane, Efferalgan, Dafalgan, Augmentin...), principes actifs sous
les noms de la table d'interactions ; chaque ligne cite le produit vérifié : la fiche emc
(SmPC) de Panadol Advance, ou un code CIS du fichier des compositions de la BDPM (ANSM, Licence
ouverte, fichier du 03/11/2026, substances actives vérifiées par script). Les marques
italiennes du jeu d'évaluation n'y sont pas ajoutées (sauf celles vendues en France :
Efferalgan, Nurofen, Gaviscon, Imodium) : l'évaluation mesure surtout la lecture de la DCI.
`config/inn_aliases.csv` : orthographes imprimées dont le nom diffère de la table
(paracetamolo -> acetaminophen, source RxNorm ; acido acetilsalicilico...). Comparaison
approximative rapidfuzz (ratio >= 86 pour une marque, >= 88 pour un principe actif), seulement
entre noms de même nombre de mots et de mêmes deux premières lettres (sinon « vitamina » donnait
« vitamin a » et « aciclovir » le famciclovir), mots de moins de 6 lettres exacts, groupes de
mots les plus longs d'abord, chaque mot utilisé une fois. RxNav (`approximateTerm` puis lien
IN) n'est interrogé que si rien n'a été trouvé, pour les 3 mots imprimés les plus grands, et
seulement avec `TUJYA_RXNAV_CONTACT_EMAIL` (même règle que le pipeline de données ; sans
adresse, le secours est désactivé et la réponse le dit). Rien de trouvé : « boîte non
reconnue ». Rejeté : envoyer tout le texte lu à un LLM (risque d'invention).

**D111. Type de photo et refus hors distribution : texte d'abord, puis détection >= 0,5.**
`decide_kind` : un texte qui nomme un médicament, ou au moins 20 lettres lues avec une
confiance >= 0,5, fait une boîte ; sinon une détection de comprimé >= 0,5 fait un comprimé ;
sinon la photo est refusée (`unknown`) et aucune classe n'est proposée ; une boîte ne reçoit
jamais de classe de comprimé. Seuil de détection choisi sur les 995 photos du split de
comparaison et les 19 images hors sujet : à 0,5, 98,1 % des comprimés restent des comprimés et
15 images hors sujet sur 19 sont refusées ; à 0,6, 95,5 % et 17 sur 19. Le gain sur le jeu hors
sujet (2 images) ne vaut pas 2,6 points de comprimés perdus. Les 4 images hors sujet acceptées
(cellule au microscope, horloge, pièces, champ d'étoiles) sont des objets ronds sur un fond uni,
comme un comprimé : limite connue. Rejeté : refuser les images en niveaux de gris (elles sont
toutes en gris dans ce petit jeu, mais une vraie photo ne l'est jamais : le score serait
artificiel) ; un classifieur « pas un comprimé » (bonne piste, demande des négatifs annotés et
un entraînement) ; la probabilité du classifieur (médiane 0,47 sur de vrais comprimés de classes
v2 : elle ne sépare pas).

**D112. Chat : les marques de la question deviennent des principes actifs cités.** Le nœud
`understand` cherche les marques de la table dans la question et ajoute leurs principes actifs ;
chaque marque devient une source citable (`doc_type` `brand`, avec sa référence BDPM ou emc),
placée en tête ; le prompt liste « Marques reconnues ». Nouvel outil `lookup_brand` (chat et
serveur MCP) : marque -> principes actifs avec la source de chaque lien, RxNav en secours,
« non reconnue » sinon. `identify_pills` (MCP) lit aussi les boîtes. Benchmark : 8 questions de
type `brand` ajoutées au jeu du module 2 (96 questions, les 88 autres inchangées), dont 6 sans
chunk attendu (paracétamol et ibuprofène ne sont pas des principes actifs des classes indexées :
l'interaction vient de DDInter) ; `bench_brands.py` les passe dans l'agent réel avec et sans la
table des marques.
Corrections trouvées par ce benchmark : le LLM planificateur passait la marque brute à
`check_interaction` (« Augmentin ») : elle est remplacée par ses principes actifs ; le modèle 3B
citait rarement la source de la marque (1 réponse sur 8) : comme pour la toxicité, une phrase
fixe « Panadol contient : acetaminophen (table des marques) [n]. » est placée en tête quand
aucune source de marque n'est citée ; enfin il a écrit que le paracétamol « n'est pas connu pour
interférer » avec la warfarine alors que sa source DDInter listait une interaction modérée :
quand `check_interaction` trouve une interaction, une phrase qui la nie est retirée et, si aucune
ligne DDInter n'est citée, une phrase fixe la donne avec sa source. Ces phrases fixes viennent
des sources, jamais du LLM.

**D113. Index du RAG dans Qdrant au lieu de pgvector (étape 3.1, décision du propriétaire).**
Remarque du professeur : la base vectorielle n'était pas accessible, on ne voyait pas les
documents utilisés ; le cours du module 2 utilise Qdrant et son tableau de bord. Service
`qdrant` (image `qdrant/qdrant:v1.20.0` épinglée, 127.0.0.1:6333 seulement, volume `qdrant`,
télémétrie coupée), collection `tujya_chunks` : un point par chunk, vecteur nommé `dense`
(cosinus, dimension du modèle), identifiant = UUID v5 du `chunk_id` (Qdrant refuse les
identifiants texte), payload = toutes les métadonnées du chunk + `content_hash` SHA-256 +
`embedding_model`, index de payload sur chaque champ filtré (le filtrage reste exact), modèle,
dimension et date d'indexation dans les métadonnées de la collection. Une seule interface de
stockage (`src/rag/store.py`) avec deux implémentations : Qdrant par défaut, la table SQL
(pgvector, SQLite dans les tests) gardée derrière `TUJYA_VECTOR_STORE=sql`, pour comparer et
revenir en arrière. Migration sans réencodage (`--copy-from-sql` copie les vecteurs) : le
benchmark de récupération donne exactement les mêmes Recall@k et MRR sur les deux stores
(écart 0 sur les 6 configurations, sur 83 puis 85 questions) ; seule la latence change.
BM25 reste en mémoire sur les textes lus dans le store (le mode sparse de Qdrant est l'objet
de l'étape 3.2). Recherche dense exacte, pas de réglage HNSW : 8 661 points. Rejeté : garder
pgvector avec un outil d'administration SQL (le professeur demande de voir documents et
métadonnées, pas des lignes de table) ; Chroma (pas de tableau de bord équivalent, absent du
cours).

**D114. Les documents indexés sont consultables dans l'application : page « Documents » et
`GET /api/documents`, reconstruits depuis les chunks de l'index.** Le catalogue
(`src/rag/documents.py`) regroupe les chunks par `doc_id` dans l'ordre : il montre donc ce que
l'agent peut retrouver, sans copie séparée qui pourrait diverger. Liste paginée avec filtres
(type, principe actif dont parle ou que cite le document, section, langue, source, mots du
titre) et facettes ; détail d'un document avec toutes ses métadonnées, le lien vers la source
et chaque chunk avec sa section et son SHA-256 ; carte de l'index (collection, points, modèle,
date, champs indexés, lien vers le tableau de bord de Qdrant). Réservé à un compte connecté,
comme le chat. Les phrases de posologie chiffrée sont masquées (« […] ») comme dans les
réponses : la règle « jamais de dose » vaut pour tout ce que l'application affiche. 503 quand
l'index est vide ou Qdrant injoignable, pour que la page dise pourquoi elle est vide.
Vérifié dans un vrai navigateur par `frontend/scripts/documents-screenshots.mjs` (captures 27
à 29).

**D115. Validation des boîtes (étape 2.7) sur des photos réelles sous licence libre de
Wikimedia Commons, en attendant les photos du propriétaire.** Le jeu italien n'a pas de
Panadol. Les photos viennent de l'API publique de Commons (`commons.wikimedia.org/w/api.php`,
sans compte ni clé, User-Agent générique, une requête toutes les 3 s) ; seules les licences
CC0, domaine public, CC BY et CC BY-SA sont acceptées, avec l'auteur et la licence exacts de la
page Commons dans `labels.csv` et les crédits dans `docs/data_pipeline/LICENSES.md`. Chaque
photo a été regardée : on garde une boîte réelle dont la marque est lisible en alphabet latin
(11 photos : Panadol x3, Doliprane, Augmentin x3, Glucophage, Levothyrox, Spasfon, Xanax SR).
Écartées : plaquettes seules (Panadol d'Indonésie, Efferalgan codéine), comprimés seuls
(Augmentin 1 g, Augmentin XR), flacons (Voltaren, Advil), rayon de pharmacie avec plusieurs
produits (Nurofen), marque imprimée seulement en chinois (Glucophage de Chine) ou en cyrillique
(Imodium de Russie), et toute image de site marchand ou de fabricant. Les fichiers sont les
miniatures de 1 920 px servies par Commons, renommées, non modifiées ; suivis par DVC
(`data/raw/medicine_boxes/web_commons`), jamais dans git. `--set web` de
`src.models.ocr.evaluate` lit ce dossier comme `--set user` (même format de `labels.csv`) ;
run MLflow `ocr-rapidocr-web`, sans Registry (le champion reste choisi sur le jeu italien).
Ce jeu sert de jeu de test de l'étape ; les photos du propriétaire viendront plus tard comme
second jeu de test (`--set user`), sans changer le code. Limite : 11 photos, prises par des
contributeurs de Commons, souvent bien éclairées ; ce n'est pas une mesure de précision, c'est
la vérification des critères de l'étape sur de vraies boîtes.

**D116. BM25 dans Qdrant par vecteurs sparse `Qdrant/bm25` (fastembed), fusion hybride faite
par Qdrant (étape 3.2).** Comme dans le cours (FastEmbedSparse), chaque point porte un vecteur
sparse `bm25` produit par le modèle `Qdrant/bm25` de fastembed (fréquences de termes saturées,
k1 = 1,2, b = 0,75) et la collection applique le modificateur IDF : Qdrant calcule le BM25 et
le tient à jour seul. Le mode hybride est une requête Qdrant (deux `prefetch` filtrés de 50
candidats, fusion RRF) ; `k = 61` chez Qdrant (positions comptées depuis 0) reproduit exactement
1 / (60 + rang), vérifié par un test sur le client en mémoire et sur le serveur. fastembed tourne
en ONNX Runtime, déjà dans l'image : pas de PyTorch. Qdrant ne sait pas ajouter un vecteur à une
collection existante : l'indexation recrée la collection en recopiant les points (payload et
vecteurs denses, aucun encodage, 9 s) et garde ses métadonnées ; les vecteurs sparse ne sont
recalculés que si le nom du modèle sparse change (`sparse_model` dans les métadonnées). Le
retriever n'utilise le BM25 de Qdrant que si l'index a été fait avec le même encodeur, sinon il
garde le BM25 en mémoire (stockage SQL compris). Mesuré : un peu moins bon que notre BM25 en
mémoire (Recall@5 0,753 contre 0,765), car `Qdrant/bm25` ne retire que les mots vides anglais ;
gardé quand même, parce qu'il fait le calcul dans la base (comme demandé par le cours) et que la
réécriture de requête (D118) supprime l'écart (0,882). Rejeté : vecteurs sparse calculés par
notre propre tokeniseur en production (gardé pour les tests seulement : `TokenBM25`, même formule,
sans téléchargement) ; SPLADE / miniCOIL (modèles appris, anglais seulement, non demandés).

**D117. Reranker cross-encoder BAAI/bge-reranker-base, en ONNX par fastembed, désactivé par
défaut sur ce PC (étape 3.2, remarque 13).** La recherche renvoie `fetch_k` candidats (20 par
défaut), le cross-encoder lit question + titre + texte de chaque candidat, les `k` meilleurs
sont gardés ; les égalités gardent l'ordre de la recherche ; chaque source garde son score de
reranker et son rang d'avant. Le modèle (1,1 Go) est téléchargé une fois depuis Hugging Face sans
compte dans `TUJYA_RAG_MODEL_CACHE_DIR` (volume `fastembed` dans Docker, l'image lit `models/` en
lecture seule). Benchmark (85 questions, fetch_k = 10) : dense 0,906 -> 0,941 de Recall@5 et
0,832 -> 0,861 de MRR ; hybride + réécriture 0,929 -> 0,965 et 0,860 -> 0,889. Mais 13 à 18 s par
question sur le processeur de ce portable (1 à 2 s par paire, le PC se bride sous charge, D96) et
le GPU de 4 Go est pris par Ollama (qwen2.5:3b + bge-m3) : `TUJYA_RAG_RERANK` reste à faux pour
l'agent et la démonstration ; à activer sur une machine avec un GPU libre ou un processeur plus
rapide. Rejeté : bge-reranker-v2-m3 (568 M paramètres, plus lent encore) ; un petit reranker
anglais (ms-marco-MiniLM, rapide mais les questions sont à moitié en français) ; le reranker en
PyTorch (pas de PyTorch dans l'image de l'API, D97).

**D118. Réécriture de requête par le LLM, en option (étape 3.2).** Le corpus est en anglais, la
moitié des questions en français. Le LLM (celui de l'agent) réécrit la question en une requête
anglaise de 20 mots au plus qui garde les noms de médicaments ; la recherche et le reranking
portent sur elle, la réponse sur la question ; le filtre de principes actifs vient toujours de la
question. Garde-fous : réécriture vide, de plus de 200 caractères ou en erreur -> question
d'origine ; mots collés par le petit LLM séparés (défaut trouvé par le benchmark : BM25 tombait à
0,612). Mesuré : BM25 0,753 -> 0,882, dense 0,906 -> 0,941 (Recall@10 1,000), hybride 0,871 ->
0,929 de Recall@5. Désactivée par défaut (`TUJYA_RAG_QUERY_REWRITE`) : un appel LLM de plus par
question, et le planificateur de l'agent écrit déjà une requête anglaise quand il choisit
`search_documents` ; l'étape 3.3 décidera du réglage de l'agent sur la génération, pas seulement
sur la récupération (même prudence que D65 : ne pas régler le système sur son propre jeu de test
sans mesurer les réponses).

## Fin de l'étape 2.3 : classifieurs (2026-11-11)

**D119. La table de classes de l'API passe aux 380 classes de C3PI v2 ; celle de v1 garde son
rôle sous un autre nom ; le corpus RAG servi reste celui de v1.** Le champion de l'étape 2.3 a
380 sorties aux identifiants v2 (D101) : `config/pill_classes.csv` décrit désormais ces 380
classes (`python -m src.data.corpus classes`, copie de `config/splits/c3pi_v2/classes.csv` avec
les mêmes identifiants et les colonnes de principes actifs), puis `ingredients` et
`interactions` refont les tables de `data/processed/pharma` avec la méthode et les sources de
v1 (RxNav : NDC -> RxCUI -> principes actifs de base ; DDInter 2.0, 14 fichiers, reconstruit
depuis les fichiers bruts du disque). Résultat : 379 classes sur 380 associées par leurs NDC
(les 149 de v1 retrouvent exactement leurs principes actifs), 176 principes actifs, 8 611 paires
entre classes et 75 653 avec tout médicament. La classe sans principe actif est la même qu'en v1
(vitamines et minéraux, NDC « ALIEN » chez RxNav) : l'API dit déjà qu'aucune vérification n'est
possible, rien n'est deviné. Huit alias USAN -> DCI ajoutés à `config/ingredient_aliases.csv`
(alendronate, clavulanate, clomiphene, ethinyl estradiol, levomefolate, medroxyprogesterone,
norethindrone, oméga-3) ; le fer (deux formes, une seule entrée « Iron » dans DDInter) n'est pas
rapproché, un nom DDInter ne pouvant désigner qu'un principe actif. La table de v1 devient
`config/pill_classes_v1.csv`, lue par les étapes qui décrivent le jeu v1 (`data_cli select`,
entraînement v1, jobs Spark `split` et `epillid`, statistiques, `prepare_classify_v2`) ; leurs
étapes DVC ne changent que de chemin (`dvc commit -f`). La galerie de démo passe aux identifiants
v2 (même produit, vérifié par `class_key`). Le corpus RAG servi (`rag_corpus`) dépend des tables
de `data/processed/pharma` : il est gelé dans `dvc.yaml` et reste celui des 150 classes de v1
(82 principes actifs), pour que l'index Qdrant et les benchmarks de l'étape 3.3 restent mesurés
sur le corpus décrit ; `check_interaction` et l'écran Vérifier, qui lisent les tables, couvrent
les 380 classes. Les notices et résumés des nouveaux principes actifs viendront de la collecte
RAG v2 (D90). Rejeté : deux tables de classes côte à côte dans l'API (deux sources de vérité,
D101) ; réécrire les sorties du modèle vers les identifiants v1 (230 classes n'en ont pas) ;
reconstruire le corpus RAG maintenant (il ajouterait environ 6 000 paires DDInter sans notice,
plus que tout le corpus actuel, juste avant le benchmark de la récupération).
