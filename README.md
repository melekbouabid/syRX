# Tujya

Tujya est une application de sécurité du médicament, réalisée comme projet étudiant solo.
On photographie un ou plusieurs comprimés (upload ou webcam) : un modèle YOLO **détecte**
chaque comprimé, puis un classifieur entraîné par *transfer learning* l'**identifie** parmi
une liste fermée de 100 à 200 comprimés du jeu de données public NLM C3PI. Chaque comprimé
est relié à son ou ses principes actifs, et Tujya vérifie les **interactions** entre ces
principes actifs dans DDInter. Un agent LangGraph répond ensuite aux questions de suivi par
RAG (notices openFDA, résumés PubMed, DDInter), en **citant ses sources**.

> Information uniquement : Tujya ne remplace pas l'avis d'un pharmacien ou d'un médecin.
> Tujya ne donne jamais de posologie et ne dit jamais qu'une association est « sûre » :
> il indique au mieux « aucune interaction connue dans cette base ».

Règles du projet pour les sessions de travail : [CLAUDE.md](CLAUDE.md). Feuille de route :
[docs/ROADMAP.md](docs/ROADMAP.md). Décisions : [docs/DECISIONS.md](docs/DECISIONS.md).

## Exigences des modules

Une case n'est cochée que lorsque l'exigence est réellement satisfaite.

### Module 1 : Industrialisation Vision & MLOps

- [ ] Axe vision : détection des comprimés avec YOLOv8/v11
- [ ] Axe vision : identification de chaque comprimé par transfer learning
- [ ] Backend FastAPI asynchrone, schémas Pydantic, Swagger sur `/docs`
- [ ] Frontend React : upload d'image, webcam, affichage des prédictions et des bounding boxes
- [ ] DVC (versionnement des données)
- [ ] MLflow : tracking des expériences
- [ ] MLflow : Model Registry
- [ ] Export ONNX et inférence ONNX Runtime
- [ ] Docker Compose
- [ ] GitHub Actions CI/CD
- [ ] Soutenance pratique (démonstration technique en direct)

### Module 2 : AI Engineering

- [ ] RAG avec gestion des documents, métadonnées, citations, documentation
- [ ] Tool calling / MCP
- [ ] LangChain et/ou LangGraph
- [ ] Benchmarking de la récupération et de la génération
- [ ] Interface web
- [ ] Dépôt GitHub documenté avec commits réguliers
- [ ] Rapport technique
- [ ] Démonstration

## Sources de données

| Jeu de données | Lien | Licence | Usage dans Tujya |
|---|---|---|---|
| NLM C3PI (Computational Photography Project for Pill Identification) | [catalog.data.gov](https://catalog.data.gov/dataset/computational-photography-project-for-pill-identification-c3pi), [fichiers](https://data.lhncbc.nlm.nih.gov/public/Pills/) | ODbL ; usage recherche et développement uniquement | Images de comprimés (photos « consumer » et images de référence) pour la détection et l'identification ; métadonnées (NDC, nom, principe actif) |
| ePillID | [arXiv:2005.14288](https://arxiv.org/abs/2005.14288) | article (benchmark construit sur C3PI) | Benchmark de référence pour comparer l'identification |
| DDInter 2 | [ddinter2.scbdd.com](https://ddinter2.scbdd.com/) | CC BY-NC-SA 4.0 (usage académique non commercial) | Interactions entre principes actifs : sévérité, description |
| RxNorm / RxNav (NLM) | [lhncbc.nlm.nih.gov/RxNav](https://lhncbc.nlm.nih.gov/RxNav/APIs/) | conditions d'utilisation UMLS / RxNorm (API publique) | Correspondance NDC / nom -> principe(s) actif(s) si les métadonnées C3PI ne suffisent pas |
| openFDA drug labels | [open.fda.gov/apis/drug/label](https://open.fda.gov/apis/drug/label/) | CC0 / domaine public (conditions openFDA) | Corpus RAG : notices officielles, avec métadonnées et citations |
| PubMed (E-utilities) | [ncbi.nlm.nih.gov/books/NBK25501](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | conditions NLM | Corpus RAG : résumés d'articles |

Les données ne sont jamais commitées dans git (plusieurs sources interdisent la
redistribution) : elles sont sous `data/`, versionnées avec DVC. Les sources de l'ancien
pipeline SynergyRx sont décrites dans l'annexe ci-dessous.

## Journal de progression

| Date | Tâche | Ce qui a été fait | Fichiers | Résultats |
| 2026-10-28 | Tâche 2 : exploration de C3PI | Téléchargement de `sampleData.zip` (517 Mo) et des 110 fichiers XML de métadonnées ; outil `vision/tujya_vision` (index des images, sélection, téléchargement vérifié par SHA-1) ; documentation complète du jeu de données | `docs/DATASET_C3PI.md`, `vision/tujya_vision/c3pi.py`, `vision/tujya_vision/data_cli.py`, `vision/tests/test_c3pi.py` | 133 458 fichiers ; 40 587 photos « consumer » de 2 400 comprimés ; 380 comprimés ont au moins 20 photos ; aucune bounding box ni masque pour les photos consumer, masques alpha dans les images de référence segmentées ; 8 tests |
| 2026-10-28 | Tâche 3 : sous-ensemble d'entraînement | 150 classes (comprimés = NDC9 + part) classées par nombre de photos consumer, hors appareils Canon G10 et Nikon S3000 (trop lourds pour le débit) ; jusqu'à 30 photos par classe + 1 image de référence RxNav par NDC ; téléchargement vérifié par SHA-1 ; split par photo 70/15/15 stratifié par classe ; DVC initialisé, images et métadonnées suivies (pas encore de remote) | `config/pill_classes.csv`, `config/splits/selected_images.csv`, `data/vision/c3pi/images.dvc`, `data/vision/c3pi/metadata.dvc`, `.dvc/`, `vision/requirements.txt` | 4 206 fichiers (4,8 Go), 24 à 30 photos par classe, 3 006 train / 600 val / 600 test, 0 échec de téléchargement ; DVC local seulement : case DVC non cochée tant qu'aucun remote n'est configuré |
| 2026-10-28 | Tâche 4 : principes actifs et interactions | Principes actifs par RxNav (NDC -> RxCUI -> ingrédients de base, sans dosage) ; connecteur DDInter existant activé par une étape Tujya (conditions CC BY-NC-SA acceptées, 14 fichiers ATC au lieu de 8) ; 3 alias de noms ; tables d'interactions propres (paire, sévérité, description, source) suivies par DVC | `src/synergyrx_data/tujya/`, `tests/unit/test_tujya.py`, `config/ingredient_aliases.csv`, `docs/INTERACTIONS.md`, `data/tujya.dvc` | 149/150 classes associées (82 principes actifs ; seule la multivitamine n'a pas d'ingrédient) ; 82/82 trouvés dans DDInter ; 2 350 paires entre classes (125 Major, 826 Moderate, 72 Minor, 1 327 Unknown) ; 40 508 paires avec n'importe quel médicament ; 279 tests du pipeline passent |
| 2026-10-28 | Tâche 5 : infrastructure | `docker-compose.yml` : PostgreSQL 16 + pgvector et serveur MLflow 3.17.2 (store PostgreSQL, artefacts dans un volume) ; schéma SQL (sources, pills, ingredients, pill_ingredients, interactions) en SQLAlchemy, séparé de `app.db` ; chargeur idempotent + tests | `docker-compose.yml`, `docker/`, `backend/app/pharma.py`, `backend/app/pharma_loader.py`, `backend/tests/test_pharma.py`, `docs/INFRA.md` | Conteneurs démarrés (postgres healthy, MLflow /health = 200) ; chargé dans PostgreSQL : 150 comprimés, 1 716 ingrédients, 161 liens, 40 508 interactions ; 37 tests backend passent ; case Docker Compose non cochée (backend et frontend pas encore dans le compose) |
|---|---|---|---|---|
| 2026-10-26 | Commit initial (`c1cb26d`) | Import du pipeline SynergyRx, du backend FastAPI, du frontend React (marque Tujya) et du plan Tujya | tout le dépôt | 270 tests unitaires du pipeline ; backend avec comptes, sessions, admin, catalogue |
| 2026-10-27 | Tâche 1 : README et règles du projet | Présentation de Tujya, exigences des modules, sources de données, journal ; règles obligatoires de session dans CLAUDE.md ; nouvelle orientation données (C3PI, pas de photos personnelles, deux étapes détection + identification, ePillID, openFDA) | `README.md`, `CLAUDE.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`, `docs/NIGHT_REPORT.md` | Documentation à jour ; aucune exigence cochée (aucune n'est encore satisfaite) |

---

## Annexe : pipeline de données SynergyRx

Data collection pipeline for **SynergyRx AI**: prediction of drug-combination synergy on
cancer cell lines, enriched with knowledge extracted from PubMed.

It collects public scientific data from a fixed list of sources, politely and reproducibly:
official APIs first, official bulk files second, and no HTML scraping. It is not a
general-purpose web scraper and refuses to talk to any host that is not one of its sources.

### What you get

Everything the pipeline produces is in one folder, `data/`, with one sub-folder per file
format. The extension of a file tells you where it is:

```
data/
  parquet/   the tables to model on
  csv/       coverage, conflicts, review queues, sentences to label
  json/      files for the frontend
  reports/   the data quality report (Markdown and HTML), the annotation guide

  raw/       files exactly as downloaded, per source      interim/   one validated table per source
  manual/    files you place yourself                     cache/ state/   bookkeeping
```

| File | Content |
|---|---|
| `data/parquet/synergy_pairs.parquet` | One row per drug pair, cell line, source and study. Bliss, Loewe, HSA and ZIP in separate columns |
| `data/parquet/drugs.parquet` | One row per drug: standardised SMILES, InChIKey, PubChem CID, ChEMBL ID, synonyms, validity flag |
| `data/parquet/cell_lines.parquet` | Names, tissue, DepMap identifier |
| `data/parquet/expression.parquet` | Expression matrix for the cell lines in use |
| `data/parquet/abstracts.parquet`, `candidate_sentences.parquet` | PubMed abstracts, and sentences mentioning two or more drugs |
| `data/csv/coverage.csv`, `conflicts.csv`, `source_agreement.csv` | Distinct drugs and cell lines per source; records that disagree; agreement between sources |
| `data/csv/drug_review_queue.csv`, `cell_line_review_queue.csv` | Names a person should check |
| `data/csv/sample_sentences.csv` | Stratified sentences to label by hand (also `data/json/sample_sentences.jsonl`) |
| `data/json/*.json` | `drugs.json`, `cell_lines.json`, `mock_predictions_seed.json`, `citations.json` for the frontend |
| `data/reports/data_quality.md` / `.html` | The data quality report; `annotation_guide.md` explains the labels |
| `manifest.json`, `LICENSES.md` | SHA-256, size, URL, version, retrieval time and licence of every file |

Every column is explained in [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

`data/raw` and `data/interim` are the pipeline's working folders and keep one sub-folder per
source. `manifest.json` and `LICENSES.md` sit next to `data/` rather than inside it: they
hold no data, they describe the folder, and unlike `data/` they can be committed.
`--data-dir` moves the whole folder; `parquet_dir`, `csv_dir`, `json_dir` and `reports_dir`
in `config.yaml` can point one format elsewhere.

### Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env                 # then edit .env
```

Put two values in `.env` (it is git-ignored and never logged):

- `CONTACT_EMAIL`: sent in the User-Agent so a provider can reach you. **Required**: live
  requests are refused without it.
- `NCBI_API_KEY`: optional, free from your NCBI account settings. Raises the E-utilities
  limit from about 3 to about 10 requests per second.

Then check the machine:

```bash
synergyrx preflight
```

### Running it

```bash
synergyrx fetch --sample 100 --dry-run   # print the plan; requests and writes nothing
synergyrx fetch --sample 100             # small sample of every enabled source
synergyrx fetch                          # full run (about 740 MB for stage 1)
synergyrx clean                          # build the processed tables (data/parquet)
synergyrx fetch --source pubmed --source europepmc
synergyrx literature                     # abstracts, candidate sentences, annotation sample
synergyrx report                         # quality report, coverage, conflicts, LICENSES.md
synergyrx export                         # frontend JSON
synergyrx summary                        # final report of everything collected
```

Or everything in one go, incrementally: `synergyrx update`.

`fetch` runs in stages, because later sources need earlier results:

1. Drug combinations and cell lines: DrugComb, TDC, O'Neil, NCI-ALMANAC, DepMap.
2. PubChem resolves the drug names found in stage 1.
3. ChEMBL looks up the InChIKeys PubChem returned.
4. PubMed, with queries built from the cleaned drug table (so run `clean` first).
5. Europe PMC (open-access status) and PubTator, for the PMIDs PubMed returned.

#### Commands

| Command | What it does |
|---|---|
| `sources` | List sources, licences and state. `--check` probes every link and rewrites `docs/SOURCES.md` |
| `fetch` | Download, parse and validate into `data/raw` and `data/interim`. `--source`, `--sample N`, `--dry-run`, `--refresh` |
| `validate` | Re-check raw file hashes against the manifest and interim rows against their schemas |
| `clean` | Build the processed tables in `data/parquet` |
| `literature` | Abstract store, sentence segmentation, drug mentions, annotation sample |
| `report` | Quality report, coverage, conflicts, licence registry |
| `export` | Frontend JSON |
| `status` | State of every connector and what is on disk |
| `summary` | Final report of everything collected (`--json` for machines) |
| `preflight` | Disk space, permissions, identity, tooling |
| `update` | All of the above, incrementally. For cron or CI |

Global options go before the command: `--config`, `--set key=value`, `--data-dir`,
`--log-level`, `--text-logs`. Any setting in `config.yaml` can be overridden:

```bash
synergyrx --set sources.tdc.enabled=false --set "http.hosts[zenodo.org].rate=0.05" fetch
```

Logs are JSON lines on stderr; tables and reports go to stdout.

#### Second runs are cheap

Nothing is downloaded or duplicated twice. Raw files are skipped when their hash matches the
manifest, API answers are cached on disk and revalidated with ETag / Last-Modified, lookups
(including "not found") are remembered, interrupted downloads resume with an HTTP Range
request, and PubMed is only asked for records entered since the previous run.

### Sources

| Source | Access | On by default | Licence |
|---|---|---|---|
| DrugComb v1.4 | Zenodo bulk file, pinned | yes | CC BY 4.0 |
| TDC DrugSyn | Harvard Dataverse file (loaded with PyTDC when installed) | yes | CC BY 4.0 |
| O'Neil / DeepSynergy | bulk CSV | yes | none stated: research use, do not redistribute |
| NCI-ALMANAC | **manual download** | yes | US Government work |
| DepMap / CCLE 24Q4 | Figshare+ API and files | yes | CC BY 4.0 |
| PubChem | PUG-REST API | yes | public domain |
| ChEMBL | web services API | yes | CC BY-SA 3.0 |
| PubMed | E-utilities API | yes | NLM terms |
| Europe PMC | REST API | yes | Europe PMC terms |
| GDSC (Cell Model Passports) | bulk file | no | not verified |
| DDInter | bulk CSV, gated | no | CC BY-NC-SA 4.0 |
| DrugBank | **file you supply** | no | DrugBank licence |
| PubMed baseline | bulk XML, very large, **manual download** | no | NLM terms |
| PubTator 3 | API | no | public domain |

Link statuses, endpoints, rate limits and what was found for each source are in
[docs/SOURCES.md](docs/SOURCES.md). Licences and citations are in `LICENSES.md`.

#### Manual steps

- **NCI-ALMANAC.** robots.txt of `wiki.nci.nih.gov` disallows automated downloads. Open the
  [NCI-ALMANAC page](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC)
  in a browser and save `ComboDrugGrowth_Nov2017.zip` (and, optionally,
  `ComboCompoundNames_small.txt`) into `data/manual/nci_almanac/`, then run `fetch` again.
- **PubMed baseline** (optional). `ftp.ncbi.nlm.nih.gov` disallows automated clients in
  robots.txt. Download the files you listed in `sources.pubmed_baseline.options.files`
  yourself and save them into `data/manual/pubmed_baseline/`.
- **DrugBank.** Never downloaded. If you hold a licence, save the vocabulary export as
  `data/manual/drugbank/drugbank vocabulary.csv` and set `sources.drugbank.enabled: true`.
- **DDInter.** Read its [terms](https://ddinter2.scbdd.com/terms/) (non-commercial,
  share-alike), then set `sources.ddinter.enabled: true` and `terms_acknowledged: true`.
- **TDC through PyTDC.** PyTDC only installs on Python 3.11 / 3.12:
  `pip install -e ".[tdc]"`. Without it the same file is read by a built-in loader.
- **Unresolved drugs and cell lines.** Work through `data/csv/drug_review_queue.csv` and
  `data/csv/cell_line_review_queue.csv`, record decisions in
  `config/manual_drug_overrides.csv` and `config/cell_line_aliases.csv`, and run `clean`.

### How it behaves on the network

- Identifies itself: `synergyrx-data/<version> (research data pipeline; mailto:<you>)`.
- Per-host token-bucket rate limits and connection caps, run at 90 % of each limit; a
  robots.txt `Crawl-delay` always wins when it is slower.
- Exponential backoff with jitter; honours `Retry-After`, 429 and 503.
- A circuit breaker pauses a source that keeps failing.
- Timeouts, maximum response sizes and content-type checks on every response.
- robots.txt is checked before any HTML page or bulk file is requested.
- A 401 / 403, CAPTCHA, login page or bot check stops the source. It is reported as
  "needs attention" and never retried or worked around.
- No cookies, no credentials, no proxies, no browser impersonation.

A source that fails does not stop the run: the others continue and `status` shows which one
needs attention and why. `fetch` and `update` exit non-zero in that case.

### Scheduling

`synergyrx update` is safe to run repeatedly. Examples:
[docs/examples/crontab.txt](docs/examples/crontab.txt) and
[docs/examples/github-actions-update.yml](docs/examples/github-actions-update.yml).

### Docker

```bash
docker build -t synergyrx-data .
docker run --rm --env-file .env -v "$PWD/data:/app/data" synergyrx-data fetch --sample 100
```

Add `--build-arg INSTALL_TDC=1` to include PyTDC. Secrets are passed at run time and are
not part of the image.

### Web app

A FastAPI backend (`backend/`) and a React frontend (`frontend/`) on top of the processed
tables. They are separate from the pipeline: the backend has its own virtual environment and
only ever reads the tables in `data/parquet`.

```bash
# Terminal 1: API on http://127.0.0.1:8000 (interactive docs at /api/docs)
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1           # Linux / macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

# Terminal 2: site on http://localhost:5173
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The dev server forwards `/api` to the backend, so the browser
talks to a single origin. Visitors first see a public introduction page; it shows the table
sizes (`/api/public/summary`), which is the only data readable without an account.

The site is branded **Tujya**. The logo files in `frontend/src/assets/` and the favicon are
cut from `photo.png`, the name is in `frontend/src/brand.ts`, and the colours, all taken from
the logo, are the tokens at the top of `frontend/src/styles.css`.

The first start creates two accounts in `backend/app.db`:

| Email | Password | Lands on |
|---|---|---|
| `admin@gmail.com` | `admin` | `/admin`: dashboard and user management |
| `user@gmail.com` | `user` | `/home`: home page, drugs, cell lines, profile |

**Change the administrator password before anyone else can reach the server** (Admin, then
My account). The API prints a warning at every start until you do. To start with other
credentials, copy `backend/.env.example` to `backend/.env` before the first start.

- Anyone can sign up; a new account is always a regular user. Only an administrator can
  create administrators, change roles, reset passwords, deactivate or delete accounts.
- Passwords are stored as Argon2 hashes and can be replaced but never read back. New
  passwords need at least 8 characters.
- The session is an `HttpOnly` cookie backed by a row in the database, so resetting a
  password or deactivating an account signs that user out immediately.
- The role is checked by the API on every request; hiding a page in the frontend is only
  for convenience.

Checks: `python -m pytest` in `backend/` (API tests on a temporary database), and
`npm run build` in `frontend/` (type check and production build).

### Development

```bash
make check        # lint, type check, unit tests (mocked HTTP, no network)
make live         # optional smoke tests against the real services, tiny samples
pre-commit install
```

Without `make` (plain Windows): `python -m ruff check src tests`, `python -m mypy`,
`python -m pytest tests/unit`, `python -m pytest tests/live -m live`.

Adding a source means writing one `Connector` subclass (`discover`, `fetch`, `parse`,
`validate`) and registering it in `src/synergyrx_data/connectors/__init__.py`. Read
[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) first.

### Layout

```
config.yaml              settings (no secrets)        .env        secrets (git-ignored)
src/synergyrx_data/
  cli.py  pipeline.py  build.py  config.py  manifest.py
  http/         client, ratelimit, retry, cache, robots, circuit, guardrails
  connectors/   one module per source
  cleaning/     smiles, drugs, cell_lines, pairs, scores
  literature/   queries, abstracts, sentences, mentions, annotation_export
  quality/      reports, drift, licenses, linkcheck
  export/       frontend
tests/unit/     mocked HTTP        tests/live/   optional smoke tests
backend/app/    FastAPI: accounts, sessions, admin routes, read access to data/parquet
frontend/src/   React: introduction, sign in / sign up, user pages, admin pages
data/           everything produced, by format: parquet, csv, json, reports; and the working
                folders raw, interim, manual, cache, state   (git-ignored)
docs/           PLAN, STATUS, DECISIONS, SOURCES, DATA_DICTIONARY, CODE_REVIEW, examples
```
