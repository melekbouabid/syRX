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

- [x] Axe vision : détection des comprimés avec YOLOv8/v11
- [x] Axe vision : identification de chaque comprimé par transfer learning
- [x] Backend FastAPI asynchrone, schémas Pydantic, Swagger sur `/docs`
- [x] Frontend React : upload d'image, webcam, affichage des prédictions et des bounding boxes
- [ ] DVC (versionnement des données)
- [x] MLflow : tracking des expériences
- [x] MLflow : Model Registry
- [x] Export ONNX et inférence ONNX Runtime
- [x] Docker Compose
- [x] GitHub Actions CI/CD
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

## Lancer l'application

Toute l'application (PostgreSQL + pgvector, MLflow, API, site web) démarre avec une seule
commande, depuis la racine du dépôt :

```bash
docker compose up -d --build
```

- Site : <http://localhost:8080> (compte de démonstration `user@gmail.com` / `user`, puis
  « Scanner ») ; Swagger : <http://localhost:8080/docs> ; MLflow : <http://localhost:5000>.
- Avant le premier démarrage : les données sous `data/` (DVC) et les modèles dans `models/`
  (`cd vision && python -m tujya_vision.publish`, depuis le Model Registry MLflow). Détails :
  [docs/INFRA.md](docs/INFRA.md) ; API : [docs/API.md](docs/API.md).
- Démonstration pas à pas : [docs/SESSION_REPORT_2.md](docs/SESSION_REPORT_2.md) ; captures
  d'écran : [docs/screenshots/](docs/screenshots/).

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
|---|---|---|---|---|
| 2026-10-26 | Commit initial (`9398f0c`) | Import du pipeline SynergyRx, du backend FastAPI, du frontend React (marque Tujya) et du plan Tujya | tout le dépôt | 270 tests unitaires du pipeline ; backend avec comptes, sessions, admin, catalogue |
| 2026-10-27 | Tâche 1 : README et règles du projet | Présentation de Tujya, exigences des modules, sources de données, journal ; règles obligatoires de session dans CLAUDE.md ; nouvelle orientation données (C3PI, pas de photos personnelles, deux étapes détection + identification, ePillID, openFDA) | `README.md`, `CLAUDE.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`, `docs/NIGHT_REPORT.md` | Documentation à jour ; aucune exigence cochée (aucune n'est encore satisfaite) |
| 2026-10-28 | Tâche 2 : exploration de C3PI | Téléchargement de `sampleData.zip` (517 Mo) et des 110 fichiers XML de métadonnées ; outil `vision/tujya_vision` (index des images, sélection, téléchargement vérifié par SHA-1) ; documentation complète du jeu de données | `docs/DATASET_C3PI.md`, `vision/tujya_vision/c3pi.py`, `vision/tujya_vision/data_cli.py`, `vision/tests/test_c3pi.py` | 133 458 fichiers ; 40 587 photos « consumer » de 2 400 comprimés ; 380 comprimés ont au moins 20 photos ; aucune bounding box ni masque pour les photos consumer, masques alpha dans les images de référence segmentées ; 8 tests |
| 2026-10-28 | Tâche 3 : sous-ensemble d'entraînement | 150 classes (comprimés = NDC9 + part) classées par nombre de photos consumer, hors appareils Canon G10 et Nikon S3000 (trop lourds pour le débit) ; jusqu'à 30 photos par classe + 1 image de référence RxNav par NDC ; téléchargement vérifié par SHA-1 ; split par photo 70/15/15 stratifié par classe ; DVC initialisé, images et métadonnées suivies (pas encore de remote) | `config/pill_classes.csv`, `config/splits/selected_images.csv`, `data/vision/c3pi/images.dvc`, `data/vision/c3pi/metadata.dvc`, `.dvc/`, `vision/requirements.txt` | 4 206 fichiers (4,8 Go), 24 à 30 photos par classe, 3 006 train / 600 val / 600 test, 0 échec de téléchargement ; DVC local seulement : case DVC non cochée tant qu'aucun remote n'est configuré |
| 2026-10-28 | Tâche 4 : principes actifs et interactions | Principes actifs par RxNav (NDC -> RxCUI -> ingrédients de base, sans dosage) ; connecteur DDInter existant activé par une étape Tujya (conditions CC BY-NC-SA acceptées, 14 fichiers ATC au lieu de 8) ; 3 alias de noms ; tables d'interactions propres (paire, sévérité, description, source) suivies par DVC | `src/synergyrx_data/tujya/`, `tests/unit/test_tujya.py`, `config/ingredient_aliases.csv`, `docs/INTERACTIONS.md`, `data/tujya.dvc` | 149/150 classes associées (82 principes actifs ; seule la multivitamine n'a pas d'ingrédient) ; 82/82 trouvés dans DDInter ; 2 350 paires entre classes (125 Major, 826 Moderate, 72 Minor, 1 327 Unknown) ; 40 508 paires avec n'importe quel médicament ; 279 tests du pipeline passent |
| 2026-10-28 | Tâche 5 : infrastructure | `docker-compose.yml` : PostgreSQL 16 + pgvector et serveur MLflow 3.17.2 (store PostgreSQL, artefacts dans un volume) ; schéma SQL (sources, pills, ingredients, pill_ingredients, interactions) en SQLAlchemy, séparé de `app.db` ; chargeur idempotent + tests | `docker-compose.yml`, `docker/`, `backend/app/pharma.py`, `backend/app/pharma_loader.py`, `backend/tests/test_pharma.py`, `docs/INFRA.md` | Conteneurs démarrés (postgres healthy, MLflow /health = 200) ; chargé dans PostgreSQL : 150 comprimés, 1 716 ingrédients, 161 liens, 40 508 interactions ; 37 tests backend passent ; case Docker Compose non cochée (backend et frontend pas encore dans le compose) |
| 2026-10-28 | Tâche 6 : pipeline vision (identification) | Matériel : RTX 2050 4 Go (PyTorch CUDA 13.0 installé après les images) ; venv `vision/.venv` ; boîtes automatiques sur les photos consumer (vision classique : fond quadratique robuste, rejet des pièces, GrabCut), photos douteuses exclues ; jeux YOLO détection + crops 224 px ; étape DVC `prepare` ; classifieur YOLO11s-cls par transfer learning, suivi MLflow, export ONNX vérifié avec ONNX Runtime, enregistré dans le Model Registry | `vision/tujya_vision/autolabel.py`, `prepare.py`, `train.py`, `evaluate.py`, `dvc.yaml`, `docs/VISION.md` | 3 306/4 056 photos étiquetées (81,5 %, précision ~90 % sur un audit de 48 photos) ; test top-1 **93,5 %**, top-5 **98,1 %** (589 crops, 148 classes) ; ONNX = PyTorch sur 100 % des crops vérifiés ; `tujya-pill-classifier` v2 dans MLflow ; essai CPU 3 époques : 19,4 % / 36,0 % |
| 2026-10-28 | Tâche 7 : CI | `.github/workflows/ci.yml` : 4 jobs (pipeline : ruff + format + mypy + tests unitaires ; backend : ruff + format + pytest ; vision : ruff + format + tests sans entraînement ; frontend : npm ci + build). Aucun test réseau en direct | `.github/workflows/ci.yml` | En local : 279 tests pipeline, 37 backend, 14 vision passent, ruff/mypy propres, build frontend OK ; résultat GitHub Actions dans le rapport de nuit ; case CI/CD non cochée (pas encore de CD) |
| 2026-10-28 | Tâche 6 (suite) : détection et bout en bout | Détecteur YOLO11n (1 classe « pill ») entraîné 30 époques sur GPU avec les boîtes automatiques, export ONNX vérifié, enregistré `tujya-pill-detector` v1 ; évaluation de bout en bout détecteur -> crop -> classifieur avec les deux modèles ONNX sur les 600 photos de test ; `dvc.lock` de l'étape `prepare` ; auto-installation d'Ultralytics désactivée | `vision/tujya_vision/evaluate.py`, `vision/tujya_vision/__init__.py`, `dvc.lock`, `.dvcignore`, `docs/VISION.md` | Détection test : mAP50 **0,972**, mAP50-95 **0,861**, précision 0,957, rappel 0,936 (par rapport aux boîtes automatiques) ; ONNX = PyTorch (même nombre de boîtes sur 100 %, IoU 0,976) ; bout en bout : comprimé trouvé sur 98,5 % des photos, top-1 **89,2 %**, top-5 **95,8 %** |
| 2026-10-28 | Session 2, tâche 1 : Swagger sur `/docs` | Swagger UI servi sur `/docs` et le schéma sur `/openapi.json` (exigence du module 1) ; `/api/docs` redirige vers `/docs`, `/api/openapi.json` sert le même schéma ; proxy Vite et futur `nginx.conf` du frontend exposent `/docs` et `/openapi.json` ; API renommée « Tujya API » ; tableau du journal remis en ordre (la ligne de séparation était au milieu) | `backend/app/main.py`, `backend/tests/test_docs.py`, `frontend/vite.config.ts`, `frontend/nginx.conf` | 4 nouveaux tests ; 41 tests backend passent |
| 2026-10-28 | Session 2, tâche 2 : API de scan | `POST /api/scan` asynchrone : image (multipart, JPEG/PNG/WebP, 10 Mo max) -> détecteur ONNX -> crop de chaque comprimé -> classifieur ONNX, avec ONNX Runtime (CPU, GPU en option) ; pour chaque comprimé : boîte en pixels et normalisée, top-5 (classe, nom, nom générique, principes actifs, probabilité), `needs_confirmation` sous le seuil 0,85 choisi sur la validation ; taille de l'image et temps d'inférence ; schémas Pydantic, connexion requise ; modèles chargés depuis `SYNERGYRX_MODELS_DIR`, remplis par `python -m tujya_vision.publish` depuis le Model Registry MLflow (503 clair si absents) | `backend/app/scanner.py`, `backend/app/routers/scan.py`, `backend/app/medicines.py`, `backend/app/schemas.py`, `backend/app/limits.py`, `backend/scripts/evaluate_scan.py`, `vision/tujya_vision/publish.py`, `backend/tests/test_scan.py`, `backend/tests/fake_models.py`, `docs/API.md`, `docs/VISION.md` | Même résultat qu'Ultralytics (40/40 photos même nombre de boîtes, IoU 0,9999) ; sur les 600 photos de test : comprimé trouvé 98,5 %, top-1 **89,3 %**, top-5 **95,8 %** ; seuil 0,85 : 70,5 % des photos de test acceptées sans confirmation, 99,3 % justes ; 12 tests avec de petits modèles ONNX factices (pas de poids réels en CI) |
| 2026-10-28 | Session 2, tâche 3 : API d'interactions | `POST /api/interactions` : 2 à 10 comprimés (class_id) et/ou principes actifs ; chaque paire avec sévérité DDInter, description, source (nom, URL, citation), principes actifs communs ; paires triées par sévérité ; sans résultat : « Aucune interaction connue dans cette base de données », jamais « sûr » ; élément inconnu ou comprimé sans principe actif : « aucune vérification n'est possible ». `GET /api/pills?q=` : recherche dans les 150 classes (nom, générique, empreinte, fabricant, principe actif, numéro). Lecture des tables PostgreSQL du chargeur | `backend/app/routers/medicines.py`, `backend/app/medicines.py`, `backend/tests/test_medicines.py`, `docs/API.md` | 16 tests sur SQLite (dont : jamais le mot « safe »/« sûr », avertissement et source toujours présents) ; 69 tests backend passent ; vérifié sur PostgreSQL (warfarine + fluconazole : Major) ; case « Backend FastAPI asynchrone, Pydantic, Swagger `/docs` » cochée |
| 2026-10-28 | Session 2, tâche 4 : page Scanner | Page « Scanner » (comptes connectés) : upload (glisser-déposer) ou webcam (getUserMedia, capture d'une image et prédictions en direct environ toutes les secondes) ou galerie de démo ; photo avec boîtes et étiquettes en superposition SVG ; top-5 de chaque comprimé avec barres de probabilité, badge « Please confirm » sous le seuil, recherche d'un autre comprimé (`GET /api/pills`), confirmation ; liste « My pills » sur plusieurs photos ; dès 2 comprimés, `/api/interactions` affiché par sévérité avec sources et avertissement. Galerie de 8 photos du split de test servies en lecture seule par le backend (`GET /api/demo/images`), jamais commitées. Preuve Playwright | `frontend/src/pages/ScanPage.tsx`, `frontend/src/api.ts`, `frontend/src/styles.css`, `frontend/scripts/screenshots.mjs`, `backend/app/routers/demo.py`, `backend/tests/test_demo.py`, `config/demo_gallery.csv`, `docs/screenshots/` | `npm run build` OK ; script Playwright : connexion, upload d'une photo de test, 2 photos de démo, interactions (fluconazole + citalopram et citalopram + tramadol : Major ; fluconazole + tramadol : Moderate), cas « à confirmer », webcam (caméra factice de Chromium), Swagger : 7 captures dans `docs/screenshots/` ; 74 tests backend ; case « Frontend React » cochée |
| 2026-10-28 | Session 2, tâche 5 : Docker Compose complet | Images backend (Python 3.12, FastAPI, ONNX Runtime CPU, utilisateur non root, healthcheck) et frontend (build Node 22 puis nginx 1.27 qui sert le site et transmet `/api`, `/docs`, `/openapi.json` au backend) ; service ponctuel `pharma-init` qui recharge les tables de médicaments ; modèles, données et config montés en lecture seule (D55) ; comptes dans PostgreSQL ; section « Lancer l'application » | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf`, `docs/INFRA.md`, `README.md` | `docker compose up -d --build` : tous les services sains ; curl via nginx : health 200, `/docs` 200, `/openapi.json` 18 routes, `/api/public/summary` 200, `/api/scan` réel sur une photo de test (fluconazole, 0,996, 59 ms), interactions Major ; parcours Playwright complet sur les conteneurs ; case « Docker Compose » cochée (Ollama viendra avec le module 2) |
| 2026-10-28 | Session 2, tâche 6 : CD | `.github/workflows/cd.yml` : quand `ci` réussit sur un push vers `main`, construit les images backend et frontend du commit testé, les teste (health, `/docs`, `/openapi.json` ; site et route `/scan`), puis les pousse sur GitHub Container Registry (`latest` et `sha-<commit>`) avec `GITHUB_TOKEN` (`packages: write`) ; documentation | `.github/workflows/cd.yml`, `docs/CD.md`, `docs/INFRA.md` | Run CD 40655933124 vert (2 jobs) après le run CI 40655719906 vert ; poussés : `ghcr.io/melekbouabid/tujya-backend` et `tujya-frontend`, tags `latest` et `sha-44b89c7` ; case « GitHub Actions CI/CD » cochée |
| 2026-10-28 | Session 3, tâche 0 : Ollama | Service `ollama` dans `docker-compose.yml` (image officielle `ollama/ollama:0.40.12`, volume nommé `ollama`, port 11434) avec le GPU NVIDIA réservé (Docker Desktop + WSL2) ; variante sans GPU `docker/ollama/cpu.yml` ; 4 petits modèles d'instruction comparés sur une même question en français ; correction du schéma de docs/VISION.md (identifiant YOLO11s-cls) | `docker-compose.yml`, `docker/ollama/cpu.yml`, `docs/INFRA.md`, `docs/DECISIONS.md` (D57, D58), `docs/VISION.md` | GPU vu dans le conteneur (`nvidia-smi`, Ollama : CUDA, 4 Gio) ; **qwen2.5:3b** retenu : 100 % GPU, 2,2 Go, 47 jetons/s, **1,7 s** pour la réponse ; llama3.2:3b 3,2 s, gemma3:4b 12,1 s, qwen3:4b 90 s (raisonnement) |
| 2026-10-28 | Session 3, tâche 1 : corpus documentaire | Connecteur openFDA `openfda_label` dans le style du pipeline (PoliteClient, 3 req/s, cache, manifeste, provenance) : une notice par principe actif selon une règle documentée (D60), sections de posologie jamais stockées ; mode `ingredients` du connecteur PubMed existant (interactions et effets indésirables, 20 résumés au plus par principe actif, tri par pertinence, D61) ; paires DDInter entre les classes (D62) ; découpage par section (notices), par paragraphe (résumés), une ligne par paire (DDInter), avec métadonnées complètes ; dossier `data/rag` suivi par DVC (`dvc add` + étape `rag_corpus`) | `src/synergyrx_data/connectors/openfda.py`, `connectors/pubmed.py`, `literature/queries.py`, `tujya/corpus.py`, `tujya/__main__.py`, `tests/unit/test_tujya_rag.py`, `dvc.yaml`, `data/rag/*.dvc`, `docs/RAG.md` | 82/82 notices (81 sur ordonnance, 1 OTC ; 975 sections) ; 1 570 résumés PubMed (17 à 20 par principe actif, 4 rétractés signalés) ; 2 350 paires DDInter ; **8 661 chunks** pour 4 002 documents ; 8 nouveaux tests, 287 tests du pipeline passent |

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
# Terminal 1: API on http://127.0.0.1:8000 (Swagger UI at /docs, schema at /openapi.json)
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

Open <http://localhost:5173>. The dev server forwards `/api`, `/docs` and `/openapi.json` to
the backend, so the browser talks to a single origin (Swagger: <http://localhost:5173/docs>;
the old address `/api/docs` redirects there). Visitors first see a public introduction page; it shows the table
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
