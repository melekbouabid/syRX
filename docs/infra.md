# Infrastructure (Docker Compose)

[`docker-compose.yml`](../docker-compose.yml), à la racine du dépôt, lance toute
l'application en une commande, y compris Ollama (LLM local et embeddings de l'agent du
module 2, sur le GPU). Les fichiers des services sont dans [infra/](../infra/).

```bash
docker compose up -d --build        # le premier démarrage construit les images (quelques minutes)
docker compose ps                   # postgres, backend, frontend sains ; pharma-init terminé (0)
# site http://localhost:8080   API http://localhost:8000/docs   MLflow http://localhost:5000
docker compose down                 # arrête ; les volumes sont gardés (down -v les supprime)
```

| Service | Image | Port hôte (127.0.0.1 seulement) | Rôle |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | volume `pgdata` ; base `tujya` (comptes, tables des médicaments, pgvector) et base `mlflow` |
| `mlflow` | construite depuis [`infra/docker/mlflow/Dockerfile`](../infra/docker/mlflow/Dockerfile) (MLflow 3.17.2 + psycopg2) | 5000 | stockage du tracking dans PostgreSQL, artefacts dans le volume `mlartifacts`, Model Registry |
| `pharma-init` | image du backend | aucun | ponctuel : `python -m app.pharma_loader` reconstruit les tables des médicaments depuis `config/` et `data/processed/pharma/`, puis s'arrête ; le backend l'attend |
| `rag-init` | image du backend | aucun | ponctuel : `python -m src.rag.index --pull` télécharge les modèles Ollama s'ils manquent, puis indexe `data/processed/rag/corpus` dans la table `rag_chunks` (idempotent ; [rag.md](module2/rag.md)) ; le backend ne l'attend pas |
| `backend` | construite depuis [`backend/Dockerfile`](../backend/Dockerfile) (Python 3.12, FastAPI, ONNX Runtime CPU, LangGraph) | 8000 | l'API et l'agent (`/api/chat`, parle à `ollama`) ; Swagger sur `/docs` |
| `frontend` | construite depuis [`frontend/Dockerfile`](../frontend/Dockerfile) (build Node 22, nginx 1.27) | 8080 | le site React ; nginx transmet `/api`, `/docs`, `/openapi.json` à `backend` ([`nginx.conf`](../frontend/nginx.conf)) |
| `ollama` | `ollama/ollama:0.40.12` | 11434 | LLM local (`qwen2.5:3b`) et embeddings (`bge-m3`) ; modèles dans le volume `ollama` ; GPU NVIDIA réservé (D57) |

L'image de l'API se construit depuis la racine du dépôt (elle importe le paquet `src`) avec une liste blanche, [`backend/Dockerfile.dockerignore`](../backend/Dockerfile.dockerignore) :
seuls le code de l'API (dont la chaîne de scan `backend/app/inference.py`), `src/{rag,agent,mcp_server}`
et `src/models/toxicity/model.py` y entrent (contexte d'environ 200 Ko), dans la même disposition
que le dépôt sous `/tujya` (D79).

**Avant le premier démarrage** (les images ne contiennent ni données ni modèles, D55) :

1. Les données sous `data/` (DVC, [dataset_c3pi.md](module1/dataset_c3pi.md),
   [interactions.md](module2/interactions.md)) : `dvc pull` depuis le
   remote par défaut (section « Remote DVC » plus bas) donne `data/processed/pharma/*.csv` (tables des
   médicaments), `data/raw/c3pi/images/` (galerie de démonstration) et `data/processed/rag/corpus`
   (index RAG).
2. Les modèles : `python -m src.models.publish` remplit
   `models/` depuis le Model Registry de MLflow (MLflow doit tourner :
   `docker compose up -d mlflow` d'abord).

Ces dossiers sont montés **en lecture seule** dans le backend (`/tujya/config`,
`/tujya/data/...`, `/tujya/models` ; l'image garde la disposition du dépôt sous `/tujya`, donc
les chemins par défaut de `app/config.py` n'ont besoin d'aucun réglage). Sans modèles,
`/api/scan` répond 503 et dit comment les publier ; sans les fichiers CSV, `pharma-init`
échoue et le backend ne démarre pas.

Comptes : dans Docker, les tables des comptes sont dans la même base PostgreSQL `tujya` (pas
dans `backend/app.db`, qui ne sert qu'au serveur de développement local). Le premier
démarrage crée l'administrateur par défaut et le compte de démonstration (`user@gmail.com` /
`user`) ; changer le mot de passe administrateur avant que quiconque d'autre puisse joindre
le serveur.

Les images sont toujours construites en local par `docker compose up -d --build`. Le workflow
de CD construit et teste les mêmes images sur GitHub mais ne les publie plus (D70,
[docs/cd.md](cd.md)).

[`infra/docker/postgres/init/01-init.sql`](../infra/docker/postgres/init/01-init.sql) s'exécute
une fois, sur un volume vide : il active `vector` dans la base `tujya` et crée la base
`mlflow`. Les identifiants sont des valeurs de développement (`tujya` / `tujya`) ; on peut
surcharger `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`,
`MLFLOW_PORT`, `BACKEND_PORT`, `WEB_PORT` dans le shell ou un fichier `.env` de compose.

## Usage strictement local (depuis le 2026-10-28)

Le projet tourne **uniquement sur ce PC** (règle de [docs/project_rules.md](project_rules.md)) : pas de déploiement public,
pas de tunnel (ngrok, Cloudflare Tunnel...), pas d'image Docker de l'application publiée
([docs/cd.md](cd.md)), et aucun service lié à `0.0.0.0` sur la machine. Le dépôt GitHub reste
public (code et documentation seulement).

| Quoi | Écoute sur | Comment |
|---|---|---|
| postgres, mlflow, backend, ollama, frontend (Docker) | `127.0.0.1:<port>` | `ports: ["127.0.0.1:${WEB_PORT:-8080}:80"]` etc. dans `docker-compose.yml` |
| serveur Vite / `vite preview` | `127.0.0.1:5173` / `127.0.0.1:4173` | `server.host` et `preview.host` dans [`frontend/vite.config.ts`](../frontend/vite.config.ts) |
| uvicorn (développement local) | `127.0.0.1:8000` | `uvicorn app.main:app --host 127.0.0.1 --reload` (aussi le défaut d'uvicorn) |
| serveur MCP, transport HTTP | `127.0.0.1:8765` | `--host` n'accepte que `127.0.0.1`, `localhost` ou `::1` ([`src/mcp_server/server.py`](../src/mcp_server/server.py), testé) |
| interface MLflow | `127.0.0.1:5000` | le port de compose ; ne jamais lancer `mlflow ui --host 0.0.0.0` sur la machine |

Dans les conteneurs, uvicorn, MLflow et nginx écoutent toujours sur `0.0.0.0` : c'est
l'interface réseau privée du conteneur, joignable seulement par la redirection de port de
Docker, qui n'écoute elle-même que sur la boucle locale de la machine.

Vérification (2026-10-28, après `docker compose up -d --build`) :

```
$ docker compose ps --format "table {{.Name}}	{{.Ports}}"
tujya-backend-1    127.0.0.1:8000->8000/tcp
tujya-frontend-1   127.0.0.1:8080->80/tcp
tujya-mlflow-1     127.0.0.1:5000->5000/tcp
tujya-ollama-1     127.0.0.1:11434->11434/tcp
tujya-postgres-1   127.0.0.1:5432->5432/tcp

$ netstat -ano | findstr LISTENING     (les cinq ports)
  TCP    127.0.0.1:5000         0.0.0.0:0              LISTENING
  TCP    127.0.0.1:5432         0.0.0.0:0              LISTENING
  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING
  TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING
  TCP    127.0.0.1:11434        0.0.0.0:0              LISTENING
```

Plus aucune ligne `0.0.0.0:<port>` ni `[::]:<port>` (avant : `0.0.0.0:8080->80/tcp,
[::]:8080->80/tcp`). `http://localhost:8080`, `/docs`, `/api/health` et `/health` de MLflow
répondent 200 ; les mêmes ports sur les adresses réseau local et WSL du PC (192.168.1.6,
172.x) refusent la connexion. Le serveur Vite n'écoute que sur `127.0.0.1:5173`.

**Utiliser `127.0.0.1`, pas `localhost`, dans les chaînes de connexion des outils lancés
sur la machine** (D72). Sur ce PC, `localhost` se résout d'abord en `::1`, et Docker n'écoute
plus qu'en IPv4 : navigateurs et curl repassent aussitôt en `127.0.0.1`, mais libpq (psycopg)
attendait **65 à 100 s** par connexion (`mcp_server/smoke.py` prenait 2 min 34 s au lieu de
5 s). Les valeurs par défaut et la documentation utilisent donc `127.0.0.1`
(`SYNERGYRX_OLLAMA_URL`, `MLFLOW_TRACKING_URI`, les URL PostgreSQL) ;
`http://localhost:8080` dans un navigateur ne pose aucun problème.

## Remote DVC (depuis le 2026-10-28)

Remote par défaut `localstore` : un dossier de ce PC, hors du dépôt (D71).

```bash
dvc remote add -d localstore C:\Users\msi\dvc-storage\tujya   # déjà dans .dvc/config
dvc push            # cache -> remote (15 306 fichiers, 5,0 Go le 2026-10-28)
dvc pull            # remote -> cache -> espace de travail (nouveau clone)
dvc status -c       # « Cache and remote 'localstore' are in sync. »
dvc remote list     # localstore  C:\Users\msi\dvc-storage\tujya  (default)
```

DVC est installé dans le venv d'entraînement
(`.venv-train/Scripts/dvc.exe`, DVC 3.67.1). Ce qui est suivi :
`data/raw/c3pi/images` et `metadata` (`dvc add`), `data/processed/pharma`, `data/rag/raw`,
`data/rag/interim`, `data/rag/manifest.json`, `data/raw/tox21` (`dvc add`), et les
sorties des étapes `prepare` (étiquettes, jeux YOLO) et `rag_corpus` (chunks) de
[`dvc.yaml`](../dvc.yaml).

Le remote contient des données C3PI (ODbL, usage de recherche), DDInter (CC BY-NC-SA, aucune
redistribution hors de ses conditions) et openFDA/PubMed : il doit rester privé. Ce dossier
n'est pas une sauvegarde : le copier sur un disque externe si le PC en est la seule copie.

**Passer plus tard à Google Drive ou S3** (commandes seulement, rien n'est configuré) :

```bash
# Google Drive (pip install "dvc[gdrive]") : identifiant pris dans l'URL du dossier
dvc remote add gdrive gdrive://<id-du-dossier>
dvc remote default gdrive
dvc push                                                         # ouvre un navigateur pour OAuth, une fois

# Amazon S3 (pip install "dvc[s3]") : identifiants du profil AWS CLI, jamais dans git
dvc remote add s3store s3://<bucket>/tujya
dvc remote modify s3store region eu-west-3
dvc remote modify --local s3store profile <profil-aws>           # .dvc/config.local, ignoré par git
dvc remote default s3store
dvc push
```

Les identifiants vont toujours dans `.dvc/config.local` (`--local`, ignoré par git), jamais
dans `.dvc/config`. Garder les données privées sur l'un ou l'autre service (licences plus
haut).

## Test de fumée (2026-10-28)

Après `docker compose up -d --build`, par nginx sur le port 8080 : `/api/health` 200, `/docs`
200 (Swagger UI), `/openapi.json` 18 chemins, `/api/docs` 307 vers `/docs`,
`/api/public/summary` 200, `/api/scan` 401 sans connexion ; connecté avec le compte de
démonstration, `/api/scan` sur une photo de test C3PI de 2560 x 1920 de fluconazole : un
comprimé, classe top-1 9 FLUCONAZOLE 100MG TABLETS, probabilité 0,996, 59 ms, modèles
détecteur v1 / classifieur v2 ; `/api/interactions` fluconazole + citalopram : Major
(DDInter) ; galerie de démonstration de 8 photos. Le script Playwright
(`BASE=http://localhost:8080 node scripts/screenshots.mjs`, depuis `frontend`) rejoue tout
le parcours du navigateur sur les conteneurs.

## Tables des médicaments

Définies dans [`backend/app/pharma.py`](../backend/app/pharma.py) avec leur propre
base SQLAlchemy, pour ne jamais toucher la base des comptes `backend/app.db` :

```
sources (name, title, url, license, citation)          une ligne par source, sert de citation
pills (class_id, class_key, ndc9, part, ndc11_list, name, labeled_by, imprint, color, shape, n_images)
ingredients (name, rxcui, ddinter_id)                  principe actif de base RxNorm, en minuscules
pill_ingredients (pill_id, ingredient_id, source_id)   RxNav
interactions (ingredient_a_id < ingredient_b_id, severity, description, source_id)   DDInter
```

Dans Docker, `pharma-init` les charge à chaque démarrage. À la main (rechargement complet,
relançable sans risque) :

```bash
cd backend
python -m app.pharma_loader --database-url postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
```

Sans `--database-url`, le chargeur utilise `SYNERGYRX_PHARMA_DATABASE_URL`, par défaut
`sqlite:///data/tujya.db`. Résultat du 2026-10-28 : 150 comprimés, 1 716 principes actifs
(82 venant des comprimés, les autres sont des partenaires DDInter), 161 liens
comprimé-principe actif, 40 508 interactions. Tests :
[`backend/tests/test_pharma.py`](../backend/tests/test_pharma.py) (SQLite).

## MLflow

Les scripts d'entraînement lisent `MLFLOW_TRACKING_URI` (par défaut
`http://127.0.0.1:5000`). L'interface sur http://localhost:5000 montre les expériences
(`tujya-vision`, `tujya-rag`, `tujya-tox`, et `tujya-synergy` de l'étude retirée, D85) et les modèles enregistrés
`tujya-pill-classifier`, `tujya-pill-detector` et `tujya-tox21`. Le tracking et les artefacts
vivent dans les volumes `pgdata` et `mlartifacts` du projet compose.

## Ollama (LLM local, module 2)

Ollama n'est pas installé sur la machine : il ne tourne que comme service `ollama` (D57).

```bash
docker compose up -d ollama
docker compose exec ollama nvidia-smi            # le GPU est visible dans le conteneur
docker compose logs ollama | grep "inference compute"   # library=CUDA ... total="4.0 GiB"
docker compose exec ollama ollama pull qwen2.5:3b       # LLM (D58)
docker compose exec ollama ollama pull bge-m3           # embeddings (D59)
docker compose exec ollama ollama ps                    # ce qui est chargé, GPU ou CPU
```

Sans GPU NVIDIA : `docker compose -f docker-compose.yml -f infra/docker/ollama/cpu.yml up -d`
(même service, sans réservation de périphérique ; les réponses sont alors plusieurs fois plus
lentes).

Mesuré le 2026-10-28 (RTX 2050 4 Go, Docker Desktop + WSL2), courte question en français,
température 0, modèle déjà chargé : **qwen2.5:3b 1,7 s** (76 jetons, 47 jetons/s, 100 % GPU,
2,2 Go) ; llama3.2:3b 3,2 s ; gemma3:4b 12,1 s ; qwen3:4b 90 s (raisonnement). Choix et
tableau : D58.
