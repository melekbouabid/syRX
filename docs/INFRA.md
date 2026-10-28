# Infrastructure (Docker Compose)

`docker-compose.yml` at the repository root runs the whole app with one command, including
Ollama (local LLM and embeddings for the Module 2 agent, on the GPU).

```bash
docker compose up -d --build        # first start builds the images (a few minutes)
docker compose ps                   # postgres, backend, frontend healthy; pharma-init exited (0)
# web app http://localhost:8080   API http://localhost:8000/docs   MLflow http://localhost:5000
docker compose down                 # stops; volumes are kept (down -v deletes them)
```

| Service | Image | Port | Role |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | volume `pgdata`; database `tujya` (accounts, medicine tables, pgvector) and `mlflow` |
| `mlflow` | built from `docker/mlflow/Dockerfile` (MLflow 3.17.2 + psycopg2) | 5000 | tracking store in PostgreSQL, artefacts in volume `mlartifacts`, Model Registry |
| `pharma-init` | `tujya-backend` | none | one-shot: `python -m app.pharma_loader` rebuilds the medicine tables from `config/` and `data/tujya/`, then exits; the backend waits for it |
| `rag-init` | `tujya-backend` | none | one-shot: `python -m app.rag.index --pull` pulls the Ollama models if missing, then indexes `data/rag/corpus` into the `rag_chunks` table (idempotent; docs/RAG.md); the backend does not wait for it |
| `backend` | built from `backend/Dockerfile` (Python 3.12, FastAPI, ONNX Runtime CPU, LangGraph) | 8000 | the API and the agent (`/api/chat`, talks to `ollama`); Swagger at `/docs` |
| `frontend` | built from `frontend/Dockerfile` (Node 22 build, nginx 1.27) | 8080 | the React app; nginx forwards `/api`, `/docs`, `/openapi.json` to `backend` (`frontend/nginx.conf`) |
| `ollama` | `ollama/ollama:0.40.12` | 11434 | local LLM (`qwen2.5:3b`) and embeddings (`bge-m3`); models in volume `ollama`; NVIDIA GPU reserved (D57) |

**Before the first start** (the images contain no data and no model, D55):

1. Data under `data/` (DVC, docs/DATASET_C3PI.md, docs/INTERACTIONS.md):
   `data/tujya/*.csv` (medicine tables) and `data/vision/c3pi/images/` (demo gallery);
   `data/parquet/` (SynergyRx tables) is optional.
2. Models: `cd vision && python -m tujya_vision.publish` fills `models/` from the MLflow
   Model Registry (MLflow must be running: `docker compose up -d mlflow` first).

These folders are mounted **read-only** into the backend (`/config`, `/data/...`, `/models`;
the app's project root is `/` in the image, so its default paths need no setting). Without
models, `/api/scan` answers 503 and says how to publish them; without the CSV files,
`pharma-init` fails and the backend does not start.

Accounts: in Docker the accounts tables live in the same PostgreSQL database `tujya` (not in
`backend/app.db`, which only the local development server uses). The first start creates the
default administrator and the demo account (`user@gmail.com` / `user`); change the
administrator password before the server is reachable by anyone else.

Images on GitHub Container Registry: docs/CD.md. To run the published images instead of
building: `BACKEND_IMAGE=ghcr.io/... FRONTEND_IMAGE=ghcr.io/... docker compose up -d --no-build`.

`docker/postgres/init/01-init.sql` runs once, on an empty volume: it enables `vector` in
the `tujya` database and creates the `mlflow` database. Credentials are development
defaults (`tujya` / `tujya`); override `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`POSTGRES_PORT`, `MLFLOW_PORT`, `BACKEND_PORT`, `WEB_PORT` in the shell or a compose `.env`
file.

## Smoke test (2026-10-28)

After `docker compose up -d --build`, through nginx on port 8080: `/api/health` 200,
`/docs` 200 (Swagger UI), `/openapi.json` 18 paths, `/api/docs` 307 to `/docs`,
`/api/public/summary` 200, `/api/scan` 401 without sign-in; signed in as the demo user,
`/api/scan` on a 2560 x 1920 C3PI test photo of fluconazole: one pill, top-1 class 9
FLUCONAZOLE 100MG TABLETS, probability 0.996, 59 ms, models detector v1 / classifier v2;
`/api/interactions` fluconazole + citalopram: Major (DDInter); demo gallery 8 photos. The
Playwright script (`BASE=http://localhost:8080 node scripts/screenshots.mjs`) runs the whole
browser flow against the containers.

## Medicine tables

Defined in `backend/app/pharma.py` with their own SQLAlchemy base, so they never touch the
accounts database `backend/app.db`:

```
sources (name, title, url, license, citation)          one row per data source, used as citation
pills (class_id, class_key, ndc9, part, ndc11_list, name, labeled_by, imprint, color, shape, n_images)
ingredients (name, rxcui, ddinter_id)                  RxNorm base ingredient, lower case
pill_ingredients (pill_id, ingredient_id, source_id)   RxNav
interactions (ingredient_a_id < ingredient_b_id, severity, description, source_id)   DDInter
```

In Docker, `pharma-init` loads them on every start. By hand (full refresh, safe to rerun):

```bash
cd backend
python -m app.pharma_loader --database-url postgresql+psycopg://tujya:tujya@localhost:5432/tujya
```

Without `--database-url` the loader uses `SYNERGYRX_PHARMA_DATABASE_URL`, default
`sqlite:///data/tujya.db`. Result on 2026-10-28: 150 pills, 1,716 ingredients (82 from the
pills, the rest are DDInter partners), 161 pill-ingredient links, 40,508 interactions.
Tests: `backend/tests/test_pharma.py` (SQLite).

## MLflow

Training scripts read `MLFLOW_TRACKING_URI` (default `http://localhost:5000`). The UI at
http://localhost:5000 shows the `tujya-vision` experiment and the registered models
`tujya-pill-classifier` and `tujya-pill-detector`.

## Ollama (local LLM, Module 2)

Ollama is not installed on the host: it only runs as the `ollama` service (D57).

```bash
docker compose up -d ollama
docker compose exec ollama nvidia-smi            # the GPU is visible in the container
docker compose logs ollama | grep "inference compute"   # library=CUDA ... total="4.0 GiB"
docker compose exec ollama ollama pull qwen2.5:3b       # LLM (D58)
docker compose exec ollama ollama pull bge-m3           # embeddings (D59)
docker compose exec ollama ollama ps                    # what is loaded, GPU or CPU
```

Without an NVIDIA GPU: `docker compose -f docker-compose.yml -f docker/ollama/cpu.yml up -d`
(same service, no device reservation; answers are then several times slower).

Measured on 2026-10-28 (RTX 2050 4 GB, Docker Desktop + WSL2), short French question,
temperature 0, model already loaded: **qwen2.5:3b 1.7 s** (76 tokens, 47 tok/s, 100 % GPU,
2.2 GB); llama3.2:3b 3.2 s; gemma3:4b 12.1 s; qwen3:4b 90 s (thinking). Choice and table: D58.
