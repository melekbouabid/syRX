# Infrastructure (Docker Compose)

`docker-compose.yml` at the repository root runs the two services the data and the models
need today. Backend, frontend and Ollama services will be added when those parts exist.

| Service | Image | Port | Data |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | volume `pgdata`; databases `tujya` (app tables, pgvector extension) and `mlflow` |
| `mlflow` | built from `docker/mlflow/Dockerfile` (MLflow 3.17.2 + psycopg2) | 5000 | tracking store in PostgreSQL (`mlflow` database), artefacts in volume `mlartifacts`, served by the server (`--serve-artifacts`) |

```bash
docker compose up -d --build        # first start builds the MLflow image
docker compose ps                   # postgres should be "healthy"
curl http://localhost:5000/health   # MLflow: OK
docker compose down                 # stops; volumes are kept (down -v deletes them)
```

`docker/postgres/init/01-init.sql` runs once, on an empty volume: it enables `vector` in
the `tujya` database and creates the `mlflow` database. Credentials are development
defaults (`tujya` / `tujya`); override `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`POSTGRES_PORT`, `MLFLOW_PORT` in the shell or a compose `.env` file.

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

Load or refresh them (full refresh in one transaction, safe to rerun):

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
