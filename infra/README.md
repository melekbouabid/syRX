# Infrastructure

Files of the local services started by [docker-compose.yml](../docker-compose.yml) at the
repository root. CI workflows stay in [.github/workflows/](../.github/workflows/) (where GitHub
Actions expects them). Services, ports and flows: [docs/architecture.md](../docs/architecture.md).

| File | Role |
|---|---|
| [mlflow/Dockerfile](mlflow/Dockerfile) | MLflow 3.17.2 server (tracking + Model Registry) with the PostgreSQL driver |
| [postgres/init/01-init.sql](postgres/init/01-init.sql) | run once on an empty volume: pgvector extension and the `mlflow` database |
| [ollama/cpu.yml](ollama/cpu.yml) | override for a machine without an NVIDIA GPU |
| [ollama/judge.Modelfile](ollama/judge.Modelfile) | the local judge of the generation benchmark (phi4-mini, every layer on the GPU) |
| [spark/Dockerfile](spark/Dockerfile), [spark/requirements.txt](spark/requirements.txt) | Spark image of the Big Data jobs (profile `spark`) |

The application images are built from their own folders: the API from the repository root
(`backend/Dockerfile`, it imports the module packages) and the web app from `frontend/`.

```bash
docker compose up -d --build                                          # everything, GPU for Ollama
docker compose -f docker-compose.yml -f infra/ollama/cpu.yml up -d    # without a GPU
docker compose --profile spark up -d spark-master spark-worker        # Spark cluster
docker compose down                                                   # keeps the volumes
```

Every published port is bound to `127.0.0.1`: the application only runs on this machine.
