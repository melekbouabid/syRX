# Infrastructure

Fichiers de l'infrastructure locale lancée par [`docker-compose.yml`](../docker-compose.yml)
à la racine du dépôt. La CI et la CD restent dans [`.github/workflows/`](../.github/workflows/)
(emplacement imposé par GitHub Actions). Documentation complète :
[docs/infra.md](../docs/infra.md) (services, ports, volumes, DVC) et
[docs/cd.md](../docs/cd.md) (livraison continue sans publication).

| Fichier | Rôle |
|---|---|
| [`docker/mlflow/Dockerfile`](docker/mlflow/Dockerfile) | serveur MLflow 3.17.2 (tracking + Model Registry) avec le pilote PostgreSQL |
| [`docker/postgres/init/01-init.sql`](docker/postgres/init/01-init.sql) | exécuté une fois, sur un volume vide : extension pgvector et base `mlflow` |
| [`docker/ollama/cpu.yml`](docker/ollama/cpu.yml) | surcharge pour une machine sans GPU NVIDIA |

Les images de l'application sont construites depuis leur dossier : l'API depuis la racine
(`app/backend/Dockerfile`, elle importe les paquets des modules) et le site depuis
`app/frontend/`.

```bash
docker compose up -d --build                                              # tout, GPU pour Ollama
docker compose -f docker-compose.yml -f infra/docker/ollama/cpu.yml up -d # sans GPU
docker compose down                                                       # garde les volumes
```

Tous les ports publiés sont liés à `127.0.0.1` : l'application ne tourne que sur ce PC.
