# Tujya

[![ci-tests](https://github.com/melekbouabid/syRX/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/melekbouabid/syRX/actions/workflows/ci-tests.yml)
[![docker-build](https://github.com/melekbouabid/syRX/actions/workflows/docker-build.yml/badge.svg)](https://github.com/melekbouabid/syRX/actions/workflows/docker-build.yml)

Tujya is a medication-safety web app. Photograph one or more pills (upload or webcam): a YOLO11s
model **detects** each pill and an EfficientNetV2-S classifier **identifies** it among 380 pills
of the public NLM C3PI dataset, with its confidence. Each pill is linked to its active
ingredients and their **interactions** (DDInter); a LangGraph agent then answers follow-up
questions by **RAG** over openFDA labels, PubMed abstracts and DDInter, **citing its sources**.
Student project evaluated in two modules: **Vision & MLOps** and **AI Engineering**.

> **For information only: ask a pharmacist or a doctor.** Tujya never gives a dose and never says
> a combination is safe; a pill identification is a suggestion, always shown with its confidence.

## Architecture

```mermaid
flowchart LR
  user["Browser"] --> fe["frontend<br/>React + nginx :8080"]
  fe --> be["backend<br/>FastAPI :8000"]
  be --> onnx["ONNX Runtime<br/>detector + classifier"]
  be --> pg[("PostgreSQL<br/>medicine tables")]
  be --> qd[("Qdrant :6333<br/>RAG index")]
  be --> ol["Ollama :11434<br/>qwen2.5:3b, bge-m3"]
  mlf["MLflow :5000<br/>Model Registry"] -. "published ONNX models" .-> onnx
```

Services, ports and the flow of a photo and of a question: [docs/architecture.md](docs/architecture.md).

## Installation (one command)

Prerequisites: Docker Desktop (an NVIDIA GPU helps Ollama), the DVC data and the published models.

```bash
git clone https://github.com/melekbouabid/syRX.git && cd syRX
cp .env.example .env                               # contact e-mail for the data pipeline, project name
dvc pull                                           # data from the DVC remote
python -m src.models.registry publish              # served ONNX models from the Model Registry -> models/
docker compose up --build                          # the whole application
```

On the first start `rag-init` pulls the Ollama models and indexes the corpus in Qdrant (a few
minutes); scan and interactions work right away. Without a GPU:
`docker compose -f docker-compose.yml -f infra/ollama/cpu.yml up -d`. `docker compose down`
keeps the volumes.

| Address | What |
|---|---|
| http://localhost:8080 | web app (Scanner, chat, Documents, Monitoring, Toxicity) |
| http://localhost:8000/docs | Swagger UI of the API ([docs/api.md](docs/api.md)) |
| http://localhost:5000 | MLflow: experiments, Model Registry, Prompt Registry |
| http://localhost:6333/dashboard | Qdrant dashboard: indexed chunks and their metadata |
| http://localhost:8090 | Spark master UI (`docker compose --profile spark up -d`) |

Demo accounts created on the first start: `user@gmail.com` / `user` and `admin@gmail.com` /
`admin` (change the latter before any other use, `backend/.env.example`).

| Cited answer of the agent | Swagger |
|---|---|
| ![Chat answer](docs/screenshots/09-chat-answer.jpg) | ![Swagger](docs/screenshots/07-swagger.jpg) |

## Documentation

| | |
|---|---|
| [Module 1: vision and MLOps](docs/module1/README.md) | data -> Kaggle + Optuna -> MLflow -> comparison -> champion -> ONNX -> API -> drift -> OCR; [results](docs/module1/results.md) |
| [Module 2: AI engineering](docs/module2/README.md) | corpus -> chunking -> Qdrant -> retrieval -> rewriting -> reranking -> RAG chain -> agent and MCP -> evaluation; [results](docs/module2/results.md) |
| [Data](docs/data.md) | sources, licences, counts before / after, cleaning, split without leakage, PySpark |
| [Key decisions](docs/decisions.md) | the main choices and the evidence behind them |

## Tests and checks

```bash
uv venv .venv -p 3.12 && uv pip install -p .venv -r requirements-dev.txt   # project venv
python -m pytest                                # tests of src/ (tests/)
cd backend && python -m pytest && cd ..         # API tests (backend/tests/)
python -m pytest tools/tests                    # Git hooks
cd frontend && npm ci && npm test && npm run lint && npm run build && cd ..
ruff check . && ruff format --check . && mypy
dvc status                                      # data and pipeline up to date
```

The CI ([.github/workflows/](.github/workflows/)) runs the same checks and builds the Docker
images without publishing them.

## Repository layout

```text
backend/            FastAPI app (app/) and its tests (tests/)
frontend/           React app (Vite, Tailwind), Vitest tests, screenshot scripts
src/
  data/             C3PI download, auto-labels, datasets, Spark jobs, polite data pipeline
  models/           training, Kaggle runners, comparison, ONNX export, quantization, Registry
  rag/              corpus, chunking, Qdrant index, retrieval, reranking, query rewriting
  agent/            LangGraph agent, tools, safety rules, MCP server
  evaluation/       retrieval and generation benchmarks, their result files
  utils/            MLflow helpers, Grad-CAM
tests/              tests of src/
config/             class tables, splits, search spaces, RAG grid, brand table
data/               DVC-tracked data (raw/, processed/, external/), never in git
notebooks/          Kaggle notebooks (module 1)
infra/              MLflow, PostgreSQL, Ollama and Spark images and settings
tools/              Git hooks and their tests
docs/               documentation (this README links every page)
docker-compose.yml  the whole stack; dvc.yaml + params.yaml: the DVC pipeline
```

## Contributing

- Branches: `main` is always stable; work happens on `feature/<topic>`, `fix/<topic>`,
  `refactor/<topic>`, `docs/<topic>`, `chore/<topic>` and one `experiment/<model>` branch per
  training experiment, merged with `git merge --no-ff`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): description`, with type `feat`, `fix`, `exp`, `data`, `docs`, `refactor`,
  `chore`, `test`, `ci`, `perf` or `style`.
- Hooks ([.pre-commit-config.yaml](.pre-commit-config.yaml), install with
  `pre-commit install --hook-type pre-commit --hook-type commit-msg`): ruff, ESLint, notebooks
  without outputs, no hard-coded secret, commit message format.
- Annotated tags mark validated milestones; models are versioned in the MLflow Model Registry
  (aliases `champion`, `challenger`, `served`), data with DVC.
