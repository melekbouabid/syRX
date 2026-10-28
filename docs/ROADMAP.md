# Tujya roadmap

Steps in order. Tick a box when it is done. Context and rules: [CLAUDE.md](../CLAUDE.md).
The professor requirements are the "Exigences des modules" checklist in the README.

## Already done (from SynergyRx)
- [x] Data pipeline `src/synergyrx_data` (14 connectors incl. DDInter and PubMed, 270 tests)
- [x] FastAPI backend: accounts, sessions, admin, drug and cell-line catalog
- [x] React + Vite frontend shell branded Tujya (mock predictions)
- [x] Planning of the new direction (two modules, scope, tech choices)

## 0. Git and project rules
- [x] First commit and push to https://github.com/melekbouabid/tujia-.git (9398f0c)
- [x] README with the module requirements, data sources and progress journal
- [x] Mandatory session rules in CLAUDE.md
- [ ] Change the default admin password (admin@gmail.com / admin)

## 1. Pill data (C3PI, no personal photos)
- [x] Explore the C3PI sample and the full archive layout (docs/DATASET_C3PI.md)
- [x] `config/pill_classes.csv`: 150 classes with the most consumer images (D32)
- [x] Download the images of these classes only (4.8 GB, SHA-1 checked)
- [x] Train / val / test split with no photo in two splits (`config/splits/`)
- [x] DVC set up; `data/vision/c3pi` images and metadata tracked (local cache)
- [x] DVC remote configured and pushed (local folder outside the repo, D71; fresh-clone `dvc pull` checked)
- [x] Each class mapped to its active ingredient(s) (RxNav, 149/150)
- [x] DDInter enabled; interactions for these ingredients in a clean table (docs/INTERACTIONS.md)

## 2. Vision model
- [x] Hardware checked, `vision/` package with its own venv (RTX 2050, CUDA 13.0)
- [x] Identification: classifier trained by transfer learning (YOLO11s-cls, test top-1 93.5 %)
- [x] Detection: pill bounding boxes (automatic labels, D38; YOLO11n, test mAP50 0.972)
- [x] MLflow runs (params, metrics, artefacts)
- [x] Best model registered in the MLflow Model Registry
- [x] ONNX export, inference with ONNX Runtime, outputs match PyTorch
- [x] Measure top-1 / top-5 (identification) and mAP (detection)
- [ ] Compare with ePillID on its own split and metrics (docs/VISION.md)
- [ ] Hand-check the test boxes; second labelling pass with the trained detector

## 3. API
- [x] SQL schema (pills, ingredients, pill_ingredients, interactions, sources) + loader
- [x] Async `POST /api/scan` (image -> boxes, top-5, confidence; confirm below 0.85, docs/API.md)
- [x] `python -m tujya_vision.publish`: registered models -> `models/` for the API
- [x] Async `POST /api/interactions` (DDInter severity, description, source; never "safe")
- [x] `GET /api/pills?q=` search of the 150 classes (manual confirmation)
- [x] Swagger at `/docs` and schema at `/openapi.json` (`/api/docs` redirects, D44)
- [x] Pydantic schemas for scan and interactions
- [x] Tests (fake ONNX models, SQLite medicine tables; 69 backend tests)

## 4. RAG corpus
- [x] openFDA drug labels for the 82 ingredients (connector `openfda_label`, D60)
- [x] PubMed abstracts: up to 20 per ingredient on interactions and adverse effects (D61)
- [x] Chunking with metadata (source, drug, section, URL, date, licence, retraction; docs/RAG.md)
- [x] bge-m3 embeddings in PostgreSQL + pgvector, BM25 hybrid search (RRF), filters (D59)
- [x] Corpus versioned with DVC (`data/rag/*.dvc`, stage `rag_corpus`)

## 5. Agent
- [x] LangGraph graph: understand -> tools -> retrieve -> answer with citations -> safety check (docs/AGENT.md)
- [x] Tools: identify_pills, check_interaction, get_drug_info, search_documents (extension point for more)
- [x] Async `POST /api/chat` (JSON or SSE stream), answer + citations + tool calls; tests with a fake LLM
- [x] Ollama local LLM: qwen2.5:3b in the `ollama` service, on the GPU (D57, D58)
- [x] Benchmark set: 88 questions FR/EN built from the corpus, checked by hand (D66)
- [x] Retrieval benchmark (Recall@k, MRR) and generation benchmark (safety, citations, RAGAS); MLflow `tujya-rag`
- [x] MCP server exposing the same tools (official SDK, stdio + HTTP; docs/MCP.md)

## 6. Web
- [x] Scan page: upload + webcam (live predictions) + bounding boxes + demo gallery (D52)
- [x] Result (interaction, severity, source) on the Scanner page; screenshots in docs/screenshots/
- [x] Chat panel with streamed answer, clickable citations, tool calls (Scanner page)
- [x] Disclaimer on the Scanner page, every interaction report, the chat panel and every answer

## 7. Delivery
- [x] Docker Compose: postgres + pgvector, mlflow (docs/INFRA.md)
- [x] Docker Compose: backend, frontend (nginx), one-shot loader; one command (docs/INFRA.md)
- [x] Docker Compose: ollama service with the NVIDIA GPU (docs/INFRA.md)
- [x] GitHub Actions CI (lint, tests, frontend build)
- [x] GitHub Actions CD: images built and smoke-tested after a green CI, not pushed (D70, docs/CD.md)
- [ ] Delete the two old packages on ghcr.io (manual: no `delete:packages` scope, docs/CD.md)
- [x] Strictly local usage: every port on 127.0.0.1, checked with netstat (D69, docs/INFRA.md)
- No deployment to a server, by design: the project runs only on this PC (D69)
- [x] README updated for Tujya
- [ ] Report Module 1 (Vision & MLOps)
- [ ] Report Module 2 (AI Engineering)
- [x] Demo script (docs/DEMO.md, rehearsed on the containers) and jury questions (docs/QUESTIONS_JURY.md)
- [ ] Backup video of the demo (to record by hand)

## Bonus (only if time remains)
- [ ] Tool `predict_toxicity(smiles)`: single-molecule toxicity (model trained on Tox21 / ClinTox)
- [ ] Tool `predict_synergy(drug_a, drug_b)`: cancer synergy from the DrugComb data in the pipeline
- [ ] OCR of pill imprints
- [ ] Cancer synergy data page
