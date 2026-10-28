# Tujya roadmap

Steps in order. Tick a box when it is done. Context and rules: [CLAUDE.md](../CLAUDE.md).
The professor requirements are the "Exigences des modules" checklist in the README.

## Already done (from SynergyRx)
- [x] Data pipeline `src/synergyrx_data` (14 connectors incl. DDInter and PubMed, 270 tests)
- [x] FastAPI backend: accounts, sessions, admin, drug and cell-line catalog
- [x] React + Vite frontend shell branded Tujya (mock predictions)
- [x] Planning of the new direction (two modules, scope, tech choices)

## 0. Git and project rules
- [x] First commit and push to https://github.com/melekbouabid/tujia-.git (c1cb26d)
- [x] README with the module requirements, data sources and progress journal
- [x] Mandatory session rules in CLAUDE.md
- [ ] Change the default admin password (admin@gmail.com / admin)

## 1. Pill data (C3PI, no personal photos)
- [x] Explore the C3PI sample and the full archive layout (docs/DATASET_C3PI.md)
- [x] `config/pill_classes.csv`: 150 classes with the most consumer images (D32)
- [x] Download the images of these classes only (4.8 GB, SHA-1 checked)
- [x] Train / val / test split with no photo in two splits (`config/splits/`)
- [x] DVC set up; `data/vision/c3pi` images and metadata tracked (local cache)
- [ ] DVC remote configured and pushed (needs a storage choice)
- [x] Each class mapped to its active ingredient(s) (RxNav, 149/150)
- [x] DDInter enabled; interactions for these ingredients in a clean table (docs/INTERACTIONS.md)

## 2. Vision model
- [ ] Hardware checked, `vision/` package with its own venv
- [ ] Identification: classifier trained by transfer learning (YOLO cls)
- [ ] Detection: pill bounding boxes (decision documented, YOLO detector)
- [ ] MLflow runs (params, metrics, artefacts)
- [ ] Best model registered in the MLflow Model Registry
- [ ] ONNX export, inference with ONNX Runtime, outputs match PyTorch
- [ ] Measure top-1 / top-5 (identification) and mAP (detection); compare with ePillID

## 3. API
- [x] SQL schema (pills, ingredients, pill_ingredients, interactions, sources) + loader
- [ ] Async `POST /api/scan` (image -> boxes, classes, confidence; confirm on low confidence)
- [ ] Async `POST /api/interactions` (DDInter severity, explanation, timing only if sourced)
- [ ] Pydantic schemas, Swagger at `/docs`
- [ ] Tests

## 4. RAG corpus
- [ ] openFDA drug labels for the listed ingredients
- [ ] PubMed abstracts for the listed ingredients
- [ ] Chunking with metadata (source, drug, section, URL, date)
- [ ] bge-m3 embeddings in PostgreSQL + pgvector, BM25 hybrid search
- [ ] Corpus versioned with DVC

## 5. Agent
- [ ] LangGraph graph: understand -> tools -> retrieve -> answer with citations -> safety check
- [ ] Tools: identify_pills, check_interaction, get_drug_info, search_literature (optional get_synergy)
- [ ] Ollama local LLM (Mistral, Llama or Qwen)
- [ ] Benchmark set of 50-100 questions
- [ ] Retrieval benchmark (Recall@k, MRR) and generation benchmark (RAGAS)
- [ ] MCP server exposing the same tools (lower priority)

## 6. Web
- [ ] Scan page: upload + webcam + bounding boxes
- [ ] Result page (interaction, severity, source)
- [ ] Chat panel with citations
- [ ] Disclaimer everywhere: information only, consult a pharmacist or doctor

## 7. Delivery
- [x] Docker Compose: postgres + pgvector, mlflow (docs/INFRA.md)
- [ ] Docker Compose: backend, frontend, ollama
- [ ] GitHub Actions CI (lint, tests, frontend build)
- [ ] GitHub Actions CD (images / deploy)
- [x] README updated for Tujya
- [ ] Report Module 1 (Vision & MLOps)
- [ ] Report Module 2 (AI Engineering)
- [ ] Demo script (C3PI test images) + backup video

## Bonus (only if time remains)
- [ ] OCR of pill imprints
- [ ] Cancer synergy data page
