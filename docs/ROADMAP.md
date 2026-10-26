# Tujya roadmap

Steps in order. Tick a box when it is done. Context and rules: [CLAUDE.md](../CLAUDE.md).

## Already done (from SynergyRx)
- [x] Data pipeline `src/synergyrx_data` (14 connectors incl. DDInter and PubMed, 270 tests)
- [x] FastAPI backend: accounts, sessions, admin, drug and cell-line catalog
- [x] React + Vite frontend shell branded Tujya (mock predictions)
- [x] Planning of the new direction (two modules, scope, tech choices)

## 0. Git
- [ ] First commit and push to https://github.com/melekbouabid/tujia-.git
- [ ] Change the default admin password (admin@gmail.com / admin)

## 1. Medicine data
- [ ] `config/medicines.csv` filled (class_id,brand,ingredient,strength,form), 10-15 medicines
  - [x] File created with header only
- [ ] Photos: 30-50 per medicine, varied angles / lights / backgrounds, 2-3 boxes together in a third, some from the webcam
- [ ] ~30 photos with no medicine
- [ ] Annotation (Label Studio or Roboflow), YOLO format
- [ ] DVC set up; `data/vision/` tracked
- [ ] DDInter enabled for the ingredients in the list

## 2. Vision model
- [ ] Fine-tune YOLOv8/v11
- [ ] MLflow runs (params, metrics, artefacts)
- [ ] Best model registered in the MLflow Model Registry
- [ ] ONNX export, inference with ONNX Runtime
- [ ] Measure mAP and identification accuracy

## 3. API
- [ ] Async `POST /api/scan` (image -> boxes, classes, confidence; confirm on low confidence)
- [ ] Async `POST /api/interactions` (DDInter severity, explanation, timing only if sourced)
- [ ] Pydantic schemas, Swagger at `/docs`
- [ ] Tests

## 4. RAG corpus
- [ ] Official leaflets collected
- [ ] PubMed abstracts for the listed ingredients
- [ ] Chunking with metadata (source, drug, section, URL, date)
- [ ] bge-m3 embeddings in PostgreSQL + pgvector, BM25 hybrid search
- [ ] Corpus versioned with DVC

## 5. Agent
- [ ] LangGraph graph: understand -> tools -> retrieve -> answer with citations -> safety check
- [ ] Tools: identify_drugs, check_interaction, get_drug_info, search_literature (optional get_synergy)
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
- [ ] Docker Compose (backend, frontend, postgres, mlflow, ollama)
- [ ] GitHub Actions CI/CD
- [ ] README updated for Tujya
- [ ] Report Module 1 (Vision & MLOps)
- [ ] Report Module 2 (AI Engineering)
- [ ] Demo script + backup video

## Bonus (only if time remains)
- [ ] OCR for unknown boxes
- [ ] Cancer synergy data page
