"""Tujya code shared by the two modules (layout of the module guide).

Sub-packages:
- `data`: C3PI subset (index, download, automatic pill boxes, YOLO datasets) and the
  collection pipeline of the external sources (openFDA labels, PubMed abstracts, DDInter);
- `models`: training, evaluation, ONNX export and publication of the vision models, and the
  Tox21 toxicity model (bonus);
- `rag`, `agent`, `evaluation`: module 2 (RAG corpus, retrieval and chain, LangGraph agent and
  its MCP server, retrieval and generation benchmarks);
- `utils`: helpers shared by the others.

The web API (package `app`, in backend/) imports `rag`, `agent` and `models.toxicity`; the
training code (`data`, `models`) runs in its own venv with PyTorch (.venv-train).
"""
