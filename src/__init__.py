"""Tujya code shared by the two modules (layout of the module guide).

Sub-packages:
- `data`: C3PI subset (index, download, automatic pill boxes, YOLO datasets) and the RAG
  corpus (openFDA labels, PubMed abstracts, DDInter interactions);
- `models`: training, evaluation, ONNX export and publication of the vision models, and the
  Tox21 toxicity model (bonus);
- `rag`, `agent`, `mcp_server`, `benchmark`: module 2 (retrieval, LangGraph agent, MCP
  server, retrieval and generation benchmarks);
- `utils`: helpers shared by the others.

The web API (package `app`, in backend/) imports `rag`, `agent` and `models.toxicity`; the
training code (`data`, `models`) runs in its own venv with PyTorch (.venv-train).
"""
