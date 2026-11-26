"""Settings of the web API, set in `backend/.env` or the environment with the `TUJYA_` prefix."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
# The repository root (data/, config/, models/). The Docker image keeps the same layout under
# /tujya (backend/Dockerfile), where docker-compose.yml mounts those folders.
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_ADMIN_PASSWORD = "admin"  # noqa: S105  first-run default, meant to be changed
ENV_PREFIX = "TUJYA_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_prefix=ENV_PREFIX, extra="ignore"
    )

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'app.db').as_posix()}"
    # Tujya medicine tables (pills, ingredients, interactions): PostgreSQL from
    # docker-compose.yml, e.g. postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya.
    # The default is a SQLite file under data/ so nothing else is needed for development.
    pharma_database_url: str = f"sqlite:///{(PROJECT_DIR / 'data' / 'tujya.db').as_posix()}"
    # ONNX models for /api/scan: detector.onnx, classifier.onnx and models.json, written by
    # `python -m src.models.registry publish` (TUJYA_MODELS_DIR). Loaded on the first scan.
    models_dir: Path = PROJECT_DIR / "models"
    # "cpu" always works; "auto" uses CUDA first when the installed onnxruntime has it.
    onnx_providers: str = "cpu"
    onnx_threads: int = 0  # 0 = ONNX Runtime default (one per core)
    # Below this top-1 probability the app asks the user to confirm the pill (chosen on the
    # validation photos).
    scan_confirm_threshold: float = 0.85
    scan_detection_confidence: float = 0.25
    scan_max_bytes: int = 10 * 1024 * 1024
    scan_max_pixels: int = 50_000_000
    # Medicine boxes (app/boxes.py). Below scan_pill_min_confidence with little text, the
    # photo is refused as neither a pill nor a box.
    box_ocr_enabled: bool = True
    ocr_threads: int = 4
    brand_table_file: Path = PROJECT_DIR / "config" / "brand_aliases.csv"
    inn_aliases_file: Path = PROJECT_DIR / "config" / "inn_aliases.csv"
    scan_pill_min_confidence: float = 0.5
    box_text_chars: int = 20
    # RxNav (public NLM API) is asked only with a contact address, like the data pipeline.
    rxnav_contact_email: str | None = None
    rxnav_url: str = "https://rxnav.nlm.nih.gov/REST"
    rxnav_timeout_seconds: float = 5.0
    # Demo gallery (app/gallery.py): photos no served model saw in training.
    demo_images_dir: Path = PROJECT_DIR / "data" / "raw" / "c3pi" / "images"
    demo_split_file: Path = PROJECT_DIR / "config" / "splits" / "selected_images.csv"
    demo_holdout_file: Path = PROJECT_DIR / "config" / "splits" / "c3pi_v2" / "test.csv"
    demo_gallery_file: Path = PROJECT_DIR / "config" / "demo_gallery.csv"
    # Grad-CAM pictures of the gallery photos (DVC stage gradcam_gallery).
    demo_gradcam_dir: Path = PROJECT_DIR / "data" / "processed" / "c3pi" / "gradcam"
    # Reference statistics of the drift monitor (DVC stage drift_reference).
    drift_reference_file: Path = (
        PROJECT_DIR / "data" / "processed" / "c3pi" / "drift" / "reference.json"
    )

    # Module 2: RAG and agent. `python -m src.rag.vector_store index` indexes these chunks
    # (380 classes) in Qdrant; collection tujya_chunks holds the 150-class corpus. "sql" keeps
    # the index in the pharma database instead. Ollama serves the LLM and the embeddings.
    rag_chunks_file: Path = PROJECT_DIR / "data" / "processed" / "rag" / "corpus_v2" / "chunks"
    vector_store: Literal["qdrant", "sql"] = "qdrant"
    # docker compose `qdrant`; ":memory:" = the client's local in-memory mode (tests).
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "tujya_chunks_v2"
    # Shown on the Documents page; the browser opens it on the host.
    qdrant_dashboard_url: str = "http://127.0.0.1:6333/dashboard"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1
    # 4096 keeps qwen2.5:3b (2.2 GB) and bge-m3 on the 4 GB GPU together; at 8192 Ollama
    # evicts one for the other on every question. Agent prompts are ~1,300 tokens.
    llm_num_ctx: int = 4096
    llm_timeout_seconds: float = 180.0
    # A small model can loop on itself; answers are 3-6 sentences, far below this.
    llm_max_tokens: int = 450
    # Version of the answer prompt (src/agent/prompts.py), chosen on the generation benchmark.
    llm_prompt_version: str = "v1"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_num_ctx: int = 2048  # chunks are at most ~400 tokens
    rag_top_k: int = 6
    # Chosen on the retrieval grid (best MRR within 2 s): hybrid search of the rewritten
    # English query, which BM25 needs.
    rag_search_mode: str = "hybrid"
    # fastembed models (BM25, reranker) are downloaded once into rag_model_cache_dir.
    rag_sparse_model: str = "Qdrant/bm25"
    # The retrieval champion (src/evaluation/results/grid/champion.json) has no reranker:
    # a cross-encoder costs seconds per question on the CPU. When on, the fastest one is used.
    rag_rerank: bool = False
    rag_rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rag_fetch_k: int = 10
    rag_query_rewrite: bool = True
    rag_model_cache_dir: Path = PROJECT_DIR / "models" / "fastembed"
    # An uncited answer sentence gets the source closest in meaning (bge-m3 cosine) above
    # this; calibrated on the first benchmark run (docs/module2/results.md).
    # 0 disables it.
    rag_attribution_threshold: float = 0.65

    # Bonus: single-molecule toxicity (docs/module2/README.md). The model bundle is published by
    # `python -m src.models.toxicity.train_tox21 publish`; ingredient names -> SMILES from PubChem.
    tox_model_dir: Path = PROJECT_DIR / "models" / "tox21"
    ingredient_smiles_file: Path = PROJECT_DIR / "config" / "ingredient_smiles.csv"

    session_cookie: str = "tujya_session"
    session_hours: int = 12
    cookie_secure: bool = False  # set to true as soon as the site is served over HTTPS

    # Activity log of the admin pages (app/activity.py): events older than this are deleted
    # at start-up; 0 keeps everything.
    activity_retention_days: int = 180
    # Saved medicines and reports (app/routers/my.py).
    max_saved_medicines: int = 50
    max_reports_per_user: int = 200
    report_share_max_days: int = 30
    # Key of the seal put on each chat answer, checked when the answer is added to a report.
    # Empty: a random key per run (answers from before a restart then need asking again).
    report_seal_key: str = ""

    # Accounts created the first time the API starts on an empty database.
    admin_email: str = "admin@gmail.com"
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    seed_demo_user: bool = True
    demo_user_email: str = "user@gmail.com"
    demo_user_password: str = "user"  # noqa: S105  demo account
