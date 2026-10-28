"""Settings of the web API.

Every value can be overridden in `backend/.env` or the environment with the `SYNERGYRX_`
prefix, for example `SYNERGYRX_ADMIN_PASSWORD=...`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_ADMIN_PASSWORD = "admin"  # noqa: S105  first-run default, meant to be changed


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_prefix="SYNERGYRX_", extra="ignore"
    )

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'app.db').as_posix()}"
    # Tujya medicine tables (pills, ingredients, interactions): PostgreSQL from
    # docker-compose.yml, e.g. postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya.
    # The default is a SQLite file under data/ so nothing else is needed for development.
    pharma_database_url: str = f"sqlite:///{(PROJECT_DIR / 'data' / 'tujya.db').as_posix()}"
    # ONNX models for /api/scan: detector.onnx, classifier.onnx and models.json, written by
    # `python -m tujya_vision.publish` (SYNERGYRX_MODELS_DIR). Loaded on the first scan.
    models_dir: Path = PROJECT_DIR / "models"
    # "cpu" always works; "auto" uses CUDA first when the installed onnxruntime has it.
    onnx_providers: str = "cpu"
    onnx_threads: int = 0  # 0 = ONNX Runtime default (one per core)
    # Below this top-1 probability the app asks the user to confirm the pill. Chosen on the
    # validation photos: docs/VISION.md, "Confirmation threshold" (decision D47).
    scan_confirm_threshold: float = 0.85
    scan_detection_confidence: float = 0.25
    scan_max_bytes: int = 10 * 1024 * 1024
    scan_max_pixels: int = 50_000_000
    # C3PI test-split photos shown in the demo gallery (read-only); see app/routers/demo.py.
    demo_images_dir: Path = PROJECT_DIR / "data" / "vision" / "c3pi" / "images"
    demo_split_file: Path = PROJECT_DIR / "config" / "splits" / "selected_images.csv"
    demo_gallery_file: Path = PROJECT_DIR / "config" / "demo_gallery.csv"
    # Tables written by `synergyrx clean` (the pipeline keeps its Parquet tables in
    # data/parquet); the API only ever reads them.
    processed_dir: Path = PROJECT_DIR / "data" / "parquet"

    # Module 2: RAG and agent (docs/RAG.md, docs/AGENT.md). The chunk index lives in the
    # pharma database (table rag_chunks, pgvector); `python -m app.rag.index` fills it from
    # this file. Ollama serves both the LLM and the embeddings (docker compose `ollama`).
    rag_chunks_file: Path = PROJECT_DIR / "data" / "rag" / "corpus" / "chunks.parquet"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1
    # 4096 keeps qwen2.5:3b (2.2 GB) and bge-m3 on the 4 GB GPU together; at 8192 Ollama
    # evicts one for the other on every question. Agent prompts are ~1,300 tokens.
    llm_num_ctx: int = 4096
    llm_timeout_seconds: float = 180.0
    # A small model can loop on itself; answers are 3-6 sentences, far below this.
    llm_max_tokens: int = 450
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_num_ctx: int = 2048  # chunks are at most ~400 tokens
    rag_top_k: int = 6
    # Chosen on the benchmark (docs/BENCHMARK.md, D65): dense + ingredient filter is best;
    # hybrid loses on French questions (BM25 only matches English words).
    rag_search_mode: str = "dense"
    # An uncited answer sentence gets the source closest in meaning (bge-m3 cosine) above
    # this; calibrated on the first benchmark run (docs/BENCHMARK.md). 0 disables it.
    rag_attribution_threshold: float = 0.65

    # Bonus: single-molecule toxicity (docs/TOXICITY.md). The model bundle is published by
    # `python scripts/train_tox21.py publish`; ingredient names -> SMILES from PubChem.
    tox_model_dir: Path = PROJECT_DIR / "models" / "tox21"
    ingredient_smiles_file: Path = PROJECT_DIR / "config" / "ingredient_smiles.csv"

    session_cookie: str = "synergyrx_session"
    session_hours: int = 12
    cookie_secure: bool = False  # set to true as soon as the site is served over HTTPS

    # Accounts created the first time the API starts on an empty database.
    admin_email: str = "admin@gmail.com"
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    seed_demo_user: bool = True
    demo_user_email: str = "user@gmail.com"
    demo_user_password: str = "user"  # noqa: S105  demo account
