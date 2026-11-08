"""Settings of the web API.

Every value can be overridden in `backend/.env` or the environment with the `TUJYA_`
prefix, for example `TUJYA_ADMIN_PASSWORD=...`. The project's first prefix, `SYNERGYRX_`, is
still read after it, so that existing `.env` files and shells keep working (decision D86).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
# The repository root (data/, config/, models/). The Docker image keeps the same layout under
# /tujya (backend/Dockerfile), where docker-compose.yml mounts those folders.
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_ADMIN_PASSWORD = "admin"  # noqa: S105  first-run default, meant to be changed
ENV_PREFIX = "TUJYA_"
# Fallback for the settings written before the rename (D86); TUJYA_ always wins.
LEGACY_ENV_PREFIX = "SYNERGYRX_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_prefix=ENV_PREFIX, extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Arguments, then TUJYA_* (environment, .env), then the legacy SYNERGYRX_* names."""
        env_file = getattr(dotenv_settings, "env_file", cls.model_config.get("env_file"))
        legacy_env = EnvSettingsSource(settings_cls, env_prefix=LEGACY_ENV_PREFIX)
        legacy_dotenv = DotEnvSettingsSource(
            settings_cls, env_file=env_file, env_prefix=LEGACY_ENV_PREFIX
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            legacy_env,
            legacy_dotenv,
            file_secret_settings,
        )

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'app.db').as_posix()}"
    # Tujya medicine tables (pills, ingredients, interactions): PostgreSQL from
    # docker-compose.yml, e.g. postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya.
    # The default is a SQLite file under data/ so nothing else is needed for development.
    pharma_database_url: str = f"sqlite:///{(PROJECT_DIR / 'data' / 'tujya.db').as_posix()}"
    # ONNX models for /api/scan: detector.onnx, classifier.onnx and models.json, written by
    # `python -m src.models.publish` (TUJYA_MODELS_DIR). Loaded on the first scan.
    models_dir: Path = PROJECT_DIR / "models"
    # "cpu" always works; "auto" uses CUDA first when the installed onnxruntime has it.
    onnx_providers: str = "cpu"
    onnx_threads: int = 0  # 0 = ONNX Runtime default (one per core)
    # Below this top-1 probability the app asks the user to confirm the pill. Chosen on the
    # validation photos: docs/module1/vision.md, "Confirmation threshold" (D47).
    scan_confirm_threshold: float = 0.85
    scan_detection_confidence: float = 0.25
    scan_max_bytes: int = 10 * 1024 * 1024
    scan_max_pixels: int = 50_000_000
    # C3PI test-split photos shown in the demo gallery (read-only); see app/routers/demo.py.
    # A photo is served only if it is in the v2 test split (detector) and not in the v1
    # train or val split (classifier): never seen in training by either served model (D105).
    demo_images_dir: Path = PROJECT_DIR / "data" / "raw" / "c3pi" / "images"
    demo_split_file: Path = PROJECT_DIR / "config" / "splits" / "selected_images.csv"
    demo_holdout_file: Path = PROJECT_DIR / "config" / "splits" / "c3pi_v2" / "test.csv"
    demo_gallery_file: Path = PROJECT_DIR / "config" / "demo_gallery.csv"
    # Grad-CAM pictures of the gallery photos, precomputed with the PyTorch weights of the
    # served classifier (`python -m src.models.gradcam_gallery`, DVC stage, D104).
    demo_gradcam_dir: Path = PROJECT_DIR / "data" / "processed" / "c3pi" / "gradcam"

    # Module 2: RAG and agent (docs/module2/rag.md, agent.md). The chunk index
    # lives in the pharma database (table rag_chunks, pgvector);
    # `python -m src.rag.index` fills it from this file. Ollama serves both
    # the LLM and the embeddings (docker compose `ollama`).
    rag_chunks_file: Path = PROJECT_DIR / "data" / "processed" / "rag" / "corpus" / "chunks.parquet"
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
    # Chosen on the benchmark (docs/module2/benchmark.md, D65): dense +
    # ingredient filter is best; hybrid loses on French questions (BM25 only matches English).
    rag_search_mode: str = "dense"
    # An uncited answer sentence gets the source closest in meaning (bge-m3 cosine) above
    # this; calibrated on the first benchmark run (docs/module2/benchmark.md).
    # 0 disables it.
    rag_attribution_threshold: float = 0.65

    # Bonus: single-molecule toxicity (docs/bonus/toxicity.md). The model bundle is published by
    # `python -m src.models.toxicity.train_tox21 publish`; ingredient names -> SMILES from PubChem.
    tox_model_dir: Path = PROJECT_DIR / "models" / "tox21"
    ingredient_smiles_file: Path = PROJECT_DIR / "config" / "ingredient_smiles.csv"

    session_cookie: str = "tujya_session"
    session_hours: int = 12
    cookie_secure: bool = False  # set to true as soon as the site is served over HTTPS

    # Accounts created the first time the API starts on an empty database.
    admin_email: str = "admin@gmail.com"
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    seed_demo_user: bool = True
    demo_user_email: str = "user@gmail.com"
    demo_user_password: str = "user"  # noqa: S105  demo account
