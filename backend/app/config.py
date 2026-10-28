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
    # docker-compose.yml, e.g. postgresql+psycopg://tujya:tujya@localhost:5432/tujya.
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
    # C3PI test-split photos shown in the demo gallery (read-only); see app/routers/scan.py.
    demo_images_dir: Path = PROJECT_DIR / "data" / "vision" / "c3pi" / "images"
    demo_split_file: Path = PROJECT_DIR / "config" / "splits" / "selected_images.csv"
    # Tables written by `synergyrx clean` (the pipeline keeps its Parquet tables in
    # data/parquet); the API only ever reads them.
    processed_dir: Path = PROJECT_DIR / "data" / "parquet"

    session_cookie: str = "synergyrx_session"
    session_hours: int = 12
    cookie_secure: bool = False  # set to true as soon as the site is served over HTTPS

    # Accounts created the first time the API starts on an empty database.
    admin_email: str = "admin@gmail.com"
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    seed_demo_user: bool = True
    demo_user_email: str = "user@gmail.com"
    demo_user_password: str = "user"  # noqa: S105  demo account
