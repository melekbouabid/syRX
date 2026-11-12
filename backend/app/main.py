"""Application factory. Run with `uvicorn app.main:app --reload` from the backend folder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import sessionmaker

from app.accounts import seed_accounts, warn_about_default_password
from app.boxes import BoxReader, OcrHolder, RxNavClient
from app.config import Settings
from app.db import Base, make_engine
from app.drift import DriftMonitor
from app.inference import ScannerHolder
from app.limits import BodySizeLimit
from app.routers import (
    admin,
    auth,
    chat,
    demo,
    documents,
    medicines,
    monitoring,
    scan,
    toxicity,
    vision,
)
from src.agent.runtime import AgentHolder
from src.models.toxicity.model import ToxHolder

DESCRIPTION = (
    "Pill identification and interaction checks for the Tujya project. "
    "Information uniquement, consultez un pharmacien ou un médecin. "
    "No doses are ever given; a pair missing from the database is reported as "
    '"aucune interaction connue dans cette base de données", never as safe.'
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    # Medicine tables: their own database (PostgreSQL in docker compose), read only here.
    pharma_engine = make_engine(settings.pharma_database_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        Base.metadata.create_all(engine)
        with session_factory() as db:
            seed_accounts(db, settings)
            warn_about_default_password(db, settings)
        yield
        engine.dispose()
        pharma_engine.dispose()

    # Swagger UI at /docs (required by the course); the older /api/docs address redirects
    # there and /api/openapi.json serves the same schema, so existing links keep working.
    app = FastAPI(
        title="Tujya API",
        version="0.3.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @app.get("/api/docs", include_in_schema=False)
    def old_docs_address() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/api/openapi.json", include_in_schema=False)
    def old_schema_address() -> JSONResponse:
        return JSONResponse(app.openapi())

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.pharma_engine = pharma_engine
    app.state.scanner = ScannerHolder(
        settings.models_dir,
        providers=settings.onnx_providers,
        threads=settings.onnx_threads,
        conf=settings.scan_detection_confidence,
    )
    # Medicine boxes (step 2.7): OCR loaded on the first scan, brand and INN tables in config/.
    app.state.boxes = BoxReader(
        OcrHolder(settings.ocr_threads, enabled=settings.box_ocr_enabled),
        settings.brand_table_file,
        settings.inn_aliases_file,
        RxNavClient(
            settings.rxnav_contact_email,
            base_url=settings.rxnav_url,
            timeout=settings.rxnav_timeout_seconds,
        ),
    )
    # Data drift (step 2.8): statistics of the scanned photos in a sliding window, in memory.
    app.state.drift = DriftMonitor.from_file(settings.drift_reference_file)
    # Bonus: Tox21 toxicity model, loaded on first use (models/tox21, docs/bonus/toxicity.md).
    app.state.toxicity = ToxHolder(settings.tox_model_dir, settings.ingredient_smiles_file)
    # Module 2 agent: built on the first /api/chat call (Ollama, chunk index in Qdrant).
    app.state.agents = AgentHolder(
        settings, pharma_engine, app.state.scanner, toxicity=app.state.toxicity
    )
    # Refuse an oversized upload from its Content-Length, before the body is read.
    app.add_middleware(BodySizeLimit, max_bytes=settings.scan_max_bytes + 64 * 1024)

    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(scan.router)
    app.include_router(medicines.router)
    app.include_router(demo.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(toxicity.router)
    # Endpoints of the course guide: /health, /model/metadata, /predict, /predict/visualize.
    app.include_router(vision.router)
    app.include_router(monitoring.router)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
