"""Application factory. Run with `uvicorn app.main:app --reload` from the backend folder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from app.accounts import seed_accounts, warn_about_default_password
from app.catalog import Catalog
from app.config import Settings
from app.db import Base, make_engine
from app.routers import admin, auth, catalog


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    session_factory = sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        Base.metadata.create_all(engine)
        with session_factory() as db:
            seed_accounts(db, settings)
            warn_about_default_password(db, settings)
        yield
        engine.dispose()

    app = FastAPI(
        title="SynergyRx API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.catalog = Catalog(settings.processed_dir)

    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(catalog.router)
    app.include_router(catalog.public_router)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
