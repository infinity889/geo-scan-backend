from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import text
from app.api.routes import router
from app.core.config import settings
from app.db.session import get_session, init_db, is_db_enabled
from app.services.store import seed_database

@asynccontextmanager
async def lifespan(_: FastAPI):
    if is_db_enabled():
        init_db()
        seed_database()
    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="GeoRAG backend prototype for geological document analysis.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        payload = {
            "status": "ok",
            "environment": settings.app_env,
            "database": "disabled",
        }
        if is_db_enabled():
            try:
                with get_session() as session:
                    session.execute(text("SELECT 1"))
                payload["database"] = "connected"
            except Exception as exc:
                payload["status"] = "degraded"
                payload["database"] = f"error: {exc}"
        return payload

    return app


app = create_app()
