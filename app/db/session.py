from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

_engine = None
_SessionLocal = None


def is_db_enabled() -> bool:
    return settings.database_enabled


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine


def init_db() -> None:
    if not is_db_enabled():
        return

    engine = _get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsvector tsvector"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING gin(content_tsvector)"))


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if not is_db_enabled():
        raise RuntimeError("Database is not configured. Set DATABASE_URL.")

    session = _get_engine()  # ensure engine exists
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
