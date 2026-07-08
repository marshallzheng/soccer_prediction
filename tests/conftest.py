import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from corner_predictor.persistence import db


@pytest.fixture(autouse=True)
def isolated_in_memory_db(monkeypatch):
    # StaticPool keeps a single shared connection alive across threads, which
    # in-memory SQLite needs -- FastAPI's TestClient runs sync routes in a
    # worker thread, and each new connection to "sqlite://" is otherwise a
    # brand new (empty) database.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", session_local)
    db.Base.metadata.create_all(engine)
    yield
