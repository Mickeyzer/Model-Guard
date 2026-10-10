import os
import tempfile

# Point the middleware at a throwaway database before middleware.db is
# imported (it creates its engine at import time), so tests never touch
# data/modelguard.db.
_tmp_dir = tempfile.mkdtemp(prefix="modelguard-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp_dir, "test.db").replace("\\", "/")

import joblib
import pytest
from fastapi.testclient import TestClient

from middleware.db import Base, SessionLocal, engine, init_db
from middleware import defense

PROTECTED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protected_model", "model.pkl"
)


@pytest.fixture(autouse=True)
def clean_state():
    init_db()
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    defense._rate_limit_store.clear()
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def protected_model(monkeypatch):
    """Answers like the real protected service, in-process, and counts calls."""
    if not os.path.exists(PROTECTED_MODEL_PATH):
        pytest.skip("protected_model/model.pkl missing; run protected_model/train_and_save.py")
    model = joblib.load(PROTECTED_MODEL_PATH)
    calls = []

    def fake_call(features):
        calls.append(features)
        return {
            "prediction": int(model.predict([features])[0]),
            "probabilities": model.predict_proba([features])[0].tolist(),
            "model": "iris-random-forest-v1",
        }

    monkeypatch.setattr("middleware.app.call_protected_model", fake_call)
    return calls


@pytest.fixture
def client(protected_model):
    from middleware.app import app
    return TestClient(app)


@pytest.fixture
def fixed_score(monkeypatch):
    """Force the fingerprint engine to return a chosen threat score."""
    from middleware import fingerprint_engine

    def set_score(score):
        monkeypatch.setattr(
            fingerprint_engine, "classify_query",
            lambda recent: {"threat_score": score, "features_used": {}, "model_loaded": True},
        )
    return set_score
