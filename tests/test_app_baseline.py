from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app, create_app

client = TestClient(app)


def test_create_app_returns_fastapi():
    from fastapi import FastAPI
    assert isinstance(create_app(), FastAPI)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_runs_ingest_all(monkeypatch):
    async def fake_ingest_all(**kwargs):
        return {"NIST": 3, "ISO": 2}

    monkeypatch.setattr(main_mod, "ingest_all", fake_ingest_all)
    r = client.post("/ingest")
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"NIST": 3, "ISO": 2}
    assert body["total"] == 5
