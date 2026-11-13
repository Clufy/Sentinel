import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.llm import FakeProvider
from app.main import app


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_db")
def test_query_streams_grounded_answer(monkeypatch):
    monkeypatch.setattr(main_mod, "get_provider", lambda: FakeProvider())
    client = TestClient(app)
    with client.stream("POST", "/query",
                       json={"question": "how do I manage user accounts?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: token" in body
    assert "event: citations" in body
    assert "NIST AC-2" in body  # grounded citation in the citations event


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_db")
def test_query_unanswerable_refuses(monkeypatch):
    monkeypatch.setattr(main_mod, "get_provider", lambda: FakeProvider())
    client = TestClient(app)
    with client.stream("POST", "/query",
                       json={"question": "what is the airspeed velocity of an unladen swallow?"}) as resp:
        body = "".join(resp.iter_text())
    # FakeProvider emits no citations when no controls are relevant -> refusal.
    assert "event: refusal" in body
    assert "event: token" not in body


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_db")
def test_map_covers_multiple_frameworks(monkeypatch):
    monkeypatch.setattr(main_mod, "get_provider", lambda: FakeProvider())
    client = TestClient(app)
    body = client.post("/map", json={"finding_text": "former employees keep active accounts"}).json()
    frameworks = {c["framework"] for c in body["controls"]}
    assert len(frameworks) >= 2
    assert all(c["rationale"] for c in body["controls"])


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_db")
def test_map_off_topic_finding_is_refused(monkeypatch):
    # Pins the MAP_MIN_SIMILARITY floor on real embeddings: an unrelated "finding"
    # returns no controls and makes no LLM call.
    class NoCall:
        async def chat(self, *a, **k):
            raise AssertionError("no LLM call expected when nothing is relevant")

    monkeypatch.setattr(main_mod, "get_provider", lambda: NoCall())
    body = TestClient(app).post("/map", json={"finding_text": "my cat is hungry"}).json()
    assert body["refused"] is True
    assert body["controls"] == []
