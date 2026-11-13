"""Ship smoke: demo page served, README curl bodies match the API request models. Offline."""
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_mod
from app.agent import AgentResult
from app.main import app
from app.models import MapRequest, QueryRequest

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _readme_bodies() -> list[dict]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [json.loads(raw) for raw in re.findall(r"-d '(\{.*?\})'", text, flags=re.DOTALL)]


def _bodies_with(key: str) -> list[dict]:
    found = [b for b in _readme_bodies() if key in b]
    assert found, f"no README curl body with a '{key}' field found"
    return found


def test_demo_page_served_at_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="answer"' in resp.text
    assert '"/query"' in resp.text


def test_every_readme_body_is_query_or_map():
    # A renamed field must not silently drop a body out of the checks below.
    for body in _readme_bodies():
        assert ("question" in body) != ("finding_text" in body), body


def test_readme_query_bodies_match_model():
    for body in _bodies_with("question"):
        assert set(body) <= set(QueryRequest.model_fields)
        QueryRequest(**body)


def test_readme_map_bodies_match_model():
    for body in _bodies_with("finding_text"):
        assert set(body) <= set(MapRequest.model_fields)
        MapRequest(**body)


def test_readme_query_bodies_accepted_by_endpoint(monkeypatch):
    async def fake_run_agent(question, provider, *, framework=None, **kw):
        return AgentResult(answer="Manage accounts [NIST AC-2].", citations=["NIST AC-2"],
                           trace=[], steps=1, input_tokens=1, output_tokens=1,
                           stop_reason="answered", refused=False)

    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "get_provider", lambda: object())
    for body in _bodies_with("question"):
        with client.stream("POST", "/query", json=body) as resp:
            assert resp.status_code == 200, body
            assert "text/event-stream" in resp.headers["content-type"]
            "".join(resp.iter_text())


def test_demo_sends_query_model_fields():
    demo = client.get("/").text
    m = re.search(r'fetch\("/query".*?JSON\.stringify\(\{([^}]*)\}\)', demo, flags=re.DOTALL)
    assert m, "demo fetch('/query') body not found"
    sent = {k.strip().split(":")[0] for k in m.group(1).split(",") if k.strip()}
    assert sent <= set(QueryRequest.model_fields)
    assert "question" in sent
    readme_keys = {k for b in _bodies_with("question") for k in b}
    assert readme_keys <= set(QueryRequest.model_fields)
    assert "question" in readme_keys and "framework" in readme_keys
