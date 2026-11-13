from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_mod
from app.agent import AgentResult
from app.main import app

client = TestClient(app)

_INDEX = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def test_root_serves_demo_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "<title>Sentinel" in body
    assert 'id="answer"' in body
    assert "/query" in body
    assert 'id="ask"' in body
    assert "JSON.stringify({ question" in body  # demo sends the real request field


def test_index_html_exists_and_utf8():
    assert _INDEX.exists()
    text = _INDEX.read_text(encoding="utf-8")
    assert "getReader" in text  # streams via fetch reader, not EventSource


def test_page_js_splits_sse_frames_crlf_tolerant():
    text = _INDEX.read_text(encoding="utf-8")
    assert r"/\r?\n\r?\n/" in text  # frame split
    assert r"/\r?\n/" in text  # line split
    assert 'indexOf("\n\n")' not in text


def test_api_routes_not_shadowed_by_static():
    assert client.get("/health").json() == {"status": "ok"}


def test_query_sse_frames_are_crlf(monkeypatch):
    async def fake_run_agent(question, provider, *, framework=None, **kw):
        return AgentResult(answer="Manage accounts [NIST AC-2].", citations=["NIST AC-2"],
                           trace=[], steps=1, input_tokens=10, output_tokens=5,
                           stop_reason="answered", refused=False)

    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "get_provider", lambda: object())
    with client.stream("POST", "/query", json={"question": "accounts"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "\r\n\r\n" in body
