import json

from fastapi.testclient import TestClient

import app.main as main_mod
from app.agent import AgentResult
from app.main import app


def _result(answer, citations, refused, stop_reason="answered"):
    return AgentResult(answer=answer, citations=citations, trace=[{"tool": "search_controls",
                       "args": {}, "result_summary": "1 controls: [NIST AC-2]"}],
                       steps=1, input_tokens=10, output_tokens=5,
                       stop_reason=stop_reason, refused=refused)


async def _collect(gen):
    return [ev async for ev in gen]


async def test_agent_events_streams_token_citations_meta_done():
    result = _result("Manage accounts [NIST AC-2].", ["NIST AC-2"], refused=False)
    events = await _collect(main_mod._agent_events(result, cost_usd=0.0))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "token"
    assert "citations" in kinds
    assert "meta" in kinds
    assert kinds[-1] == "done"
    cite = next(e for e in events if e["event"] == "citations")
    assert json.loads(cite["data"])["citations"] == ["NIST AC-2"]
    meta = next(e for e in events if e["event"] == "meta")
    md = json.loads(meta["data"])
    assert md["stop_reason"] == "answered"
    assert md["input_tokens"] == 10
    assert "cost_usd" in md


async def test_agent_events_refusal_only():
    result = _result("", [], refused=True, stop_reason="max_steps")
    events = await _collect(main_mod._agent_events(result, cost_usd=0.0))
    assert [e["event"] for e in events] == ["refusal"]
    assert "message" in json.loads(events[0]["data"])


def test_query_endpoint_streams(monkeypatch):
    async def fake_run_agent(question, provider, *, framework=None, **kw):
        return _result("Manage accounts [NIST AC-2].", ["NIST AC-2"], refused=False)

    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "get_provider", lambda: object())
    client = TestClient(app)
    with client.stream("POST", "/query", json={"question": "accounts"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: meta" in body
    assert "event: done" in body


def test_query_endpoint_refuses(monkeypatch):
    async def fake_run_agent(question, provider, *, framework=None, **kw):
        return _result("", [], refused=True, stop_reason="answered")

    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "get_provider", lambda: object())
    client = TestClient(app)
    with client.stream("POST", "/query", json={"question": "nonsense"}) as resp:
        body = "".join(resp.iter_text())
    assert "event: refusal" in body


async def test_meta_event_carries_tool_trace():
    result = _result("Manage accounts [NIST AC-2].", ["NIST AC-2"], refused=False)
    events = await _collect(main_mod._agent_events(result, cost_usd=0.0))
    meta = json.loads(next(e for e in events if e["event"] == "meta")["data"])
    assert meta["trace"] == [{"tool": "search_controls", "args": {},
                              "result_summary": "1 controls: [NIST AC-2]"}]
