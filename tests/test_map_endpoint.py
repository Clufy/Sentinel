# tests/test_map_endpoint.py
import json

from fastapi.testclient import TestClient

import app.main as main_mod
from app.llm import ProviderResponse
from app.main import app


class StubProvider:
    async def chat(self, messages, tools=None):
        return ProviderResponse(
            content=json.dumps({
                "[NIST AC-2]": "Manages account lifecycle.",
                "[ISO A.5.15]": "Governs access rights.",
            }),
            tool_calls=[], input_tokens=10, output_tokens=5,
        )


def _patch(monkeypatch):
    async def fake_map_finding(text):
        return {"controls": [
            {"framework": "NIST", "control_id": "AC-2", "title": "Account Management",
             "summary": "Manage accounts.", "source_url": "u", "score": 0.9, "citation": "[NIST AC-2]"},
            {"framework": "ISO", "control_id": "A.5.15", "title": "Access control",
             "summary": "Control access.", "source_url": "u", "score": 0.8, "citation": "[ISO A.5.15]"},
        ]}

    monkeypatch.setattr(main_mod, "map_finding", fake_map_finding)
    monkeypatch.setattr(main_mod, "get_provider", lambda: StubProvider())


def test_map_returns_controls_across_two_frameworks(monkeypatch):
    _patch(monkeypatch)
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    frameworks = {c["framework"] for c in body["controls"]}
    assert len(frameworks) >= 2
    nist = next(c for c in body["controls"] if c["control_id"] == "AC-2")
    assert nist["rationale"] == "Manages account lifecycle."


def test_map_reports_tokens_trace_citations_cost(monkeypatch):
    _patch(monkeypatch)
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    assert body["input_tokens"] == 10
    assert body["output_tokens"] == 5
    assert "cost_usd" in body
    assert "NIST AC-2" in body["citations"]
    assert any(t["tool"] == "map_finding" for t in body["tool_trace"])


def test_map_rejects_empty_finding(monkeypatch):
    _patch(monkeypatch)
    resp = TestClient(app).post("/map", json={"finding_text": ""})
    assert resp.status_code == 422


def test_map_falls_back_on_bad_json(monkeypatch):
    _patch(monkeypatch)

    class BadProvider:
        async def chat(self, messages, tools=None):
            return ProviderResponse(content="not json", tool_calls=[], input_tokens=1, output_tokens=1)

    monkeypatch.setattr(main_mod, "get_provider", lambda: BadProvider())
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    assert all(c["rationale"] for c in body["controls"])


# --- I5: relevance floor and honest rationale -----------------------------
def _patch_controls(monkeypatch, controls, provider):
    async def fake_map_finding(text, framework=None):
        return {"controls": controls}

    monkeypatch.setattr(main_mod, "map_finding", fake_map_finding)
    monkeypatch.setattr(main_mod, "get_provider", lambda: provider)


def _ctl(fw, cid, score):
    return {"framework": fw, "control_id": cid, "title": f"{cid} title", "summary": "s",
            "source_url": "u", "score": score, "citation": f"[{fw} {cid}]"}


class _Content:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        return ProviderResponse(content=self.content, tool_calls=[],
                                input_tokens=3, output_tokens=2)


def test_map_returns_what_map_finding_kept(monkeypatch):
    # The similarity floor lives in map_finding (tests/test_agent_tools.py); the route
    # passes its controls through unchanged rather than re-filtering on another scale.
    provider = _Content("{}")
    _patch_controls(monkeypatch, [_ctl("NIST", "AC-2", 0.45), _ctl("ISO", "A.8.2", 0.21)],
                    provider)
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    assert [c["control_id"] for c in body["controls"]] == ["AC-2", "A.8.2"]
    assert body["citations"] == ["NIST AC-2", "ISO A.8.2"]
    assert body["refused"] is False


def test_map_nothing_relevant_refuses_without_llm_call(monkeypatch):
    from app.obs import metrics_store

    provider = _Content("{}")
    _patch_controls(monkeypatch, [], provider)  # map_finding kept nothing
    metrics_store.reset()
    resp = TestClient(app).post("/map", json={"finding_text": "my cat is hungry"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["controls"] == []
    assert body["citations"] == []
    assert body["refused"] is True
    assert provider.calls == 0
    assert body["input_tokens"] == 0 and body["cost_usd"] == 0.0
    assert metrics_store.records[-1].refused is True


def test_map_parses_code_fenced_json_with_unbracketed_keys(monkeypatch):
    content = ('```json\n{"NIST AC-2": "Covers account lifecycle.",'
               ' "CIS 5": "Account inventory.", "[iso a.5.15]": "Access rules."}\n```')
    _patch_controls(monkeypatch, [_ctl("NIST", "AC-2", 0.9), _ctl("CIS", "CIS-5", 0.8),
                                  _ctl("ISO", "A.5.15", 0.7)], _Content(content))
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    rationale = {c["control_id"]: c["rationale"] for c in body["controls"]}
    assert rationale == {"AC-2": "Covers account lifecycle.", "CIS-5": "Account inventory.",
                         "A.5.15": "Access rules."}


def test_map_fallback_rationale_is_honest(monkeypatch):
    _patch_controls(monkeypatch, [_ctl("NIST", "AC-2", 0.9), _ctl("ISO", "A.5.15", 0.8)],
                    _Content('{"[NIST AC-2]": "Manages accounts."}'))
    body = TestClient(app).post("/map", json={"finding_text": "stale accounts"}).json()
    iso = next(c for c in body["controls"] if c["control_id"] == "A.5.15")
    assert iso["rationale"] == "No model rationale; matched by retrieval (score 0.80)."
    assert "addresses this finding" not in json.dumps(body)
