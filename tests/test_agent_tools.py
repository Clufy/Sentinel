# tests/test_agent_tools.py
import pytest

import app.agent as agent
from app.models import Chunk


def _c(framework, cid, title, summary, score):
    return Chunk(framework=framework, control_id=cid, title=title, summary=summary,
                 source_url=f"https://x/{cid}", score=score, id=1)


@pytest.fixture
def fake_retrieve(monkeypatch):
    calls = []

    async def fake(query, framework, top_k):
        calls.append({"query": query, "framework": framework, "top_k": top_k})
        catalog = {
            "NIST": [_c("NIST", "AC-2", "Account Management", "Manage accounts.", 0.9)],
            "ISO": [_c("ISO", "A.5.15", "Access control", "Control access.", 0.8)],
            "CIS": [_c("CIS", "CIS-5", "Account Management", "Manage accounts.", 0.7)],
        }
        if framework is None:
            return [c for rows in catalog.values() for c in rows]
        return catalog.get(framework, [])

    monkeypatch.setattr(agent, "retrieve", fake)
    return calls


@pytest.fixture
def fake_vector_search(monkeypatch):
    """map_finding ranks by embedding cosine (vector_search), never the cross-encoder."""
    calls = []

    async def fake(query, framework, limit):
        calls.append({"query": query, "framework": framework, "limit": limit})
        rows = [_c("CIS", "CIS-5", "Account Management", "Manage accounts.", 0.51),
                _c("NIST", "AC-2", "Account Management", "Manage accounts.", 0.45),
                _c("ISO", "A.8.2", "Privileged access rights", "Limit privilege.", 0.34),
                _c("ISO", "A.5.24", "Incident planning", "Plan for incidents.", 0.12)]
        return [c for c in rows if framework is None or c.framework == framework]

    async def no_retrieve(*a, **k):
        raise AssertionError("map_finding must not use the cross-encoder retrieve()")

    monkeypatch.setattr(agent, "vector_search", fake)
    monkeypatch.setattr(agent, "retrieve", no_retrieve)
    return calls


@pytest.fixture
def fake_db(monkeypatch):
    catalog = {
        ("NIST", "AC-2"): _c("NIST", "AC-2", "Account Management", "Manage accounts.", 0.0),
        ("ISO", "A.5.15"): _c("ISO", "A.5.15", "Access control", "Control access.", 0.0),
        ("ISO", "A.5.16"): _c("ISO", "A.5.16", "Identity management", "Manage identities.", 0.0),
        ("CIS", "CIS-5"): _c("CIS", "CIS-5", "Account Management", "Manage accounts.", 0.0),
    }

    class FakeConn:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class FakePool:
        def acquire(self):
            return FakeConn()

    async def fake_pool():
        return FakePool()

    async def fake_get_by_control_id(conn, framework, control_id):
        return catalog.get((framework, control_id))

    monkeypatch.setattr(agent, "get_pool", fake_pool)
    monkeypatch.setattr(agent, "get_by_control_id", fake_get_by_control_id)


async def test_search_controls_returns_dicts_with_citation(fake_retrieve):
    result = await agent.search_controls("access control", framework="NIST")
    assert result["controls"][0]["citation"] == "[NIST AC-2]"
    assert fake_retrieve[0]["framework"] == "NIST"


async def test_search_controls_defaults_framework_none(fake_retrieve):
    await agent.search_controls("access control")
    assert fake_retrieve[0]["framework"] is None


async def test_compare_frameworks_uses_deterministic_lookup(fake_db):
    # AC-2 -> ISO A.5.15/A.5.16 + CIS CIS-5 per corpus/mappings.yaml seed.
    result = await agent.compare_frameworks("AC-2")
    assert result["source"]["citation"] == "[NIST AC-2]"
    related = {c["citation"] for c in result["related"]}
    assert "[ISO A.5.15]" in related
    assert "[CIS CIS-5]" in related


async def test_compare_frameworks_unknown_control(fake_db):
    result = await agent.compare_frameworks("ZZ-99")
    assert result["source"] is None
    assert result["related"] == []


async def test_map_finding_ranks_across_frameworks(fake_vector_search):
    result = await agent.map_finding("employees keep active accounts after leaving")
    frameworks = {c["framework"] for c in result["controls"]}
    assert frameworks >= {"NIST", "ISO", "CIS"}
    scores = [c["score"] for c in result["controls"]]
    assert scores == sorted(scores, reverse=True)
    assert fake_vector_search[0]["framework"] is None


async def test_map_finding_drops_below_map_min_similarity(fake_vector_search, monkeypatch):
    # Findings are statements, which the MS-MARCO cross-encoder scores ~0 even for the
    # right control (measured live), so /map gates on embedding cosine instead.
    monkeypatch.setattr(agent, "get_settings", lambda: type("S", (), {
        "retrieval_top_k": 8, "map_min_similarity": 0.2})())
    result = await agent.map_finding("former employees keep active accounts")
    assert [c["control_id"] for c in result["controls"]] == ["CIS-5", "AC-2", "A.8.2"]


def test_tool_schemas_cover_three_tools():
    assert {s["function"]["name"] for s in agent.TOOL_SCHEMAS} == \
        {"search_controls", "compare_frameworks", "map_finding"}


def test_tool_functions_map_names():
    assert set(agent.TOOL_FUNCTIONS) == {"search_controls", "compare_frameworks", "map_finding"}
    assert callable(agent.TOOL_FUNCTIONS["search_controls"])


async def test_compare_frameworks_normalises_control_id(fake_db, monkeypatch):
    seen = []

    async def lookup(conn, framework, control_id):
        seen.append((framework, control_id))
        return {("NIST", "AC-2.1"): _c("NIST", "AC-2.1", "Automated", "Auto.", 0.0),
                ("CIS", "CIS-5"): _c("CIS", "CIS-5", "Account Management", "M.", 0.0),
                }.get((framework, control_id))

    monkeypatch.setattr(agent, "get_by_control_id", lookup)
    assert (await agent.compare_frameworks("ac-2(1)"))["source"]["citation"] == "[NIST AC-2.1]"
    assert (await agent.compare_frameworks("5"))["source"]["citation"] == "[CIS CIS-5]"
    assert (await agent.compare_frameworks("CIS 5"))["source"]["citation"] == "[CIS CIS-5]"


async def test_map_finding_accepts_framework(fake_vector_search):
    result = await agent.map_finding("stale accounts", framework="ISO")
    assert fake_vector_search[0]["framework"] == "ISO"
    assert {c["framework"] for c in result["controls"]} == {"ISO"}


async def test_map_finding_lists_each_control_once(monkeypatch):
    # A long summary is split into several chunks of one control; /map should list
    # the control once, at its best-scoring chunk.
    async def split(query, framework, limit):
        return [_c("NIST", "AC-2", "Account Management", "part one", 0.50),
                _c("NIST", "AC-2", "Account Management", "part two", 0.44),
                _c("CIS", "CIS-5", "Account Management", "Manage accounts.", 0.47)]

    monkeypatch.setattr(agent, "vector_search", split)
    monkeypatch.setattr(agent, "get_settings", lambda: type("S", (), {
        "retrieval_top_k": 8, "map_min_similarity": 0.35})())
    result = await agent.map_finding("stale accounts")
    assert [(c["control_id"], c["score"]) for c in result["controls"]] == [
        ("AC-2", 0.50), ("CIS-5", 0.47)]
