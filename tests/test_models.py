from app.models import Chunk, MapRequest, QueryRequest


def test_chunk_keyword_construction_with_defaults():
    c = Chunk(framework="NIST", control_id="AC-2", title="Account Management",
             summary="Manage accounts.")
    assert c.chunk_idx == 0
    assert c.source_url == ""
    assert c.score == 0.0
    assert c.id is None
    assert c.embedding is None


def test_chunk_full_construction():
    c = Chunk(framework="ISO", control_id="A.5.1", chunk_idx=1, title="Policies",
             summary="Policy set.", source_url="http://x", score=0.9, id=42,
             embedding=[0.1, 0.2])
    assert c.id == 42
    assert c.embedding == [0.1, 0.2]
    assert c.score == 0.9


def test_query_request_defaults_framework_none():
    q = QueryRequest(question="how to manage accounts?")
    assert q.framework is None


def test_map_request_requires_finding_text():
    m = MapRequest(finding_text="stale accounts remain active")
    assert m.finding_text == "stale accounts remain active"
