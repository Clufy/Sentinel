from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _doc():
    return (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")


def test_exists():
    assert (ROOT / "docs" / "architecture.md").exists()


def test_has_mermaid():
    assert "```mermaid" in _doc()


def test_covers_retrieval_pipeline():
    for stage in ("FTS", "vector", "RRF", "rerank", "MMR"):
        assert stage in _doc(), stage


def test_covers_agent_and_grounding():
    t = _doc().lower()
    assert "agent" in t
    assert "min_relevance" in t or "refus" in t
    assert "citation" in t


def test_lists_endpoints():
    for ep in ("/query", "/map", "/ingest", "/health", "/metrics"):
        assert ep in _doc(), ep
