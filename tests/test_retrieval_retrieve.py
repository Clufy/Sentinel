import pytest

from app.models import Chunk
import app.retrieval as R


def _c(cid, emb=None):
    return Chunk(framework="NIST", control_id=cid, title=f"t-{cid}", summary=f"s-{cid}",
                 embedding=emb or [1.0, 0.0])


@pytest.fixture(autouse=True)
def wire(monkeypatch):
    async def fake_fts(query, framework, limit):
        return ["NIST:AC-2:0", "NIST:AC-1:0"]

    async def fake_vec(query, framework, top_k):
        return [_c("AC-1"), _c("SI-4")]  # carry .embedding already

    async def fake_by_keys(conn, keys):
        return {k: _c(k.split(":")[1]) for k in keys}

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

    monkeypatch.setattr(R, "fts_search", fake_fts)
    monkeypatch.setattr(R, "vector_search", fake_vec)
    monkeypatch.setattr(R, "_chunks_by_keys", fake_by_keys)
    monkeypatch.setattr(R, "get_pool", fake_pool)
    monkeypatch.setattr(R, "embed_query", lambda q: [1.0, 0.0])

    def fake_rerank(query, candidates):
        order = {"AC-1": 0.9, "SI-4": 0.5, "AC-2": 0.4}
        for c in candidates:
            c.score = order[c.control_id]
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    monkeypatch.setattr(R, "rerank", fake_rerank)
    monkeypatch.setattr(R, "mmr", lambda cands, lambda_=0.7, k=8: list(cands)[:k])


async def test_retrieve_returns_reranked_chunks():
    out = await R.retrieve("password policy", framework=None, top_k=3)
    assert [c.control_id for c in out] == ["AC-1", "SI-4", "AC-2"]
    assert out[0].score == pytest.approx(0.9)


async def test_retrieve_respects_top_k():
    out = await R.retrieve("q", framework=None, top_k=2)
    assert [c.control_id for c in out] == ["AC-1", "SI-4"]


async def test_retrieve_empty_when_no_hits(monkeypatch):
    async def empty_fts(q, f, limit):
        return []

    async def empty_vec(q, f, t):
        return []

    monkeypatch.setattr(R, "fts_search", empty_fts)
    monkeypatch.setattr(R, "vector_search", empty_vec)
    assert await R.retrieve("nothing matches") == []


async def test_retrieve_uses_lambda_0_7(monkeypatch):
    seen = {}

    def spy(cands, lambda_=0.5, k=8):
        seen["lambda_"] = lambda_
        return list(cands)[:k]

    monkeypatch.setattr(R, "mmr", spy)
    await R.retrieve("q", framework=None, top_k=3)
    assert seen["lambda_"] == 0.7
