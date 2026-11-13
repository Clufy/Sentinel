"""M1 + M2, offline: vector_search's filtered-scan setting and thread offload.

A recording fake connection stands in for asyncpg. These tests pin what we send
and in what order; the live integration pass must still run the SQL against a
real pgvector >= 0.8 (see test_retrieval_search.py's integration tests).
"""
import threading

import pytest

import app.retrieval as R
from app.models import Chunk


class _Txn:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append(("begin",))
        return self

    async def __aexit__(self, *exc):
        self.log.append(("commit",))
        return False


class _Conn:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def transaction(self):
        return _Txn(self.log)

    async def execute(self, sql, *args):
        self.log.append(("execute", " ".join(sql.split())))

    async def fetch(self, sql, *args):
        self.log.append(("fetch", args))
        return []


@pytest.fixture
def conn_log(monkeypatch):
    log = []

    class _Pool:
        def acquire(self):
            return _Conn(log)

    async def fake_pool():
        return _Pool()

    monkeypatch.setattr(R, "get_pool", fake_pool)
    monkeypatch.setattr(R, "embed_query", lambda q: [1.0, 0.0])
    return log


async def test_vector_search_with_framework_sets_iterative_scan_in_transaction(conn_log):
    await R.vector_search("q", "ISO", top_k=5)
    kinds = [e[0] for e in conn_log]
    assert kinds == ["begin", "execute", "fetch", "commit"]
    assert conn_log[1][1] == "SET LOCAL hnsw.iterative_scan = strict_order"
    assert conn_log[2][1][1:] == (5, "ISO")


async def test_vector_search_without_framework_has_no_transaction(conn_log):
    await R.vector_search("q", None, top_k=5)
    assert [e[0] for e in conn_log] == ["fetch"]


async def test_vector_search_embeds_off_the_event_loop(conn_log, monkeypatch):
    seen = {}

    def embed(q):
        seen["thread"] = threading.get_ident()
        return [1.0, 0.0]

    monkeypatch.setattr(R, "embed_query", embed)
    await R.vector_search("q", None, top_k=5)
    assert seen["thread"] != threading.get_ident()


async def test_retrieve_reranks_off_the_event_loop(monkeypatch):
    seen = {}

    async def fake_fts(query, framework, limit):
        return []

    async def fake_vec(query, framework, top_k):
        return [Chunk(framework="NIST", control_id="AC-1", title="t", summary="s",
                      embedding=[1.0, 0.0])]

    def fake_rerank(query, candidates):
        seen["thread"] = threading.get_ident()
        return candidates

    monkeypatch.setattr(R, "fts_search", fake_fts)
    monkeypatch.setattr(R, "vector_search", fake_vec)
    monkeypatch.setattr(R, "rerank", fake_rerank)
    out = await R.retrieve("q", None, top_k=3)
    assert [c.control_id for c in out] == ["AC-1"]
    assert seen["thread"] != threading.get_ident()
