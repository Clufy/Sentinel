import pytest

from app.db import close_pool, get_pool, init_db
from app.models import Chunk
from app.ingest.store import upsert_controls
import app.retrieval as R


@pytest.fixture
async def seeded():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    near = [1.0] + [0.0] * 383
    far = [0.0] * 383 + [1.0]
    await upsert_controls([
        Chunk(framework="NIST", control_id="AC-1", title="Access Control Policy",
              summary="password authentication policy access", embedding=near),
        Chunk(framework="NIST", control_id="AC-2", title="Account Management",
              summary="user account provisioning lifecycle", embedding=[0.9, 0.1] + [0.0] * 382),
        Chunk(framework="NIST", control_id="SI-4", title="System Monitoring",
              summary="intrusion detection network monitoring", embedding=far),
    ])
    yield {"near": near}
    await close_pool()


@pytest.mark.integration
async def test_fts_search_ranks_and_keys(seeded):
    keys = await R.fts_search("password authentication", None, limit=10)
    assert keys[0] == "NIST:AC-1:0"


@pytest.mark.integration
async def test_fts_search_framework_filter(seeded):
    assert await R.fts_search("monitoring", "ISO", limit=10) == []


@pytest.mark.integration
async def test_vector_search_returns_chunks_with_embedding(seeded, monkeypatch):
    monkeypatch.setattr(R, "embed_query", lambda q: seeded["near"])
    chunks = await R.vector_search("accounts", None, top_k=3)
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].control_id == "AC-1"
    assert chunks[0].embedding is not None and len(chunks[0].embedding) == 384
    assert chunks[0].score >= chunks[1].score


@pytest.mark.integration
async def test_chunks_by_keys(seeded):
    pool = await get_pool()
    async with pool.acquire() as con:
        got = await R._chunks_by_keys(con, ["NIST:AC-2:0", "NIST:SI-4:0"])
    assert set(got.keys()) == {"NIST:AC-2:0", "NIST:SI-4:0"}
    assert got["NIST:AC-2:0"].title == "Account Management"


@pytest.mark.integration
async def test_vector_search_framework_filter_runs_iterative_scan(seeded, monkeypatch):
    # M1 live check: SET LOCAL hnsw.iterative_scan needs pgvector >= 0.8.
    monkeypatch.setattr(R, "embed_query", lambda q: seeded["near"])
    nist = await R.vector_search("accounts", "NIST", top_k=3)
    assert [c.control_id for c in nist] == ["AC-1", "AC-2", "SI-4"]
    assert await R.vector_search("accounts", "ISO", top_k=3) == []
