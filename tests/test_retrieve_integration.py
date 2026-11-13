# tests/test_retrieve_integration.py
import pytest

from app.db import close_pool, get_pool, init_db
from app.embeddings import embed_texts
from app.grounding import assemble_citations, is_refusal
from app.models import Chunk
from app.ingest.store import upsert_controls
from app.retrieval import retrieve


@pytest.fixture
async def seeded():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    texts = [
        "Manage the lifecycle of information system accounts and access.",
        "Enforce approved authorizations for logical access to resources.",
        "Monitor and control communications at external network boundaries.",
    ]
    embs = embed_texts(texts)
    await upsert_controls([
        Chunk(framework="NIST", control_id="AC-2", title="Account Management",
              summary=texts[0], embedding=embs[0]),
        Chunk(framework="NIST", control_id="AC-3", title="Access Enforcement",
              summary=texts[1], embedding=embs[1]),
        Chunk(framework="NIST", control_id="SC-7", title="Boundary Protection",
              summary=texts[2], embedding=embs[2]),
    ])
    yield
    await close_pool()


@pytest.mark.integration
async def test_retrieve_finds_account_control(seeded):
    chunks = await retrieve("how do I manage user accounts", top_k=3)
    assert chunks, "expected at least one retrieved chunk"
    assert "NIST AC-2" in assemble_citations(chunks)
    assert chunks[0].control_id == "AC-2"
    assert is_refusal(chunks, 0.0) is False
