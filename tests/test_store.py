import pytest

from app.db import close_pool, get_pool, init_db
from app.models import Chunk
from app.ingest.store import upsert_controls


@pytest.fixture
async def db():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    yield pool
    await close_pool()


def _chunk(title, summary):
    return Chunk(framework="NIST", control_id="AC-2", chunk_idx=0, title=title,
                 summary=summary, source_url="http://x", embedding=[0.1] * 384)


@pytest.mark.integration
async def test_upsert_inserts_then_updates_idempotent(db):
    n = await upsert_controls([_chunk("Account Management", "Manage accounts.")])
    assert n == 1
    async with db.acquire() as con:
        assert await con.fetchval("SELECT count(*) FROM controls") == 1

    await upsert_controls([_chunk("Account Mgmt v2", "Manage accounts better.")])
    async with db.acquire() as con:
        assert await con.fetchval("SELECT count(*) FROM controls") == 1
        title = await con.fetchval("SELECT title FROM controls WHERE control_id='AC-2'")
        assert title == "Account Mgmt v2"
