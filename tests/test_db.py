import pytest

from app.db import close_pool, get_by_control_id, get_pool, init_db


class _FakeConnection:
    """Fake asyncpg connection for exercising get_by_control_id's row mapping
    without a real database."""

    def __init__(self, row):
        self._row = row

    async def fetchrow(self, query, framework, control_id):
        if self._row is None:
            return None
        if (
            self._row["framework"] == framework
            and self._row["control_id"] == control_id
        ):
            return self._row
        return None


def test_get_by_control_id_maps_row_to_chunk_with_fake_connection():
    row = {
        "id": 7,
        "framework": "NIST",
        "control_id": "AC-2",
        "chunk_idx": 0,
        "title": "Account Management",
        "summary": "Manage accounts.",
        "source_url": "http://x",
    }
    conn = _FakeConnection(row)

    import asyncio

    chunk = asyncio.run(get_by_control_id(conn, "NIST", "AC-2"))
    assert chunk is not None
    assert chunk.id == 7
    assert chunk.framework == "NIST"
    assert chunk.control_id == "AC-2"
    assert chunk.chunk_idx == 0
    assert chunk.title == "Account Management"
    assert chunk.summary == "Manage accounts."
    assert chunk.source_url == "http://x"

    missing = asyncio.run(get_by_control_id(conn, "NIST", "ZZ-99"))
    assert missing is None


def test_get_by_control_id_defaults_source_url_when_none():
    row = {
        "id": 8,
        "framework": "ISO",
        "control_id": "A.5.1",
        "chunk_idx": 0,
        "title": "Policies",
        "summary": "Policy set.",
        "source_url": None,
    }
    conn = _FakeConnection(row)

    import asyncio

    chunk = asyncio.run(get_by_control_id(conn, "ISO", "A.5.1"))
    assert chunk is not None
    assert chunk.source_url == ""


@pytest.fixture
async def db():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    yield pool
    await close_pool()


@pytest.mark.integration
async def test_table_exists_and_insert_roundtrip(db):
    async with db.acquire() as con:
        exists = await con.fetchval(
            "SELECT to_regclass('public.controls') IS NOT NULL"
        )
        assert exists is True
        await con.execute(
            "INSERT INTO controls "
            "(framework, control_id, chunk_idx, title, summary, source_url, embedding) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            "NIST", "AC-2", 0, "Account Management", "Manage accounts.",
            "http://x", "[" + ",".join(["0.1"] * 384) + "]",
        )
        row = await con.fetchrow(
            "SELECT title, fts FROM controls WHERE control_id='AC-2'"
        )
        assert row["title"] == "Account Management"
        assert row["fts"] is not None


@pytest.mark.integration
async def test_get_by_control_id_returns_chunk(db):
    async with db.acquire() as con:
        await con.execute(
            "INSERT INTO controls "
            "(framework, control_id, chunk_idx, title, summary, source_url, embedding) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            "ISO", "A.5.1", 0, "Policies", "Policy set.", "http://x",
            "[" + ",".join(["0.2"] * 384) + "]",
        )
        chunk = await get_by_control_id(con, "ISO", "A.5.1")
        assert chunk is not None
        assert chunk.framework == "ISO"
        assert chunk.control_id == "A.5.1"
        assert chunk.title == "Policies"
        assert chunk.id is not None
        missing = await get_by_control_id(con, "ISO", "ZZ-99")
        assert missing is None
