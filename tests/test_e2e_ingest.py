"""M1 end-to-end smoke: ingest a small fixture corpus into real Postgres and
read it back, proving load_yaml -> embed -> upsert -> read works together.

Requires a real pgvector-backed Postgres (see docker-compose.yml). Locally,
tests/conftest.py detects that Postgres is unreachable/unprovisioned and
auto-skips anything marked `integration` — that is the expected state until
the integration pass runs against `docker compose up -d db`.
"""
import pytest

from app.db import close_pool, get_by_control_id, get_pool, init_db
from app.ingest.run import ingest_all

_FIXTURE_YAML = """\
framework: ISO
controls:
  - control_id: a.5.1
    title: Policies for information security
    summary: Write down the organisation's security policy and the narrower topic policies that support it.
    source_url: http://iso.example/a5-1
  - control_id: a.5.2
    title: Information security roles and responsibilities
    summary: Decide who owns each security duty and assign those duties to named people.
    source_url: http://iso.example/a5-2
"""


@pytest.fixture
async def db():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    yield pool
    await close_pool()


@pytest.mark.integration
async def test_ingest_all_then_lookup_by_control_id(db, tmp_path):
    fixture = tmp_path / "mini_corpus.yaml"
    fixture.write_text(_FIXTURE_YAML, encoding="utf-8")

    counts = await ingest_all(fixture=fixture, include_nist=False)
    assert counts == {"ISO": 2}

    async with db.acquire() as con:
        total = await con.fetchval("SELECT count(*) FROM controls")
        found = await get_by_control_id(con, "ISO", "A.5.1")

    assert total == 2
    assert found is not None
    assert found.title == "Policies for information security"
    assert found.id is not None

    async with db.acquire() as con:
        row = await con.fetchrow(
            "SELECT embedding FROM controls WHERE framework=$1 AND control_id=$2 AND chunk_idx=0",
            "ISO", "A.5.1",
        )
    assert row is not None
    assert row["embedding"] is not None


@pytest.mark.integration
async def test_ingest_all_is_idempotent_on_reingest(db, tmp_path):
    fixture = tmp_path / "mini_corpus.yaml"
    fixture.write_text(_FIXTURE_YAML, encoding="utf-8")

    await ingest_all(fixture=fixture, include_nist=False)
    await ingest_all(fixture=fixture, include_nist=False)

    async with db.acquire() as con:
        total = await con.fetchval("SELECT count(*) FROM controls")

    assert total == 2
