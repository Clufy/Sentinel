import asyncio
import json
from pathlib import Path

import pytest

from scripts.ingest_fixture import load_rows, main

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"


def test_load_rows_covers_three_frameworks():
    rows = load_rows()
    frameworks = {r["framework"] for r in rows}
    assert frameworks == {"NIST", "ISO", "CIS"}
    assert len(rows) >= 15
    for r in rows:
        assert {"framework", "control_id", "title", "summary"} <= set(r.keys())


def test_fixture_covers_every_eval_dataset_control():
    have = {f"{r['framework']} {r['control_id']}" for r in load_rows()}
    expected: set[str] = set()
    for line in _DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected |= set(json.loads(line)["expected_control_ids"])
    assert expected, "dataset has no expected ids"
    assert expected <= have, f"missing from fixture: {sorted(expected - have)}"


def test_fixture_summaries_avoid_shall_phrasing():
    for r in load_rows():
        assert "shall" not in r["summary"].lower(), r["control_id"]


@pytest.mark.integration
def test_ingest_fixture_is_idempotent():
    n1 = asyncio.run(main())
    n2 = asyncio.run(main())
    assert n1 == n2 and n1 >= 15


# --- M9: the eval always runs on the fixture alone ------------------------
def test_main_truncates_controls_before_ingesting(monkeypatch):
    import scripts.ingest_fixture as mod

    log = []

    class Conn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, sql, *args):
            log.append(("execute", " ".join(sql.split())))

    class Pool:
        def acquire(self):
            return Conn()

    async def fake_pool():
        return Pool()

    async def fake_init_db():
        log.append(("init_db",))

    async def fake_upsert(chunks):
        log.append(("upsert", len(chunks)))
        return len(chunks)

    async def fake_close():
        log.append(("close",))

    monkeypatch.setattr(mod, "init_db", fake_init_db)
    monkeypatch.setattr(mod, "get_pool", fake_pool)
    monkeypatch.setattr(mod, "upsert_controls", fake_upsert)
    monkeypatch.setattr(mod, "close_pool", fake_close)
    monkeypatch.setattr(mod, "build_chunks", lambda rows, n: list(rows))

    n = asyncio.run(main())
    assert [e[0] for e in log] == ["init_db", "execute", "upsert", "close"]
    assert log[1][1] == "TRUNCATE controls RESTART IDENTITY"
    assert n == len(load_rows())


@pytest.mark.integration
def test_ingest_fixture_replaces_existing_rows():
    from app.db import close_pool, get_pool

    async def add_stray_row_then_count():
        pool = await get_pool()
        async with pool.acquire() as con:
            await con.execute(
                "INSERT INTO controls (framework, control_id, chunk_idx, title, summary) "
                "VALUES ('NIST', 'ZZ-99', 0, 'stray', 'stray') ON CONFLICT DO NOTHING")
        await close_pool()

    async def count():
        pool = await get_pool()
        async with pool.acquire() as con:
            total = await con.fetchval("SELECT count(*) FROM controls")
        await close_pool()
        return total

    asyncio.run(main())
    asyncio.run(add_stray_row_then_count())
    n = asyncio.run(main())
    assert asyncio.run(count()) == n
