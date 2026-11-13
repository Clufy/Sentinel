import asyncio
import os
import socket
import pytest

import asyncpg
import pytest_asyncio

import app.db as db_mod
from app.db import close_pool, get_pool, init_db
from app.embeddings import embed_texts
from app.ingest.store import upsert_controls
from app.models import Chunk


def _db_reachable() -> bool:
    url = os.environ.get("DATABASE_URL", "postgresql://sentinel:sentinel@127.0.0.1:5432/sentinel")
    # crude host:port parse first: cheaply rule out "nothing listening at all"
    try:
        hostport = url.split("@", 1)[1].split("/", 1)[0]
        host, _, port = hostport.partition(":")
        with socket.create_connection((host, int(port or "5432")), timeout=0.5):
            pass
    except Exception:
        return False

    # A port can be open (e.g. an unrelated local Postgres/service on 5432)
    # while auth, the target database, or our schema don't match this
    # project's config. Actually connect *and* confirm our schema (the
    # `controls` table) is present, so a bare/unprovisioned Postgres is
    # treated the same as "unreachable" for this project's integration tests.
    async def _try_connect() -> bool:
        try:
            conn = await asyncio.wait_for(asyncpg.connect(dsn=url), timeout=1.5)
        except Exception:
            return False
        try:
            exists = await asyncio.wait_for(
                conn.fetchval("SELECT to_regclass('public.controls')"), timeout=1.5
            )
            return exists is not None
        except Exception:
            return False
        finally:
            try:
                await conn.close()
            except Exception:
                pass

    try:
        return asyncio.run(_try_connect())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _db_reachable():
        return
    skip = pytest.mark.skip(reason="Postgres unreachable; integration tests run in the integration pass")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest_asyncio.fixture
async def seeded_db():
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute("TRUNCATE controls RESTART IDENTITY")
    texts = [
        "Manage the lifecycle of information system accounts and access privileges.",
        "Establish access control rules based on business and security requirements.",
        "Maintain an inventory of all managed accounts across the enterprise.",
    ]
    embs = embed_texts(texts)
    await upsert_controls([
        Chunk(framework="NIST", control_id="AC-2", title="Account Management",
              summary=texts[0], embedding=embs[0]),
        Chunk(framework="ISO", control_id="A.5.15", title="Access control",
              summary=texts[1], embedding=embs[1]),
        Chunk(framework="CIS", control_id="CIS-5", title="Account Management",
              summary=texts[2], embedding=embs[2]),
    ])
    # Close the seeding pool on THIS loop before yielding: the sync TestClient used by the
    # test body runs requests on its own loop and must create its own pool there.
    await close_pool()
    yield
    # Any pool created by a request lives on TestClient's dead loop; abandon it rather than
    # awaiting a cross-loop close (same pattern as tests/test_seeded_smoke.py).
    db_mod._pool = None
