"""asyncpg pool, schema DDL, and deterministic lookups."""
from pathlib import Path

import asyncpg

from app.config import get_settings
from app.models import Chunk

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_pool: asyncpg.Pool | None = None


def load_ddl() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=get_settings().database_url, min_size=1, max_size=5
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_db() -> None:
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute(load_ddl())


async def get_by_control_id(
    conn: asyncpg.Connection, framework: str, control_id: str
) -> Chunk | None:
    """Deterministic exact lookup of the chunk_idx=0 row (no semantic search)."""
    row = await conn.fetchrow(
        """
        SELECT id, framework, control_id, chunk_idx, title, summary, source_url
        FROM controls
        WHERE framework = $1 AND control_id = $2 AND chunk_idx = 0
        """,
        framework, control_id,
    )
    if row is None:
        return None
    return Chunk(**{**dict(row), "source_url": row["source_url"] or ""})
