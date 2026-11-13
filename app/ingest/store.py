"""Idempotent upsert of Chunk rows into the controls table."""
from app.db import get_pool
from app.models import Chunk
from app.retrieval import _vec_literal

_UPSERT = """
INSERT INTO controls
    (framework, control_id, chunk_idx, title, summary, source_url, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (framework, control_id, chunk_idx) DO UPDATE SET
    title = EXCLUDED.title,
    summary = EXCLUDED.summary,
    source_url = EXCLUDED.source_url,
    embedding = EXCLUDED.embedding
"""


async def upsert_controls(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    pool = await get_pool()
    payload = [
        (c.framework, c.control_id, c.chunk_idx, c.title, c.summary,
         c.source_url, _vec_literal(c.embedding or []))
        for c in chunks
    ]
    async with pool.acquire() as con:
        await con.executemany(_UPSERT, payload)
    return len(chunks)
