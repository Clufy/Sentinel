"""Seed the DB from tests/fixtures/mini_*.yaml. Run: python -m scripts.ingest_fixture.

Replaces whatever is in `controls` (TRUNCATE first), so the eval always runs on
the fixture corpus even after a full /ingest on the same database.
"""
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db import close_pool, get_pool, init_db
from app.ingest.load_yaml import build_chunks, parse_yaml_file
from app.ingest.store import upsert_controls

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def load_rows(path: Path | None = None) -> list[dict]:
    """Rows from one fixture file, or from every mini_*.yaml (one framework each)."""
    if path is not None:
        return parse_yaml_file(path)
    rows: list[dict] = []
    for p in sorted(_FIXTURE_DIR.glob("mini_*.yaml")):
        rows.extend(parse_yaml_file(p))
    return rows


async def main() -> int:
    try:
        await init_db()
        chunks = build_chunks(load_rows(), get_settings().max_chunk_tokens)
        pool = await get_pool()
        async with pool.acquire() as con:
            await con.execute("TRUNCATE controls RESTART IDENTITY")
        n = await upsert_controls(chunks)
    finally:
        await close_pool()
    print(f"ingested {n} fixture rows")
    return n


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
