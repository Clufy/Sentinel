"""Full-ingest orchestration: init schema, load, embed, upsert; return counts."""
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.db import init_db
from app.ingest.load_nist import build_chunks as nist_build
from app.ingest.load_nist import fetch_catalog, parse_catalog
from app.ingest.load_yaml import build_chunks as yaml_build
from app.ingest.load_yaml import load_corpus_dir, parse_yaml_file
from app.ingest.store import upsert_controls


async def ingest_all(
    *,
    fixture: str | Path | None = None,
    corpus_dir: str = "corpus",
    include_nist: bool = True,
) -> dict[str, int]:
    await init_db()
    mct = get_settings().max_chunk_tokens
    counts: dict[str, int] = defaultdict(int)

    if fixture is not None:
        parsed = parse_yaml_file(fixture)
        chunks = yaml_build(parsed, mct)
        await upsert_controls(chunks)
        for p in parsed:
            counts[p["framework"]] += 1
        return dict(counts)

    if include_nist:
        catalog = await fetch_catalog()
        parsed = parse_catalog(catalog)
        chunks = nist_build(parsed, mct)
        await upsert_controls(chunks)
        for p in parsed:
            counts[p["framework"]] += 1

    yaml_parsed = load_corpus_dir(corpus_dir)
    if yaml_parsed:
        chunks = yaml_build(yaml_parsed, mct)
        await upsert_controls(chunks)
        for p in yaml_parsed:
            counts[p["framework"]] += 1

    return dict(counts)
