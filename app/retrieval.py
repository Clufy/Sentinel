"""Hybrid retrieval: FTS + vector -> RRF -> cross-encoder rerank -> MMR -> top_k.

Row keys are strings of the form f"{framework}:{control_id}:{chunk_idx}".
"""
from __future__ import annotations

import asyncio
import math
from functools import lru_cache

from app.db import get_pool
from app.embeddings import embed_query
from app.models import Chunk


def fuse_rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over best-first ranked key lists.

    A key at 1-indexed position ``rank`` contributes ``1 / (k + rank)``;
    contributions across rankings are summed.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for zero_idx, key in enumerate(ranking):
            rank = zero_idx + 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 if either vector has zero norm."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr(
    candidates: list[Chunk],
    lambda_: float = 0.7,
    k: int = 8,
) -> list[Chunk]:
    """Greedy MMR: relevance is the candidate's rerank ``.score`` (None -> 0.0);
    redundancy is the max cosine to already-selected ``.embedding``s (no re-embed)."""
    n = len(candidates)
    if n == 0:
        return []
    limit = min(k, n)
    vecs = [c.embedding or [] for c in candidates]
    rel = [c.score or 0.0 for c in candidates]

    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < limit:
        best_i, best_score = None, -math.inf
        for i in remaining:
            redundancy = max((_cosine(vecs[i], vecs[j]) for j in selected), default=0.0)
            score = lambda_ * rel[i] - (1.0 - lambda_) * redundancy
            if score >= best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return [candidates[i] for i in selected]


@lru_cache(maxsize=1)
def _cross_encoder():
    """Process-wide CrossEncoder singleton (lazy import)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[Chunk]) -> list[Chunk]:
    """Score candidates with the cross-encoder; set .score; return sorted best-first."""
    if not candidates:
        return []
    model = _cross_encoder()
    # Label AFTER the text: MS-MARCO reads a leading "NIST AC-2" as noise and scores
    # relevant passages far lower; as a suffix it still helps framework-named questions.
    pairs = [(query, f"{c.title}. {c.summary} ({c.framework} {c.control_id})")
             for c in candidates]
    scores = model.predict(pairs)
    for chunk, score in zip(candidates, scores):
        # Sigmoid maps the raw cross-encoder logit into (0, 1), matching the
        # cosine-similarity scale that grounding.is_refusal's min_relevance
        # threshold is calibrated for. Monotonic, so ranking order is unchanged.
        chunk.score = 1.0 / (1.0 + math.exp(-float(score)))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def _key(framework: str, control_id: str, chunk_idx: int) -> str:
    return f"{framework}:{control_id}:{chunk_idx}"


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


async def fts_search(query: str, framework: str | None, limit: int) -> list[str]:
    """Return row keys best-first by ts_rank_cd over the fts column."""
    where = "AND framework = $3" if framework is not None else ""
    # plainto_tsquery ANDs every lexeme, so a natural-language question matches almost
    # nothing. Reuse its stemming/stopword handling but OR the lexemes instead.
    tsq = "replace(plainto_tsquery('english', $1)::text, ' & ', ' | ')::tsquery"
    sql = f"""
        SELECT framework, control_id, chunk_idx,
               ts_rank_cd(fts, {tsq}) AS rank
        FROM controls
        WHERE fts @@ {tsq} {where}
        ORDER BY rank DESC
        LIMIT $2
    """
    params = [query, limit] + ([framework] if framework is not None else [])
    pool = await get_pool()
    async with pool.acquire() as con:
        rows = await con.fetch(sql, *params)
    # Rows are UNIQUE on (framework, control_id, chunk_idx), so keys are distinct.
    return [_key(r["framework"], r["control_id"], r["chunk_idx"]) for r in rows]


async def vector_search(query: str, framework: str | None, top_k: int) -> list[Chunk]:
    """Embed the query and cosine-rank; return Chunks with .embedding + similarity .score.

    This single function is the eval vector-only path AND a step of retrieve().
    With a framework filter, HNSW filters AFTER its index scan, so a selective filter
    could return far fewer rows than LIMIT; pgvector >= 0.8's iterative scan keeps
    scanning (in exact distance order) until LIMIT filtered rows are found.
    """
    # CPU-bound embedding runs in a worker thread so the event loop stays free.
    qvec = _vec_literal(await asyncio.to_thread(embed_query, query))
    where = "WHERE framework = $3" if framework is not None else ""
    sql = f"""
        SELECT id, framework, control_id, chunk_idx, title, summary, source_url,
               embedding, 1 - (embedding <=> $1) AS score
        FROM controls
        {where}
        ORDER BY embedding <=> $1
        LIMIT $2
    """
    params = [qvec, top_k] + ([framework] if framework is not None else [])
    pool = await get_pool()
    async with pool.acquire() as con:
        if framework is None:
            rows = await con.fetch(sql, *params)
        else:
            # SET LOCAL only lasts for (and only works inside) a transaction.
            async with con.transaction():
                await con.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                rows = await con.fetch(sql, *params)
    return [_row_to_chunk(r, score=float(r["score"])) for r in rows]


async def _chunks_by_keys(conn, keys: list[str]) -> dict[str, Chunk]:
    """Load full rows (incl. embedding) for the given keys, keyed by key string."""
    if not keys:
        return {}
    triples = [k.split(":", 2) for k in keys]
    frameworks = [t[0] for t in triples]
    control_ids = [t[1] for t in triples]
    chunk_idxs = [int(t[2]) for t in triples]
    rows = await conn.fetch(
        """
        SELECT id, framework, control_id, chunk_idx, title, summary, source_url, embedding
        FROM controls
        WHERE (framework, control_id, chunk_idx) IN (
            SELECT * FROM unnest($1::text[], $2::text[], $3::int[])
        )
        """,
        frameworks, control_ids, chunk_idxs,
    )
    return {_key(r["framework"], r["control_id"], r["chunk_idx"]): _row_to_chunk(r) for r in rows}


def _row_to_chunk(r, score: float = 0.0) -> Chunk:
    emb = r["embedding"]
    if emb is not None and not isinstance(emb, list):
        # pgvector returns a string like "[0.1,0.2]"; parse to list[float].
        emb = [float(x) for x in str(emb).strip("[]").split(",") if x]
    return Chunk(
        id=r["id"],
        framework=r["framework"],
        control_id=r["control_id"],
        chunk_idx=r["chunk_idx"],
        title=r["title"],
        summary=r["summary"],
        source_url=r["source_url"] or "",
        score=score,
        embedding=emb,
    )


_CANDIDATE_LIMIT = 20


async def retrieve(query: str, framework: str | None = None, top_k: int = 8) -> list[Chunk]:
    """Hybrid pipeline: FTS + vector -> RRF -> rerank -> MMR -> top_k.

    Reuses vector_search Chunks (they already carry .embedding); fetches any
    FTS-only keys from the DB. MMR reuses .embedding (no re-embedding).
    """
    fts_keys = await fts_search(query, framework, _CANDIDATE_LIMIT)
    vec_chunks = await vector_search(query, framework, _CANDIDATE_LIMIT)
    vec_by_key = {_key(c.framework, c.control_id, c.chunk_idx): c for c in vec_chunks}

    fused = fuse_rrf([fts_keys, list(vec_by_key.keys())])
    if not fused:
        return []
    fused_keys = sorted(fused, key=lambda kkey: fused[kkey], reverse=True)

    missing = [k for k in fused_keys if k not in vec_by_key]
    fetched: dict[str, Chunk] = {}
    if missing:
        pool = await get_pool()
        async with pool.acquire() as conn:
            fetched = await _chunks_by_keys(conn, missing)

    candidates: list[Chunk] = []
    for k in fused_keys:
        chunk = vec_by_key.get(k) or fetched.get(k)
        if chunk is not None:
            candidates.append(chunk)
    if not candidates:
        return []

    # Cross-encoder inference is CPU-bound; keep it off the event loop.
    reranked = await asyncio.to_thread(rerank, query, candidates)
    return mmr(reranked, lambda_=0.7, k=top_k)
