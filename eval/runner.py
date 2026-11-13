from __future__ import annotations

from typing import Awaitable, Callable

from pydantic import BaseModel

from app.agent import _chunk_to_dict, run_agent
from app.models import Chunk
from eval.dataset import EvalItem
from eval.metrics import (
    citation_correctness,
    faithfulness,
    norm_id,
    recall_at_k,
    refusal_correct,
)

RetrieveFn = Callable[[str, str | None, int], Awaitable[list[Chunk]]]


class ItemResult(BaseModel):
    recall: float
    citation: float
    faithful: float
    refusal_ok: bool
    refused: bool
    answerable: bool


class Scores(BaseModel):
    recall: float
    citation_correctness: float
    faithfulness: float
    refusal_accuracy: float
    n: int


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


async def score_item(item: EvalItem, retrieve_fn: RetrieveFn, provider,
                     top_k: int, min_relevance: float) -> ItemResult:
    first_call: list[Chunk] | None = None

    async def adapter(query: str, framework: str | None = None) -> dict:
        nonlocal first_call
        chunks = await retrieve_fn(query, framework, top_k)
        if first_call is None:
            first_call = chunks
        return {"controls": [_chunk_to_dict(c) for c in chunks if c.score >= min_relevance]}

    result = await run_agent(item.question, provider, tool_functions={"search_controls": adapter})

    if first_call is None:
        first_call = await retrieve_fn(item.question, None, top_k)
    retrieved_ids = _dedupe([norm_id(c.framework, c.control_id) for c in first_call])
    cited_ids = [] if result.refused else _dedupe(list(result.citations))
    expected_ids = _dedupe(list(item.expected_control_ids))

    return ItemResult(
        recall=recall_at_k(retrieved_ids, expected_ids, top_k),
        citation=citation_correctness(cited_ids, expected_ids),
        faithful=faithfulness(cited_ids, retrieved_ids),
        refusal_ok=refusal_correct(item.answerable, result.refused),
        refused=result.refused,
        answerable=item.answerable,
    )


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 1.0


def aggregate(results: list[ItemResult]) -> Scores:
    n = len(results)
    answered = [r for r in results if not r.refused]
    return Scores(
        recall=_mean([r.recall for r in results if r.answerable]),
        citation_correctness=_mean([r.citation for r in answered]),
        faithfulness=_mean([r.faithful for r in answered]),
        refusal_accuracy=sum(1 for r in results if r.refusal_ok) / n if n else 1.0,
        n=n,
    )
