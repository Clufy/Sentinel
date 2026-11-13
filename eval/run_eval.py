from __future__ import annotations

import asyncio

from app.config import get_settings
from app.db import close_pool
from app.llm import FakeProvider
from app.retrieval import retrieve, vector_search
from eval.dataset import EvalItem, load_dataset
from eval.report import gate, render_table
from eval.runner import Scores, aggregate, score_item


async def run(retrieve_fns: dict, provider, dataset: list[EvalItem],
              top_k: int, min_relevance: float) -> dict[str, Scores]:
    cards: dict[str, Scores] = {}
    for name, fn in retrieve_fns.items():
        results = [await score_item(item, fn, provider, top_k, min_relevance) for item in dataset]
        cards[name] = aggregate(results)
    return cards


async def main() -> int:
    settings = get_settings()
    try:
        cards = await run(
            {"hybrid": retrieve, "vector": vector_search},
            FakeProvider(), load_dataset(),
            top_k=settings.retrieval_top_k, min_relevance=settings.min_relevance,
        )
        print(render_table(cards))
        code = gate(cards["hybrid"], cards["vector"])
        if code == 0:
            print("\nGATE PASSED")
        return code
    finally:
        await close_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
