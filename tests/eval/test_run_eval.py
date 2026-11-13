import asyncio

from app.llm import FakeProvider
from app.models import Chunk
from eval.dataset import EvalItem
from eval.report import gate
from eval.run_eval import main, run


def _perfect(item_map):
    async def fn(query, framework, top_k):
        return [Chunk(framework=f, control_id=c, title="t", summary="s", score=0.9)
                for f, c in item_map.get(query, [])][:top_k]
    return fn


def _empty():
    async def fn(query, framework, top_k):
        return []
    return fn


def _dataset():
    return [
        EvalItem(question="q-hit", expected_control_ids=["NIST AC-2"], gold_answer_sketch="x", answerable=True),
        EvalItem(question="q-refuse", expected_control_ids=[], gold_answer_sketch="", answerable=False),
    ]


def test_run_returns_card_per_system():
    fns = {"hybrid": _perfect({"q-hit": [("NIST", "AC-2")]}), "vector": _empty()}
    cards = asyncio.run(run(fns, FakeProvider(), _dataset(), top_k=8, min_relevance=0.3))
    assert set(cards.keys()) == {"hybrid", "vector"}


def test_hybrid_perfect_passes_gate_over_vector():
    hybrid = _perfect({"q-hit": [("NIST", "AC-2")]})
    cards = asyncio.run(run({"hybrid": hybrid, "vector": _empty()},
                            FakeProvider(), _dataset(), top_k=8, min_relevance=0.3))
    assert gate(cards["hybrid"], cards["vector"]) == 0


def test_empty_retriever_scores_refusals():
    cards = asyncio.run(run({"vector": _empty()}, FakeProvider(), _dataset(), top_k=8, min_relevance=0.3))
    # 1 correct refusal + 1 wrong refusal => 0.5
    assert cards["vector"].refusal_accuracy == 0.5


def test_main_closes_pool_and_returns_gate_code(monkeypatch):
    import eval.run_eval as mod

    closed = []

    async def fake_close():
        closed.append(True)

    monkeypatch.setattr(mod, "load_dataset", _dataset)
    monkeypatch.setattr(mod, "close_pool", fake_close)
    monkeypatch.setattr(mod, "retrieve", _perfect({"q-hit": [("NIST", "AC-2")]}))
    monkeypatch.setattr(mod, "vector_search", _empty())
    assert asyncio.run(main()) == 0
    assert closed == [True]


def test_main_closes_pool_on_error(monkeypatch):
    import eval.run_eval as mod

    closed = []

    async def fake_close():
        closed.append(True)

    async def boom(q, f, k):
        raise RuntimeError("db down")

    monkeypatch.setattr(mod, "load_dataset", _dataset)
    monkeypatch.setattr(mod, "close_pool", fake_close)
    monkeypatch.setattr(mod, "retrieve", boom)
    monkeypatch.setattr(mod, "vector_search", boom)
    try:
        asyncio.run(main())
    except RuntimeError:
        pass
    assert closed == [True]
