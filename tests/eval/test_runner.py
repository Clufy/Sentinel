import asyncio

from app.llm import FakeProvider
from app.models import Chunk
from eval.dataset import EvalItem
from eval.runner import ItemResult, Scores, aggregate, score_item


def _retriever(chunks):
    async def fn(query, framework, top_k):
        return chunks[:top_k]
    return fn


def _c(framework, cid, score=0.9):
    return Chunk(framework=framework, control_id=cid, title="t", summary="s", score=score)


def test_answerable_hit_is_high():
    item = EvalItem(question="account management", expected_control_ids=["NIST AC-2"],
                    gold_answer_sketch="x", answerable=True)
    res = asyncio.run(score_item(item, _retriever([_c("NIST", "AC-2")]),
                                 FakeProvider(), top_k=8, min_relevance=0.3))
    assert res.recall == 1.0
    assert res.citation == 1.0
    assert res.faithful == 1.0
    assert res.refused is False and res.refusal_ok is True and res.answerable is True


def test_low_relevance_refuses():
    item = EvalItem(question="nonsense", expected_control_ids=[], gold_answer_sketch="", answerable=False)
    res = asyncio.run(score_item(item, _retriever([_c("NIST", "AC-2", score=0.05)]),
                                 FakeProvider(), top_k=8, min_relevance=0.3))
    assert res.refused is True and res.refusal_ok is True


def test_answerable_but_refused_is_wrong():
    item = EvalItem(question="answerable", expected_control_ids=["NIST AC-2"],
                    gold_answer_sketch="x", answerable=True)
    res = asyncio.run(score_item(item, _retriever([_c("NIST", "AC-2", score=0.1)]),
                                 FakeProvider(), top_k=8, min_relevance=0.3))
    assert res.refused is True and res.refusal_ok is False
    assert res.recall == 1.0  # raw retrieval still contained the expected control


def test_empty_retrieval_refuses():
    item = EvalItem(question="q", expected_control_ids=[], gold_answer_sketch="", answerable=False)
    res = asyncio.run(score_item(item, _retriever([]), FakeProvider(), top_k=8, min_relevance=0.3))
    assert res.refused is True and res.refusal_ok is True


def test_duplicate_expected_ids_not_double_weighted():
    item = EvalItem(question="q", expected_control_ids=["NIST AC-2", "NIST AC-2", "NIST AC-3"],
                    gold_answer_sketch="x", answerable=True)
    res = asyncio.run(score_item(item, _retriever([_c("NIST", "AC-2")]),
                                 FakeProvider(), top_k=8, min_relevance=0.3))
    assert res.recall == 0.5


def _r(recall, citation, faithful, refusal_ok, refused, answerable=True):
    return ItemResult(recall=recall, citation=citation, faithful=faithful,
                      refusal_ok=refusal_ok, refused=refused, answerable=answerable)


def test_aggregate_means():
    results = [
        _r(1.0, 1.0, 1.0, True, False),
        _r(0.0, 0.0, 0.0, False, False),
        _r(1.0, 1.0, 1.0, True, True, answerable=False),
    ]
    s = aggregate(results)
    assert isinstance(s, Scores)
    assert s.n == 3
    assert s.refusal_accuracy == 2 / 3
    assert s.recall == 0.5  # answerable items only
    assert s.citation_correctness == 0.5  # answered items only
    assert s.faithfulness == 0.5


def test_aggregate_all_refused_defaults_one():
    s = aggregate([_r(1.0, 1.0, 1.0, True, True, answerable=False)])
    assert s.recall == 1.0 and s.citation_correctness == 1.0 and s.faithfulness == 1.0
    assert s.refusal_accuracy == 1.0


def test_answerable_but_refused_pulls_recall_down():
    s = aggregate([
        _r(1.0, 1.0, 1.0, True, False),
        _r(0.0, 1.0, 1.0, False, True),  # answerable, refused, nothing recalled
    ])
    assert s.recall == 0.5
    assert s.citation_correctness == 1.0
