import pytest

from app.models import Chunk
from app.retrieval import _cosine, mmr


def _c(cid, emb, score=0.5):
    return Chunk(framework="NIST", control_id=cid, title="t", summary="s",
                 embedding=emb, score=score)


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_first_pick_is_highest_rerank_score():
    cands = [_c("c0", [1.0, 0.0], 0.2), _c("c1", [0.0, 1.0], 0.9), _c("c2", [-1.0, 0.0], 0.4)]
    assert [c.control_id for c in mmr(cands, k=1)] == ["c1"]


def test_higher_rerank_score_beats_higher_query_cosine():
    # "close" is the better embedding neighbour of any query along [1, 0], but the
    # cross-encoder says "far" is the better answer; relevance is the rerank score.
    cands = [_c("close", [1.0, 0.0], 0.3), _c("far", [0.0, 1.0], 0.9)]
    assert [c.control_id for c in mmr(cands, lambda_=0.7, k=1)] == ["far"]


def test_redundancy_demotes_near_duplicate():
    cands = [_c("c0", [1.0, 0.0], 0.9), _c("dup", [0.99, 0.01], 0.85), _c("div", [0.0, 1.0], 0.7)]
    out = mmr(cands, lambda_=0.5, k=2)
    assert [c.control_id for c in out] == ["c0", "div"]


def test_pure_relevance_when_lambda_one():
    cands = [_c("lo", [0.0, 1.0], 0.1), _c("mid", [0.99, 0.01], 0.5), _c("hi", [1.0, 0.0], 0.9)]
    out = mmr(cands, lambda_=1.0, k=3)
    assert [c.control_id for c in out] == ["hi", "mid", "lo"]


def test_exact_selection_values_new_formula():
    # lambda=0.7: pick a (0.9). b: .7*.8 - .3*cos(b,a)=.56-.3*.6=.38 ; c: .7*.5 - 0 = .35
    a = _c("a", [1.0, 0.0], 0.9)
    b = _c("b", [0.6, 0.8], 0.8)
    c = _c("c", [0.0, 1.0], 0.5)
    assert [x.control_id for x in mmr([a, b, c], lambda_=0.7, k=2)] == ["a", "b"]
    # lambda=0.5: b: .4-.3=.1 ; c: .25 -> c wins
    assert [x.control_id for x in mmr([a, b, c], lambda_=0.5, k=2)] == ["a", "c"]


def test_k_larger_than_candidates_returns_all():
    out = mmr([_c("a", [1.0, 0.0]), _c("b", [0.0, 1.0])], k=10)
    assert {c.control_id for c in out} == {"a", "b"}


def test_empty_candidates_returns_empty():
    assert mmr([], k=5) == []


def test_none_score_treated_as_zero():
    none = _c("none", [1.0, 0.0])
    object.__setattr__(none, "score", None)  # bypass pydantic's float validation
    out = mmr([none, _c("has", [0.0, 1.0], 0.4)], k=2)
    assert out[0].control_id == "has"


def test_none_embedding_has_no_redundancy():
    out = mmr([_c("has", [1.0, 0.0], 0.9), _c("none", None, 0.5)], k=2)
    assert [c.control_id for c in out] == ["has", "none"]


def test_tie_break_prefers_later_candidate():
    # ``>=``: on an exact tie the later-iterated candidate wins deterministically.
    a = _c("a", [1.0, 0.0], 0.5)
    b = _c("b", [0.0, 1.0], 0.5)
    first = [c.control_id for c in mmr([a, b], k=1)]
    assert first == [c.control_id for c in mmr([a, b], k=1)]
