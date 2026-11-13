import pytest

from app.retrieval import fuse_rrf


def test_single_ranking_scores_by_rank():
    out = fuse_rrf([["a", "b"]], k=60)
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62)
    assert out["a"] > out["b"]


def test_key_in_two_rankings_sums_contributions():
    out = fuse_rrf([["a", "b"], ["b", "a"]], k=60)
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(out["b"])


def test_custom_k_changes_denominator():
    assert fuse_rrf([["x"]], k=10)["x"] == pytest.approx(1 / 11)


def test_empty_rankings_returns_empty_dict():
    assert fuse_rrf([], k=60) == {}
    assert fuse_rrf([[], []], k=60) == {}


def test_disjoint_keys_all_present():
    out = fuse_rrf([["a"], ["b"]], k=60)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 61)


def test_dead_dedupe_helper_removed():
    # M11: fts_search keys come from UNIQUE (framework, control_id, chunk_idx) rows.
    import app.retrieval as R

    assert not hasattr(R, "_dedupe")
