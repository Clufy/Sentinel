from eval.report import THRESHOLDS, gate, render_table
from eval.runner import Scores


def _s(recall=0.9, citation=0.8, faithful=0.9, refusal=0.9, n=35):
    return Scores(recall=recall, citation_correctness=citation, faithfulness=faithful,
                  refusal_accuracy=refusal, n=n)


def test_thresholds():
    assert THRESHOLDS == {"recall": 0.7, "refusal_accuracy": 0.8, "citation": 0.6}


def test_render_table_has_both_systems_and_headers():
    table = render_table({"hybrid": _s(), "vector": _s(recall=0.5)})
    assert "hybrid" in table and "vector" in table
    assert "recall@8" in table and "refusal_acc" in table


def test_gate_passes_when_all_hold():
    assert gate(_s(recall=0.9), _s(recall=0.6)) == 0


def test_gate_fails_below_recall():
    assert gate(_s(recall=0.5), _s(recall=0.4)) == 1


def test_gate_fails_when_hybrid_not_better_than_vector():
    # hybrid meets absolute thresholds but vector.recall exceeds hybrid.recall
    assert gate(_s(recall=0.75), _s(recall=0.80)) == 1


def test_gate_fails_below_citation_or_refusal():
    assert gate(_s(citation=0.1), _s(recall=0.1)) == 1
    assert gate(_s(refusal=0.1), _s(recall=0.1)) == 1
