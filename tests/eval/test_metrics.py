from eval.metrics import (
    citation_correctness,
    faithfulness,
    norm_id,
    recall_at_k,
    refusal_correct,
)


def test_norm_id():
    assert norm_id("NIST", "AC-2") == "NIST AC-2"


def test_recall_full_and_partial():
    assert recall_at_k(["NIST AC-2", "ISO A.5.1"], ["NIST AC-2", "ISO A.5.1"], 8) == 1.0
    assert recall_at_k(["NIST AC-2", "Z 9"], ["NIST AC-2", "ISO A.5.1"], 8) == 0.5


def test_recall_respects_cutoff():
    assert recall_at_k(["X 1", "Y 2", "NIST AC-2"], ["NIST AC-2"], 2) == 0.0
    assert recall_at_k(["X 1", "Y 2", "NIST AC-2"], ["NIST AC-2"], 3) == 1.0


def test_recall_empty_expected_is_one():
    assert recall_at_k(["A 1"], [], 8) == 1.0


def test_citation_correctness():
    assert citation_correctness(["NIST AC-2", "BOGUS 9"], ["NIST AC-2"]) == 0.5
    assert citation_correctness([], []) == 1.0
    assert citation_correctness([], ["NIST AC-2"]) == 0.0


def test_faithfulness():
    assert faithfulness(["NIST AC-2"], ["NIST AC-2", "ISO A.5.1"]) == 1.0
    assert faithfulness(["NIST AC-2", "FAKE 1"], ["NIST AC-2"]) == 0.5
    assert faithfulness([], ["NIST AC-2"]) == 1.0


def test_refusal_correct_truth_table():
    assert refusal_correct(True, False) is True
    assert refusal_correct(True, True) is False
    assert refusal_correct(False, True) is True
    assert refusal_correct(False, False) is False
