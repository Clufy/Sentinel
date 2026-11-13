from __future__ import annotations


def norm_id(framework: str, control_id: str) -> str:
    return f"{framework} {control_id}"


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    if not expected_ids:
        return 1.0
    top = set(retrieved_ids[:k])
    return sum(1 for e in expected_ids if e in top) / len(expected_ids)


def citation_correctness(cited_ids: list[str], expected_ids: list[str]) -> float:
    if not cited_ids:
        return 1.0 if not expected_ids else 0.0
    expected = set(expected_ids)
    return sum(1 for c in cited_ids if c in expected) / len(cited_ids)


def faithfulness(cited_ids: list[str], retrieved_ids: list[str]) -> float:
    if not cited_ids:
        return 1.0
    retrieved = set(retrieved_ids)
    return sum(1 for c in cited_ids if c in retrieved) / len(cited_ids)


def refusal_correct(answerable: bool, refused: bool) -> bool:
    return refused == (not answerable)
