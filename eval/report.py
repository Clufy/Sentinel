from __future__ import annotations

from eval.runner import Scores

THRESHOLDS: dict[str, float] = {"recall": 0.7, "refusal_accuracy": 0.8, "citation": 0.6}


def render_table(cards: dict[str, Scores]) -> str:
    header = (f"{'system':<8} | {'recall@8':>8} | {'citation':>8} | "
              f"{'faithful':>8} | {'refusal_acc':>11} | {'n':>3}")
    lines = [header, "-" * len(header)]
    for name, s in cards.items():
        lines.append(
            f"{name:<8} | {s.recall:>8.3f} | {s.citation_correctness:>8.3f} | "
            f"{s.faithfulness:>8.3f} | {s.refusal_accuracy:>11.3f} | {s.n:>3d}"
        )
    return "\n".join(lines)


def gate(hybrid: Scores, vector: Scores) -> int:
    failures: list[str] = []
    if hybrid.recall < THRESHOLDS["recall"]:
        failures.append(f"recall@8 {hybrid.recall:.3f} < {THRESHOLDS['recall']}")
    if hybrid.refusal_accuracy < THRESHOLDS["refusal_accuracy"]:
        failures.append(f"refusal_accuracy {hybrid.refusal_accuracy:.3f} < {THRESHOLDS['refusal_accuracy']}")
    if hybrid.citation_correctness < THRESHOLDS["citation"]:
        failures.append(f"citation_correctness {hybrid.citation_correctness:.3f} < {THRESHOLDS['citation']}")
    if hybrid.recall < vector.recall:
        failures.append(f"hybrid.recall {hybrid.recall:.3f} < vector.recall {vector.recall:.3f} "
                        "(hybrid must not be worse than vector)")
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0
