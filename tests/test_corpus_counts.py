"""Count invariants for the governance corpus. Marked `corpus` (select with
-m corpus); they are fast YAML reads and run in CI with the full suite."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.corpus

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
ISO_PATH = CORPUS_DIR / "iso27001.yaml"
CIS_PATH = CORPUS_DIR / "cis_v8.yaml"


def load_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["controls"]


def _iso_clause(control_id: str) -> str:
    return ".".join(control_id.split(".")[:2])  # "A.5.1" -> "A.5"


def test_iso_has_93_controls() -> None:
    controls = load_corpus(ISO_PATH)
    assert len(controls) == 93, f"ISO Annex A 2022 has 93 controls; found {len(controls)}"


def test_iso_covers_all_four_clauses() -> None:
    clauses = {_iso_clause(e["control_id"]) for e in load_corpus(ISO_PATH)}
    for expected in ("A.5", "A.6", "A.7", "A.8"):
        assert expected in clauses, f"ISO clause {expected} has no controls"


def test_iso_clause_family_counts() -> None:
    counts = Counter(_iso_clause(e["control_id"]) for e in load_corpus(ISO_PATH))
    assert counts["A.5"] == 37, f"A.5 Organizational: expected 37, got {counts['A.5']}"
    assert counts["A.6"] == 8, f"A.6 People: expected 8, got {counts['A.6']}"
    assert counts["A.7"] == 14, f"A.7 Physical: expected 14, got {counts['A.7']}"
    assert counts["A.8"] == 34, f"A.8 Technological: expected 34, got {counts['A.8']}"


def test_cis_has_exactly_18_top_level_controls() -> None:
    ids = [e["control_id"] for e in load_corpus(CIS_PATH)]
    top = [cid for cid in ids if re.fullmatch(r"CIS-\d{1,2}", cid)]
    assert len(top) == 18, f"CIS v8 has 18 controls; found {len(top)}: {sorted(top)}"


def test_cis_top_level_ids_are_1_through_18() -> None:
    ids = {e["control_id"] for e in load_corpus(CIS_PATH)}
    for n in range(1, 19):
        assert f"CIS-{n}" in ids, f"missing top-level control CIS-{n}"


# Safeguards per control in CIS Controls v8 (sums to 153). Exact, not a floor:
# a looser `>= 153` check once let 13 invented safeguard ids through.
CIS_SAFEGUARDS_PER_CONTROL = {
    1: 5, 2: 7, 3: 14, 4: 12, 5: 6, 6: 8, 7: 7, 8: 12, 9: 7,
    10: 7, 11: 5, 12: 8, 13: 11, 14: 9, 15: 7, 16: 14, 17: 9, 18: 5,
}


def test_cis_has_exactly_153_safeguards() -> None:
    ids = [e["control_id"] for e in load_corpus(CIS_PATH)]
    safeguards = [cid for cid in ids if re.fullmatch(r"CIS-\d{1,2}\.\d{1,2}", cid)]
    assert sum(CIS_SAFEGUARDS_PER_CONTROL.values()) == 153
    assert len(safeguards) == 153, f"CIS v8 has 153 safeguards; found {len(safeguards)}"


def test_cis_safeguard_ids_match_v8_exactly() -> None:
    ids = {e["control_id"] for e in load_corpus(CIS_PATH)}
    actual = {cid for cid in ids if re.fullmatch(r"CIS-\d{1,2}\.\d{1,2}", cid)}
    expected = {
        f"CIS-{n}.{m}"
        for n, count in CIS_SAFEGUARDS_PER_CONTROL.items()
        for m in range(1, count + 1)
    }
    assert not actual - expected, f"safeguards not in CIS v8: {sorted(actual - expected)}"
    assert not expected - actual, f"missing CIS v8 safeguards: {sorted(expected - actual)}"


def test_cis_safeguards_belong_to_a_known_control() -> None:
    ids = {e["control_id"] for e in load_corpus(CIS_PATH)}
    for cid in ids:
        m = re.fullmatch(r"(CIS-\d{1,2})\.\d{1,2}", cid)
        if m:
            assert m.group(1) in ids, f"safeguard {cid} has no parent control {m.group(1)}"
