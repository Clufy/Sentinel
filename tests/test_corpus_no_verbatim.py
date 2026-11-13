"""Copyright guardrail: ISO/CIS summaries must be original paraphrase, not verbatim.

Marked `corpus` (select with -m corpus); it runs in CI with the full suite.
This enforces the 'no copyrighted standard text' acceptance criterion.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.corpus

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
ISO_PATH = CORPUS_DIR / "iso27001.yaml"
CIS_PATH = CORPUS_DIR / "cis_v8.yaml"

# A small blocklist of distinctive verbatim phrases from the published standards.
# Our summaries paraphrase intent; none of these exact strings may appear.
BLOCKLIST = [
    "shall be defined, approved by management",
    "information security policy and topic-specific policies shall be defined",
    "roles and responsibilities shall be defined and allocated",
    "controls shall be implemented and operated",
    "the organization shall determine the boundaries and applicability",
    "an inventory of information and other associated assets, including owners",
    "actively manage (inventory, track, and correct) all enterprise assets",
    "actively manage (inventory, track, and correct) all software",
]


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and map 'should' to 'shall' so modal swaps can't evade."""
    return re.sub(r"should", "shall", " ".join(text.lower().split()))


def _summaries(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(e["control_id"], e["summary"]) for e in data["controls"]]


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_no_verbatim_blocklisted_phrases(path: Path) -> None:
    for cid, summary in _summaries(path):
        low = _normalise(summary)
        for phrase in BLOCKLIST:
            assert phrase not in low, (
                f"{path.name} {cid}: summary contains verbatim standard text: {phrase!r}"
            )


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_summaries_are_original_prose_min_words(path: Path) -> None:
    for cid, summary in _summaries(path):
        assert len(summary.split()) >= 8, (
            f"{path.name} {cid}: summary is only {len(summary.split())} words; "
            f"write an original paraphrase of at least 8 words"
        )


# M5: the guard covers test data too ("no copyrighted standard text in repo").
_TEST_SOURCES = [
    *sorted((Path(__file__).parent / "data").glob("*.yaml")),
    *sorted((Path(__file__).parent / "fixtures").glob("*.yaml")),
    Path(__file__).parent / "test_e2e_ingest.py",
]


@pytest.mark.parametrize("path", _TEST_SOURCES, ids=lambda p: p.name)
def test_test_data_has_no_blocklisted_phrases(path):
    text = _normalise(path.read_text(encoding="utf-8"))
    for phrase in BLOCKLIST:
        assert _normalise(phrase) not in text, f"{path.name}: {phrase!r}"
