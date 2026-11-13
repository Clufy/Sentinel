from app.grounding import (
    REFUSAL_MESSAGE,
    assemble_citations,
    format_citation,
    is_refusal,
    verify_grounded,
)
from app.models import Chunk


def _c(fw, cid, score=0.9):
    return Chunk(framework=fw, control_id=cid, title="t", summary="s", score=score)


def test_format_citation():
    assert format_citation("NIST", "AC-2") == "[NIST AC-2]"
    assert format_citation("ISO", "A.5.1") == "[ISO A.5.1]"


def test_assemble_citations_bare_and_deduped():
    chunks = [_c("NIST", "AC-2"), _c("ISO", "A.5.1"), _c("NIST", "AC-2")]
    assert assemble_citations(chunks) == ["NIST AC-2", "ISO A.5.1"]


def test_assemble_citations_empty():
    assert assemble_citations([]) == []


def test_is_refusal_empty():
    assert is_refusal([], 0.3) is True


def test_is_refusal_below_threshold():
    assert is_refusal([_c("NIST", "AC-2", score=0.1)], 0.3) is True


def test_is_refusal_above_threshold():
    assert is_refusal([_c("NIST", "AC-2", score=0.9)], 0.3) is False


def test_refusal_message_is_explicit():
    assert isinstance(REFUSAL_MESSAGE, str)
    assert "grounded" in REFUSAL_MESSAGE.lower() or "relevant" in REFUSAL_MESSAGE.lower()


def test_verify_grounded_all_cited_present():
    retrieved = [_c("NIST", "AC-2"), _c("ISO", "A.5.1")]
    assert verify_grounded("Use [NIST AC-2] and [ISO A.5.1].", retrieved) is True


def test_verify_grounded_flags_hallucinated_citation():
    retrieved = [_c("NIST", "AC-2")]
    assert verify_grounded("Use [NIST AC-2] and [CIS 3.1].", retrieved) is False


def test_verify_grounded_no_citations_is_true():
    assert verify_grounded("no citations here", [_c("NIST", "AC-2")]) is True


# --- I1/M7: citation normalisation ---------------------------------------
from app.grounding import normalize_citation, normalize_control_id  # noqa: E402
from app.llm import extract_cited_ids  # noqa: E402


def test_normalize_citation_cis_gets_prefix():
    assert normalize_citation("CIS 3.1") == "CIS CIS-3.1"
    assert normalize_citation("CIS CIS-3.1") == "CIS CIS-3.1"
    assert normalize_citation("CIS cis-3.1") == "CIS CIS-3.1"


def test_normalize_citation_nist_parenthesised_enhancement():
    assert normalize_citation("NIST AC-2(1)") == "NIST AC-2.1"
    assert normalize_citation("NIST ac-2") == "NIST AC-2"
    assert normalize_citation("NIST AC-2.1") == "NIST AC-2.1"


def test_normalize_citation_iso_and_brackets():
    assert normalize_citation("[ISO a.5.1]") == "ISO A.5.1"
    assert normalize_citation("ISO A.5.1") == "ISO A.5.1"


def test_normalize_control_id():
    assert normalize_control_id("NIST", "ac-2(12)") == "AC-2.12"
    assert normalize_control_id("CIS", "5") == "CIS-5"
    assert normalize_control_id("ISO", "A.8.24") == "A.8.24"


def test_extract_cited_ids_accepts_parenthesised_ids():
    assert extract_cited_ids("See [NIST AC-2(1)].") == ["NIST AC-2(1)"]


def test_verify_grounded_normalises_both_sides():
    assert verify_grounded("[CIS 3.1] and [NIST AC-2(1)]",
                           [_c("CIS", "CIS-3.1"), _c("NIST", "AC-2.1")]) is True
