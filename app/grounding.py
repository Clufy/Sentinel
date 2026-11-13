"""Citation assembly, refusal decision, and faithfulness verification."""
from __future__ import annotations

import re

from app.models import Chunk

REFUSAL_MESSAGE = (
    "I could not find sufficiently relevant, grounded information in the "
    "governance corpus to answer this question confidently. Rather than guess, "
    "I am declining to answer. Try rephrasing or narrowing to a specific "
    "framework or control."
)

# Accepts parenthesised enhancement ids such as [NIST AC-2(1)]; normalise after.
_CITATION_RE = re.compile(r"\[([A-Z0-9]+(?:-[A-Z0-9]+)?)\s+([A-Za-z0-9.\-()]+)\]")
_PAREN_RE = re.compile(r"\((\w+)\)")


def normalize_control_id(framework: str, control_id: str) -> str:
    """Canonical stored id: upper-case, AC-2(1) -> AC-2.1, CIS 3.1 -> CIS-3.1."""
    cid = _PAREN_RE.sub(r".\1", control_id.strip()).upper()
    if framework.strip().upper() == "CIS" and not cid.startswith("CIS-"):
        cid = f"CIS-{cid}"
    return cid


def normalize_citation(cited: str) -> str:
    """Canonical bare citation 'FW id', e.g. 'CIS 3.1' -> 'CIS CIS-3.1'."""
    parts = cited.strip().strip("[]").split(None, 1)
    if len(parts) != 2:
        return cited.strip()
    framework = parts[0].upper()
    return f"{framework} {normalize_control_id(framework, parts[1])}"


def format_citation(framework: str, control_id: str) -> str:
    return f"[{framework} {control_id}]"


def assemble_citations(chunks: list[Chunk]) -> list[str]:
    """Ordered, de-duplicated bare tags like 'NIST AC-2' (no brackets)."""
    return list(dict.fromkeys(f"{c.framework} {c.control_id}" for c in chunks))


def is_refusal(chunks: list[Chunk], min_relevance: float) -> bool:
    """True if no chunks, or the top chunk's score is below min_relevance."""
    if not chunks:
        return True
    return max(c.score for c in chunks) < min_relevance


def verify_grounded(answer: str, retrieved: list[Chunk]) -> bool:
    """True iff every [FW id] cited in the answer maps to a retrieved chunk."""
    grounded = {normalize_citation(f"{c.framework} {c.control_id}") for c in retrieved}
    for framework, control_id in _CITATION_RE.findall(answer):
        if normalize_citation(f"{framework} {control_id}") not in grounded:
            return False
    return True
