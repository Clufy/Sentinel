import math

import pytest

from app.models import Chunk
import app.retrieval as R


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class FakeCE:
    def __init__(self, table):
        self.table = table
        self.seen = []

    def predict(self, pairs):
        self.seen = list(pairs)
        return [self.table[passage] for (_q, passage) in pairs]


def _c(cid, title, summary):
    return Chunk(framework="NIST", control_id=cid, title=title, summary=summary,
                 source_url="http://x")


def test_rerank_sorts_descending_and_sets_score(monkeypatch):
    c1, c2 = _c("AC-1", "Access", "alpha"), _c("AC-2", "Account", "beta")
    fake = FakeCE({"Access. alpha (NIST AC-1)": 0.2, "Account. beta (NIST AC-2)": 0.9})
    monkeypatch.setattr(R, "_cross_encoder", lambda: fake)
    out = R.rerank("q", [c1, c2])
    assert [c.control_id for c in out] == ["AC-2", "AC-1"]
    assert out[0].score == pytest.approx(_sig(0.9))
    assert out[1].score == pytest.approx(_sig(0.2))


def test_rerank_builds_query_passage_pairs(monkeypatch):
    c1 = _c("AC-1", "Access", "alpha")
    fake = FakeCE({"Access. alpha (NIST AC-1)": 0.5})
    monkeypatch.setattr(R, "_cross_encoder", lambda: fake)
    R.rerank("my-query", [c1])
    assert fake.seen == [("my-query", "Access. alpha (NIST AC-1)")]


def test_rerank_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(R, "_cross_encoder", lambda: FakeCE({}))
    assert R.rerank("q", []) == []


def test_rerank_pair_text_includes_framework_and_control_id(monkeypatch):
    # The label goes AFTER the text: measured on the live eval with the MS-MARCO
    # cross-encoder, a leading "NIST AC-2 ..." prefix cut relevant scores from ~0.78
    # to ~0.25 (refusing grounded questions), while the suffix kept them and still
    # lets framework-named questions favour that framework.
    nist = _c("AC-2", "Account Management", "same")
    cis = Chunk(framework="CIS", control_id="CIS-5", title="Account Management",
                summary="same", source_url="http://x")
    fake = FakeCE({"Account Management. same (NIST AC-2)": 0.1,
                   "Account Management. same (CIS CIS-5)": 0.8})
    monkeypatch.setattr(R, "_cross_encoder", lambda: fake)
    out = R.rerank("Which CIS control addresses account management?", [nist, cis])
    assert [c.control_id for c in out] == ["CIS-5", "AC-2"]
