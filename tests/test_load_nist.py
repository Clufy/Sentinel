import json
from pathlib import Path

from app.ingest.load_nist import NIST_CATALOG_URL, build_chunks, parse_catalog

DATA = Path(__file__).parent / "data" / "nist_sample_catalog.json"


def _catalog():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_parse_walks_controls_and_enhancements():
    parsed = parse_catalog(_catalog())
    assert {p["control_id"] for p in parsed} == {"AC-2", "AC-2.1"}


def test_parse_control_id_uppercased_and_source_url():
    parsed = parse_catalog(_catalog())
    ac2 = next(p for p in parsed if p["control_id"] == "AC-2")
    assert ac2["title"] == "Account Management"
    assert ac2["framework"] == "NIST"
    assert ac2["source_url"] == NIST_CATALOG_URL


def test_summary_joins_statement_and_subparts():
    ac2 = next(p for p in parse_catalog(_catalog()) if p["control_id"] == "AC-2")
    assert "Manage system accounts" in ac2["summary"]
    assert "Defining account types;" in ac2["summary"]
    assert "Assigning account managers." in ac2["summary"]


def test_build_chunks_embeds_and_sets_chunk_idx(monkeypatch):
    import app.ingest.load_yaml as ln  # build_chunks (and its embed_texts) live there
    monkeypatch.setattr(ln, "embed_texts", lambda texts: [[0.0, 1.0] for _ in texts])
    chunks = build_chunks(parse_catalog(_catalog()), max_chunk_tokens=512)
    assert len(chunks) == 2  # each short control -> one chunk
    for c in chunks:
        assert c.framework == "NIST"
        assert c.chunk_idx == 0
        assert c.embedding == [0.0, 1.0]


def test_build_chunks_embeds_title_prefixed_text(monkeypatch):
    import app.ingest.load_yaml as ln  # build_chunks (and its embed_texts) live there
    seen = []
    monkeypatch.setattr(ln, "embed_texts", lambda texts: seen.extend(texts) or [[0.0] for _ in texts])
    parsed = [{"framework": "NIST", "control_id": "XX-1", "title": "My Title",
               "summary": " ".join(f"w{i}" for i in range(600)), "source_url": ""}]
    chunks = build_chunks(parsed, max_chunk_tokens=512)
    assert len(seen) == 2 and all(t.startswith("My Title. ") for t in seen)
    assert all(not c.summary.startswith("My Title") for c in chunks)  # stored summary unchanged


def test_build_chunks_splits_long_summary(monkeypatch):
    import app.ingest.load_yaml as ln  # build_chunks (and its embed_texts) live there
    monkeypatch.setattr(ln, "embed_texts", lambda texts: [[0.0] for _ in texts])
    parsed = [{
        "framework": "NIST", "control_id": "XX-1", "title": "T",
        "summary": " ".join(f"w{i}" for i in range(600)),
        "source_url": NIST_CATALOG_URL,
    }]
    chunks = build_chunks(parsed, max_chunk_tokens=512)
    assert [c.chunk_idx for c in chunks] == [0, 1]


# --- I4: withdrawn controls and OSCAL parameter placeholders --------------
def test_withdrawn_controls_and_enhancements_are_skipped():
    ids = {p["control_id"] for p in parse_catalog(_catalog())}
    assert "AC-2.10" not in ids  # withdrawn enhancement
    assert "AC-13" not in ids  # withdrawn base control (value case differs)


def test_param_placeholders_are_rendered_from_catalog_params():
    parsed = {p["control_id"]: p["summary"] for p in parse_catalog(_catalog())}
    assert "Requiring [Assignment: prerequisites and criteria] for group membership;" \
        in parsed["AC-2"]
    assert "Choosing to [Selection: disable; remove] unused accounts within " \
        "[Assignment: organization-defined value];" in parsed["AC-2"]
    # enhancement sees its own params and its parent's
    assert parsed["AC-2.1"] == ("Support account management using [Assignment: automated "
                                "mechanisms] and [Assignment: prerequisites and criteria].")


def test_no_summary_contains_param_placeholders():
    for p in parse_catalog(_catalog()):
        assert "{{" not in p["summary"], p["control_id"]
        assert "insert: param" not in p["summary"], p["control_id"]
