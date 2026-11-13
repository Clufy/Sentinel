from pathlib import Path

from app.ingest.load_yaml import build_chunks, load_corpus_dir, parse_yaml_file

DATA = Path(__file__).parent / "data" / "sample_corpus.yaml"


def test_parse_yaml_uppercases_and_defaults():
    parsed = parse_yaml_file(DATA)
    assert {p["control_id"] for p in parsed} == {"A.5.1", "A.5.2"}
    a51 = next(p for p in parsed if p["control_id"] == "A.5.1")
    assert a51["framework"] == "ISO"
    assert a51["source_url"] == "http://iso.example/a5-1"
    a52 = next(p for p in parsed if p["control_id"] == "A.5.2")
    assert a52["source_url"] == ""  # default when missing


def test_parse_yaml_framework_override():
    parsed = parse_yaml_file(DATA, framework="CIS")
    assert all(p["framework"] == "CIS" for p in parsed)


def test_load_corpus_dir_missing_returns_empty(tmp_path):
    assert load_corpus_dir(tmp_path / "does_not_exist") == []


def test_load_corpus_dir_reads_yaml(tmp_path):
    (tmp_path / "iso.yaml").write_text(DATA.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(load_corpus_dir(tmp_path)) == 2


def test_build_chunks(monkeypatch):
    import app.ingest.load_yaml as ly
    monkeypatch.setattr(ly, "embed_texts", lambda texts: [[1.0] for _ in texts])
    chunks = build_chunks(parse_yaml_file(DATA), max_chunk_tokens=512)
    assert len(chunks) == 2
    assert all(c.framework == "ISO" for c in chunks)
    assert all(c.chunk_idx == 0 for c in chunks)
    assert all(c.embedding == [1.0] for c in chunks)


def test_build_chunks_embeds_title_prefixed_text(monkeypatch):
    import app.ingest.load_yaml as ly
    seen = []
    monkeypatch.setattr(ly, "embed_texts", lambda texts: seen.extend(texts) or [[1.0] for _ in texts])
    parsed = parse_yaml_file(DATA)
    chunks = build_chunks(parsed, max_chunk_tokens=512)
    assert seen == [f"{p['title']}. {p['summary']}" for p in parsed]
    assert [c.summary for c in chunks] == [p["summary"] for p in parsed]  # stored unchanged
