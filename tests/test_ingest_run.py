import pytest

import app.ingest.run as run_mod
from app.models import Chunk


@pytest.fixture(autouse=True)
def stub_io(monkeypatch):
    async def fake_init():
        return None

    upserted = []

    async def fake_upsert(chunks):
        upserted.extend(chunks)
        return len(chunks)

    monkeypatch.setattr(run_mod, "init_db", fake_init)
    monkeypatch.setattr(run_mod, "upsert_controls", fake_upsert)
    return upserted


async def test_ingest_fixture_only_reads_single_file(monkeypatch, tmp_path):
    fixture = tmp_path / "mini.yaml"
    fixture.write_text("framework: ISO\ncontrols:\n  - {control_id: a.5.1, title: T, summary: S}\n",
                       encoding="utf-8")

    def fake_yaml_build(parsed, mct):
        return [Chunk(framework=p["framework"], control_id=p["control_id"], title="T",
                      summary="S", embedding=[0.0]) for p in parsed]

    monkeypatch.setattr(run_mod, "yaml_build", fake_yaml_build)
    counts = await run_mod.ingest_all(fixture=fixture, include_nist=False)
    assert counts == {"ISO": 1}


async def test_ingest_all_nist_plus_corpus(monkeypatch):
    async def fake_fetch(url=None):
        return {"catalog": {"groups": []}}

    monkeypatch.setattr(run_mod, "fetch_catalog", fake_fetch)
    monkeypatch.setattr(run_mod, "parse_catalog", lambda cat: [
        {"framework": "NIST", "control_id": "AC-2", "title": "T", "summary": "S", "source_url": "u"}])
    monkeypatch.setattr(run_mod, "nist_build", lambda parsed, mct: [
        Chunk(framework="NIST", control_id="AC-2", title="T", summary="S", embedding=[0.0])])
    monkeypatch.setattr(run_mod, "load_corpus_dir", lambda d: [
        {"framework": "ISO", "control_id": "A.5.1", "title": "T", "summary": "S", "source_url": ""}])
    monkeypatch.setattr(run_mod, "yaml_build", lambda parsed, mct: [
        Chunk(framework="ISO", control_id="A.5.1", title="T", summary="S", embedding=[0.0])])

    counts = await run_mod.ingest_all()
    assert counts == {"NIST": 1, "ISO": 1}
