from app.db import load_ddl


def test_ddl_has_columns_and_indexes():
    ddl = load_ddl()
    assert "vector(384)" in ddl
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "CREATE TABLE IF NOT EXISTS controls" in ddl
    assert "UNIQUE (framework, control_id, chunk_idx)" in ddl
    assert "GENERATED ALWAYS AS" in ddl
    assert "to_tsvector('english'" in ddl
    assert "USING gin (fts)" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "USING ivfflat" not in ddl


def test_fts_covers_framework_and_control_id():
    ddl = load_ddl()
    fts_def = ddl.split("tsvector GENERATED")[1].split("UNIQUE")[0]
    assert "framework" in fts_def and "control_id" in fts_def
    assert "setweight" in ddl
    assert "docker compose down -v" in ddl
