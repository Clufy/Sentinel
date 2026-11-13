CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS controls (
    id          serial PRIMARY KEY,
    framework   text NOT NULL,
    control_id  text NOT NULL,
    chunk_idx   int  NOT NULL DEFAULT 0,
    title       text NOT NULL,
    summary     text NOT NULL,
    source_url  text,
    embedding   vector(384),
    -- NOTE: CREATE ... IF NOT EXISTS will not alter an existing table/index. An existing local
    -- DB must be recreated (`docker compose down -v`) to pick up the fts column and HNSW index.
    fts         tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('english',
                      coalesce(framework, '') || ' ' || coalesce(control_id, '') || ' ' || coalesce(title, '')), 'A')
                  || setweight(to_tsvector('english', coalesce(summary, '')), 'B')
                ) STORED,
    UNIQUE (framework, control_id, chunk_idx)
);

CREATE INDEX IF NOT EXISTS controls_fts_gin ON controls USING gin (fts);
-- HNSW needs no training data (ivfflat on an empty table returns too few rows at real size).
CREATE INDEX IF NOT EXISTS controls_embedding_hnsw
    ON controls USING hnsw (embedding vector_cosine_ops);
