import importlib

import app.config as config_mod


def test_defaults(monkeypatch, tmp_path):
    for var in ("DATABASE_URL", "LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_MODEL",
                "OLLAMA_BASE_URL", "MAX_AGENT_STEPS", "MAX_REQUEST_TOKENS",
                "RETRIEVAL_TOP_K", "MIN_RELEVANCE", "MAP_MIN_SIMILARITY",
                "MAX_CHUNK_TOKENS", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Settings reads ".env" from the working directory; the README tells users to
    # create one, so run from an empty dir to see the code defaults, not their edits.
    monkeypatch.chdir(tmp_path)
    importlib.reload(config_mod)
    s = config_mod.get_settings()
    assert s.llm_provider == "ollama"
    assert s.llm_model == "llama3.1"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.max_agent_steps == 4
    assert s.max_request_tokens == 8000
    assert s.retrieval_top_k == 8  # aligns with eval recall@8
    assert s.min_relevance == 0.3
    assert s.map_min_similarity == 0.35
    assert s.max_chunk_tokens == 512
    assert s.ollama_base_url == "http://localhost:11434"


def test_env_override_and_types(monkeypatch):
    monkeypatch.setenv("MAX_AGENT_STEPS", "9")
    monkeypatch.setenv("MIN_RELEVANCE", "0.75")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    importlib.reload(config_mod)
    s = config_mod.get_settings()
    assert s.max_agent_steps == 9
    assert isinstance(s.max_agent_steps, int)
    assert s.min_relevance == 0.75
    assert s.llm_provider == "anthropic"
    assert s.database_url == "postgresql://u:p@db:5432/x"
