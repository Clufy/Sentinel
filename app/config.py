from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql://sentinel:sentinel@127.0.0.1:5432/sentinel"
    llm_provider: str = "ollama"
    llm_api_key: str = ""
    llm_model: str = "llama3.1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    max_agent_steps: int = 4
    max_request_tokens: int = 8000
    retrieval_top_k: int = 8
    min_relevance: float = 0.3
    # /map floor on embedding cosine (the cross-encoder scores declarative findings ~0).
    # On the full corpus real findings top 0.41-0.64 and off-topic text at most 0.33.
    map_min_similarity: float = 0.35
    max_chunk_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
