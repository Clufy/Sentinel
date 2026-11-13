"""Embedding via a cached SentenceTransformer singleton."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embedding_dim() -> int:
    return int(_model().get_sentence_embedding_dimension())


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    raw = _model().encode(texts, convert_to_numpy=False)
    return [[float(x) for x in row] for row in raw]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
