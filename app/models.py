"""The single Chunk model plus API request/response models."""
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    framework: str
    control_id: str
    chunk_idx: int = 0
    title: str
    summary: str
    source_url: str = ""
    score: float = 0.0
    id: int | None = None  # DB row id; None before persistence
    embedding: list[float] | None = None  # populated by vector_search for MMR reuse


class QueryRequest(BaseModel):
    question: str
    framework: str | None = None


class MapRequest(BaseModel):
    finding_text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    counts: dict[str, int]
    total: int
