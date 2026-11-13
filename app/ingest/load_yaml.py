"""Loader for authored YAML corpus files under corpus/."""
from pathlib import Path

import yaml

from app.ingest.chunk import chunk_control, embed_input
from app.embeddings import embed_texts
from app.models import Chunk


def parse_yaml_file(path: str | Path, framework: str | None = None) -> list[dict]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    fw = framework if framework is not None else str(data.get("framework", "")).strip()
    out: list[dict] = []
    for ctrl in data.get("controls", []) or []:
        out.append({
            "framework": fw,
            "control_id": str(ctrl["control_id"]).upper(),
            "title": ctrl.get("title", ""),
            "summary": ctrl.get("summary", ""),
            "source_url": ctrl.get("source_url", ""),
        })
    return out


def load_corpus_dir(corpus_dir: str | Path) -> list[dict]:
    d = Path(corpus_dir)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(d.glob("*.yaml")):
        out.extend(parse_yaml_file(path))
    return out


def build_chunks(parsed: list[dict], max_chunk_tokens: int) -> list[Chunk]:
    chunk_texts: list[str] = []
    meta: list[tuple[dict, int]] = []
    for p in parsed:
        for idx, text in enumerate(chunk_control(p["title"], p["summary"], max_chunk_tokens)):
            chunk_texts.append(text)
            meta.append((p, idx))
    embeddings = embed_texts([embed_input(p['title'], t) for (p, _), t in zip(meta, chunk_texts)])
    chunks: list[Chunk] = []
    for (p, idx), text, emb in zip(meta, chunk_texts, embeddings):
        chunks.append(Chunk(
            framework=p["framework"],
            control_id=p["control_id"],
            chunk_idx=idx,
            title=p["title"],
            summary=text,
            source_url=p["source_url"],
            embedding=emb,
        ))
    return chunks
