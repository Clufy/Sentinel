from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_PATH = Path(__file__).with_name("dataset.jsonl")


class EvalItem(BaseModel):
    question: str
    expected_control_ids: list[str] = Field(default_factory=list)
    gold_answer_sketch: str = ""
    answerable: bool = True


def load_dataset(path: str | None = None) -> list[EvalItem]:
    target = Path(path) if path is not None else _DEFAULT_PATH
    items: list[EvalItem] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(EvalItem(**json.loads(line)))
    return items
