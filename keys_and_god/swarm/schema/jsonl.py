"""JSONL helpers for records (events, snapshots, llm_calls, scores)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def to_jsonl_line(rec: BaseModel) -> str:
    """Serialize one record as a JSONL line."""
    return rec.model_dump_json() + "\n"


def write_jsonl(path: str | Path, records: Iterable[BaseModel], *, append: bool = True) -> int:
    """Write records as JSONL (append by default); returns the count."""
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(to_jsonl_line(r))
            n += 1
    return n


def read_jsonl[T: BaseModel](path: str | Path, model: type[T]) -> Iterator[T]:
    """Yield records of `model` from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


def read_jsonl_raw(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw dicts from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
