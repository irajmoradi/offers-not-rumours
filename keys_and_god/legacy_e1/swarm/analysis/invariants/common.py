"""Shared pieces of the invariant checker: regexes, Violation, the event-replay state, JSONL reading."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from swarm.config import RunConfig

KEYS_RE = re.compile(r"\{info_id: (\d+), key: (\d+)\}")
NOTICE_KEY_RE = re.compile(r"info #(\d+) \(key (\d+)\)")
NOTE_RE = re.compile(r"Your (note|thought) \(round (\d+)\): ")
FOUR_DIGITS = re.compile(r"(?<!\d)\d{4}(?!\d)")
# lines whose text an agent wrote (DMs, board posts, own messages/notes): they may quote any number
AGENT_TEXT_TAGS = ("DM from ", "BOARD ", "posted on the board", ") board_post: ", "You messaged ", "You posted ", " -> ",
                   "Your note (", "Your thought (", "Your summary (")


@dataclass
class Violation:
    game: str
    check: str
    detail: str


@dataclass
class Replay:
    """The truth state re-derived from events (holdings by copy id)."""
    keys: dict[str, dict[str, tuple[int, int]]]          # agent -> copy_id -> (info_id, key)
    needs: dict[str, list[int]]
    burned: dict[str, list[int]] = field(default_factory=dict)
    lost: dict[str, list[int]] = field(default_factory=dict)
    won: dict[str, int | None] = field(default_factory=dict)

    def open_needs(self, a: str) -> list[int]:
        return [i for i in self.needs[a] if i not in self.burned[a] and i not in self.lost[a]]

    def held(self, a: str) -> set[int]:
        return {i for i, _ in self.keys[a].values()}

    def holdings(self, a: str) -> list[tuple[int, int]]:
        return sorted(self.keys[a].values())

    def score(self, a: str) -> float:
        if not self.needs[a]:
            return 0.0
        h = self.held(a)
        return sum(1 for i in self.open_needs(a) if i in h) / len(self.needs[a])

    def winning(self, a: str) -> bool:
        on = self.open_needs(a)
        return bool(on) and set(on) <= self.held(a)

    def copy(self) -> Replay:
        return Replay({a: dict(k) for a, k in self.keys.items()}, dict(self.needs),
                      {a: list(v) for a, v in self.burned.items()}, {a: list(v) for a, v in self.lost.items()},
                      dict(self.won))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except ValueError:
                    continue                      # a torn last line of a running game


def load_cfg(run_dir: Path) -> RunConfig:
    raw = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    raw.pop("runner", None)
    return RunConfig.model_validate(raw)
