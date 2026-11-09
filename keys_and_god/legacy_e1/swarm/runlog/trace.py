"""Run reader (WP5): load a run directory back into typed records. Read-only, no side effects.

    run = load_run("runs/<run_id>")
    run.manifest            dict
    run.games               list[GameTrace] (sorted by label)
    run.pairs()             {game_idx: {branch: GameTrace}} for paired forks
    g.events / g.llm_calls / g.snapshots / g.outcome / g.meta
    g.of_type("turn")       -> list[Event]

Lines that fail typed validation are kept out of `events` and counted in `g.parse_errors`
(so a single bad line never breaks a report).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from swarm.schema import Event, LLMCall, Outcome, Snapshot


@dataclass
class GameTrace:
    dir: Path
    meta: dict[str, Any]
    events: list[Event]
    llm_calls: list[LLMCall]
    snapshots: list[Snapshot]
    outcome: Outcome | None
    parse_errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return str(self.meta.get("label", self.dir.name))

    @property
    def game_id(self) -> str:
        if self.events:
            return self.events[0].game_id
        return str(self.meta.get("game_id", self.dir.name))

    @property
    def branch(self) -> str:
        return str(self.meta.get("branch", "main"))

    @property
    def game_idx(self) -> int:
        return int(self.meta.get("game_idx", 0))

    def of_type(self, *types: str) -> list[Event]:
        return [e for e in self.events if e.type in types]

    @property
    def agents(self) -> list[str]:
        gs = self.of_type("game_start")
        if gs:
            return list(gs[0].data.order)  # type: ignore[union-attr]
        return sorted({e.actor for e in self.events if e.type == "turn"})


def _read_typed(path: Path, model: type, errors: list[str]) -> list[Any]:
    out: list[Any] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                out.append(model.model_validate_json(line))
            except ValidationError as e:
                errors.append(f"{path.name}:{i + 1}: {str(e).splitlines()[0]}")
    return out


def load_game(game_dir: str | Path, *, with_llm_calls: bool = True, with_snapshots: bool = True) -> GameTrace:
    d = Path(game_dir)
    errors: list[str] = []
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"label": d.name}
    events = _read_typed(d / "events.jsonl", Event, errors)
    calls = _read_typed(d / "llm_calls.jsonl", LLMCall, errors) if with_llm_calls else []
    snaps = _read_typed(d / "snapshots.jsonl", Snapshot, errors) if with_snapshots else []
    outcome = None
    if (d / "outcome.json").exists():
        outcome = Outcome.model_validate_json((d / "outcome.json").read_text())
    else:
        ends = [e for e in events if e.type == "game_end"]
        if ends:
            outcome = ends[-1].data.outcome  # type: ignore[union-attr]
    return GameTrace(d, meta, events, calls, snaps, outcome, errors)


@dataclass
class RunTrace:
    dir: Path
    manifest: dict[str, Any]
    games: list[GameTrace]

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", self.dir.name))

    def pairs(self) -> dict[int, dict[str, GameTrace]]:
        out: dict[int, dict[str, GameTrace]] = defaultdict(dict)
        for g in self.games:
            out[g.game_idx][g.branch] = g
        return dict(out)

    def game(self, label: str) -> GameTrace:
        for g in self.games:
            if g.label == label:
                return g
        raise KeyError(label)


def load_run(run_dir: str | Path, **kw: Any) -> RunTrace:
    d = Path(run_dir)
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    gdirs = sorted(p for p in (d / "games").iterdir() if p.is_dir()) if (d / "games").is_dir() else []
    if not gdirs and (d / "events.jsonl").exists():   # single-game dir (doc layout)
        gdirs = [d]
    return RunTrace(d, man, [load_game(p, **kw) for p in gdirs])
