"""Run directory writers (WP5). The ONLY place that appends to events/llm_calls/snapshots files.

Layout (one run = many games; paired forks are sibling game dirs):

    runs/<run_id>/
      manifest.json            run-level (see manifest.py), atomic rewrite
      config.resolved.yaml     resolved RunConfig (+ the `runner:` block)
      prompts/                 copy of every prompt set used
      games/<label>/           label = g<idx:02d>-<branch>, e.g. g00-god, g00-nogod, g00-noise
        events.jsonl           typed Event stream; seq contiguous from 0; fsync at round_end/game_end
        llm_calls.jsonl        one LLMCall per call (cache hits included)
        snapshots.jsonl        one Snapshot per round end (+ round 0)
        outcome.json           Outcome
        meta.json              {label, game_id, game_idx, branch, god, seed, status, error}
      llm_calls.jsonl          calls whose meta.game_id is unknown (e.g. doctor pings)
      report.html, figs/       derived (viz)

Every serialized line passes through `swarm.llm.secrets.redact`, so a registered API key can never
reach disk even if some upstream code put it into a message.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from swarm.llm.secrets import redact
from swarm.schema import (
    Event,
    EventBody,
    GameState,
    LLMCall,
    Outcome,
    Snapshot,
    make_event,
)

FSYNC_TYPES = frozenset({"round_end", "game_end"})

# The GameWriter of the asyncio task currently playing a game. Set by the runner per game task;
# LLM calls made inside that task are routed to it (paired branches share a game_id, so routing by
# meta.game_id alone would be ambiguous when they run concurrently).
CURRENT_GAME: contextvars.ContextVar[GameWriter | None] = contextvars.ContextVar("swarm_current_game",
                                                                                   default=None)


def _dump(rec: BaseModel | dict[str, Any]) -> str:
    s = rec.model_dump_json() if isinstance(rec, BaseModel) else json.dumps(rec, sort_keys=True, default=str)
    return redact(s) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(redact(text))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


class _Appender:
    """Line-buffered JSONL appender with explicit fsync."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, line: str, *, sync: bool = False) -> None:
        with self._lock:
            self._f.write(line)
            self._f.flush()
            if sync:
                os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                os.fsync(self._f.fileno())
                self._f.close()


class UsageTotals:
    """Running totals per (role, model) for the manifest `usage` block."""

    def __init__(self) -> None:
        self.by: dict[str, dict[str, float]] = defaultdict(lambda: {
            "calls": 0, "cache_hits": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})

    def add(self, call: LLMCall) -> None:
        k = f"{call.request.meta.role}/{call.model}"
        t = self.by[k]
        t["calls"] += 1
        if call.error:
            t["errors"] += 1
        r = call.response
        if r is not None:
            t["cache_hits"] += int(r.cache_hit)
            t["prompt_tokens"] += r.usage.prompt_tokens
            t["completion_tokens"] += r.usage.completion_tokens
            t["reasoning_tokens"] += r.usage.reasoning_tokens
            t["cost_usd"] += r.cost_usd
            t["latency_s"] += r.latency_s

    def merge(self, other: UsageTotals) -> None:
        for k, t in other.by.items():
            for f, v in t.items():
                self.by[k][f] += v

    def to_dict(self, budget_usd: float | None = None) -> dict[str, Any]:
        by = {k: dict(v) for k, v in sorted(self.by.items())}
        return {"by_role_model": by, "total_cost_usd": round(sum(v["cost_usd"] for v in by.values()), 6),
                "total_calls": int(sum(v["calls"] for v in by.values())), "budget_usd": budget_usd}


class GameWriter:
    """Per-game sink. Implements `schema.EventSink` (emit stamps v/run_id/game_id/seq)."""

    def __init__(self, game_dir: Path, run_id: str, game_id: str, meta: dict[str, Any] | None = None) -> None:
        self.dir = Path(game_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.game_id = run_id, game_id
        self.seq = 0
        self.usage = UsageTotals()
        self.meta: dict[str, Any] = {"game_id": game_id, "status": "running", **(meta or {})}
        self._events = _Appender(self.dir / "events.jsonl")
        self._calls = _Appender(self.dir / "llm_calls.jsonl")
        self._snaps = _Appender(self.dir / "snapshots.jsonl")
        self._lock = threading.Lock()
        self.write_meta()

    # ------------------------------------------------------------ EventSink
    def emit(self, body: EventBody) -> Event:
        with self._lock:
            ev = make_event(body, run_id=self.run_id, game_id=self.game_id, seq=self.seq)
            self.seq += 1
        self._events.write(_dump(ev), sync=ev.type in FSYNC_TYPES)
        return ev

    def emit_event(self, ev: Event) -> Event:
        """Write an already-stamped Event (re-stamps seq/run/game so the stream stays contiguous)."""
        with self._lock:
            ev = ev.model_copy(update={"run_id": self.run_id, "game_id": self.game_id, "seq": self.seq})
            self.seq += 1
        self._events.write(_dump(ev), sync=ev.type in FSYNC_TYPES)
        return ev

    # ------------------------------------------------------------ other streams
    def log_llm_call(self, call: LLMCall) -> None:
        self.usage.add(call)
        self._calls.write(_dump(call))

    def snapshot(self, state: GameState) -> None:
        snap = Snapshot(run_id=self.run_id, game_id=self.game_id, round=state.round, state=state)
        self._snaps.write(_dump(snap), sync=True)

    def write_outcome(self, outcome: Outcome) -> None:
        atomic_write_text(self.dir / "outcome.json", outcome.model_dump_json(indent=2) + "\n")
        self.meta.update(end_reason=outcome.end_reason, rounds=outcome.rounds_played,
                         welfare=outcome.welfare, win_rate=outcome.win_rate)

    def write_meta(self, **update: Any) -> None:
        self.meta.update(update)
        atomic_write_json(self.dir / "meta.json", self.meta)

    def close(self, status: str = "done", error: str | None = None) -> None:
        for a in (self._events, self._calls, self._snaps):
            a.close()
        self.write_meta(status=status, error=error, n_events=self.seq,
                        usage=self.usage.to_dict())


class RunWriter:
    """Owns runs/<run_id>/. Hands out GameWriters and routes LLM calls to them by meta.game_id."""

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "games").mkdir(exist_ok=True)
        self.run_id = run_id
        self.games: dict[str, GameWriter] = {}          # game_id -> first writer (fallback routing)
        self.all_games: list[GameWriter] = []
        self._orphan_calls: _Appender | None = None
        self.orphan_usage = UsageTotals()
        self._lock = threading.Lock()

    def game(self, label: str, game_id: str, meta: dict[str, Any] | None = None) -> GameWriter:
        gw = GameWriter(self.dir / "games" / label, self.run_id, game_id, {"label": label, **(meta or {})})
        with self._lock:
            self.games.setdefault(game_id, gw)
            self.all_games.append(gw)
        return gw

    def log_llm_call(self, call: LLMCall) -> None:
        """Router: pass this as the Accounting middleware's on_call callback (shared clients)."""
        gw = CURRENT_GAME.get()
        if gw is None:
            gid = call.request.meta.game_id
            gw = self.games.get(gid) if gid else None
        if gw is not None:
            gw.log_llm_call(call)
            return
        with self._lock:
            if self._orphan_calls is None:
                self._orphan_calls = _Appender(self.dir / "llm_calls.jsonl")
        self.orphan_usage.add(call)
        self._orphan_calls.write(_dump(call))

    def usage(self) -> UsageTotals:
        tot = UsageTotals()
        for gw in self.all_games:
            tot.merge(gw.usage)
        tot.merge(self.orphan_usage)
        return tot

    def close(self) -> None:
        if self._orphan_calls is not None:
            self._orphan_calls.close()
