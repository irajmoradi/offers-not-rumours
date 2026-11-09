"""Event-sink and LLM-request wrappers for one real game (split out of runner/wiring.py, which
re-exports them).

  LedgerSink    keeps the per-agent context ledgers, rebuilt from `turn` and `action_result` events
  ProgressSink  rewrites runs/<id>/progress.txt on every turn (cosmetic, best effort)
  OracleClient  leak-checks every agent-bound ChatRequest BEFORE it is sent
Imports: stdlib only at module level (context.base lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class LedgerSink:
    """EventSink wrapper that keeps the engine-owned per-agent ledgers (CONTRACTS §1).

    Ledgers live outside GameState and are rebuilt from the event stream: a `turn` event adds the
    items of the observation it logs (items_from_observation, the only constructor), and the
    matching `action_result` events add the agent's own action items. Forwards everything."""

    def __init__(self, inner: Any, counter: Any = None) -> None:
        self.inner, self.counter = inner, counter
        self.ledgers: dict[str, list[Any]] = {}
        self._own_n: dict[tuple[str, int], int] = {}   # own actions seen per (agent, round)

    def emit(self, body: Any) -> Any:
        from swarm.context.base import items_from_observation, own_action_items

        ev = self.inner.emit(body)
        d = body.data
        if d.type == "turn":
            self.ledgers.setdefault(d.agent, []).extend(items_from_observation(d.obs, self.counter))
        elif d.type == "action_result" and d.action.kind != "grep_log":
            # Number own actions per (agent, round): own_action_items() numbers within ONE call, and
            # we call it once per action, so without this every action got item_id ...:own_action:0
            # and the context policy (items keyed by id) dropped all but the last (CODE_REVIEW #1).
            k = (d.agent, body.round)
            n = self._own_n.get(k, 0)
            self._own_n[k] = n + 1
            (item,) = own_action_items(d.agent, body.round, [d.action], [(d.ok, d.reason)], self.counter)
            self.ledgers.setdefault(d.agent, []).append(
                item.model_copy(update={"item_id": f"{d.agent}:{body.round}:own_action:{n}"}))
        return ev

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class ProgressSink:
    """EventSink wrapper that rewrites runs/<id>/progress.txt on every `turn` event: one line per
    game (label, round/turn, $ and calls so far for the shared agent client). Best effort only."""

    _lines: dict[Path, dict[str, str]] = {}

    def __init__(self, inner: Any, label: str, max_rounds: int, client: Any) -> None:
        self.inner, self.label, self.max_rounds, self.client = inner, label, max_rounds, client
        self.path = Path(inner.dir).parent.parent / "progress.txt"

    def emit(self, body: Any) -> Any:
        ev = self.inner.emit(body)
        if body.data.type in ("turn", "game_end"):
            try:
                import time

                tot = getattr(getattr(self.client, "inner", self.client), "totals", None)
                cost = getattr(tot, "cost_usd", 0.0)
                calls = getattr(tot, "calls", 0)
                what = "done" if body.data.type == "game_end" else f"round {body.round}/{self.max_rounds} turn {body.turn_index}"
                lines = self._lines.setdefault(self.path, {})
                lines[self.label] = f"{self.label}: {what}"
                summary = f"{time.strftime('%H:%M:%S')} run-wide agent client: ${cost:.4f}, {calls} calls"
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text("\n".join([summary, *lines.values()]) + "\n", encoding="utf-8")
                tmp.replace(self.path)
            except Exception:   # progress is cosmetic; never break a game over it
                pass
        return ev

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class OracleClient:
    """Per-game LLMClient wrapper: every agent-bound ChatRequest is scanned by the LeakOracle BEFORE
    it is sent (a TruthLeak aborts the game, so a leak can never be silently logged as a result)."""

    def __init__(self, inner: Any, oracle: Any) -> None:
        self.inner, self.oracle = inner, oracle

    async def chat(self, req: Any) -> Any:
        self.oracle.assert_clean(req)
        return await self.inner.chat(req)
