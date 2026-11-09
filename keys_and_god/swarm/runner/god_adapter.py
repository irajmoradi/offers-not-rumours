"""God timing adapter for the real engine (split out of runner/wiring.py, which re-exports it).

GameEngine calls a legacy god_hook(state, round, next_turn) with state.turn_index = next_turn - 1;
GodRunner/ScriptedGod schedule by the slot ABOUT to be played. `_SlotGod` + `make_god_hook` bridge the
two (FROZEN convention: forged DMs are stamped next_turn - 1, visible at next_turn), and
`check_schedule` rejects ScriptedGod entries that this timing can never fire.
Imports: config.
"""

from __future__ import annotations

from typing import Any

from swarm.config import RunConfig


class _SlotGod:
    """Adapter between the engine's god_hook timing and GodRunner's.

    The engine calls the hook with state.turn_index = next_turn - 1 (so forged DMs are stamped like
    the previous slot and are visible at next_turn). ScriptedGod schedules by the slot ABOUT to be
    played. We expose next_turn while God decides, then restore next_turn - 1 before powers apply."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "god")
        self.next_turn = 0

    def wakes_at(self, state: Any) -> bool:
        return bool(self.inner.wakes_at(state))

    async def step(self, state: Any, view: Any) -> list[Any]:
        try:
            return list(await self.inner.step(state, view))
        finally:
            state.turn_index = self.next_turn - 1


def make_god_hook(runner: Any, slot_god: _SlotGod) -> Any:
    """-> god_hook(state, round, next_turn) for GameEngine."""

    async def hook(state: Any, r: int, next_turn: int) -> list[Any]:
        slot_god.next_turn = next_turn
        state.turn_index = next_turn
        try:
            if not runner.should_wake(state):
                return []
            res = await runner.wake(state)
            return list(res.events)
        finally:
            state.turn_index = next_turn - 1

    return hook


def check_schedule(cfg: RunConfig) -> None:
    """Reject ScriptedGod schedule entries the real engine can never fire (they used to be silently
    skipped, so a run looked like it had an intervention that never happened). The engine consults
    God before slot after_turns of the round: only slot 0 in simultaneous mode, slots 0..n-1 in
    sequential mode. (round < start_round entries are documented as never firing and stay allowed.)"""
    if cfg.god.kind != "scripted":
        return
    n = cfg.game.n_agents
    for i, sp in enumerate(cfg.god.schedule):
        if cfg.game.turn_mode == "simultaneous" and sp.after_turns > 0:
            raise ValueError(f"god.schedule[{i}].after_turns={sp.after_turns}: simultaneous mode wakes God only "
                             "at the start of a round (after_turns must be 0)")
        if sp.after_turns >= n:
            raise ValueError(f"god.schedule[{i}].after_turns={sp.after_turns} >= game.n_agents={n}: never fires")
