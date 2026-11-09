"""God interface and the per-game runner (WP4).

    runner = GodRunner(god, cfg.god, memory, max_message_chars=cfg.game.max_message_chars,
                       winners_leave=cfg.game.winner_policy == "sit_out")
    runner.init_game(state)                      # once, after generate_world (sets the budget)
    ... before each agent slot (sequential) or at round start (simultaneous):
    if runner.should_wake(state):
        res = await runner.wake(state)           # mutates state (inbox / memory / budget)
        for body in res.events: sink.emit(body)

`state.turn_index` must already be the slot S about to be played: God acts *between* turns.
ONE stamp convention (CODE_REVIEW #7): God DECIDES at S (its schedule and free tools see
turn_index = S), and everything it DELIVERS (forged DMs, memory overwrites, power and post-step tool
events) is stamped S - 1, i.e. like the previous slot, so it is visible to the observation at S.
This is what every real run used (formerly via runner/wiring's _SlotGod adapter); wake() restores
turn_index = S before returning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from swarm.adversary.god_tools import GodView
from swarm.adversary.powers import PowerResult, apply_power, init_budget, is_power, refresh_budget
from swarm.config import GodCfg
from swarm.memory import MemoryStore
from swarm.schema import EventBody, GameState, GodAction


def turns_into_round(state: GameState) -> int:
    """Agent slots already played in the current round (schema.turn_index contract)."""
    return state.turn_index - (state.round - 1) * len(state.order)


@runtime_checkable
class God(Protocol):
    name: str

    def wakes_at(self, state: GameState) -> bool:
        """Should God wake before the slot `state.turn_index`? (start_round is checked by the runner.)"""
        ...

    async def step(self, state: GameState, view: GodView) -> list[GodAction]:
        """Observe through `view` (free) and return the GodActions to apply (tools and/or powers).
        Must not mutate `state`; the runner applies powers via adversary.powers."""
        ...


@dataclass
class WakeResult:
    actions: list[GodAction] = field(default_factory=list)
    results: list[PowerResult] = field(default_factory=list)
    events: list[EventBody] = field(default_factory=list)
    ceiling_hit: bool = False


class GodRunner:
    """Owns budget refresh, the tool ceiling and power application for one game."""

    def __init__(self, god: God, cfg: GodCfg, memory: MemoryStore, *, max_message_chars: int = 500,
                 winners_leave: bool = False) -> None:
        self.god, self.cfg, self.memory = god, cfg, memory
        self.max_message_chars = max_message_chars
        self.winners_leave = winners_leave      # game.winner_policy == "sit_out": no fraud via departed agents
        self._budget_round: int | None = None

    def init_game(self, state: GameState) -> None:
        init_budget(state, self.cfg)
        self._budget_round = None

    def should_wake(self, state: GameState) -> bool:
        if self.cfg.kind == "none" or state.round < self.cfg.start_round:
            return False
        return self.god.wakes_at(state)

    async def wake(self, state: GameState) -> WakeResult:
        self._budget_round = refresh_budget(state, self.cfg, self._budget_round)
        view = GodView(state, ceiling=self.cfg.tool_call_safety_ceiling)
        slot = state.turn_index
        try:
            actions = await self.god.step(state, view)     # decides at the slot about to be played
        finally:
            state.turn_index = slot - 1                    # ... delivers stamped like the slot before
        try:
            out = WakeResult(actions=list(actions))
            for a in actions:
                if is_power(a):
                    continue
                view.run(a)                       # tool actions returned by step (LLM God style)
            out.events.extend(view.events)        # tool events happened before the powers
            for a in actions:
                if not is_power(a):
                    continue
                res = apply_power(state, a, self.cfg, self.memory, max_message_chars=self.max_message_chars,
                                  winners_leave=self.winners_leave)
                out.results.append(res)
                out.events.extend(res.events)
            out.ceiling_hit = view.exhausted
            return out
        finally:
            state.turn_index = slot
