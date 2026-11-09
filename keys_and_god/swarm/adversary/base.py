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

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from swarm.adversary.god_tools import GodView
from swarm.adversary.powers import PowerResult, apply_power, init_budget, is_power, refresh_budget
from swarm.config import GodCfg
from swarm.memory import MemoryStore
from swarm.schema import EventBody, GameState, GodAction

log = logging.getLogger("swarm.god")


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
                 winners_leave: bool = False, framer: Any = None) -> None:
        self.god, self.cfg, self.memory = god, cfg, memory
        # game.framing (facility): God's DM texts are delivered in portal terms (powers framer=) and each
        # delivered text is checked for game vocabulary (framing_hits: [(token, snippet)]; warn + count)
        self.framer = framer
        self.framing_hits: list[tuple[str, str]] = []
        self.guard_drops = 0                    # scripted God texts dropped by the framing guardrail
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
        if getattr(self.god, "interactive", False):
            return await self._wake_interactive(state, view, slot)
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
                res = self._apply(state, a)
                out.results.append(res)
                out.events.extend(res.events)
            out.ceiling_hit = view.exhausted
            return out
        finally:
            state.turn_index = slot

    def _apply(self, state: GameState, action: GodAction) -> PowerResult:
        if self.framer is not None and not getattr(self.god, "interactive", False):
            # scripted God under game.framing: its fixed templates pass the same output guardrail as the LLM
            # God's texts (no rewrite turn for a template: a hit drops the power, nothing is spent)
            from swarm.adversary.powers import _fail
            from swarm.schema import MessageFraud, RevealNeeds

            text = getattr(action, "text", "") or ""
            if isinstance(action, RevealNeeds):
                text = text.replace("{target}", "").replace("{items}", "")
            if isinstance(action, (MessageFraud, RevealNeeds)) and text.strip():
                bad = self.framer.god_guard(text)
                if bad:
                    log.warning("framing: scripted God text dropped (%s)", bad)
                    self.guard_drops += 1
                    return _fail(state, action, action.kind, "guard_drop")
        res = apply_power(state, action, self.cfg, self.memory, max_message_chars=self.max_message_chars,
                          winners_leave=self.winners_leave, framer=self.framer)
        if self.framer is not None and res.ok:
            from swarm.framing.facility import banned_hits

            texts = {d.data.message.text for d in res.events if getattr(d.data, "type", "") == "message"}
            for t in texts:
                hits = banned_hits(t)
                if hits:
                    log.warning("framing: God text has game vocabulary %s", [h[0] for h in hits])
                self.framing_hits += hits
        return res

    async def _wake_interactive(self, state: GameState, view: GodView, slot: int) -> WakeResult:
        """LLM God: it applies each power inside its own loop through `view.power` (so it reads the
        result and the budget left before its next step). Same stamp convention: it decides at S and
        every power delivers stamped S - 1; tool / thought / power events stay in the order they
        happened. The budget, enabled flags and validation are adversary.powers', as for ScriptedGod."""
        out = WakeResult()

        def power(action: GodAction) -> PowerResult:
            if not is_power(action):
                raise TypeError(f"not a God power: {action.kind}")
            state.turn_index = slot - 1
            try:
                res = self._apply(state, action)
            finally:
                state.turn_index = slot
            out.actions.append(action)
            out.results.append(res)
            view.events.extend(res.events)
            return res

        view.power = power
        try:
            extra = await self.god.step(state, view)
            for a in extra:                                # an interactive God may still return actions
                if is_power(a):
                    power(a)
                else:
                    state.turn_index = slot - 1
                    try:
                        view.run(a)
                    finally:
                        state.turn_index = slot
                    out.actions.append(a)
            out.events.extend(view.events)
            out.ceiling_hit = view.exhausted
            return out
        finally:
            view.power = None
            state.turn_index = slot
