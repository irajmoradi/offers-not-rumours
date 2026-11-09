"""Shared builders for game tests: tiny configs, hand-built states, scripted decision stubs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from swarm.config import RunConfig, from_dict
from swarm.schema import Action, AgentOutput, AgentState, GameState, GrepResult, KeyCopy, Observation

FAKE_ROLES = {r: {"backend": "fake"} for r in ("agent", "god", "judge", "summarizer")}


def run_cfg(game: dict[str, Any] | None = None, view_log: dict[str, Any] | None = None,
            **top: Any) -> RunConfig:
    g = {"n_agents": 4, "max_rounds": 6, "turn_order": "fixed", **(game or {})}
    if view_log:
        g["view_log"] = view_log
    top.setdefault("god", {"kind": "none"})       # engine tests inject God via god= / god_hook=
    return from_dict({"run": {"seed": 3}, "game": g, "roles": FAKE_ROLES, "context": {"tokenizer": "words"},
                      "llm": {"cache": {"mode": "off"}}, **top})


def mk_state(spec: dict[str, tuple[list[int], list[int]]], game_id: str = "gt", round: int = 1) -> GameState:
    """spec: agent -> (held info ids, needs). Key value of info i is 1000 + i.
    round=1 (default) is a state mid-round-1, for applying rules directly; use `fresh_game` for the
    engine, which (like generate_world) expects round 0 = setup and plays rounds 1, 2, ..."""
    agents = {}
    for a, (held, needs) in spec.items():
        agents[a] = AgentState(agent=a, keys=[KeyCopy(copy_id=f"{a}-{i}", info_id=i, key=1000 + i)
                                              for i in held], needs=list(needs))
    return GameState(game_id=game_id, round=round, turn_index=0, order=list(spec), agents=agents)


def fresh_game(spec: dict[str, tuple[list[int], list[int]]], game_id: str = "gt") -> GameState:
    """A hand-built round-0 (setup) state for GameEngine / play_game."""
    return mk_state(spec, game_id, round=0)


FOUR = {"ann": ([1, 2], [3, 5]), "bob": ([3, 4], [1, 6]), "cat": ([5, 6], [2, 4]), "dan": ([7, 8], [1, 3])}


Script = dict[tuple[int, str], list[Action]]   # (round, agent) -> actions


class Scripted:
    """Decision stub: plays scripted actions per (round, agent), noop otherwise; records every call."""

    def __init__(self, script: Script | None = None,
                 policy: Callable[[Observation, GrepResult | None], list[Action]] | None = None) -> None:
        self.script = script or {}
        self.policy = policy
        self.calls: list[tuple[Observation, GrepResult | None]] = []

    def __call__(self, obs: Observation, grep: GrepResult | None) -> AgentOutput:
        self.calls.append((obs, grep))
        if self.policy is not None:
            return AgentOutput(actions=self.policy(obs, grep))
        if grep is not None:
            return AgentOutput(actions=[])
        return AgentOutput(actions=list(self.script.get((obs.round, obs.agent), [])))

    def obs_of(self, agent: str) -> list[Observation]:
        return [o for o, g in self.calls if o.agent == agent and g is None]
