"""LLM God (WP4): INTERFACE STUB ONLY. Not used in the demo.

Planned behaviour (ARCHITECTURE_v2 section 8): prompt set `god/<set>`, JSON tool-call output parsed into
GodActions, free observation via GodView up to the safety ceiling (each tool result is fed back
in a follow-up call), budget enforced by adversary.powers through GodRunner. Wakes at round start,
or every `wake_every_turns` agent turns.
"""

from __future__ import annotations

from typing import Any

from swarm.adversary.base import turns_into_round
from swarm.adversary.god_tools import GodView
from swarm.config import GodCfg
from swarm.schema import GameState, GodAction


class LLMGod:
    name = "llm"

    def __init__(self, cfg: GodCfg, client: Any = None, prompts: Any = None, *, seed: int = 0) -> None:
        self.cfg, self.client, self.prompts, self.seed = cfg, client, prompts, seed

    def wakes_at(self, state: GameState) -> bool:
        t = turns_into_round(state)
        every = self.cfg.wake_every_turns
        return t == 0 if every is None else t % every == 0

    async def step(self, state: GameState, view: GodView) -> list[GodAction]:
        raise NotImplementedError("LLMGod is a stub; use god.kind=scripted for the demo")


def make_god(cfg: GodCfg, *, seed: int = 0, client: Any = None, prompts: Any = None):
    """god.kind -> God instance (None for kind=none)."""
    from swarm.adversary.scripted_god import ScriptedGod

    if cfg.kind == "none":
        return None
    if cfg.kind == "scripted":
        return ScriptedGod(cfg, seed=seed)
    return LLMGod(cfg, client, prompts, seed=seed)
