"""Code-review regressions for game/engine.py (plan/CODE_REVIEW.md)."""

from __future__ import annotations

import asyncio

from swarm.game.engine import GameEngine
from swarm.llm.base import BudgetExceeded
from swarm.memory import MemoryStore
from swarm.schema import AgentOutput, MemorySink, Noop, turn_index

from .helpers import FOUR, fresh_game, run_cfg


class _Budgeted:
    """ann raises BudgetExceeded at once; the others are slow LLM calls that would keep running."""

    def __init__(self, agent: str, started: list[str], finished: list[str]) -> None:
        self.agent, self.started, self.finished = agent, started, finished

    async def act(self, obs, **_kw) -> AgentOutput:
        self.started.append(self.agent)
        if self.agent == "ann":
            raise BudgetExceeded("run", 1.0, 0.1, 1.0)
        await asyncio.sleep(0.2)
        self.finished.append(self.agent)
        return AgentOutput(actions=[Noop()])


def test_simultaneous_failure_cancels_sibling_turns():
    """Review: asyncio.gather does not cancel the other act() calls when one raises, so after
    BudgetExceeded they kept running (and spending) after game_end. They must be cancelled."""
    cfg = run_cfg(game={"turn_mode": "simultaneous", "max_rounds": 3})
    started: list[str] = []
    finished: list[str] = []
    agents = {a: _Budgeted(a, started, finished) for a in FOUR}

    async def main() -> tuple[str, list[asyncio.Task]]:
        eng = GameEngine(cfg, fresh_game(FOUR), agents, sink=MemorySink())
        out = await eng.run()
        await asyncio.sleep(0.3)                 # give leaked tasks time to finish
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return out.end_reason, others

    end_reason, leftover = asyncio.run(main())
    assert end_reason == "budget_exceeded"
    assert sorted(started) == sorted(FOUR)
    assert finished == [] and leftover == []


def test_empty_ledgers_dict_is_used_not_replaced():
    """Review: `ledgers or {...}` dropped a caller's (still empty) dict, so it never filled up."""
    cfg = run_cfg(game={"max_rounds": 1})
    ledgers: dict = {a: [] for a in FOUR}
    passed: dict = {}
    eng = GameEngine(cfg, fresh_game(FOUR), lambda obs, g: AgentOutput(actions=[Noop()]), ledgers=passed)
    assert eng.ledgers is passed
    eng2 = GameEngine(cfg, fresh_game(FOUR), lambda obs, g: AgentOutput(actions=[Noop()]), ledgers=ledgers)
    asyncio.run(eng2.run())
    assert all(ledgers[a] for a in FOUR)


def test_budget_stop_closes_the_partial_round():
    """Review #11: a BudgetExceeded mid-round still emits that round's round_end and snapshot."""
    cfg = run_cfg(game={"max_rounds": 5})
    calls = {"n": 0}

    def decide(obs, g):
        calls["n"] += 1
        if obs.round == 2 and calls["n"] > 5:          # second agent of round 2
            raise BudgetExceeded("run", 1.0, 0.1, 1.0)
        return AgentOutput(actions=[Noop()])
    eng = GameEngine(cfg, fresh_game(FOUR), decide, sink=MemorySink())
    out = asyncio.run(eng.run())
    ends = [e.round for e in eng.sink.events if e.type == "round_end"]
    assert out.end_reason == "budget_exceeded" and out.rounds_played == 2
    assert ends == [1, 2] and [s.round for s in eng.snapshots] == [0, 1, 2]
    assert eng.sink.events[-1].type == "game_end"


def test_one_god_stamp_convention_god_and_hook_paths_agree():
    """Review #7: GodRunner via `god=` now stamps what God delivers at S - 1, like the hook path the
    real runner used, so a forged DM is in the victim's inbox at the wake slot (simultaneous: the
    same round) on both paths, with identical event streams."""
    from swarm.adversary.base import GodRunner
    from swarm.adversary.scripted_god import ScriptedGod
    from swarm.runner.god_adapter import _SlotGod, make_god_hook

    cfg = run_cfg(game={"turn_mode": "simultaneous", "max_rounds": 3},
                  god={"kind": "scripted", "start_round": 2, "budget": {"k": 1},
                       "schedule": [{"round": 2, "power": "message_fraud", "args": {"text": "psst"}}]})

    def play(path):
        st = fresh_game(FOUR)
        if path == "god":
            kw = {"god": GodRunner(ScriptedGod(cfg.god, seed=1), cfg.god, MemoryStore(cfg.reasoning))}
        else:
            slot = _SlotGod(ScriptedGod(cfg.god, seed=1))
            runner = GodRunner(slot, cfg.god, MemoryStore(cfg.reasoning))
            runner.init_game(st)
            kw = {"god_hook": make_god_hook(runner, slot)}
        eng = GameEngine(cfg, st, lambda obs, g: AgentOutput(actions=[Noop()]), sink=MemorySink(), **kw)
        asyncio.run(eng.run())
        return eng
    a, b = play("god"), play("hook")
    assert [e.model_dump_json() for e in a.sink.events] == [e.model_dump_json() for e in b.sink.events]
    forged = next(m for m in a.state.messages if m.forged)
    assert forged.turn_index == turn_index(2, 0, 4) - 1
    seen = [e.data.obs for e in a.sink.events if e.type == "turn" and e.data.agent == forged.receiver]
    assert any(m.text == "psst" for o in seen if o.round == 2 for m in o.inbox)


def test_engine_ledger_skips_grep_and_numbers_per_turn():
    """R5: the engine is the one ledger builder; it renders what real runs always rendered
    (no own-action item for grep_log, own actions numbered 0, 1, ... per turn)."""
    from swarm.schema import GiveKey, GrepLog, SendMessage

    cfg = run_cfg(game={"max_rounds": 1})

    def decide(obs, g):
        if obs.agent == "ann":
            return AgentOutput(actions=[GrepLog(pattern="x"), GiveKey(to="bob", info_id=1),
                                        SendMessage(to="bob", text="hi")])
        return AgentOutput(actions=[Noop()])
    eng = GameEngine(cfg, fresh_game(FOUR), decide, sink=MemorySink())
    asyncio.run(eng.run())
    own = [it for it in eng.ledgers["ann"] if it.kind == "own_action"]
    assert [it.item_id for it in own] == ["ann:1:own_action:0", "ann:1:own_action:1"]
    assert "searched" not in " ".join(it.text for it in own)
