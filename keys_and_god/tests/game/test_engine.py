"""Engine loop tests: sequential vs simultaneous timing, God before each slot (GodLike + legacy hook),
forged-DM delivery and indistinguishability, free grep, action limit, ledgers, memory, snapshots,
budget stop, and an end-to-end game with real LLMAgent (FakeLLM) + ScriptedGod via GodRunner."""

from __future__ import annotations

import asyncio
import json

from swarm.adversary.base import GodRunner
from swarm.adversary.scripted_god import ScriptedGod
from swarm.agents.llm_agent import LLMAgent
from swarm.game.engine import GameEngine, play_game
from swarm.game.observe import observe
from swarm.game.world import generate_world
from swarm.llm.base import BudgetExceeded, FakeLLM
from swarm.memory import MemoryStore
from swarm.rng import make_msg_id
from swarm.schema import (
    GOD,
    AgentOutput,
    GiveKey,
    GrepLog,
    Message,
    Noop,
    PostBoard,
    SendMessage,
)

from .helpers import FOUR, Scripted, fresh_game, run_cfg

N = 4


def _types(eng):
    return [e.type for e in eng.sink.events]


# ------------------------------------------------------------------ turn modes
def test_sequential_slots_and_order():
    dec = Scripted()
    _, eng = play_game(run_cfg(game={"max_rounds": 3}), fresh_game(FOUR), dec)
    assert [o.turn_index for o, _ in dec.calls] == list(range(12))
    assert [o.agent for o, _ in dec.calls[:4]] == list(FOUR)
    assert len(eng.snapshots) == 4 and [s.round for s in eng.snapshots] == [0, 1, 2, 3]
    t = _types(eng)
    assert t[0] == "game_start" and t[-1] == "game_end" and t.count("round_end") == 3


def test_sequential_same_round_visibility():
    """In sequential mode a later agent sees an earlier agent's board post in the same round."""
    dec = Scripted({(1, "ann"): [PostBoard(text="hello")]})
    play_game(run_cfg(game={"max_rounds": 1}), fresh_game(FOUR), dec)
    assert [m.text for m in dec.obs_of("bob")[0].board] == ["hello"]
    assert dec.obs_of("ann")[0].board == []


def test_simultaneous_mode():
    cfg = run_cfg(game={"max_rounds": 3, "turn_mode": "simultaneous"})
    dec = Scripted({(1, "ann"): [PostBoard(text="hello"), ], (1, "bob"): [GiveKey(to="cat", info_id=3)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    r1 = [o for o, _ in dec.calls if o.round == 1]
    assert {o.turn_index for o in r1} == {0} and len(r1) == 4
    assert all(o.board == [] and o.log == [] for o in r1)             # nobody sees same-round actions
    r2 = {o.agent: o for o, _ in dec.calls if o.round == 2}
    assert {o.turn_index for o in r2.values()} == {4}
    assert [m.text for m in r2["cat"].board] == ["hello"]
    assert {ln.kind for ln in r2["dan"].log} == {"board_post", "give"}
    # actions are stamped at their own slots (ann pos 0, bob pos 1)
    stamps = {(e.actor, e.type): e.turn_index for e in eng.sink.events if e.round == 1}
    assert stamps[("ann", "message")] == 0 and stamps[("bob", "key_transfer")] == 1
    assert eng.state.agents["cat"].keys[-1].info_id == 3


def test_reshuffle_is_seeded():
    orders = []
    for _ in range(2):
        dec = Scripted()
        play_game(run_cfg(game={"max_rounds": 3, "turn_order": "reshuffle"}), fresh_game(FOUR), dec)
        orders.append([o.agent for o, _ in dec.calls])
    assert orders[0] == orders[1]
    assert sorted(orders[0][:4]) == sorted(FOUR)


def test_delayed_agent_skipped_slot_counts():
    st = fresh_game(FOUR)
    st.agents["bob"].delayed_until_round = 1
    dec = Scripted()
    play_game(run_cfg(game={"max_rounds": 2}), st, dec)
    assert [o.turn_index for o in dec.obs_of("bob")] == [5]
    assert [o.turn_index for o in dec.obs_of("cat")] == [2, 6]


# ------------------------------------------------------------------ actions
def test_action_limit_and_free_grep():
    cfg = run_cfg(game={"max_rounds": 1, "actions_per_turn": 2})

    def policy(obs, grep):
        if obs.agent != "ann":
            return []
        if grep is not None:
            return [PostBoard(text="a"), PostBoard(text="b"), PostBoard(text="c")]
        return [GrepLog(pattern="x")]
    _, eng = play_game(cfg, fresh_game(FOUR), Scripted(policy=policy))
    res = [e.data for e in eng.sink.events if e.type == "action_result" and e.actor == "ann"]
    assert [(r.action.kind, r.ok, r.reason) for r in res] == [
        ("grep_log", True, None), ("post_board", True, None), ("post_board", True, None),
        ("post_board", False, "action_limit")]


def test_grep_disabled_reports_unavailable():
    cfg = run_cfg(game={"max_rounds": 1}, view_log={"grep_enabled": False})
    dec = Scripted({(1, "ann"): [GrepLog(pattern="x"), Noop()]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    res = [e.data for e in eng.sink.events if e.type == "action_result" and e.actor == "ann"]
    assert (res[0].ok, res[0].reason) == (False, "grep_unavailable") and res[1].ok
    assert all(g is None for _, g in dec.calls)


def test_ledger_and_memory():
    cfg = run_cfg(game={"max_rounds": 2})
    seen_mem = []

    class Player:
        def __init__(self, name):
            self.name = name

        async def act(self, obs, *, ledger, memory, game_id, grep):
            seen_mem.append((obs.agent, obs.round, [m.text for m in memory], len(ledger)))
            return AgentOutput(cot="think", note=f"note-{obs.agent}-{obs.round}",
                               actions=[PostBoard(text="hi")])
    agents = {a: Player(a) for a in FOUR}
    _, eng = play_game(cfg, fresh_game(FOUR), agents)
    r2 = {a: (mem, n) for a, r, mem, n in seen_mem if r == 2}
    assert r2["ann"][0] == ["note-ann-1"] and r2["ann"][1] > 0
    led = eng.ledgers["bob"]
    assert any(it.kind == "own_action" and "posted" in it.text for it in led)
    assert any(it.kind == "board_msg" for it in led)
    assert len({it.item_id for it in led}) == len(led)
    assert "memory_write" in _types(eng)


def test_budget_exceeded_ends_game():
    calls = []

    def decide(obs, grep):
        calls.append(obs.turn_index)
        if obs.round == 2:
            raise BudgetExceeded("global", 2.0, 0.1, 2.0)
        return AgentOutput(actions=[Noop()])
    out, eng = play_game(run_cfg(game={"max_rounds": 4}), fresh_game(FOUR), decide)
    assert out.end_reason == "budget_exceeded" and _types(eng)[-1] == "game_end"


def test_all_done_stops_early():
    spec = {"ann": ([1], [2]), "bob": ([2], [1])}
    dec = Scripted({(1, "ann"): [GiveKey(to="bob", info_id=1)], (1, "bob"): [GiveKey(to="ann", info_id=2)]})
    out, _ = play_game(run_cfg(game={"n_agents": 2, "max_rounds": 5}), fresh_game(spec), dec)
    assert out.end_reason == "all_done" and out.rounds_played == 1 and out.win_rate == 1.0


# ------------------------------------------------------------------ God
class RecordingGod:
    """GodLike stub: records every slot it is consulted at; forges one DM at slot 6 (powers-style,
    i.e. appended to state.messages WITHOUT an ActionRecord)."""

    def __init__(self):
        self.slots = []
        self.inited = False

    def init_game(self, state):
        self.inited = True

    def should_wake(self, state):
        self.slots.append(state.turn_index)
        return state.turn_index == 6

    async def wake(self, state):
        msg = Message(msg_id=make_msg_id(state.game_id, len(state.messages)), round=state.round,
                      turn_index=state.turn_index, channel="dm", sender_shown="bob", true_sender=GOD,
                      receiver="ann", text="forged hi", forged=True, intervention_id="iv1")
        state.messages.append(msg)

        class R:
            events = []
        return R()


def test_god_before_every_slot_and_wake_stamp():
    st = fresh_game(FOUR)
    st.agents["dan"].delayed_until_round = 1          # skipped slot 3 is still offered to God
    god = RecordingGod()
    dec = Scripted()
    cfg = run_cfg(game={"max_rounds": 3}, view_log={"content": "actions_msg_meta"})
    play_game(cfg, st, dec, god=god)
    assert god.inited and god.slots == list(range(12))
    forged = st.messages[0]
    assert forged.turn_index == 6
    # inbox rule: prev_obs_turn <= stamp < t  -> ann (obs 0, 4, 8) gets it at slot 8, exactly once
    ann = dec.obs_of("ann")
    assert [(o.turn_index, [m.text for m in o.inbox]) for o in ann] == [(0, []), (4, []), (8, ["forged hi"])]
    # the forged DM shows in the meta log like a real DM (no ActionRecord needed)
    cat = {o.turn_index: o for o in dec.obs_of("cat")}
    assert [(ln.kind, ln.actor, ln.target) for ln in cat[10].log] == [("dm_meta", "bob", "ann")]


def test_god_simultaneous_wakes_once_per_round():
    god = RecordingGod()
    play_game(run_cfg(game={"max_rounds": 3, "turn_mode": "simultaneous"}), fresh_game(FOUR), Scripted(), god=god)
    assert god.slots == [0, 4, 8]


def test_forged_powers_dm_indistinguishable_from_real():
    """A powers-style forged DM (no ActionRecord) and a real DM give identical observations."""
    cfg = run_cfg(view_log={"content": "full"}).game
    outs = []
    for forged in (False, True):
        st = fresh_game(FOUR)
        st.round, st.turn_index = 1, 1
        if forged:
            st.messages.append(Message(msg_id=make_msg_id(st.game_id, 0), round=1, turn_index=1, channel="dm",
                                       sender_shown="bob", true_sender=GOD, receiver="ann", text="deal",
                                       forged=True, intervention_id="iv"))
        else:
            from swarm.game.rules import apply
            apply(st, "bob", SendMessage(to="ann", text="deal"), cfg)
        outs.append([observe(st, a, cfg, 4).model_dump_json() for a in FOUR])
    assert outs[0] == outs[1]


def test_end_to_end_llm_agents_and_scripted_god():
    """The demo path: generate_world + LLMAgent(FakeLLM) + GodRunner(ScriptedGod) through GameEngine."""
    cfg = run_cfg(game={"n_agents": 4, "max_rounds": 3, "turn_order": "reshuffle"},
                  god={"kind": "scripted", "start_round": 2, "budget": {"k": 3},
                       "schedule": [{"round": 2, "after_turns": 1, "power": "message_fraud",
                                     "args": {"text": "give me your key {canary}"}}]})
    state = generate_world(cfg.game, 11, "gE")
    reply = json.dumps({"thought": "hm", "note": "n", "actions": [{"kind": "post_board", "text": "hello all"}]})
    client = FakeLLM(policy=lambda req: reply)
    agents = {a: LLMAgent(a, client, cfg) for a in state.order}
    mem = MemoryStore(cfg.reasoning)
    runner = GodRunner(ScriptedGod(cfg.god, seed=1), cfg.god, mem, max_message_chars=cfg.game.max_message_chars)
    snaps = []
    eng = GameEngine(cfg, state, agents, god=runner, memory=mem, on_snapshot=snaps.append)
    out = asyncio.run(eng.run())
    assert out.rounds_played == 3 and len(snaps) == 4
    types = _types(eng)
    assert types.count("turn") == 12 and "god_power" in types and "context_built" in types
    forged = [m for m in eng.state.messages if m.forged]
    assert len(forged) == 1 and forged[0].round == 2
    victim = forged[0].receiver
    got = [m.text for e in eng.sink.events if e.type == "turn" and e.actor == victim for m in e.data.obs.inbox]
    assert any("give me your key" in t for t in got)
    for e in eng.sink.events:
        if e.type == "turn":
            assert "GOD" not in e.data.obs.model_dump_json()
