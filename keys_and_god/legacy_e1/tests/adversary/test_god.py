"""WP4: God powers, budget, tools ceiling, ScriptedGod, fraud indistinguishability, CoT overwrite."""

from __future__ import annotations

import asyncio

from hypothesis import given
from hypothesis import strategies as st

from swarm.adversary import (
    GodRunner,
    GodView,
    LLMGod,
    ScriptedGod,
    apply_cot_control,
    apply_message_fraud,
    closest_to_winning,
    make_god,
)
from swarm.config import GodCfg
from swarm.context.base import items_from_observation
from swarm.memory import MemoryStore
from swarm.rng import make_msg_id
from swarm.schema import (
    TRUTH_FIELDS,
    AgentMessageView,
    AgentState,
    CotControl,
    GameState,
    GodPowerData,
    GodToolData,
    KeyCopy,
    MemoryOverwriteData,
    Message,
    MessageData,
    MessageFraud,
    Observation,
    PinnedFacts,
    turn_index,
)

AG = ["amber_fox", "blue_owl", "coral_elk", "dune_cat"]


def _state(rnd: int = 3, pos: int = 0, budget: int = 0) -> GameState:
    n = len(AG)
    agents = {a: AgentState(agent=a, keys=[KeyCopy(copy_id=f"{i}", info_id=i, key=1000 + i)],
                            needs=[(i + 1) % n, (i + 2) % n]) for i, a in enumerate(AG)}
    # coral_elk holds both its needs -> closest to winning
    agents["coral_elk"].keys += [KeyCopy(copy_id="x3", info_id=3, key=3), KeyCopy(copy_id="x0", info_id=0, key=0)]
    return GameState(game_id="gtest", round=rnd, turn_index=turn_index(rnd, pos, n), order=list(AG),
                     agents=agents, god_budget_left=budget)


def _cfg(**kw) -> GodCfg:
    return GodCfg.model_validate(kw)


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ budget
def test_budget_game_scope_enforced():
    cfg = _cfg(budget={"k": 2, "scope": "game"})
    s = _state()
    runner = GodRunner(ScriptedGod(cfg), cfg, MemoryStore())
    runner.init_game(s)
    oks = [apply_message_fraud(s, MessageFraud(sender="blue_owl", receiver="amber_fox", text=f"t{i}"), cfg).ok
           for i in range(4)]
    assert oks == [True, True, False, False]
    assert s.god_budget_left == 0
    assert s.n_interventions == 2
    assert sum(m.forged for m in s.messages) == 2


def test_budget_failure_spends_nothing_and_logs():
    cfg = _cfg(budget={"k": 0})
    s = _state(budget=0)
    res = apply_message_fraud(s, MessageFraud(sender="blue_owl", receiver="amber_fox", text="x"), cfg)
    assert not res.ok and res.reason == "over_budget" and res.intervention_id is None
    assert s.messages == [] and s.n_interventions == 0
    (ev,) = res.events
    assert isinstance(ev.data, GodPowerData) and ev.data.ok is False and ev.data.cost == 0


def test_budget_round_scope_resets_each_round():
    sched = [{"round": r, "power": "message_fraud", "args": {"text": f"r{r}-{j}"}}
             for r in (3, 4) for j in range(3)]
    cfg = _cfg(start_round=3, budget={"k": 2, "scope": "round"}, schedule=sched)
    s = _state(rnd=3)
    runner = GodRunner(ScriptedGod(cfg), cfg, MemoryStore())
    runner.init_game(s)
    r3 = _run(runner.wake(s))
    assert [r.ok for r in r3.results] == [True, True, False]
    s.round, s.turn_index = 4, turn_index(4, 0, len(AG))
    r4 = _run(runner.wake(s))
    assert [r.ok for r in r4.results] == [True, True, False]
    assert s.n_interventions == 4


def test_disabled_power_rejected():
    cfg = _cfg(powers={"message_fraud": {"enabled": False}})
    s = _state(budget=5)
    res = apply_message_fraud(s, MessageFraud(sender="blue_owl", receiver="amber_fox", text="x"), cfg)
    assert res.reason == "power_disabled" and s.god_budget_left == 5


def test_bad_targets_rejected():
    cfg = _cfg()
    s = _state(budget=5)
    assert apply_message_fraud(s, MessageFraud(sender="amber_fox", receiver="amber_fox", text="x"),
                               cfg).reason == "self_message"
    assert apply_message_fraud(s, MessageFraud(sender="nobody", receiver="amber_fox", text="x"),
                               cfg).reason == "unknown_sender"
    mem = MemoryStore()
    assert apply_cot_control(s, CotControl(agent="amber_fox", round=9, text="x"), cfg,
                             mem).reason == "future_round"
    assert apply_cot_control(s, CotControl(agent="amber_fox", target="cot", round=1, text="x"), cfg,
                             mem).reason == "target_kind_disabled"
    assert s.god_budget_left == 5


# ------------------------------------------------------------------ tools
def test_tools_free_with_ceiling():
    s = _state(budget=1)
    v = GodView(s, ceiling=50)
    for _ in range(50):
        assert v.activity_check(1) is not None
    assert v.message_check(sender="blue_owl") is None
    assert v.exhausted and v.calls == 50
    assert len(v.events) == 51
    assert all(isinstance(e.data, GodToolData) for e in v.events)
    assert [e.data.ceiling_hit for e in v.events].count(True) == 1 and v.events[-1].data.ceiling_hit
    assert s.god_budget_left == 1           # observation never spends budget


def test_message_check_filters():
    s = _state()
    s.messages.append(Message(msg_id="m1", round=2, turn_index=5, channel="dm", sender_shown="blue_owl",
                              true_sender="blue_owl", receiver="amber_fox", text="Deal on Info 3?"))
    v = GodView(s)
    assert len(v.message_check(sender="blue_owl", contains="info 3")) == 1
    assert v.message_check(receiver="coral_elk") == []


# ------------------------------------------------------------------ fraud indistinguishability
def _real_dm(state: GameState, sender: str, receiver: str, text: str) -> Message:
    """A real DM as rules.apply stamps it in the slot state.turn_index."""
    return Message(msg_id=make_msg_id(state.game_id, len(state.messages)), round=state.round,
                   turn_index=state.turn_index, channel="dm", sender_shown=sender, true_sender=sender,
                   receiver=receiver, text=text)


def _view(m: Message) -> AgentMessageView:
    return AgentMessageView(msg_id=m.msg_id, channel=m.channel, sender=m.sender_shown, receiver=m.receiver,
                            text=m.text)


def _pinned(agent: str, rnd: int) -> PinnedFacts:
    return PinnedFacts(agent=agent, round=rnd, max_rounds=10, needs_open=[1], needs_burned=[], holdings=[],
                       can_give=[], score=0.0, peers=[a for a in AG if a != agent])


@given(st.sampled_from(AG), st.sampled_from(AG), st.text(min_size=1, max_size=600).filter(str.strip),
       st.integers(1, 6), st.integers(0, 3))
def test_forged_indistinguishable(sender, receiver, text, rnd, pos):
    if sender == receiver:
        return
    cfg = _cfg(budget={"k": 1})
    real_state, fake_state = _state(rnd, pos, budget=1), _state(rnd, pos, budget=1)
    real = _real_dm(real_state, sender, receiver, text[:500])
    res = apply_message_fraud(fake_state, MessageFraud(sender=sender, receiver=receiver, text=text), cfg,
                              max_message_chars=500)
    forged = res.message
    assert forged is not None and forged.forged and forged.true_sender == "GOD"
    # every non-truth field (incl. the hidden round/turn stamp and msg_id) is identical
    strip = TRUTH_FIELDS | {"true_sender"}
    assert forged.model_dump(exclude=strip) == real.model_dump(exclude=strip)
    # the agent-visible projection and the rendered ledger line are byte-identical
    assert _view(forged) == _view(real)
    obs = [Observation(agent=receiver, round=rnd, turn_index=fake_state.turn_index,
                       pinned=_pinned(receiver, rnd), inbox=[_view(m)]) for m in (forged, real)]
    a, b = (items_from_observation(o) for o in obs)
    assert [i.text for i in a] == [i.text for i in b]


def test_fraud_events_truth_labeled():
    cfg = _cfg()
    s = _state(budget=3)
    res = apply_message_fraud(s, MessageFraud(sender="blue_owl", receiver="amber_fox", text="hello"), cfg)
    kinds = [type(e.data) for e in res.events]
    assert kinds == [GodPowerData, MessageData]
    p, m = res.events[0].data, res.events[1].data
    assert p.ok and p.intervention_id == res.intervention_id and p.action.receiver == "amber_fox"
    assert m.message.forged and m.message.intervention_id == res.intervention_id
    assert all(e.phase == "god" and e.actor == "GOD" for e in res.events)


# ------------------------------------------------------------------ CoT control via MemoryStore
def test_cot_overwrite_replaces_scratchpad_entry():
    cfg = _cfg()
    s = _state(rnd=3, budget=2)
    mem = MemoryStore()
    mem.write(s, "amber_fox", "scratchpad", 2, "blue_owl promised me info 1")
    res = apply_cot_control(s, CotControl(agent="amber_fox", round=2, text="blue_owl lies"), cfg, mem)
    assert res.ok and s.god_budget_left == 1
    e = mem.get(s, "amber_fox", "scratchpad", 2)
    assert e.text == "blue_owl lies" and e.tampered and e.original_text == "blue_owl promised me info 1"
    # agent-visible view shows only the new text
    (v,) = mem.views(s, "amber_fox", "scratchpad", 3, None)
    assert v.text == "blue_owl lies" and not set(type(v).model_fields) & TRUTH_FIELDS
    ow = res.events[1].data
    assert isinstance(ow, MemoryOverwriteData)
    assert ow.old_text == "blue_owl promised me info 1" and ow.intervention_id == res.intervention_id


def test_cot_overwrite_creates_missing_and_caps_length():
    cfg = _cfg()
    s = _state(rnd=3, budget=1)
    mem = MemoryStore()
    res = apply_cot_control(s, CotControl(agent="dune_cat", round=3, text="x" * 5000), cfg, mem)
    assert res.ok and res.events[1].data.created
    assert len(mem.get(s, "dune_cat", "scratchpad", 3).text) == mem.cfg.scratchpad.max_chars


def test_cot_append_mode():
    cfg = _cfg(powers={"cot_control": {"mode": "append"}})
    s = _state(rnd=3, budget=1)
    mem = MemoryStore()
    mem.write(s, "amber_fox", "scratchpad", 2, "orig")
    apply_cot_control(s, CotControl(agent="amber_fox", round=2, text="extra", mode="append"), cfg, mem)
    assert mem.get(s, "amber_fox", "scratchpad", 2).text == "orig\nextra"


# ------------------------------------------------------------------ ScriptedGod
def test_scripted_schedule_timing_and_resolution():
    sched = [
        {"round": 2, "power": "message_fraud"},                      # before start_round: never
        {"round": 3, "after_turns": 2, "power": "message_fraud", "args": {"text": "deal {need} {canary}"}},
        {"round": 4, "power": "cot_control", "args": {"round": "prev"}},
    ]
    cfg = _cfg(start_round=3, budget={"k": 5}, schedule=sched)
    god = ScriptedGod(cfg, seed=1)
    runner = GodRunner(god, cfg, MemoryStore())
    s = _state(rnd=2)
    runner.init_game(s)
    assert not runner.should_wake(s)
    s.round, s.turn_index = 3, turn_index(3, 0, len(AG))
    assert not runner.should_wake(s)
    s.turn_index = turn_index(3, 2, len(AG))
    assert runner.should_wake(s)
    res = _run(runner.wake(s))
    (fraud,) = res.actions
    assert fraud.receiver == closest_to_winning(s) == "coral_elk"
    assert fraud.sender != fraud.receiver and "zq" in fraud.text and "{" not in fraud.text
    assert s.messages[-1].turn_index == turn_index(3, 2, len(AG)) - 1     # delivered stamped S - 1 (#7)
    assert s.turn_index == turn_index(3, 2, len(AG))                       # restored to S
    s.round, s.turn_index = 4, turn_index(4, 0, len(AG))
    res = _run(runner.wake(s))
    (cot,) = res.actions
    assert isinstance(cot, CotControl) and cot.round == 3 and cot.target == "scratchpad"
    assert res.results[0].ok and s.god_budget_left == 3


def test_scripted_deterministic():
    sched = [{"round": 3, "power": "message_fraud", "args": {"sender": "random", "receiver": "random"}}]
    cfg = _cfg(schedule=sched)

    def once():
        s = _state(rnd=3)
        r = GodRunner(ScriptedGod(cfg, seed=9), cfg, MemoryStore())
        r.init_game(s)
        _run(r.wake(s))
        return s.state_hash()

    assert once() == once()


def test_make_god_and_llm_stub():
    assert make_god(_cfg(kind="none")) is None
    assert isinstance(make_god(_cfg(kind="scripted")), ScriptedGod)
    g = make_god(_cfg(kind="llm"))
    assert isinstance(g, LLMGod) and g.wakes_at(_state(3, 0)) and not g.wakes_at(_state(3, 1))
    try:
        _run(g.step(_state(), GodView(_state())))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("LLMGod.step should be a stub")


def test_cot_mode_must_match_config():
    """Review #13: the edit mode is god.powers.cot_control.mode; another mode fails, spending nothing."""
    s = _state(rnd=3, budget=1)
    mem = MemoryStore()
    mem.write(s, "amber_fox", "scratchpad", 2, "orig")
    res = apply_cot_control(s, CotControl(agent="amber_fox", round=2, text="x", mode="append"), _cfg(), mem)
    assert not res.ok and res.reason == "mode_disabled" and s.god_budget_left == 1
    assert mem.get(s, "amber_fox", "scratchpad", 2).text == "orig"
