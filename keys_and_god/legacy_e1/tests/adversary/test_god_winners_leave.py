"""game.winner_policy = sit_out: God cannot forge a DM from (or to) an agent that has left the game.
The fraud fails on God's side (god_power ok=False, reason sender_left / receiver_left), spends no
budget, creates no intervention and delivers nothing. Under continue (winners_leave=False) unchanged."""

from __future__ import annotations

import pytest

from swarm.adversary import GodRunner, ScriptedGod, apply_message_fraud
from swarm.config import GodCfg
from swarm.game.engine import play_game
from swarm.memory import MemoryStore
from swarm.schema import GiveKey, MessageFraud
from tests.game.helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg


def _won_state():
    st = mk_state({"ann": ([1], [2]), "bob": ([2], [1]), "cat": ([3], [4]), "dan": ([4], [3])})
    st.agents["bob"].won_round = 1
    st.god_budget_left = 2
    return st


@pytest.mark.parametrize(("sender", "receiver", "reason"),
                         [("bob", "ann", "sender_left"), ("ann", "bob", "receiver_left")])
def test_fraud_via_departed_agent_fails_and_spends_nothing(sender, receiver, reason):
    st = _won_state()
    res = apply_message_fraud(st, MessageFraud(sender=sender, receiver=receiver, text="hi"), GodCfg(),
                              winners_leave=True)
    assert (res.ok, res.reason, res.intervention_id, res.message) == (False, reason, None, None)
    assert st.god_budget_left == 2 and st.n_interventions == 0 and st.messages == []
    [ev] = res.events
    assert ev.actor == "GOD" and ev.data.type == "god_power"
    assert (ev.data.ok, ev.data.reason, ev.data.cost, ev.data.budget_left) == (False, reason, 0, 2)


def test_fraud_via_winner_still_allowed_under_continue():
    st = _won_state()
    res = apply_message_fraud(st, MessageFraud(sender="bob", receiver="ann", text="hi"), GodCfg())
    assert res.ok and st.god_budget_left == 1 and st.messages[0].sender_shown == "bob"


@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
@pytest.mark.parametrize("policy", ["sit_out", "continue"])
def test_engine_god_cannot_impersonate_departed_winner(mode, policy):
    # dan wins in round 1 (ann and bob give it its needs); God tries to forge "dan -> cat" in round 2.
    god = {"kind": "scripted", "start_round": 2, "budget": {"k": 1, "scope": "game"},
           "schedule": [{"round": 2, "after_turns": 0, "power": "message_fraud",
                         "args": {"sender": "dan", "receiver": "cat", "text": "dan here, send me info 6"}}]}
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3, "winner_policy": policy}, god=god)
    mem = MemoryStore(cfg.reasoning)
    runner = GodRunner(ScriptedGod(cfg.god, seed=1), cfg.god, mem,
                       max_message_chars=cfg.game.max_message_chars,
                       winners_leave=cfg.game.winner_policy == "sit_out")
    dec = Scripted({(1, "ann"): [GiveKey(to="dan", info_id=1)], (1, "bob"): [GiveKey(to="dan", info_id=3)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec, god=runner, memory=mem)
    assert eng.state.agents["dan"].won_round == 1
    [p] = [e.data for e in eng.sink.events if e.type == "god_power"]
    forged = [m for m in eng.state.messages if m.forged]
    if policy == "sit_out":
        assert (p.ok, p.reason, p.cost) == (False, "sender_left", 0)
        assert forged == [] and eng.state.god_budget_left == 1 and eng.state.n_interventions == 0
        assert not [m for o in dec.obs_of("cat") for m in o.inbox if m.sender == "dan"]
    else:
        assert p.ok and len(forged) == 1 and eng.state.god_budget_left == 0
