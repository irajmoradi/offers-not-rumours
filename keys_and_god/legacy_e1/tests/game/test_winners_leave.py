"""game.winner_policy = "sit_out": winners leave the game (rules.departed), in both turn modes.

The winner takes no more turns, its score is locked, its keys leave with it, actions aimed at it fail
with a "left" notice, every other agent is told, it drops out of the players list, the prompt states
the rule only under sit_out, and the game still ends when every remaining agent is done.
"""

from __future__ import annotations

import pytest

from swarm.agents.llm_agent import system_context
from swarm.game import rules
from swarm.game.engine import play_game
from swarm.game.resolve import resolve_turn
from swarm.prompts import load_prompt_set
from swarm.runner.run import load_experiment
from swarm.schema import DefectAction, GiveKey, Noop, SendMessage

from .helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg

MODES = ["sequential", "simultaneous"]
LEFT_DAN = "dan has won and left the game; it can no longer trade."

# FOUR: ann ([1,2] needs [3,5]), bob ([3,4] needs [1,6]), cat ([5,6] needs [2,4]), dan ([7,8] needs [1,3]).
# Fixed order ann, bob, cat, dan. Round 1: ann and bob give dan its two needs -> dan wins at bob's
# slot. cat then tries to give dan a key; dan itself scripted to give keys away (must never happen).
# Round 2: cat defects on dan, ann messages dan, bob gives dan a key: all aimed at a departed agent.
SCRIPT = {
    (1, "ann"): [GiveKey(to="dan", info_id=1)],
    (1, "bob"): [GiveKey(to="dan", info_id=3)],
    (1, "cat"): [GiveKey(to="dan", info_id=6)],
    (1, "dan"): [GiveKey(to="cat", info_id=1), SendMessage(to="cat", text="bye")],
    (2, "ann"): [SendMessage(to="dan", text="still there?")],
    (2, "bob"): [GiveKey(to="dan", info_id=4)],
    (2, "cat"): [DefectAction(target="dan", info_id=7)],
    (2, "dan"): [GiveKey(to="ann", info_id=7)],
    (3, "dan"): [GiveKey(to="ann", info_id=8)],
}


def _play(mode, policy="sit_out", rounds=3):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": rounds, "winner_policy": policy})
    dec = Scripted(SCRIPT)
    out, eng = play_game(cfg, fresh_game(FOUR), dec)
    return cfg, dec, out, eng


def _results(eng, actor):
    return [(e.round, e.data.action.kind, e.data.ok, e.data.reason) for e in eng.sink.events
            if e.type == "action_result" and e.actor == actor]


@pytest.mark.parametrize("mode", MODES)
def test_winner_leaves_and_takes_no_more_turns(mode):
    _, dec, out, eng = _play(mode)
    assert out.per_agent["dan"].won_round == 1
    turns = [e.round for e in eng.sink.events if e.type == "turn" and e.actor == "dan"]
    if mode == "sequential":
        assert turns == []                         # dan's r1 slot comes after bob's: already gone
        assert _results(eng, "dan") == []
    else:
        # chosen at s0 before it won; resolves after bob's gift -> every action fails actor_left
        assert turns == [1]
        assert _results(eng, "dan") == [(1, "give_key", False, "actor_left"),
                                        (1, "send_message", False, "actor_left")]
        dan_notices = [e.data.notice for e in eng.sink.events if e.type == "notice" and e.data.notice.agent == "dan"]
        # no conflict notice; one "left" notice listing the actions that did not happen
        assert [n.kind for n in dan_notices] == ["gift_received", "gift_received", "won", "left"]
        assert dan_notices[-1].text == rules.midturn_text(SCRIPT[(1, "dan")])
        assert dan_notices[-1].text == ("You won and left the game during this turn, so these actions did "
                                        "not happen: give_key of info #1 to cat; message to cat.")
    assert not [o for o in dec.obs_of("dan") if o.round >= 2]


@pytest.mark.parametrize("mode", MODES)
def test_score_locked_and_keys_leave_with_winner(mode):
    _, _, out, eng = _play(mode)
    dan = eng.state.agents["dan"]
    assert sorted(k.info_id for k in dan.keys) == [1, 3, 7, 8]           # nothing given away, nothing taken
    assert dan.burned == [] and dan.lost == []
    assert out.per_agent["dan"].score == 1.0
    assert [s.scores["dan"] for s in (e.data for e in eng.sink.events if e.type == "round_end")] == [1.0] * 3
    assert not [e for e in eng.sink.events if e.type == "key_transfer" and e.round >= 2
                and "dan" in (e.data.from_agent, e.data.to_agent)]
    assert out.welfare == pytest.approx(sum(o.score for o in out.per_agent.values()) / 4)
    assert out.win_rate == pytest.approx(0.25)


@pytest.mark.parametrize("mode", MODES)
def test_actions_aimed_at_departed_agent_fail_with_notice(mode):
    _, dec, _, eng = _play(mode)
    assert (1, "give_key", False, "recipient_left") in _results(eng, "cat")
    assert (2, "defect", False, "target_left") in _results(eng, "cat")
    assert _results(eng, "ann")[-1] == (2, "send_message", False, "recipient_left")
    assert _results(eng, "bob")[-1] == (2, "give_key", False, "recipient_left")
    assert sorted(k.info_id for k in eng.state.agents["bob"].keys) == [4]   # bob kept its key 4
    assert eng.state.agents["cat"].lost == []                               # no wrong-defect penalty
    assert not [m for m in eng.state.messages if m.receiver == "dan"]
    # cat observes in r2 (its r1 failure) and r3 (its r2 failures): one "left" notice per failed action,
    # on top of the departure announcement; no extra simultaneous-mode conflict notice
    cat = [n for o in dec.obs_of("cat") for n in o.notices]
    assert [n.text for n in cat if n.kind == "system"] == []
    left = [n.text for n in cat if n.kind == "left"]
    assert left.count(LEFT_DAN) == 1
    assert LEFT_DAN + " Your give_key of info #6 did not happen (you keep the key)." in left
    assert LEFT_DAN + " Your defect had no effect (no penalty)." in left
    ann = [n.text for o in dec.obs_of("ann") for n in o.notices]
    assert LEFT_DAN + " Your message was not delivered." in ann


@pytest.mark.parametrize("mode", MODES)
def test_everyone_is_told_and_players_list_drops_winner(mode):
    _, dec, _, eng = _play(mode)
    for a in ("ann", "bob", "cat"):
        told = False                     # sequential: cat's r1 slot is after dan's win, so it is told in r1
        for o in dec.obs_of(a):
            told = told or LEFT_DAN in [n.text for n in o.notices if n.kind == "left"]
            assert ("dan" in o.pinned.peers) == (not told)
            assert len(o.pinned.peers) == (3 if not told else 2)
        assert told
        assert [n.text for o in dec.obs_of(a) for n in o.notices].count(LEFT_DAN) == 1
    first_told = {a: next(o.round for o in dec.obs_of(a) if LEFT_DAN in [n.text for n in o.notices])
                  for a in ("ann", "bob", "cat")}
    assert first_told == ({"ann": 2, "bob": 2, "cat": 1} if mode == "sequential" else {"ann": 2, "bob": 2, "cat": 2})
    won = next(e.data.notice for e in eng.sink.events if e.type == "notice" and e.data.notice.kind == "won")
    assert won.agent == "dan" and "You have left the game" in won.text


@pytest.mark.parametrize("mode", MODES)
def test_continue_default_unchanged(mode):
    cfg, dec, out, eng = _play(mode, policy="continue")
    assert cfg.game.winner_policy == "continue" == run_cfg().game.winner_policy
    assert not [e for e in eng.sink.events if e.type == "notice" and e.data.notice.kind == "left"]
    assert (2, "defect", False, "target_won") in _results(eng, "cat")
    assert (2, "give_key", True, None) in _results(eng, "bob")              # winners still receive
    assert [o.round for o in dec.obs_of("dan")] == [1, 2, 3]
    assert all("dan" in o.pinned.peers for a in ("ann", "bob", "cat") for o in dec.obs_of(a))


def test_mid_turn_win_ends_the_turn():
    """A defect that mints the actor's last need makes it win and leave: the rest of its turn fails."""
    cfg = run_cfg(game={"winner_policy": "sit_out", "defect_gain": "mint", "actions_per_turn": 2}).game
    st = mk_state({"x": ([1], [2]), "y": ([3], [2]), "z": ([2], [3])})
    res = resolve_turn(st, "x", [DefectAction(target="y", info_id=2), GiveKey(to="z", info_id=1)], None, cfg)
    assert res.results == [(True, None), (False, "actor_left")]
    assert st.agents["x"].won_round == 1 and sorted(k.info_id for k in st.agents["x"].keys) == [1, 2]
    assert sorted(k.info_id for k in st.agents["z"].keys) == [2]          # the give did not execute
    assert not [e for e in res.events if e.data.type == "key_transfer" and e.data.cause == "give"]
    left = [n for n in st.notices if n.kind == "left"]
    assert {n.agent for n in left if n.text == rules.left_text("x")} == {"y", "z"}
    assert [n.text for n in left if n.agent == "x"] == [rules.midturn_text([GiveKey(to="z", info_id=1)])]


@pytest.mark.parametrize("mode", MODES)
def test_give_before_last_need_then_win_mid_turn(mode):
    """Generosity must come first: give a spare, then complete the last need (mint) -> both happen;
    a give AFTER the completing action does not execute (engine path, both turn modes)."""
    spec = {"x": ([1, 4], [2]), "y": ([3], [2]), "z": ([2], [3, 4])}
    script = {(1, "x"): [GiveKey(to="z", info_id=4), DefectAction(target="y", info_id=2)],
              (2, "x"): [GiveKey(to="z", info_id=1)]}
    cfg = run_cfg(game={"n_agents": 3, "turn_mode": mode, "max_rounds": 2, "winner_policy": "sit_out"})
    out, eng = play_game(cfg, fresh_game(spec), Scripted(script))
    assert _results(eng, "x") == [(1, "give_key", True, None), (1, "defect", True, None)]
    assert out.per_agent["x"].won_round == 1 and 4 in [k.info_id for k in eng.state.agents["z"].keys]
    # same turn, reversed order: the give after the completing defect fails actor_left
    script2 = {(1, "x"): [DefectAction(target="y", info_id=2), GiveKey(to="z", info_id=4)]}
    out2, eng2 = play_game(cfg, fresh_game(spec), Scripted(script2))
    assert _results(eng2, "x") == [(1, "defect", True, None), (1, "give_key", False, "actor_left")]
    assert 4 in [k.info_id for k in eng2.state.agents["x"].keys]
    assert out2.per_agent["x"].score == 1.0
    x_left = [e.data.notice.text for e in eng2.sink.events if e.type == "notice"
              and e.data.notice.agent == "x" and e.data.notice.kind == "left"]
    assert x_left == [rules.midturn_text([GiveKey(to="z", info_id=4)])]


def test_steal_never_takes_from_departed_winner():
    cfg = run_cfg(game={"winner_policy": "sit_out", "defect_gain": "steal"}).game
    st = mk_state({"a_won": ([2, 5], [5]), "x": ([1], [2]), "y": ([3], [2]), "z": ([2], [3])})
    st.agents["a_won"].won_round = 1
    _, evs = rules.apply(st, "x", DefectAction(target="y", info_id=2), cfg)
    tr = [e.data for e in evs if e.data.type == "key_transfer"]
    assert [(t.from_agent, t.to_agent) for t in tr] == [("z", "x")]
    assert sorted(k.info_id for k in st.agents["a_won"].keys) == [2, 5]


def test_departed_noop_and_continue_policy_rules():
    st = mk_state({"w": ([1], [1]), "x": ([2], [3])})
    st.agents["w"].won_round = 1
    sit = run_cfg(game={"winner_policy": "sit_out"}).game
    cont = run_cfg().game
    assert rules.departed(st, "w", sit) and not rules.departed(st, "w", cont) and not rules.departed(st, "x", sit)
    assert rules.apply(st, "w", Noop(), sit, copy=True)[1][0].data.reason == "actor_left"
    assert rules.apply(st, "w", Noop(), cont, copy=True)[1][0].data.ok
    assert rules.apply(st, "x", SendMessage(to="w", text="hi"), cont, copy=True)[1][0].data.ok


@pytest.mark.parametrize("mode", MODES)
def test_game_ends_when_all_remaining_agents_done(mode):
    # r1: ann gives bob its need (bob wins, leaves with keys 1 and 2), cat (no needs: done) gives ann
    # its need -> ann wins. Everyone done after round 1.
    spec = {"ann": ([1], [2]), "bob": ([2], [1]), "cat": ([2], [])}
    script = {(1, "ann"): [GiveKey(to="bob", info_id=1)], (1, "cat"): [GiveKey(to="ann", info_id=2)]}
    cfg = run_cfg(game={"n_agents": 3, "turn_mode": mode, "max_rounds": 6, "winner_policy": "sit_out"})
    out, _ = play_game(cfg, fresh_game(spec), Scripted(script))
    assert (out.end_reason, out.rounds_played) == ("all_done", 1)
    assert {a: o.won_round for a, o in out.per_agent.items()} == {"ann": 1, "bob": 1, "cat": None}
    assert out.per_agent["ann"].score == out.per_agent["bob"].score == 1.0


@pytest.mark.parametrize("policy", ["continue", "sit_out"])
def test_prompt_states_rule_only_under_sit_out(policy):
    cfg = run_cfg(game={"winner_policy": policy})
    text = load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))
    assert (rules.LEFT_RULE in text) == (policy == "sit_out")
    assert (rules.MIDTURN_RULE in text) == (policy == "sit_out")
    assert ("give them before you complete your last need" in text) == (policy == "sit_out")
    assert ("leave the game" in text) == (policy == "sit_out")


def test_winners_leave_config():
    cfg, _ = load_experiment("configs/winners_leave_sim.yaml")
    base, _ = load_experiment("configs/grid_oss_sim_k2.yaml")
    assert cfg.game.winner_policy == "sit_out" and base.game.winner_policy == "continue"
    assert cfg.game.model_dump(exclude={"winner_policy"}) == base.game.model_dump(exclude={"winner_policy"})
    assert cfg.god == base.god and cfg.roles == base.roles and cfg.config_hash() != base.config_hash()
