"""Rules table tests (give, message, defect x gain x penalty, win, winner policies) and closure fuzz."""

from __future__ import annotations

import random

import pytest

from swarm.game.rules import apply, deliver_message, is_winning, open_needs, score
from swarm.schema import (
    BurnData,
    DefectAction,
    GiveKey,
    GrepLog,
    KeyTransferData,
    Noop,
    PostBoard,
    SendMessage,
    WinData,
)

from .helpers import FOUR, mk_state, run_cfg


def g(**kw):
    return run_cfg(game=kw).game


def kinds(evs):
    return [e.data.type for e in evs]


def test_give_moves_one_copy_and_notifies():
    st = mk_state(FOUR)
    _, evs = apply(st, "ann", GiveKey(to="bob", info_id=1), g())
    assert evs[0].data.type == "action_result" and evs[0].data.ok
    assert 1 not in [k.info_id for k in st.agents["ann"].keys]
    assert 1 in [k.info_id for k in st.agents["bob"].keys]
    tr = [e.data for e in evs if isinstance(e.data, KeyTransferData)]
    assert tr[0].cause == "give" and tr[0].from_agent == "ann"
    assert st.notices[-1].agent == "bob" and "#1" in st.notices[-1].text
    assert st.history[-1].kind == "give" and st.history[-1].info_id == 1


@pytest.mark.parametrize("action,reason", [
    (GiveKey(to="bob", info_id=9), "no_such_key"),
    (GiveKey(to="ann", info_id=1), "self_target"),
    (GiveKey(to="zed", info_id=1), "unknown_agent"),
    (SendMessage(to="ann", text="hi"), "self_target"),
    (DefectAction(target="ann", info_id=3), "self_target"),
    (GrepLog(pattern="x"), "grep_handled_by_engine"),
])
def test_invalid_actions_leave_state_unchanged(action, reason):
    st = mk_state(FOUR)
    h = st.state_hash()
    _, evs = apply(st, "ann", action, g())
    assert len(evs) == 1 and not evs[0].data.ok and evs[0].data.reason == reason
    assert st.state_hash() == h


def test_copy_flag_does_not_mutate_input():
    st = mk_state(FOUR)
    h = st.state_hash()
    new, _ = apply(st, "ann", GiveKey(to="bob", info_id=1), g(), copy=True)
    assert st.state_hash() == h and new.state_hash() != h


def test_message_and_board_truncated_and_recorded():
    st = mk_state(FOUR)
    cfg = g(max_message_chars=5)
    apply(st, "ann", SendMessage(to="bob", text="hello world"), cfg)
    apply(st, "ann", PostBoard(text="abcdefgh"), cfg)
    assert [m.text for m in st.messages] == ["hello", "abcde"]
    assert st.messages[0].true_sender == "ann" and not st.messages[0].forged
    assert [r.kind for r in st.history] == ["dm", "board_post"]


def test_forged_message_record_uses_shown_sender():
    st = mk_state(FOUR)
    m, evs = deliver_message(st, channel="dm", sender_shown="bob", receiver="ann", text="t",
                             true_sender="GOD", forged=True, intervention_id="i1")
    assert st.history[-1].actor == "bob" and m.forged and evs[0].actor == "GOD"


@pytest.mark.parametrize("gain", ["mint", "steal", "none"])
def test_correct_defect(gain):
    st = mk_state(FOUR)
    _, evs = apply(st, "ann", DefectAction(target="bob", info_id=6), g(defect_gain=gain))
    assert st.agents["bob"].burned == [6] and 6 not in open_needs(st, "bob")
    burn = next(e.data for e in evs if isinstance(e.data, BurnData))
    assert burn.correct and burn.penalty == "none" and burn.gain == gain
    ann_has = [k.info_id for k in st.agents["ann"].keys]
    if gain == "mint":
        assert ann_has.count(6) == 1 and st.agents["cat"].keys[1].info_id == 6   # cat keeps its copy
        assert any(k.origin == "defect_mint" for k in st.agents["ann"].keys)
    elif gain == "steal":
        assert ann_has.count(6) == 1 and 6 not in [k.info_id for k in st.agents["cat"].keys]
        assert any(n.agent == "cat" for n in st.notices)
    else:
        assert 6 not in ann_has
    victim = [n for n in st.notices if n.agent == "bob"]
    assert victim and victim[0].kind == "burned" and "#6" in victim[0].text
    # the gained key is never recorded as a give
    assert all(r.kind != "give" for r in st.history)


def test_wrong_defect_lose_need():
    st = mk_state(FOUR)
    _, evs = apply(st, "ann", DefectAction(target="bob", info_id=5), g(wrong_defect_penalty="lose_need"))
    assert evs[0].data.ok                                       # executed, just wrong
    assert st.agents["bob"].burned == []
    assert st.agents["ann"].lost == [3]                         # first open need, permanent
    burn = next(e.data for e in evs if isinstance(e.data, BurnData))
    assert not burn.correct and burn.penalty == "lose_need" and burn.lost_info_id == 3
    assert st.history[-1].ok is False
    assert [n.kind for n in st.notices if n.agent == "bob"] == ["defect_attempt"]


def test_wrong_defect_delay():
    st = mk_state(FOUR)
    st.round = 2
    apply(st, "ann", DefectAction(target="bob", info_id=5), g(wrong_defect_penalty="delay", delay_n=3))
    assert st.agents["ann"].delayed_until_round == 5 and st.agents["ann"].lost == []


def test_defect_against_winner_fails_without_penalty():
    st = mk_state(FOUR)
    st.agents["bob"].won_round = 1
    h = st.state_hash()
    _, evs = apply(st, "ann", DefectAction(target="bob", info_id=6), g())
    assert evs[0].data.reason == "target_won" and st.state_hash() == h


def test_win_on_receiving_last_need_and_score():
    st = mk_state({"a": ([1, 2], [3]), "b": ([3], [1, 2])})
    cfg = run_cfg(game={"n_agents": 2, "needs_per_agent": 1, "keys_per_agent": 1}).game
    assert score(st, "b") == 0.0
    _, evs = apply(st, "a", GiveKey(to="b", info_id=1), cfg)
    assert score(st, "b") == 0.5 and st.agents["b"].won_round is None
    _, evs = apply(st, "a", GiveKey(to="b", info_id=2), cfg)
    assert st.agents["b"].won_round == 1 and any(isinstance(e.data, WinData) for e in evs)
    assert is_winning(st, "b") and score(st, "b") == 1.0


def test_burn_counts_zero():
    st = mk_state({"a": ([1], [2, 3]), "b": ([2, 3], [1])})
    cfg = run_cfg(game={"n_agents": 2, "needs_per_agent": 1, "keys_per_agent": 1}).game
    apply(st, "b", DefectAction(target="a", info_id=3), cfg)
    apply(st, "b", GiveKey(to="a", info_id=2), cfg)
    assert score(st, "a") == 0.5 and st.agents["a"].won_round == 1   # all OPEN needs held


def test_fuzz_rule_closure_and_conservation():
    rng = random.Random(0)
    names = list(FOUR) + ["zed"]
    for gain in ("mint", "steal", "none"):
        for pen in ("lose_need", "delay"):
            st = mk_state(FOUR)
            cfg = g(defect_gain=gain, wrong_defect_penalty=pen)
            initial = {k.copy_id for s in st.agents.values() for k in s.keys}
            for _ in range(300):
                actor = rng.choice(list(FOUR))
                act = rng.choice([
                    GiveKey(to=rng.choice(names), info_id=rng.randint(0, 9)),
                    DefectAction(target=rng.choice(names), info_id=rng.randint(0, 9)),
                    SendMessage(to=rng.choice(names), text="x" * rng.randint(1, 900)),
                    PostBoard(text="p"), Noop(),
                ])
                h = st.state_hash()
                _, evs = apply(st, actor, act, cfg)
                assert evs and evs[0].data.type == "action_result"
                if not evs[0].data.ok:
                    assert st.state_hash() == h
                now = [k.copy_id for s in st.agents.values() for k in s.keys]
                assert len(now) == len(set(now))                       # no copy duplicated
                assert initial <= set(now)                             # initial copies never vanish
                for s in st.agents.values():                           # monotone: burned/lost only grow
                    assert set(s.burned) <= set(s.needs) and set(s.lost) <= set(s.needs)


def test_internal_rule_error_is_loud(monkeypatch, caplog):
    """Review #12: an internal KeyError inside a rule is logged at ERROR and re-raised under
    SWARM_STRICT_RULES=1 (the test default); without it the action fails as error:KeyError."""
    from swarm.game import rules

    def boom(*a, **k):
        raise KeyError("bug")
    monkeypatch.setattr(rules, "_give", boom)
    st = mk_state(FOUR)
    cfg = run_cfg().game
    with pytest.raises(KeyError):
        apply(st, "ann", GiveKey(to="bob", info_id=1), cfg)
    monkeypatch.setenv("SWARM_STRICT_RULES", "0")
    with caplog.at_level("ERROR", logger="swarm.game.rules"):
        _, evs = apply(st, "ann", GiveKey(to="bob", info_id=1), cfg)
    assert evs[0].data.reason == "error:KeyError" and "internal error" in caplog.text
