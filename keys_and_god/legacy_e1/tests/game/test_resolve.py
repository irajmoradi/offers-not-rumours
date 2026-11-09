"""game.resolve.resolve_turn called directly (no engine): slot limit, free grep, board options."""

from __future__ import annotations

from swarm.game.board import BOARD_WHOLE_TURN
from swarm.game.resolve import resolve_turn
from swarm.schema import GiveKey, GrepLog, GrepResult, Noop, PostBoard, SendMessage

from .helpers import FOUR, mk_state, run_cfg


def test_slot_limit_and_free_grep():
    st = mk_state(FOUR)
    g = run_cfg(game={"actions_per_turn": 1}).game
    res = resolve_turn(st, "ann", [GrepLog(pattern="x"), GiveKey(to="bob", info_id=1),
                                   SendMessage(to="bob", text="hi"), GrepLog(pattern="y")],
                       GrepResult(pattern="x", hits=[]), g)
    assert res.results == [(True, None), (True, None), (False, "action_limit"), (False, "grep_unavailable")]
    assert [e.data.type for e in res.events][:1] == ["action_result"]
    assert not any(k.info_id == 1 for k in st.agents["ann"].keys)


def test_board_whole_turn_and_skip():
    st = mk_state(FOUR)
    g = run_cfg(game={"board": {"post_costs_turn": True, "post_skip_rounds": 2}}).game
    res = resolve_turn(st, "ann", [SendMessage(to="bob", text="a"), PostBoard(text="p"), Noop()], None, g)
    assert res.results == [(False, BOARD_WHOLE_TURN), (True, None), (True, None)]
    assert st.agents["ann"].delayed_until_round == st.round + 2
    notices = [e.data.notice.text for e in res.events if e.data.type == "notice"]
    assert len(notices) == 3 and "whole turn" in notices[0] and "skip your turns" in notices[1]
