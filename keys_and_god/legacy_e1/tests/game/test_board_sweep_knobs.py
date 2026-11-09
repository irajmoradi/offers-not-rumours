"""Sweep board knobs (game.board.enabled, game.board.visible_delay_rounds) in sequential and
simultaneous modes, plus their defaults (original rules, prompts and config hash unchanged)."""

from __future__ import annotations

import pytest

from swarm.agents.llm_agent import system_context
from swarm.config import BoardCfg, RunConfig
from swarm.game.engine import play_game
from swarm.game.observe import grep_log
from swarm.prompts import load_prompt_set
from swarm.schema import PostBoard

from .helpers import FOUR, Scripted, fresh_game, run_cfg

MODES = ["sequential", "simultaneous"]


def _board_seen(dec, agent):
    """[(round observed, text)] of every board post `agent` saw."""
    return [(o.round, m.text) for o, g in dec.calls if o.agent == agent and g is None for m in o.board]


def _log_seen(dec, agent):
    return [(o.round, ln.text) for o, g in dec.calls if o.agent == agent and g is None
            for ln in o.log if ln.kind == "board_post"]


def test_defaults_are_original_rules_and_hash_neutral():
    assert BoardCfg().enabled is True and BoardCfg().visible_delay_rounds == 0
    explicit = RunConfig.model_validate({"game": {"board": {"enabled": True, "visible_delay_rounds": 0}}})
    assert explicit.config_hash() == RunConfig().config_hash()
    anon = RunConfig.model_validate({"game": {"board": {"anonymous": True}}})
    anon2 = RunConfig.model_validate({"game": {"board": {"anonymous": True, "enabled": True,
                                                         "visible_delay_rounds": 0}}})
    assert anon.config_hash() == anon2.config_hash()                 # pre-existing board configs keep ids
    for b in ({"enabled": False}, {"visible_delay_rounds": 2}):
        assert RunConfig.model_validate({"game": {"board": b}}).config_hash() != RunConfig().config_hash()


def test_board_off_rejects_other_board_options():
    with pytest.raises(ValueError):
        BoardCfg(enabled=False, anonymous=True)
    with pytest.raises(ValueError):
        BoardCfg(enabled=False, visible_delay_rounds=1)


@pytest.mark.parametrize("mode", MODES)
def test_board_off_post_fails_and_nothing_is_seen(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3, "board": {"enabled": False}})
    dec = Scripted({(1, "ann"): [PostBoard(text="hello all")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    res = [(e.data.action.kind, e.data.ok, e.data.reason) for e in eng.sink.events
           if e.type == "action_result" and e.actor == "ann" and e.round == 1]
    assert res == [("post_board", False, "board_disabled")]
    assert not [m for m in eng.state.messages if m.channel == "board"]
    for a in ("bob", "cat"):
        assert _board_seen(dec, a) == [] and _log_seen(dec, a) == []


def test_board_off_system_prompt_has_no_board():
    off = load_prompt_set("agent/v1").render("system.j2", **system_context(
        run_cfg(game={"board": {"enabled": False}})))
    assert "post_board" not in off and "board" not in off.lower()
    on = load_prompt_set("agent/v1").render("system.j2", **system_context(run_cfg()))
    assert "post_board" in on and "public board posts, and a log" in on and "delayed" not in on


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [1, 2])
def test_delayed_board_shows_posts_k_rounds_later(mode, k):
    base = run_cfg(game={"turn_mode": mode, "max_rounds": 6})
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 6, "board": {"visible_delay_rounds": k}})
    script = {(1, "ann"): [PostBoard(text="p1")], (3, "ann"): [PostBoard(text="p3")]}
    d0, d1 = Scripted(dict(script)), Scripted(dict(script))
    play_game(base, fresh_game(FOUR), d0)
    _, eng = play_game(cfg, fresh_game(FOUR), d1)
    for a in ("bob", "cat"):
        undelayed = _board_seen(d0, a)
        assert [t for _, t in undelayed] == ["p1", "p3"]
        assert _board_seen(d1, a) == [(r + k, t) for r, t in undelayed]          # each shown once, k later
        assert _log_seen(d1, a) == [(r + k, t) for r, t in _log_seen(d0, a)]
    # grep only finds a delayed post once it is visible
    st = eng.state
    early = grep_log(st, "bob", cfg.game, "p3", obs_turns=[(3 + k - 1) * 4 - 1])
    assert early.hits == []


def test_delayed_board_rule_in_system_prompt():
    text = load_prompt_set("agent/v1").render("system.j2", **system_context(
        run_cfg(game={"board": {"visible_delay_rounds": 2}})))
    assert "Board posts are delayed" in text and "2 round(s)" in text
