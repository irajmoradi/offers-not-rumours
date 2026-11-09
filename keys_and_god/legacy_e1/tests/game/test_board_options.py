"""Board options (game.board): post_costs_turn and anonymous, in sequential and simultaneous modes,
plus the defaults (both off = the original rules, config hash unchanged)."""

from __future__ import annotations

import asyncio
import json

import pytest

from swarm.agents.llm_agent import LLMAgent, system_context
from swarm.config import HASH_GLOBAL_CAP_USD, BoardCfg, RunConfig
from swarm.context.base import items_from_observation
from swarm.game.engine import GameEngine, play_game
from swarm.game.observe import ANON_SENDER, grep_log
from swarm.game.world import generate_world
from swarm.llm.base import FakeLLM
from swarm.prompts import load_prompt_set
from swarm.rng import canonical_json, sha256_hex
from swarm.schema import GiveKey, GrepLog, Noop, PostBoard, SendMessage

from .helpers import FOUR, Scripted, fresh_game, run_cfg

MODES = ["sequential", "simultaneous"]
WHOLE = "board_post_uses_whole_turn"
COST_RULE = "Posting to the public board uses your whole turn; a direct message costs one action."


def _results(eng, rnd, actor=None):
    return [(e.actor, e.data.action.kind, e.data.ok, e.data.reason) for e in eng.sink.events
            if e.round == rnd and e.type == "action_result" and (actor is None or e.actor == actor)]


class _Seen:
    """Everything `agent` observed over the game (board, inbox, log, notices), merged."""

    def __init__(self, dec, agent):
        obs = [o for o, g in dec.calls if o.agent == agent and g is None]
        self.board = [m for o in obs for m in o.board]
        self.inbox = [m for o in obs for m in o.inbox]
        self.log = [ln for o in obs for ln in o.log]
        self.notices = [n for o in obs for n in o.notices]
        self.obs = obs


def _cfg(mode, **board):
    return run_cfg(game={"turn_mode": mode, "max_rounds": 2, "board": board})


# ------------------------------------------------------------------ defaults
def test_defaults_off_and_hash_unchanged():
    cfg = RunConfig()
    assert cfg.game.board == BoardCfg(post_costs_turn=False, anonymous=False)
    d = cfg.model_dump(mode="json", exclude={"sweep"})
    d["game"].pop("board")
    d["game"].pop("network")                                            # later field, off by default
    d["llm"]["openrouter"]["global_cap_usd"] = HASH_GLOBAL_CAP_USD   # cap is hashed as legacy $2
    for k in ("reasoning_effort", "provider_sort", "provider_order", "allow_fallbacks"):  # later, off
        d["llm"]["openrouter"].pop(k)
    d["run"].pop("resample_from_round")                                 # later field, off by default
    d["run"].pop("log_prompt_trace")                                    # logging only, never hashed
    d["reasoning"].pop("salvage_truncated")
    assert cfg.config_hash() == sha256_hex(canonical_json(d))       # pre-option hash preserved
    on = RunConfig.model_validate({"game": {"board": {"anonymous": True}}})
    assert on.config_hash() != cfg.config_hash()


@pytest.mark.parametrize("mode", MODES)
def test_defaults_post_and_dm_same_turn_and_author_shown(mode):
    cfg = _cfg(mode)
    dec = Scripted({(1, "ann"): [PostBoard(text="hello"), SendMessage(to="bob", text="hi")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _results(eng, 1, "ann") == [("ann", "post_board", True, None),
                                       ("ann", "send_message", True, None)]
    seen = _Seen(dec, "bob")
    assert [(m.sender, m.text) for m in seen.board] == [("ann", "hello")]
    assert any(ln.kind == "board_post" and ln.actor == "ann" for ln in seen.log)


def test_default_system_prompt_has_no_board_rules():
    text = load_prompt_set("agent/v1").render("system.j2", **system_context(run_cfg()))
    assert COST_RULE not in text and "anonymous" not in text


# ------------------------------------------------------------------ post_costs_turn
@pytest.mark.parametrize("mode", MODES)
def test_post_first_blocks_the_rest(mode):
    cfg = _cfg(mode, post_costs_turn=True)
    dec = Scripted({(1, "ann"): [PostBoard(text="hello"), GiveKey(to="bob", info_id=1)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _results(eng, 1, "ann") == [("ann", "post_board", True, None),
                                       ("ann", "give_key", False, WHOLE)]
    assert {k.info_id for k in eng.state.agents["ann"].keys} == {1, 2}      # the give did not happen
    notes = [n.text for n in _Seen(dec, "ann").notices]
    assert any("give_key" in t and "whole turn" in t for t in notes)
    assert [m.text for m in _Seen(dec, "bob").board] == ["hello"]


@pytest.mark.parametrize("mode", MODES)
def test_post_second_still_blocks_the_other_action(mode):
    """The post takes the whole turn wherever it is in the list: an earlier DM fails too."""
    cfg = _cfg(mode, post_costs_turn=True)
    dec = Scripted({(1, "ann"): [SendMessage(to="bob", text="psst"), PostBoard(text="hello")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _results(eng, 1, "ann") == [("ann", "send_message", False, WHOLE),
                                       ("ann", "post_board", True, None)]
    assert not [m for m in eng.state.messages if m.channel == "dm"]
    assert not _Seen(dec, "bob").inbox


@pytest.mark.parametrize("mode", MODES)
def test_two_posts_only_first_applies(mode):
    cfg = _cfg(mode, post_costs_turn=True)
    dec = Scripted({(1, "ann"): [PostBoard(text="one"), PostBoard(text="two")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert [r[2:] for r in _results(eng, 1, "ann")] == [(True, None), (False, WHOLE)]
    assert [m.text for m in eng.state.messages if m.channel == "board"] == ["one"]


@pytest.mark.parametrize("mode", MODES)
def test_dms_still_cost_one_slot_and_grep_noop_are_free(mode):
    cfg = _cfg(mode, post_costs_turn=True)
    dec = Scripted({(1, "ann"): [SendMessage(to="bob", text="a"), SendMessage(to="cat", text="b")],
                    (1, "bob"): [PostBoard(text="p"), Noop()]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert [r[2] for r in _results(eng, 1, "ann")] == [True, True]
    assert [r[2] for r in _results(eng, 1, "bob")] == [True, True]
    assert not [n for n in eng.state.notices if n.agent in ("ann", "bob")]


def test_post_with_grep_follow_up_keeps_grep():
    cfg = _cfg("sequential", post_costs_turn=True)
    dec = Scripted(policy=lambda obs, g: (
        [GrepLog(pattern="x")] if g is None and obs.agent == "ann" and obs.round == 1
        else ([PostBoard(text="p"), SendMessage(to="bob", text="m")] if g is not None else [])))
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _results(eng, 1, "ann") == [("ann", "grep_log", True, None), ("ann", "post_board", True, None),
                                       ("ann", "send_message", False, WHOLE)]


def test_cost_rule_only_in_system_prompt_when_on():
    on = run_cfg(game={"board": {"post_costs_turn": True}})
    text = load_prompt_set("agent/v1").render("system.j2", **system_context(on))
    assert COST_RULE in text and "anonymous" not in text


# ------------------------------------------------------------------ anonymous
@pytest.mark.parametrize("mode", MODES)
def test_anonymous_board_view_log_and_context(mode):
    cfg = _cfg(mode, anonymous=True)
    dec = Scripted({(1, "ann"): [PostBoard(text="selling 1"), SendMessage(to="bob", text="dm")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    seen = _Seen(dec, "bob")
    assert [(m.sender, m.text) for m in seen.board] == [(ANON_SENDER, "selling 1")]
    assert [m.sender for m in seen.inbox] == ["ann"]                       # DMs keep the sender
    posts = [ln for ln in seen.log if ln.kind == "board_post"]
    assert posts and all(ln.actor is None for ln in posts)
    texts = [it.text for o in seen.obs for it in items_from_observation(o)]
    assert any("selling 1" in t for t in texts)
    assert not any("ann" in t for t in texts if "selling" in t)
    # truth keeps the author: state, message events
    msg = next(m for m in eng.state.messages if m.channel == "board")
    assert msg.sender_shown == "ann" and msg.true_sender == "ann"
    ev = next(e for e in eng.sink.events if e.type == "message" and e.data.message.channel == "board")
    assert ev.actor == "ann" and ev.data.message.sender_shown == "ann"
    # grep_log over the visible history
    res = grep_log(eng.state, "bob", cfg.game, "selling", eng.obs_turns["bob"])
    assert res.hits and all(h.actor is None for h in res.hits)
    assert grep_log(eng.state, "bob", cfg.game, "ann posted", eng.obs_turns["bob"]).hits == []


def test_anonymous_only_affects_board():
    cfg = run_cfg(game={"max_rounds": 2, "board": {"anonymous": True}},
                  view_log={"content": "full"})
    dec = Scripted({(1, "ann"): [PostBoard(text="pub"), SendMessage(to="bob", text="priv")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    log = _Seen(dec, "cat").log
    assert [(ln.kind, ln.actor) for ln in log] == [("board_post", None), ("dm", "ann")]


@pytest.mark.parametrize("mode", MODES)
def test_anonymous_author_never_reaches_llm_prompts(mode):
    """LLMAgent path: the rendered prompts (current, window/earlier sections) never pair a board post
    with its author."""
    cfg = run_cfg(game={"n_agents": 4, "max_rounds": 3, "turn_mode": mode, "board": {"anonymous": True}},
                  context={"tokenizer": "words", "policy": "window", "window_rounds": 1})
    state = generate_world(cfg.game, 5, "gA")
    author = state.order[0]

    def reply(req):
        me = req.meta.agent
        acts = [{"kind": "post_board", "text": f"CANARY-{me}"}] if me == author else [{"kind": "noop"}]
        return json.dumps({"thought": "t", "note": "n", "actions": acts})

    client = FakeLLM(policy=reply)
    agents = {a: LLMAgent(a, client, cfg) for a in state.order}
    asyncio.run(GameEngine(cfg, state, agents).run())
    seen = 0
    for req in client.requests:
        if req.meta.agent == author:
            continue
        for m in req.messages:
            for line in m.content.splitlines():
                if f"CANARY-{author}" in line:
                    seen += 1
                    assert author not in line.replace(f"CANARY-{author}", ""), line
    assert seen > 0
    sys_text = client.requests[0].messages[0].content
    assert "anonymous" in sys_text and COST_RULE not in sys_text


# ------------------------------------------------------------------ post_skip_rounds
SKIP_RULE = "Posting to the public board costs you your next 3 turns."


def _turn_rounds(eng, agent):
    return [e.round for e in eng.sink.events if e.type == "turn" and e.actor == agent]


@pytest.mark.parametrize("mode", MODES)
def test_post_skip_rounds_skips_k_rounds_then_resumes(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 6, "board": {"post_skip_rounds": 3}})
    dec = Scripted({(1, "ann"): [PostBoard(text="hello"), SendMessage(to="bob", text="dm")],
                    (2, "bob"): [GiveKey(to="ann", info_id=3), SendMessage(to="ann", text="for you")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _results(eng, 1, "ann") == [("ann", "post_board", True, None),
                                       ("ann", "send_message", True, None)]   # the post itself is normal
    assert _turn_rounds(eng, "ann") == [1, 5, 6]
    assert _turn_rounds(eng, "bob") == [1, 2, 3, 4, 5, 6]
    assert 3 in {k.info_id for k in eng.state.agents["ann"].keys}            # received while skipped
    resumed = [o for o in _Seen(dec, "ann").obs if o.round == 5][0]
    assert [m.text for m in resumed.inbox] == ["for you"]
    texts = [n.text for n in resumed.notices]
    assert any("skip your turns in rounds 2 to 4" in t for t in texts)
    assert any("are over" in t for t in texts)
    assert any("gave you a key" in t for t in texts)
    later = [n.text for o in _Seen(dec, "ann").obs if o.round == 6 for n in o.notices]
    assert not any("skip" in t for t in later)                               # each shown once


@pytest.mark.parametrize("mode", MODES)
def test_post_skip_needs_a_successful_post_and_dm_does_not_skip(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3,
                        "board": {"post_skip_rounds": 2, "post_costs_turn": True}})
    dec = Scripted({(1, "ann"): [SendMessage(to="bob", text="dm"), SendMessage(to="cat", text="dm")],
                    (1, "bob"): [PostBoard(text="p"), GiveKey(to="ann", info_id=3)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert _turn_rounds(eng, "ann") == [1, 2, 3]
    assert _turn_rounds(eng, "bob") == [1]
    assert _results(eng, 1, "bob") == [("bob", "post_board", True, None),
                                       ("bob", "give_key", False, WHOLE)]


def test_post_skip_combines_with_anonymous_and_prompt_rules():
    cfg = run_cfg(game={"max_rounds": 2, "board": {"post_skip_rounds": 3, "anonymous": True}})
    dec = Scripted({(1, "ann"): [PostBoard(text="anon post")]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    assert [(m.sender, m.text) for m in _Seen(dec, "bob").board] == [(ANON_SENDER, "anon post")]
    assert _turn_rounds(eng, "ann") == [1]
    text = load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))
    assert SKIP_RULE in text and "anonymous" in text and COST_RULE not in text
    off = load_prompt_set("agent/v1").render("system.j2", **system_context(run_cfg()))
    assert "costs you your next" not in off


def test_post_skip_hash_unchanged_at_default():
    base = RunConfig.model_validate({"game": {"board": {"post_costs_turn": True}}})
    d = base.model_dump(mode="json", exclude={"sweep"})
    for k in ("post_skip_rounds", "enabled", "visible_delay_rounds", "scope"):   # later board fields
        d["game"]["board"].pop(k)
    d["game"].pop("network")                                            # later field, off by default
    d["run"].pop("resample_from_round")
    d["run"].pop("log_prompt_trace")                                    # logging only, never hashed
    d["llm"]["openrouter"]["global_cap_usd"] = HASH_GLOBAL_CAP_USD
    for k in ("reasoning_effort", "provider_sort", "provider_order", "allow_fallbacks"):
        d["llm"]["openrouter"].pop(k)
    d["reasoning"].pop("salvage_truncated")
    assert base.config_hash() == sha256_hex(canonical_json(d))
    on = RunConfig.model_validate({"game": {"board": {"post_skip_rounds": 3}}})
    assert on.config_hash() != RunConfig().config_hash()
