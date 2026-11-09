"""Visibility and leak tests: view_log content modes, actor identity, rolling window, defect timing
(D12), forged-DM indistinguishability, grep_log (D5), and a truth-leak scan over whole games."""

from __future__ import annotations

import json
import random

import pytest

from swarm.game.engine import GameEngine, play_game
from swarm.game.observe import grep_log, observe
from swarm.game.rules import apply, deliver_message
from swarm.memory import MemoryStore
from swarm.schema import (
    TRUTH_FIELDS,
    DefectAction,
    GiveKey,
    GrepLog,
    Noop,
    PostBoard,
    SendMessage,
)

from .helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg

N = 4
CANARY_FORGED = "zq-forged-canary-71"
CANARY_NOTE = "zq-note-canary-52"
CANARY_DM = "zq-dm-canary"


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def defect_lines(obs):
    return [ln for ln in obs.log if ln.kind == "defect"]


# ------------------------------------------------------------------ content modes / identity
def _state_with_dm_and_give():
    st = mk_state(FOUR)
    st.turn_index = 0
    apply(st, "ann", SendMessage(to="bob", text="secret plan"), run_cfg().game)
    apply(st, "ann", GiveKey(to="bob", info_id=1), run_cfg().game)
    deliver_message(st, channel="dm", sender_shown="cat", receiver="dan", text=CANARY_FORGED,
                    true_sender="GOD", forged=True, intervention_id="iCANARY")
    return st


def test_actions_only_has_no_message_lines():
    st = _state_with_dm_and_give()
    obs = observe(st, "cat", run_cfg().game, 2)
    assert [ln.kind for ln in obs.log] == ["give"]
    assert "secret plan" not in obs.model_dump_json() and CANARY_FORGED not in obs.model_dump_json()


def test_meta_has_no_text_and_full_has_text():
    st = _state_with_dm_and_give()
    meta = observe(st, "cat", run_cfg(view_log={"content": "actions_msg_meta"}).game, 2)
    dms = [ln for ln in meta.log if ln.kind == "dm_meta"]
    assert len(dms) == 2 and all(ln.text is None for ln in dms)
    assert {(ln.actor, ln.target) for ln in dms} == {("ann", "bob"), ("cat", "dan")}  # shown sender only
    full = observe(st, "cat", run_cfg(view_log={"content": "full"}).game, 2)
    assert {ln.text for ln in full.log if ln.kind == "dm"} == {"secret plan", CANARY_FORGED}
    assert "[FORGED]" not in full.model_dump_json()


def test_reveal_provenance_tags_forged():
    st = _state_with_dm_and_give()
    o = observe(st, "dan", run_cfg(view_log={"reveal_provenance": True}).game, 3)
    assert o.inbox[0].text.startswith("[FORGED]")


def test_none_disables_log_and_grep():
    st = _state_with_dm_and_give()
    cfg = run_cfg(view_log={"content": "none"}).game
    assert observe(st, "cat", cfg, 2).log == []
    assert grep_log(st, "cat", cfg, "gave", [2]).hits == []


def test_hidden_actor():
    st = _state_with_dm_and_give()
    cfg = run_cfg(view_log={"actor_identity": "hidden", "content": "actions_msg_meta"}).game
    obs = observe(st, "cat", cfg, 2)
    assert obs.log and all(ln.actor is None for ln in obs.log)
    st2 = mk_state(FOUR)
    apply(st2, "ann", GiveKey(to="bob", info_id=1), cfg)       # notice text is fixed at apply time
    bob = observe(st2, "bob", cfg, 1)
    assert bob.notices[0].text.startswith("Someone gave you")


def test_key_ids_visible():
    st = _state_with_dm_and_give()
    o = observe(st, "bob", run_cfg().game, 1)
    assert {(k.info_id, k.key) for k in o.pinned.holdings} == {(1, 1001), (3, 1003), (4, 1004)}
    assert o.pinned.needs_open == [1, 6] and "#1" in o.notices[0].text
    assert o.log[0].info_id == 1
    hits = grep_log(st, "bob", run_cfg().game, "#1", [1]).hits
    assert len(hits) == 1 and hits[0].kind == "give"


def test_forged_indistinguishable():
    """A forged DM and a real DM with the same sender/text/slot give byte-identical observations."""
    outs = []
    for forged in (False, True):
        st = mk_state(FOUR)
        st.turn_index = 0
        apply(st, "cat", SendMessage(to="dan", text="other"), run_cfg().game)
        deliver_message(st, channel="dm", sender_shown="bob", receiver="ann", text="I will give you 3",
                        true_sender="GOD" if forged else "bob", forged=forged,
                        intervention_id="iX" if forged else None)
        apply(st, "dan", SendMessage(to="ann", text="hello"), run_cfg().game)
        outs.append(observe(st, "ann", run_cfg(view_log={"content": "actions_msg_meta"}).game, 4)
                    .model_dump_json())
    assert outs[0] == outs[1]


# ------------------------------------------------------------------ rolling window (D16)
@pytest.mark.parametrize("order", ["fixed", "reshuffle"])
def test_log_window_turns(order):
    cfg = run_cfg(game={"turn_order": order, "max_rounds": 4})
    dec = Scripted(policy=lambda o, g: [PostBoard(text=f"post-{o.turn_index}")])
    play_game(cfg, fresh_game(FOUR), dec)
    for obs, _ in dec.calls:
        slots = sorted(int(ln.text.split("-")[1]) for ln in obs.log if ln.kind == "board_post")
        assert slots == list(range(max(0, obs.turn_index - N), obs.turn_index))


# ------------------------------------------------------------------ defect visibility (D12)
@pytest.mark.parametrize("mode", ["public", "victim_only", "delayed", "victim_then_delayed"])
@pytest.mark.parametrize("delay", [1, 3])
@pytest.mark.parametrize("correct", [True, False])
def test_defect_visibility_table(mode, delay, correct):
    cfg = run_cfg(game={"max_rounds": 7},
                  view_log={"defect_visibility": mode, "defect_delay_rounds": delay})
    info = 6 if correct else 5                    # bob needs 6, not 5
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=info)]})
    play_game(cfg, fresh_game(FOUR), dec)
    t_d = 0
    release = t_d + 1 + delay * N
    for agent in FOUR:
        obs = dec.obs_of(agent)
        counts = [len(defect_lines(o)) for o in obs]
        notice_kind = "burned" if correct else "defect_attempt"
        notices = [o.turn_index for o in obs for n in o.notices if n.kind == notice_kind]
        if agent == "bob":                         # victim
            if mode in ("public", "victim_only", "victim_then_delayed"):
                assert notices == [obs[0].turn_index] == [1]
            else:
                assert notices == [next(o.turn_index for o in obs if o.turn_index >= release)]
        else:
            assert notices == []
        sees_line = (mode == "public" or mode == "delayed"
                     or (mode == "victim_then_delayed" and agent != "bob"))
        if not sees_line:
            assert sum(counts) == 0
            continue
        assert sum(counts) == 1                    # exactly once, never repeated
        first = next(o for o in obs if defect_lines(o))
        if mode == "public":
            assert first.turn_index == obs[0].turn_index or first.turn_index <= N
        else:
            assert first.turn_index == next(o.turn_index for o in obs if o.turn_index >= release)
            for o in obs:                         # nothing engine-generated before release
                if o.turn_index < release and agent != "bob":
                    assert not defect_lines(o)
                    if agent != "ann":            # third parties: no hint at all via notices
                        assert all(f"#{info}" not in n.text for n in o.notices)
        if agent == "ann":                         # the actor learns its own outcome at once, only once
            results = [(o.turn_index, n.text) for o in obs for n in o.notices if n.kind == "defect_result"]
            assert len(results) == 1 and results[0][0] == obs[1].turn_index
            assert ("correct" if correct else "wrong") in results[0][1]
        assert defect_lines(first)[0].ok is correct
        assert defect_lines(first)[0].info_id == info


def test_release_queued_when_inactive():
    cfg = run_cfg(game={"max_rounds": 7})
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=6)]})
    st = fresh_game(FOUR)
    st.agents["dan"].delayed_until_round = 4       # dan misses slot 15 (release = 13)
    play_game(cfg, st, dec)
    dan = dec.obs_of("dan")
    assert [o.turn_index for o in dan if defect_lines(o)] == [19]


def test_defect_actor_hidden_and_no_side_channel():
    cfg = run_cfg(game={"max_rounds": 6}, view_log={"defect_actor_identity": "hidden"})
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=6)]})
    play_game(cfg, fresh_game(FOUR), dec)
    bob_notice = dec.obs_of("bob")[0].notices[0]
    assert "ann" not in bob_notice.text and bob_notice.text.startswith("Someone")
    line = next(ln for o in dec.obs_of("cat") for ln in defect_lines(o))
    assert line.actor is None and line.target == "bob"
    for o in dec.obs_of("cat") + dec.obs_of("dan"):        # minted key never rendered as a give
        assert all(ln.kind != "give" for ln in o.log)


def test_multiple_defects_once_each():
    cfg = run_cfg(game={"max_rounds": 7, "defect_gain": "none"})
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=6), DefectAction(target="cat", info_id=2)],
                    (2, "dan"): [DefectAction(target="cat", info_id=4)]})
    play_game(cfg, fresh_game(FOUR), dec)
    for agent in ("dan",):
        seen = [(ln.target, ln.info_id) for o in dec.obs_of(agent) for ln in defect_lines(o)]
        assert sorted(seen) == [("bob", 6), ("cat", 2), ("cat", 4)]


# ------------------------------------------------------------------ grep_log (D5)
def _defect_game(rounds=6):
    cfg = run_cfg(game={"max_rounds": rounds})
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=6), GiveKey(to="dan", info_id=1)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    return cfg, eng


def test_grep_unreleased_defect():
    cfg = run_cfg(game={"max_rounds": 6})
    dec = Scripted({(1, "ann"): [DefectAction(target="bob", info_id=6)]})
    eng = GameEngine(cfg, fresh_game(FOUR), dec)
    import asyncio
    for _ in range(3):
        asyncio.run(eng.run_round())
    assert grep_log(eng.state, "cat", cfg.game, "#6", eng.obs_turns["cat"]).hits == []   # cat obs 2,6,10
    asyncio.run(eng.run_round())                                                        # cat obs 14 >= 13
    hits = grep_log(eng.state, "cat", cfg.game, "burned", eng.obs_turns["cat"]).hits
    assert len(hits) == 1 and hits[0].info_id == 6
    assert grep_log(eng.state, "bob", cfg.game, "#6", eng.obs_turns["bob"]).hits == []  # victim: notice only


def test_grep_literal_cap_and_case():
    cfg, eng = _defect_game()
    g = cfg.game
    turns = eng.obs_turns["cat"]
    assert grep_log(eng.state, "cat", g, ".*", turns).hits == []
    assert len(grep_log(eng.state, "cat", g, "GAVE DAN", turns).hits) == 1
    assert grep_log(eng.state, "cat", g, "x" * 81, turns).hits == []
    st = mk_state(FOUR)
    for i in range(10):
        st.turn_index = i
        apply(st, "ann", PostBoard(text=f"post {i}"), g)
    res = grep_log(st, "bob", run_cfg(view_log={"grep_max_results": 3}).game, "post", [10])
    assert len(res.hits) == 3 and res.truncated
    off = run_cfg(view_log={"grep_enabled": False}).game
    assert grep_log(st, "bob", off, "post", [10]).hits == []


def test_grep_actions_only_no_dm_text():
    st = _state_with_dm_and_give()
    for content in ("actions_only", "actions_msg_meta"):
        g = run_cfg(view_log={"content": content}).game
        for agent in FOUR:
            assert grep_log(st, agent, g, "secret", [2]).hits == []
            assert grep_log(st, agent, g, "canary", [3]).hits == []


def test_grep_followup_in_engine():
    cfg = run_cfg(game={"max_rounds": 2})
    seen = []

    def policy(obs, grep):
        if grep is not None:
            seen.append(grep)
            return [SendMessage(to="ann", text="thanks")]
        if obs.agent == "ann" and obs.round == 1:
            return [GiveKey(to="dan", info_id=1)]
        if obs.agent == "cat" and obs.round == 1:
            return [GrepLog(pattern="#1"), Noop()]
        return []
    _, eng = play_game(cfg, fresh_game(FOUR), Scripted(policy=policy))
    assert len(seen) == 1 and len(seen[0].hits) == 1 and seen[0].hits[0].target == "dan"
    res = [e.data for e in eng.sink.events if e.type == "action_result" and e.actor == "cat"
           and e.round == 1]
    assert res[0].action.kind == "grep_log" and res[0].ok and res[0].n_hits == 1
    assert res[1].action.kind == "send_message" and res[1].ok


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("content", ["actions_only", "actions_msg_meta", "full"])
@pytest.mark.parametrize("mode", ["public", "delayed", "victim_then_delayed"])
def test_grep_subset_of_visible_history(seed, content, mode):
    rng = random.Random(seed)
    cfg = run_cfg(game={"max_rounds": 6, "turn_order": "reshuffle"},
                  view_log={"content": content, "defect_visibility": mode, "defect_delay_rounds": 1})
    shown: dict[str, list] = {a: [] for a in FOUR}

    def policy(obs, grep):
        shown[obs.agent] += [ln.model_dump_json() for ln in obs.log] if grep is None else []
        if grep is not None:
            for h in grep.hits:
                assert h.model_dump_json() in shown[obs.agent]
            return []
        return _random_actions(rng, obs) + [GrepLog(pattern=rng.choice(["#", "gave", "burn", "post", "a"]))]
    play_game(cfg, fresh_game(FOUR), Scripted(policy=policy))


# ------------------------------------------------------------------ truth-leak scan over whole games
def _random_actions(rng, obs):
    peers = obs.pinned.peers
    acts = []
    r = rng.random()
    if r < 0.3 and obs.pinned.can_give:
        acts.append(GiveKey(to=rng.choice(peers), info_id=rng.choice(obs.pinned.can_give)))
    elif r < 0.5:
        acts.append(DefectAction(target=rng.choice(peers), info_id=rng.randint(1, 8)))
    elif r < 0.8:
        acts.append(SendMessage(to=rng.choice(peers), text=f"{CANARY_DM}-{obs.agent}-{obs.turn_index}"))
    else:
        acts.append(PostBoard(text=f"board {obs.turn_index}"))
    return acts


@pytest.mark.parametrize("content", ["actions_only", "actions_msg_meta", "full"])
@pytest.mark.parametrize("ident", ["shown", "hidden"])
@pytest.mark.parametrize("mode", ["public", "victim_only", "delayed", "victim_then_delayed"])
@pytest.mark.parametrize("winner", ["continue", "sit_out"])
def test_no_truth_leak(content, ident, mode, winner):
    rng = random.Random(hash((content, ident, mode, winner)) % 1000)
    cfg = run_cfg(game={"max_rounds": 6, "turn_order": "reshuffle", "winner_policy": winner},
                  view_log={"content": content, "actor_identity": ident, "defect_visibility": mode})
    mem = MemoryStore(cfg.reasoning)
    iids = []

    def god(state, r, next_turn):
        if r < 2 or next_turn % 3:
            return []
        iid = f"iv{len(iids)}zz"
        iids.append(iid)
        _, evs = deliver_message(state, channel="dm", sender_shown="bob", receiver="ann",
                                 text=f"{CANARY_FORGED} {iid[:0]}", true_sender="GOD", forged=True,
                                 intervention_id=iid, phase="god")
        mem.write(state, "cat", "scratchpad", r - 1, CANARY_NOTE)
        mem.overwrite(state, "cat", "scratchpad", r - 1, "edited", intervention_id=iid, now_round=r)
        return evs

    dec = Scripted(policy=lambda o, g: [] if g is not None else _random_actions(rng, o) + [GrepLog(pattern="a")])
    _, eng = play_game(cfg, fresh_game(FOUR), dec, god_hook=god)
    assert iids
    for obs, grep in dec.calls:
        blob = obs.model_dump(mode="json")
        keys = _keys(blob, set()) | (_keys(grep.model_dump(mode="json"), set()) if grep else set())
        assert not keys & TRUTH_FIELDS
        text = obs.model_dump_json() + (grep.model_dump_json() if grep else "")
        for bad in [*iids, CANARY_NOTE, "true_sender", "tampered", "taint", "GOD", "[FORGED]"]:
            assert bad not in text
        if content != "full":
            others = json.dumps({k: v for k, v in blob.items() if k != "inbox"})
            assert CANARY_FORGED not in others and CANARY_DM not in others
            assert CANARY_DM not in (grep.model_dump_json() if grep else "")
            for m in obs.inbox:
                assert m.receiver == obs.agent
            if obs.agent != "ann":
                assert CANARY_FORGED not in text
