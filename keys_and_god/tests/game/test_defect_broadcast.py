"""game.view_log.defect_broadcast (gossip knob): default off is byte-identical to showcase 9deff9d
(config hashes and system prompts of every config, scripted event streams with defects); when on, every
agent of a sparse network gets the anonymous notice at the start of round r + delay (round_summary /
per_defect), nothing names the attacker or victim, and every existing notice is unchanged."""

from __future__ import annotations

import hashlib
import json

import pytest

from swarm.agents.llm_agent import system_context
from swarm.config import REPO_ROOT, from_dict
from swarm.game import rules
from swarm.game.engine import play_game
from swarm.prompts import load_prompt_set
from swarm.rng import canonical_json
from swarm.runner.run import load_experiment
from swarm.schema import DefectAction, GiveKey, SendMessage
from tests.game.helpers import Scripted, fresh_game, run_cfg

LINE = [f"p{i}" for i in range(8)]                     # path graph p0 - p1 - ... - p7
SPEC = {a: ([10 + i, 20 + i], [10 + (i + 1) % 8, 20 + (i + 2) % 8]) for i, a in enumerate(LINE)}
SCRIPT = {(1, "p0"): [DefectAction(target="p1", info_id=11)],          # correct (p1 needs 11)
          (1, "p2"): [DefectAction(target="p3", info_id=99)],          # wrong
          (2, "p4"): [DefectAction(target="p5", info_id=16), SendMessage(to="p5", text="sorry")],
          (3, "p6"): [SendMessage(to="p7", text="hi p5")]}
# sha256 of this scripted game's event stream at showcase 9deff9d (before the knob): (default, delayed)
EVENTS_BEFORE = {
    "sequential": ("839ce3e0f2990f11e9c741b4678c8ed410b65aadff6e60e3cc4cb00d025eab0e",
                   "d4da1e8a78d4f6e7a7f97683aaf02a48af374927b3a906732a269d0dc54f2325"),
    "simultaneous": ("0b3262961e1b5fdd8f0c0be453839890a29eb518525ec0a7f3ae46914a77102f",
                     "64ac308a03240aff6908fd2826eab176219a210782cba94a63cdb322c4ea96c1"),
}
SUMMARY_R1 = "In round 1, 2 defects occurred somewhere in the game."
SUMMARY_R2 = "In round 2, 1 defect occurred somewhere in the game."


def _line_state():
    st = fresh_game(SPEC)
    for i, a in enumerate(LINE):
        st.agents[a].contacts = [LINE[j] for j in (i - 1, i + 1) if 0 <= j < 8]
    return st


def _play(mode="sequential", **vl):
    cfg = run_cfg(game={"n_agents": 8, "max_rounds": 6, "turn_mode": mode, "view_log": vl})
    p = Scripted(SCRIPT)
    _, eng = play_game(cfg, _line_state(), p)
    return eng, p


def _digest(eng) -> str:
    return hashlib.sha256(canonical_json([e.model_dump(mode="json") for e in eng.sink.events]).encode()).hexdigest()


def _bcast(eng):
    return [e.data.notice for e in eng.sink.events
            if e.type == "notice" and (e.data.notice.ref or "").startswith(rules.BROADCAST_REF)]


# ------------------------------------------------------------------ off: byte-identical
@pytest.mark.parametrize("mode", sorted(EVENTS_BEFORE))
def test_off_event_streams_byte_identical(mode):
    default, delayed = EVENTS_BEFORE[mode]
    assert _digest(_play(mode)[0]) == default
    assert _digest(_play(mode, defect_visibility="delayed")[0]) == delayed
    off = {"defect_broadcast": {"enabled": False, "delay_rounds": 1, "mode": "per_defect"}}
    assert _digest(_play(mode, **off)[0]) == default


def test_off_config_hashes_and_prompts_unchanged():
    pinned = json.loads((REPO_ROOT / "tests" / "fixtures" / "defect_broadcast_before.json").read_text())["configs"]
    assert len(pinned) >= 50
    for name, (h, sp) in pinned.items():
        cfg = load_experiment(REPO_ROOT / name, [])[0]
        ps = load_prompt_set(cfg.roles.agent.prompt_set, framing=cfg.roles.agent.framing)
        got = hashlib.sha256(ps.render("system.j2", **system_context(cfg)).encode()).hexdigest()
        assert (cfg.config_hash(), got) == (h, sp), name
        off = load_experiment(REPO_ROOT / name, ["game.view_log.defect_broadcast.delay_rounds=7"])[0]
        assert off.config_hash() == h, name


def test_on_changes_hash_and_validates():
    base = from_dict({})
    on = from_dict({}, ["game.view_log.defect_broadcast.enabled=true"])
    assert on.config_hash() != base.config_hash()
    assert from_dict({}, ["game.view_log.defect_broadcast.enabled=true",
                          "game.view_log.defect_broadcast.mode=per_defect"]).config_hash() != on.config_hash()
    assert from_dict({}, ["game.view_log.defect_broadcast.enabled=true",
                          "game.view_log.defect_broadcast.delay_rounds=2"]).config_hash() != on.config_hash()
    with pytest.raises(ValueError):
        from_dict({}, ["game.view_log.defect_broadcast.delay_rounds=0"])
    with pytest.raises(ValueError):
        from_dict({}, ["game.view_log.defect_broadcast.mode=names"])


# ------------------------------------------------------------------ on: reach, timing, no leak
@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
def test_round_summary_reaches_every_agent_at_round_r_plus_delay(mode):
    eng, p = _play(mode, defect_broadcast={"enabled": True, "delay_rounds": 2})
    for a in LINE:                                      # p7 is 4+ hops from every defect
        seen = {o.round: [n.text for n in o.notices if "somewhere in the game" in n.text] for o in p.obs_of(a)}
        assert seen[1] == seen[2] == [] and seen[3] == [SUMMARY_R1] and seen[4] == [SUMMARY_R2], (a, seen)
        assert seen[5] == seen[6] == []                 # round 3 had no defect: no notice
    b = _bcast(eng)
    assert len(b) == 2 * len(LINE)
    assert {(n.round, n.ref) for n in b} == {(3, "defect_broadcast:r1"), (4, "defect_broadcast:r2")}
    assert all(n.kind == "system" for n in b)
    # the far agent still sees no defect log line (local-only view_log unchanged)
    assert not any(ln.kind == "defect" for o in p.obs_of("p7") for ln in o.log)


def test_per_defect_and_no_identity_leak():
    eng, p = _play(defect_broadcast={"enabled": True, "delay_rounds": 1, "mode": "per_defect"})
    for a in LINE:
        seen = {o.round: [n.text for n in o.notices if n.text.startswith("A defect occurred")] for o in p.obs_of(a)}
        assert seen[2] == ["A defect occurred in round 1."] * 2 and seen[3] == ["A defect occurred in round 2."]
        assert seen[1] == seen[4] == []
    b = _bcast(eng)
    assert len(b) == 3 * len(LINE) and len({(n.agent, n.ref) for n in b}) == len(b)
    for n in b:
        assert not any(x in n.text for x in LINE) and "correct" not in n.text and "wrong" not in n.text
        assert "#" not in n.text and "burn" not in n.text


def test_existing_notices_unchanged():
    def plain(eng):
        return [(e.round, e.turn_index, e.data.notice.agent, e.data.notice.kind, e.data.notice.text)
                for e in eng.sink.events if e.type == "notice"
                and not (e.data.notice.ref or "").startswith(rules.BROADCAST_REF)]
    for vis in ("victim_then_delayed", "delayed", "public"):
        off, _ = _play(defect_visibility=vis)
        on, _ = _play(defect_visibility=vis, defect_broadcast={"enabled": True, "delay_rounds": 1})
        assert plain(on) == plain(off), vis
        assert [e.type for e in on.sink.events if e.type != "notice"] == \
               [e.type for e in off.sink.events if e.type != "notice"]


def test_departed_winner_gets_it_too():
    spec = {"x": ([1], [2]), "y": ([3], [1]), "z": ([2], [9])}
    script = {(1, "x"): [GiveKey(to="y", info_id=1)],                 # y wins and leaves
              (1, "z"): [DefectAction(target="x", info_id=2)]}
    cfg = run_cfg(game={"n_agents": 3, "max_rounds": 3, "winner_policy": "sit_out",
                        "view_log": {"defect_broadcast": {"enabled": True, "delay_rounds": 1}}})
    p = Scripted(script)
    _, eng = play_game(cfg, fresh_game(spec), p)
    assert eng.state.agents["y"].won_round == 1
    text = "In round 1, 1 defect occurred somewhere in the game."
    assert sorted(n.agent for n in eng.state.notices if n.ref == "defect_broadcast:r1") == ["x", "y", "z"]
    for a in ("x", "z"):
        assert [n.text for o in p.obs_of(a) if o.round == 2 for n in o.notices if n.text == text] == [text]


def test_example_config_is_showcase_plus_broadcast():
    sc = load_experiment(REPO_ROOT / "configs" / "showcase100_sim.yaml", [])[0]
    bc = load_experiment(REPO_ROOT / "configs" / "showcase100_broadcast_sim.yaml", [])[0]
    vl = bc.game.view_log
    assert vl.defect_broadcast.enabled and vl.defect_broadcast.mode == "round_summary"
    assert vl.defect_broadcast.delay_rounds == 3
    assert bc.game.model_copy(update={"view_log": sc.game.view_log}) == sc.game
    assert bc.god == sc.god and bc.roles == sc.roles and bc.run.seed == sc.run.seed
    assert bc.config_hash() != sc.config_hash()
