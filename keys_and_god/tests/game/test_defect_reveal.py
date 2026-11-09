"""Game-fixes v2 switches, all default off: wrong_defect_penalty score_penalty / lose_held_key,
game.defect.win_rule all_needs (the "a lost or burned need creates a win" loophole), and God's
reveal_needs power (truthful / false). Rules, prompt text, budget, hashes at the defaults, and FakeLLM
runs that hold every invariant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.adversary.powers import apply_power, apply_reveal_needs, init_budget, refresh_budget
from swarm.adversary.scripted_god import ScriptedGod
from swarm.agents.llm_agent import system_context
from swarm.analysis.invariants import check_run
from swarm.config import RunConfig
from swarm.game import rules
from swarm.memory import MemoryStore
from swarm.prompts import load_prompt_set
from swarm.schema import DefectAction, RevealNeeds
from tests.runner.conftest import _run

from .helpers import mk_state, run_cfg


def _system(cfg: RunConfig) -> str:
    return load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))


def _wrong(state, game: dict, target="bob", info=9):
    g = run_cfg(game=game).game
    st, evs = rules.apply(state, "ann", DefectAction(target=target, info_id=info), g)
    burn = next(e.data for e in evs if e.data.type == "burn")
    notes = [e.data.notice.text for e in evs if e.data.type == "notice" and e.data.notice.agent == "ann"]
    return st, burn, notes, evs, g


# ------------------------------------------------------------------ defaults
SWITCHES = [{"game": {"wrong_defect_penalty": "score_penalty"}},
            {"game": {"wrong_defect_penalty": "lose_held_key"}},
            {"game": {"wrong_defect_penalty": "score_penalty", "defect": {"score_penalty": 0.5}}},
            {"game": {"defect": {"win_rule": "all_needs"}}},
            {"god": {"powers": {"reveal_needs": {"enabled": True}}}},
            {"god": {"powers": {"reveal_needs": {"enabled": True, "variant": "false"}}}}]


def test_defaults_keep_hash_and_prompt_and_switches_change_them():
    base = RunConfig()
    explicit = RunConfig.model_validate({"game": {"defect": {"score_penalty": None, "win_rule": "open_needs"}},
                                         "god": {"powers": {"reveal_needs": {"enabled": False, "cost": 1,
                                                                             "variant": "truthful",
                                                                             "max_receivers": 3}}}})
    assert explicit.config_hash() == base.config_hash()
    assert base.config_hash() == "b442ad7af4a87da8bb2ff289c1e794cd7b98f44742c8a7066966921cfad35ee1"   # pinned
    hashes = {RunConfig.model_validate(s).config_hash() for s in SWITCHES}
    assert base.config_hash() not in hashes and len(hashes) == len(SWITCHES)
    off = _system(base)
    assert "You win when you hold a key for every one of your needs that has not been burned." in off
    assert "you are penalised." in off and "drops by" not in off and "spare" not in off
    assert _system(explicit) == off


# (config_hash, sha256 of the rendered agent/v1 system prompt) of existing configs BEFORE score_penalty,
# lose_held_key, game.defect and reveal_needs existed (git 9a14abe)
BEFORE = {
    "demo": ("16cc6969159a1e07620cd888f028f77641fd929d396ce98f83ac3fbae1fd3059",
             "2342500060a5ebaca74c3e396e9487a5984b07799b7990f57b3ca1ec55fa1d60"),
    "grid_oss_sim_k2": ("15f60423be9731158c2ea9e95947800474162f34990f778fd201af584334eb0f",
                        "8a9b6ee4477ef8eb4926a06efadb7c1ca747c5c0f9c03447d29ddb2c0bd8da25"),
    "g1_v2_costly_sim": ("7621efabf3e206e5ba64da902b9dc02ebc844f2ae769c56e5a3f7d92174b5b63",
                         "3e126a2c82abfb2c8ef26e919a71502f709a916808f0d20055d9481d65c45618"),
    "net40_er3_contacts_team03_sim": ("d9d8c70e54bbcab19acf5798e8714991b0141404feb4e32307a65e1ad885bce6",
                                      "5cd26bb48e91eb75034d7fc143cb4569bad48aafce2d6badc8cedeaeccaf4ada"),
    "net40_er3_contacts_connect2_sim": ("ecefc20f17ce9c74f27c9dcb4b50817b7d3e5846a71090fccf54bde428e39695",
                                        "83c508d0444177327fa7ec3d7cebe2f74a4cda3488bd97747df50586917e9d7a"),
}


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_existing_configs_keep_hash_and_prompt_byte_identical(name):
    import hashlib

    from swarm.config import REPO_ROOT
    from swarm.runner.run import load_experiment
    cfg = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", [])[0]
    assert (cfg.config_hash(), hashlib.sha256(_system(cfg).encode()).hexdigest()) == BEFORE[name]


def test_bad_switch_combinations_are_rejected():
    with pytest.raises(ValueError, match="score_penalty"):
        RunConfig.model_validate({"game": {"defect": {"score_penalty": 0.2}}})        # lose_need ignores it
    with pytest.raises(ValueError, match="reveal_needs but it is disabled"):
        RunConfig.model_validate({"god": {"schedule": [{"round": 3, "power": "reveal_needs"}]}})


def test_prompt_states_each_penalty_in_one_sentence():
    sp = _system(RunConfig.model_validate(SWITCHES[0]))
    assert ("If they do not need it, you are penalised: your score drops by 0.33 (the value of one need) "
            "and none of your needs is removed.") in sp
    assert "you gain a new key for that info id" in sp
    hk = _system(RunConfig.model_validate(SWITCHES[1]))
    line = next(x for x in hk.splitlines() if x.startswith("- defect:"))
    assert "you lose one of your spare keys" in line and "goes to the player you named" in line
    assert line.count(". ") == 2 and line.endswith(".")          # rule, gain and penalty: one sentence each
    assert "drops by 0.50 and none" in _system(RunConfig.model_validate(SWITCHES[2]))
    assert ("You win only when you hold a key for every one of your needs; once one of your needs is burned "
            "you can no longer win.") in _system(RunConfig.model_validate(SWITCHES[3]))


# ------------------------------------------------------------------ score_penalty
def test_score_penalty_removes_no_need_and_never_creates_a_win():
    # ann holds 5 and misses 3: lose_need burns 3 (its only missing need) and makes it a winner at 1/2
    spec = {"ann": ([5], [3, 5]), "bob": ([3], [1])}
    st, burn, _, _, _ = _wrong(mk_state(spec), {"wrong_defect_penalty": "lose_need"})
    assert st.agents["ann"].won_round == 1 and st.agents["ann"].lost == [3]
    st, burn, notes, _, g = _wrong(mk_state(spec), {"wrong_defect_penalty": "score_penalty"})
    a = st.agents["ann"]
    assert a.lost == [] and a.burned == [] and a.won_round is None
    assert (burn.penalty, burn.lost_info_id) == ("score_penalty", None)
    assert rules.score(st, "ann", g) == pytest.approx(0.5)             # the needs score is untouched
    assert rules.penalty(st, "ann") == pytest.approx(0.5)              # one need of two
    assert rules.open_needs(st, "ann") == [3, 5]
    assert "Your score drops by 0.50; none of your needs is removed." in notes[-1]


def test_score_penalty_is_not_regressive_and_stacks():
    """A hopeless agent (holds nothing) pays the same as a nearly-done one; no floor at 0."""
    g = run_cfg(game={"wrong_defect_penalty": "score_penalty"}).game
    st = mk_state({"ann": ([], [3, 5, 6]), "bob": ([3], [1])})
    for _ in range(2):
        st, _ = rules.apply(st, "ann", DefectAction(target="bob", info_id=9), g)
    assert st.agents["ann"].score_penalty == pytest.approx(2 / 3)
    assert rules.penalty(st, "ann") == pytest.approx(2 / 3) and st.agents["ann"].lost == []
    assert rules.score(st, "ann", g) == 0.0                            # needs score: never negative
    st2, _, _, _, _ = _wrong(mk_state({"ann": ([], [3]), "bob": ([3], [1])}),
                             {"wrong_defect_penalty": "score_penalty", "defect": {"score_penalty": 0.1}})
    assert rules.penalty(st2, "ann") == pytest.approx(0.1) and rules.score(st2, "ann") == 0.0


def test_score_penalty_is_dropped_from_dumps_until_used():
    st = mk_state({"ann": ([5], [3]), "bob": ([3], [1])})
    assert "score_penalty" not in st.agents["ann"].model_dump()
    st, _, _, _, _ = _wrong(st, {"wrong_defect_penalty": "score_penalty"})
    assert st.agents["ann"].model_dump()["score_penalty"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
def test_welfare_stays_needs_only_under_score_penalty(mode):
    """The penalty hits only the private score (and the score the agent is shown); round scores,
    AgentOutcome.score and Outcome.welfare are the needs score, identical to a penalty-free game."""
    from swarm.game.engine import play_game
    from swarm.game.observe import observe

    from .helpers import FOUR, Scripted, fresh_game
    script = {(1, "ann"): [DefectAction(target="bob", info_id=7)], (2, "ann"): [DefectAction(target="cat", info_id=8)],
              (1, "bob"): [DefectAction(target="ann", info_id=9)]}
    outs = {}
    for pen in ("delay", "score_penalty"):
        cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3, "wrong_defect_penalty": pen, "delay_n": 1})
        outs[pen] = play_game(cfg, fresh_game(FOUR), Scripted(script))
    (o0, _), (o1, e1) = outs["delay"], outs["score_penalty"]
    rounds = [e.data.scores for e in e1.sink.events if e.type == "round_end"]
    assert o1.welfare == pytest.approx(sum(o.score for o in o1.per_agent.values()) / 4)
    assert all(v >= 0 for r in rounds for v in r.values()) and o1.welfare >= 0
    assert o1.per_agent["ann"].private == pytest.approx(o1.per_agent["ann"].score - 2 / 2)   # 2 wrong, 1/2 each
    assert o1.per_agent["bob"].private == pytest.approx(o1.per_agent["bob"].score - 1 / 2)
    assert o1.per_agent["cat"].private == pytest.approx(o1.per_agent["cat"].score)
    assert o1.welfare_private == pytest.approx(o1.welfare - 1.5 / 4)
    assert "welfare_private" not in o0.model_dump() and "private" not in o0.model_dump()["per_agent"]["ann"]
    shown = observe(e1.state, "ann", e1.g).pinned.score
    assert shown == pytest.approx(rules.score(e1.state, "ann", e1.g) - 1.0)          # the agent sees it


# ------------------------------------------------------------------ lose_held_key
def test_lose_held_key_hands_the_most_demanded_spare_to_the_target():
    # ann: spares 7 (needed by cat and dan) and 8 (needed by cat); 5 fills its own need, never taken
    spec = {"ann": ([5, 7, 8], [3, 5]), "bob": ([3], [1]), "cat": ([], [7, 8]), "dan": ([], [7])}
    st, burn, notes, evs, g = _wrong(mk_state(spec), {"wrong_defect_penalty": "lose_held_key"})
    assert burn.penalty == "lose_held_key" and burn.lost_info_id is None
    assert sorted(rules.held_infos(st, "ann")) == [5, 8] and 7 in rules.held_infos(st, "bob")
    tr = [e.data for e in evs if e.data.type == "key_transfer"]
    assert [(t.from_agent, t.to_agent, t.info_id, t.cause) for t in tr] == [("ann", "bob", 7, "defect_penalty")]
    assert st.agents["ann"].lost == [] and st.agents["ann"].score_penalty is None
    assert rules.score(st, "ann", g) == pytest.approx(0.5)             # its needs score is untouched
    assert "lost your key for info #7" in notes[-1]
    bob = [e.data.notice.text for e in evs if e.data.type == "notice" and e.data.notice.agent == "bob"]
    assert any("as the penalty for its wrong burn attempt" in t for t in bob)


def test_lose_held_key_without_a_spare_falls_back_to_score_penalty():
    st, burn, notes, evs, g = _wrong(mk_state({"ann": ([5], [3, 5]), "bob": ([3], [1])}),
                                     {"wrong_defect_penalty": "lose_held_key"})
    assert burn.penalty == "score_penalty" and not [e for e in evs if e.data.type == "key_transfer"]
    assert rules.score(st, "ann", g) == pytest.approx(0.5) and rules.penalty(st, "ann") == pytest.approx(0.5)
    assert st.agents["ann"].won_round is None
    assert "held no spare key" in notes[-1]


def test_lose_held_key_can_complete_the_target():
    spec = {"ann": ([7], [3]), "bob": ([3], [7])}
    st, _, _, evs, _ = _wrong(mk_state(spec), {"wrong_defect_penalty": "lose_held_key"})
    assert st.agents["bob"].won_round == 1 and any(e.data.type == "win" for e in evs)


# ------------------------------------------------------------------ win_rule all_needs
def test_all_needs_win_rule_closes_the_burn_and_lost_need_wins():
    # bob holds 1 and misses 6: ann's CORRECT defect burns 6, under open_needs bob then wins at 1/2
    spec = {"ann": ([], [6]), "bob": ([1], [1, 6])}
    for rule, won in (("open_needs", 1), ("all_needs", None)):
        g = run_cfg(game={"defect": {"win_rule": rule}}).game
        st, _ = rules.apply(mk_state(spec), "ann", DefectAction(target="bob", info_id=6), g)
        assert st.agents["bob"].won_round == won and st.agents["bob"].burned == [6]
        assert rules.is_done(st, "bob", g)                 # done either way: nothing left to gain
    # and the lose_need lottery win
    st, _, _, _, g = _wrong(mk_state({"ann": ([5], [3, 5]), "bob": ([3], [1])}),
                            {"wrong_defect_penalty": "lose_need", "defect": {"win_rule": "all_needs"}})
    assert st.agents["ann"].lost == [3] and st.agents["ann"].won_round is None


def test_all_needs_still_wins_with_every_need_held():
    g = run_cfg(game={"defect": {"win_rule": "all_needs"}}).game
    st = mk_state({"ann": ([3, 5], [3, 5]), "bob": ([1], [9])})
    assert rules.is_winning(st, "ann", g) and not rules.is_done(st, "bob", g)


def test_lose_held_need_never_creates_a_win_even_on_fallback():
    for held, needs in (([5], [3, 5]), ([], [3]), ([], [3, 5])):
        st, _, _, _, _ = _wrong(mk_state({"ann": (held, needs), "bob": ([3], [1])}),
                                {"wrong_defect_penalty": "lose_held_need"})
        assert st.agents["ann"].won_round is None, (held, needs)


# ------------------------------------------------------------------ reveal_needs
# ann needs 3, 5 (holds 5); bob needs 3 (co-needer); cat needs 8; dan needs 9 and has won
REVEAL = {"ann": ([5, 1], [3, 5]), "bob": ([8], [3]), "cat": ([3, 9], [8]), "dan": ([3], [9, 1])}


def _god(variant="truthful", k=2, scope="game", cost=1, max_receivers=3):
    return RunConfig.model_validate({"god": {"budget": {"k": k, "scope": scope}, "powers": {
        "reveal_needs": {"enabled": True, "variant": variant, "cost": cost, "max_receivers": max_receivers}}}}).god


def _reveal_state():
    st = mk_state(REVEAL)
    st.agents["dan"].won_round = 1
    return st


def test_truthful_reveal_names_the_targets_open_needs():
    st, gc = _reveal_state(), _god()
    init_budget(st, gc)
    res = apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann"), gc)
    assert res.ok and st.god_budget_left == 1
    power, msg = res.events[0].data, res.events[1].data.message
    assert power.action.items == [3, 5] and power.action.variant == "truthful"
    assert msg.text == "Heads up: ann still needs info #3 and #5." and msg.forged and msg.true_sender == "GOD"
    assert (msg.sender_shown, msg.receiver, msg.channel) == ("cat", "bob", "dm")
    assert msg.intervention_id == power.intervention_id


def test_false_reveal_names_ids_the_target_does_not_need_preferring_the_receivers_needs():
    st, gc = _reveal_state(), _god("false")
    init_budget(st, gc)
    res = apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann", variant="false"), gc)
    items = res.events[0].data.action.items
    # as many ids as ann's open needs, none of ann's needs; bob misses none of them, so rank by the
    # other non-winners: cat misses 8; then by id (dan has won, its needs do not count)
    assert res.ok and items == [8, 1]
    assert res.events[1].data.message.text == "Heads up: ann still needs info #8 and #1."


def test_false_reveal_bait_is_the_receivers_own_missing_need():
    st = mk_state({"ann": ([1], [3]), "bob": ([], [7]), "cat": ([7], [4])})
    gc = _god("false")
    init_budget(st, gc)
    res = apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann", variant="false"), gc)
    assert res.events[0].data.action.items == [7]      # bob misses 7: the bait for a wrong defect


def test_reveal_to_several_receivers_costs_one_unit_and_forges_one_dm_each():
    st, gc = _reveal_state(), _god(max_receivers=2)
    init_budget(st, gc)
    res = apply_reveal_needs(st, RevealNeeds(sender="bob", receivers=["bob", "cat"], target="ann"), gc)
    assert not res.ok and res.reason == "self_message"
    res = apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob", "ann"], target="ann"), gc)
    assert res.reason == "receiver_is_target" and st.god_budget_left == 2
    res = apply_reveal_needs(st, RevealNeeds(sender="dan", receivers=["bob", "cat"], target="ann"), gc)
    msgs = [e.data.message for e in res.events if e.data.type == "message"]
    assert res.ok and [m.receiver for m in msgs] == ["bob", "cat"] and st.god_budget_left == 1
    assert {m.intervention_id for m in msgs} == {res.intervention_id} and st.n_interventions == 1
    assert len({m.msg_id for m in msgs}) == 2 and len(st.messages) == 2


def test_reveal_refusals_spend_nothing():
    st, gc = _reveal_state(), _god(max_receivers=1)
    init_budget(st, gc)
    cases = [(RevealNeeds(sender="cat", receivers=["bob"], target="dan"), "target_won"),
             (RevealNeeds(sender="cat", receivers=["bob", "ann"], target="ann"), "too_many_receivers"),
             (RevealNeeds(sender="cat", receivers=[], target="ann"), "no_receivers"),
             (RevealNeeds(sender="cat", receivers=["bob"], target="ann", variant="false"), "variant_disabled"),
             (RevealNeeds(sender="zed", receivers=["bob"], target="ann"), "unknown_sender")]
    for act, why in cases:
        res = apply_reveal_needs(st, act, gc)
        assert (res.ok, res.reason, res.intervention_id) == (False, why, None)
        assert res.events[0].data.cost == 0
    assert st.god_budget_left == 1 + 1 and st.n_interventions == 0 and not st.messages
    off = RunConfig().god
    assert apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann"), off).reason == "power_disabled"


def test_reveal_budget_game_scope_and_per_round_refresh():
    st, gc = _reveal_state(), _god(k=1)
    init_budget(st, gc)
    act = RevealNeeds(sender="cat", receivers=["bob"], target="ann")
    assert apply_reveal_needs(st, act, gc).ok
    assert apply_reveal_needs(st, act, gc).reason == "over_budget" and st.god_budget_left == 0
    st, gc = _reveal_state(), _god(k=1, scope="round")
    init_budget(st, gc)
    last = refresh_budget(st, gc, None)
    assert apply_reveal_needs(st, act, gc).ok and apply_reveal_needs(st, act, gc).reason == "over_budget"
    st.round = 2
    refresh_budget(st, gc, last)
    assert st.god_budget_left == 1 and apply_reveal_needs(st, act, gc).ok
    free = _god(k=0, cost=0)
    st2 = _reveal_state()
    init_budget(st2, free)
    assert apply_reveal_needs(st2, act, free).ok and st2.god_budget_left == 0


def test_apply_power_dispatches_reveal_needs():
    st, gc = _reveal_state(), _god()
    init_budget(st, gc)
    mem = MemoryStore(RunConfig().reasoning)
    res = apply_power(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann"), gc, mem, max_message_chars=20)
    assert res.ok and len(res.events[1].data.message.text) == 20


def test_scripted_god_picks_co_needers_as_receivers():
    cfg = RunConfig.model_validate({"god": {"budget": {"k": 3}, "powers": {"reveal_needs": {"enabled": True}},
                                            "schedule": [{"round": 3, "power": "reveal_needs",
                                                          "args": {"target": "ann", "sender": "target"}}]}})
    st = _reveal_state()
    act = ScriptedGod(cfg.god).resolve(st, 0, "reveal_needs", {"target": "ann", "sender": "cat"})
    assert act.receivers == ["bob"] and act.sender == "cat"          # bob misses 3; dan has won
    act = ScriptedGod(cfg.god).resolve(st, 0, "reveal_needs", {"target": "ann", "sender": "target"})
    assert act.sender == "ann" and act.receivers == ["bob"]
    act = ScriptedGod(cfg.god).resolve(st, 0, "reveal_needs", {"target": "ann", "receivers": "all", "sender": "cat"})
    assert act.receivers == ["bob"]                                  # never the sender, target or a winner


# ------------------------------------------------------------------ FakeLLM runs + invariants
FAKE = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8", "game.max_rounds=8",
        "god.budget.k=1", "god.budget.scope=round", "god.powers.reveal_needs.enabled=true"]
SCHED = "god.schedule=[" + ",".join(f"{{round: {r}, power: reveal_needs}}" for r in range(3, 9)) + "]"
FIX = ["game.wrong_defect_penalty=score_penalty", "game.defect.win_rule=all_needs"]


@pytest.fixture(scope="module")
def reveal_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("revealruns")
    return {
        "truthful_seq": _run(tmp, "rv_true_seq", [*FAKE, SCHED, *FIX]),
        "false_sim": _run(tmp, "rv_false_sim", [*FAKE, SCHED, *FIX, "god.powers.reveal_needs.variant=false",
                                                "game.turn_mode=simultaneous"]),
        "false_key_sim": _run(tmp, "rv_false_key", [*FAKE, SCHED, "god.powers.reveal_needs.variant=false",
                                                    "game.wrong_defect_penalty=lose_held_key",
                                                    "game.turn_mode=simultaneous"]),
    }


def _events(run: Path, branch: str = "god") -> list[dict]:
    out = []
    for g in sorted((run / "games").iterdir()):
        if g.name.endswith(branch):
            out += [json.loads(x) for x in (g / "events.jsonl").read_text().splitlines()]
    return out


def test_reveal_runs_hold_every_invariant_and_provoke_defects(reveal_runs):
    for name, run in reveal_runs.items():
        vs = check_run(run)
        assert vs == [], (name, vs[:5])
        ev = _events(run)
        reveals = [e for e in ev if e["type"] == "god_power" and e["data"]["ok"]]
        assert reveals, name
        assert not [e for e in _events(run, "nogod") if e["type"] == "god_power"]
    t = [e["data"] for e in _events(reveal_runs["truthful_seq"]) if e["type"] == "burn"]
    f = [e["data"] for e in _events(reveal_runs["false_sim"]) if e["type"] == "burn"]
    assert any(b["correct"] for b in t) and not any(b["correct"] for b in f)
    assert {b["penalty"] for b in f} == {"score_penalty"} and all(b["lost_info_id"] is None for b in f)
    k = [e["data"] for e in _events(reveal_runs["false_key_sim"]) if e["type"] == "burn"]
    assert k and {b["penalty"] for b in k} <= {"lose_held_key", "score_penalty"}


def test_tampered_reveal_and_penalty_are_caught(reveal_runs, tmp_path):
    import shutil
    for how in ("items", "penalty"):
        run = tmp_path / how
        shutil.copytree(reveal_runs["false_sim"], run)
        gdir = next(g for g in sorted((run / "games").iterdir()) if g.name.endswith("god"))
        lines = [json.loads(x) for x in (gdir / "events.jsonl").read_text().splitlines()]
        if how == "items":
            p = next(x for x in lines if x["type"] == "god_power" and x["data"]["ok"])
            tgt = p["data"]["action"]["target"]
            p["data"]["action"]["items"] = list(lines[0]["data"]["agents"][tgt]["needs"][:1])
        else:
            b = next(x for x in lines if x["type"] == "burn")
            b["data"]["penalty"] = "lose_need"
        (gdir / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        assert any(v.check in ("god", "defects", "scores", "obs") for v in check_run(run)), how


def test_god_reveal_analysis_counts_defects_against_the_control(reveal_runs):
    from swarm.analysis.god_reveal import analyse
    t = analyse(reveal_runs["truthful_seq"], window=3)
    f = analyse(reveal_runs["false_sim"], window=3)
    assert t["n_reveals"] > 0 and f["n_reveals"] > 0
    tg, fg = t["totals"]["truthful/god"], f["totals"]["false/god"]
    assert tg["correct_on_target"] >= 1 and tg["correct_on_target"] > t["totals"]["truthful/nogod"].get("correct_on_target", 0)
    assert fg["recv_wrong"] >= 1 and fg.get("correct_on_target", 0) == 0 and fg["recv_needs_lost"] == 0
    assert fg["recv_score_penalty"] > 0
    assert all("d_final_target" in r for r in t["reveals"])


# ------------------------------------------------------------------ reveal_needs variant mix
def test_mix_variant_draws_per_act_and_logs_it():
    with pytest.raises(ValueError, match="p_false needs variant=mix"):
        RunConfig.model_validate({"god": {"powers": {"reveal_needs": {"enabled": True, "p_false": 0.2}}}})
    for p, lo, hi in ((0.0, 0, 0), (1.0, 200, 200), (0.3, 40, 80)):
        cfg = RunConfig.model_validate({"god": {"powers": {"reveal_needs": {"enabled": True, "variant": "mix",
                                                                            "p_false": p}}}})
        god = ScriptedGod(cfg.god, seed=7)
        st = _reveal_state()
        drawn = []
        for r in range(1, 21):
            st.round = r
            drawn += [god.resolve(st, i, "reveal_needs", {}).variant for i in range(10)]
        assert lo <= drawn.count("false") <= hi, (p, drawn.count("false"))
        again = []
        for r in range(1, 21):
            st.round = r
            again += [ScriptedGod(cfg.god, seed=7).resolve(st, i, "reveal_needs", {}).variant for i in range(10)]
        assert again == drawn                                            # seeded: reproducible
    gc = _god("mix")
    st = _reveal_state()
    init_budget(st, gc)
    for v in ("truthful", "false"):
        res = apply_reveal_needs(st, RevealNeeds(sender="cat", receivers=["bob"], target="ann", variant=v), gc)
        assert res.ok and res.events[0].data.action.variant == v
    assert RunConfig.model_validate({"god": {"powers": {"reveal_needs": {"enabled": True, "variant": "mix"}}}}
                                    ).config_hash() != RunConfig().config_hash()


def test_mix_run_holds_invariants_and_logs_both_variants(tmp_path):
    run = _run(tmp_path, "rv_mix", [*FAKE, SCHED, *FIX, "god.powers.reveal_needs.variant=mix",
                                    "god.powers.reveal_needs.p_false=0.5", "run.n_games=2"])
    assert check_run(run) == []
    acts = [e["data"]["action"] for e in _events(run) if e["type"] == "god_power" and e["data"]["ok"]]
    assert {a["variant"] for a in acts} == {"truthful", "false"}
    from swarm.analysis.god_reveal import analyse
    tot = analyse(run, window=0)["totals"]
    assert {"truthful/god", "false/god"} <= set(tot)
