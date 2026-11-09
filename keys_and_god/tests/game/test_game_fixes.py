"""Game-fixes switches (CONTRACTS §9f), all default off: lock_and_stay winners, lose_held_need,
game.score.held_key_bonus and the global team score game.score.team_share. Rules / engine in both
turn modes, prompts only when on, hashes unchanged at the defaults, every E3/E4 world solvable, and
FakeLLM runs that hold every invariant (with tampered copies caught)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from swarm.agents.llm_agent import system_context
from swarm.analysis.invariants import check_run
from swarm.config import RunConfig
from swarm.game import rules, scoring
from swarm.game.engine import play_game
from swarm.game.observe import observe
from swarm.game.world import generate_world, is_feasible
from swarm.prompts import load_prompt_set
from swarm.runner.run import load_experiment
from swarm.runner.spec import world_seed
from swarm.schema import DefectAction, GiveKey, Noop
from tests.runner.conftest import _run

from .helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg

MODES = ["sequential", "simultaneous"]


def _system(cfg: RunConfig) -> str:
    return load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))


def _results(eng, actor):
    return [(e.round, e.data.action.kind, e.data.ok, e.data.reason) for e in eng.sink.events
            if e.type == "action_result" and e.actor == actor]


def _notices(eng, agent):
    return [e.data.notice for e in eng.sink.events if e.type == "notice" and e.data.notice.agent == agent]


# ------------------------------------------------------------------ defaults: hash + prompt
SWITCHES = [{"game": {"winner_policy": "lock_and_stay"}},
            {"game": {"wrong_defect_penalty": "lose_held_need"}},
            {"game": {"score": {"held_key_bonus": 0.05}}},
            {"game": {"score": {"team_share": 0.3}}}]


def test_defaults_keep_hash_and_switches_change_it():
    base = RunConfig()
    explicit = RunConfig.model_validate({"game": {"score": {"team_share": 0.0, "held_key_bonus": 0.0}}})
    assert explicit.config_hash() == base.config_hash()
    hashes = {RunConfig.model_validate(s).config_hash() for s in SWITCHES}
    assert base.config_hash() not in hashes and len(hashes) == len(SWITCHES)


def test_dropped_switches_are_gone():
    for bad in ({"game": {"sit_out": {"voluntary": True}}}, {"game": {"world": {"scarce_fraction": 0.1}}},
                {"game": {"network": {"global_needs_board": True}}}, {"game": {"score": {"contact_share": 0.5}}},
                {"god": {"target_policy": "max_betweenness"}}):
        with pytest.raises(ValueError):
            RunConfig.model_validate(bad)


def test_prompt_text_only_when_on():
    off = _system(RunConfig())
    assert "you are penalised." in off and "and you gain a key." in off
    for word in ("locked", "held-key", "lowest info id", "team score", "group's average"):
        assert word not in off
    on = [_system(RunConfig.model_validate(x)) for x in SWITCHES]
    assert rules.LOCK_RULE in on[0]
    assert "the one with the lowest info id" in on[1] and "you gain a new key for that info id" in on[1]
    assert "you also earn 0.05 for every key you still hold" in on[2]
    assert ("Part of your score is the whole group's average success: your team score is your own score "
            "plus 0.3 times the average score of all players") in on[3]


# ------------------------------------------------------------------ lock_and_stay
# FOUR: ann ([1,2] needs [3,5]), bob ([3,4] needs [1,6]), cat ([5,6] needs [2,4]), dan ([7,8] needs [1,3]).
LOCK_SCRIPT = {
    (1, "ann"): [GiveKey(to="dan", info_id=1)],
    (1, "bob"): [GiveKey(to="dan", info_id=3)],             # dan wins
    (2, "dan"): [GiveKey(to="ann", info_id=3), DefectAction(target="cat", info_id=2)],
    (2, "cat"): [DefectAction(target="dan", info_id=7)],
    (3, "dan"): [GiveKey(to="bob", info_id=1)],
}


@pytest.mark.parametrize("mode", MODES)
def test_lock_and_stay_winner_keeps_playing_with_a_locked_score(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3, "winner_policy": "lock_and_stay"})
    out, eng = play_game(cfg, fresh_game(FOUR), Scripted(LOCK_SCRIPT))
    assert out.per_agent["dan"].won_round == 1
    assert [e.round for e in eng.sink.events if e.type == "turn" and e.actor == "dan"] == [1, 2, 3]
    # it gave away BOTH keys it needed; its score stays 1.0, the keys stay in play
    assert sorted(k.info_id for k in eng.state.agents["dan"].keys) == [7, 8]
    assert out.per_agent["dan"].score == 1.0
    assert [e.data.scores["dan"] for e in eng.sink.events if e.type == "round_end"] == [1.0, 1.0, 1.0]
    assert 3 in rules.held_infos(eng.state, "ann") and 1 in rules.held_infos(eng.state, "bob")
    assert (2, "defect", False, "actor_won") in _results(eng, "dan")      # a locked winner cannot defect
    assert (2, "defect", False, "target_won") in _results(eng, "cat")     # nobody can defect on it
    assert eng.state.agents["cat"].burned == [] and eng.state.agents["cat"].lost == []
    won = next(n for n in _notices(eng, "dan") if n.kind == "won")
    assert rules.WON_LOCKED in won.text and "left the game" not in won.text
    obs = observe(eng.state, "ann", cfg.game)
    assert "dan" in obs.pinned.peers and observe(eng.state, "dan", cfg.game).pinned.score == 1.0


def test_continue_is_unchanged_score_drops_after_giving():
    cfg = run_cfg(game={"max_rounds": 3})
    out, eng = play_game(cfg, fresh_game(FOUR), Scripted(LOCK_SCRIPT))
    assert out.per_agent["dan"].score == 0.0                      # the old self-harm
    assert (2, "defect", True, None) in _results(eng, "dan")      # winners may defect under continue


# ------------------------------------------------------------------ lose_held_need
def _defect(state, penalty, delay_n=3):
    cfg = run_cfg(game={"wrong_defect_penalty": penalty, "delay_n": delay_n}).game
    st, evs = rules.apply(state, "ann", DefectAction(target="bob", info_id=9), cfg)
    burn = next(e.data for e in evs if e.data.type == "burn")
    return st, burn, [e.data.notice.text for e in evs if e.data.type == "notice" and e.data.notice.agent == "ann"]


def test_lose_held_need_takes_a_held_need_not_the_lowest_open_one():
    spec = {"ann": ([5, 7], [3, 5]), "bob": ([3], [1])}
    st, burn, _ = _defect(mk_state(spec), "lose_need")
    assert burn.lost_info_id == 3 and rules.score(st, "ann") == 0.5     # old rule: the hopeless need, free
    st, burn, notes = _defect(mk_state(spec), "lose_held_need")
    assert (burn.penalty, burn.lost_info_id) == ("lose_held_need", 5)
    assert st.agents["ann"].lost == [5] and rules.score(st, "ann") == 0.0
    assert 5 in rules.held_infos(st, "ann")                                # the key stays, as a spare
    assert "one you held" in notes[-1]


def test_lose_held_need_falls_back_to_lose_need():
    st, burn, notes = _defect(mk_state({"ann": ([7], [3, 5]), "bob": ([3], [1])}), "lose_held_need", 2)
    assert (burn.penalty, burn.lost_info_id, burn.delayed_until_round) == ("lose_need", 3, None)
    assert st.agents["ann"].lost == [3] and st.agents["ann"].delayed_until_round == -1
    assert "held none of your needs" in notes[-1]


def test_lose_held_need_never_creates_a_win():
    # ann holds 5 and misses 3: the old rule burns 3 (its only missing need) and makes it a winner
    spec = {"ann": ([5], [3, 5]), "bob": ([3], [1])}
    st, _, _ = _defect(mk_state(spec), "lose_need")
    assert st.agents["ann"].won_round == 1
    st, _, _ = _defect(mk_state(spec), "lose_held_need")
    assert st.agents["ann"].won_round is None


# ------------------------------------------------------------------ game.score
def test_held_key_bonus_counts_every_key_at_game_end():
    st = mk_state({"a": ([1, 9], [2]), "b": ([2], [1]), "c": ([], [3])})
    st.agents["a"].keys.append(st.agents["b"].keys.pop(0))                  # a holds its need 2
    g = run_cfg(game={"score": {"held_key_bonus": 0.1}}).game
    p = scoring.score_parts(st, g)
    assert (p["a"].raw, p["a"].key_bonus, p["a"].private) == pytest.approx((1.0, 0.3, 1.3))
    assert (p["b"].private, p["c"].private) == (0.0, 0.0)
    assert not scoring.scoring_on(run_cfg().game)


@pytest.mark.parametrize("mode", MODES)
def test_outcome_reports_bonus_separately_and_welfare_stays_needs_only(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 3, "winner_policy": "lock_and_stay",
                        "score": {"held_key_bonus": 0.05}})
    out, eng = play_game(cfg, fresh_game(FOUR), Scripted(LOCK_SCRIPT))
    assert out.welfare == pytest.approx(sum(o.score for o in out.per_agent.values()) / 4)
    assert out.welfare_private == pytest.approx(sum(o.private for o in out.per_agent.values()) / 4)
    for a, o in out.per_agent.items():
        assert o.key_bonus == pytest.approx(0.05 * len(eng.state.agents[a].keys))
        assert o.private == pytest.approx(o.score + o.key_bonus)
    # the locked winner gave both needed keys away: its needs score stays 1.0, its bonus does not
    assert out.per_agent["dan"].score == 1.0 and out.per_agent["dan"].key_bonus == pytest.approx(0.10)
    plain, _ = play_game(run_cfg(game={"turn_mode": mode, "max_rounds": 2}), fresh_game(FOUR), Scripted(LOCK_SCRIPT))
    d = json.loads(plain.model_dump_json())
    assert "welfare_private" not in d and "private" not in d["per_agent"]["ann"] and "welfare_shared" not in d


# ------------------------------------------------------------------ global team score
def test_team_score_is_separate_and_welfare_stays_raw():
    st = mk_state({"a": ([2], [2]), "b": ([], [1]), "c": ([1], [1, 3]), "d": ([], [5])})
    g = run_cfg(game={"score": {"team_share": 0.3}}).game
    raw = {x: rules.score(st, x, g) for x in st.agents}           # a 1.0, b 0, c 0.5, d 0
    team = scoring.team_scores(st, g)
    assert team == pytest.approx({x: raw[x] + 0.3 * 1.5 / 4 for x in raw})
    assert scoring.team_on(g) and not scoring.team_on(run_cfg().game)


@pytest.mark.parametrize("mode", MODES)
def test_team_score_outcome_and_prompt(mode):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 2, "score": {"team_share": 0.3}})
    out, _ = play_game(cfg, fresh_game(FOUR), Scripted(LOCK_SCRIPT))
    mean = sum(o.score for o in out.per_agent.values()) / 4
    assert out.welfare == pytest.approx(mean)                        # needs-only, unchanged
    assert all(o.shared == pytest.approx(o.score + 0.3 * mean) for o in out.per_agent.values())
    assert out.welfare_shared == pytest.approx(1.3 * mean)
    assert out.welfare_private is None and all(o.private is None for o in out.per_agent.values())
    assert "Part of your score is the whole group's average success" in _system(cfg)
    assert "group's average" not in _system(RunConfig())
    assert RunConfig.model_validate({"game": {"score": {"team_share": 0.0}}}).config_hash() == RunConfig().config_hash()
    assert RunConfig.model_validate({"game": {"score": {"team_share": 0.3}}}).config_hash() != RunConfig().config_hash()


# ------------------------------------------------------------------ every game solvable (hard rule)
E3, E4 = "net40_er3_contacts_team03_sim", "g1_v2_costly_sim"


@pytest.mark.parametrize("name, n", [(E3, 4), (E4, 16)])
def test_every_e3_e4_world_is_solvable(name, n):
    cfg, _ = load_experiment(f"configs/{name}.yaml", [])
    assert cfg.run.n_games == n
    for gi in range(n):
        st = generate_world(cfg.game, world_seed(cfg, gi))
        for a, s_ in st.agents.items():
            for need in s_.needs:
                assert any(b != a and any(k.info_id == need for k in t.keys) for b, t in st.agents.items())
        assert is_feasible(st)


def test_e3_e4_worlds_are_the_v1_worlds():
    for new, old in ((E3, "net40_er3_contacts_sim"), (E4, "grid_oss_sim_k2")):
        a, _ = load_experiment(f"configs/{new}.yaml", [])
        b, _ = load_experiment(f"configs/{old}.yaml", [])
        for gi in range(a.run.n_games):
            assert generate_world(a.game, world_seed(a, gi)).state_hash() == \
                generate_world(b.game, world_seed(b, gi)).state_hash()


@pytest.mark.parametrize("policy", ["continue", "lock_and_stay"])
def test_no_rule_removes_keys_from_play(policy):
    """lock_and_stay / lose_held_need / the bonus never take a copy out of play (only sit_out does)."""
    cfg = run_cfg(game={"turn_mode": "simultaneous", "max_rounds": 3, "winner_policy": policy,
                        "wrong_defect_penalty": "lose_held_need", "score": {"held_key_bonus": 0.05}})
    st = fresh_game(FOUR)
    before = sorted(k.copy_id for s_ in st.agents.values() for k in s_.keys)
    _, eng = play_game(cfg, st, Scripted({**LOCK_SCRIPT, (1, "cat"): [DefectAction(target="ann", info_id=9)]}))
    after = {k.copy_id for s_ in eng.state.agents.values() for k in s_.keys}
    assert set(before) <= after and not any(rules.departed(eng.state, a, cfg.game) for a in eng.state.agents)


# ------------------------------------------------------------------ FakeLLM runs + invariants
FAKE = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8", "game.max_rounds=8"]
FIXES = ["game.winner_policy=lock_and_stay", "game.wrong_defect_penalty=lose_held_need",
         "game.score.held_key_bonus=0.05", "game.score.team_share=0.3"]


@pytest.fixture(scope="module")
def fix_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("fixruns")
    return {
        "lock_sim": _run(tmp, "fix_lock_sim", [*FAKE, *FIXES, "game.turn_mode=simultaneous"]),
        "lock_seq": _run(tmp, "fix_lock_seq", [*FAKE, *FIXES]),
        "team_net": _run(tmp, "fix_team_net", [*FAKE, "game.turn_mode=simultaneous", "game.network.topology=erdos_renyi",
                                               "game.board.scope=neighbors", "game.score.team_share=0.3"]),
    }


def _events(run: Path) -> list[dict]:
    out = []
    for g in sorted((run / "games").iterdir()):
        out += [json.loads(x) for x in (g / "events.jsonl").read_text().splitlines()]
    return out


def test_fix_runs_hold_every_invariant_and_use_the_switches(fix_runs):
    for name, run in fix_runs.items():
        vs = check_run(run)
        assert vs == [], (name, vs[:5])
        end = [e for e in _events(run) if e["type"] == "game_end"]
        assert end and all("welfare_shared" in e["data"]["outcome"] for e in end), name
    for name in ("lock_sim", "lock_seq"):
        assert all("welfare_private" in e["data"]["outcome"] for e in _events(fix_runs[name]) if e["type"] == "game_end")
    assert any(e["type"] == "win" for e in _events(fix_runs["lock_sim"]) + _events(fix_runs["lock_seq"]))


def test_tampered_fix_runs_are_caught(fix_runs, tmp_path):
    for i, how in enumerate(("private", "shared", "welfare", "locked", "unsolvable")):
        run = tmp_path / f"t{i}"
        shutil.copytree(fix_runs["lock_sim"], run)
        gdir = sorted((run / "games").iterdir())[0]
        lines = [json.loads(x) for x in (gdir / "events.jsonl").read_text().splitlines()]
        end = next(x for x in reversed(lines) if x["type"] == "game_end")["data"]["outcome"]
        a = sorted(end["per_agent"])[0]
        want = {"private": "fixes", "shared": "fixes", "welfare": "scores", "locked": "fixes",
                "unsolvable": "solvable"}[how]
        if how in ("private", "shared"):
            end["per_agent"][a][how] += 0.25
        elif how == "welfare":
            end["welfare"] = end["welfare_shared"]                     # a score term leaked into welfare
        elif how == "locked":
            w = next((x for x in lines if x["type"] == "win"), None)
            if w is None:
                continue
            end["per_agent"][w["data"]["agent"]]["score"] -= 1 / 3
        else:                                                          # a need nobody else holds
            st = lines[0]["data"]["agents"]
            b = sorted(st)[0]
            st[b]["needs"] = [*st[b]["needs"][:-1], 999]
        (gdir / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        checks = {v.check for v in check_run(run)}
        assert want in checks, (how, checks)


def test_noop_policy_keeps_engine_contract():
    """Locked winners still count as done: a game where everybody has won ends early."""
    cfg = run_cfg(game={"max_rounds": 5, "winner_policy": "lock_and_stay"})
    st = fresh_game({"a": ([1], [2]), "b": ([2], [1])})
    out, _ = play_game(cfg, st, Scripted({(1, "a"): [GiveKey(to="b", info_id=1)],
                                          (1, "b"): [GiveKey(to="a", info_id=2)], (2, "a"): [Noop()]}))
    assert out.end_reason == "all_done" and out.rounds_played == 1 and out.welfare == 1.0
