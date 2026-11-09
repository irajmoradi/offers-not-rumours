"""Showcase features (configs/showcase100_sim.yaml): key_placement far and the community threshold bonus
(game.score.community_bonus). Both default off: hashes, worlds and prompts byte-identical; far keeps every
world solvable and puts every holder >= far_min_hops away; the community bonus is reported next to the
needs-only welfare, never inside it; the invariant checker replays it; the showcase and control configs
pair by world."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.agents.llm_agent import system_context
from swarm.analysis.invariants import check_run
from swarm.config import REPO_ROOT, from_dict
from swarm.game import network, scoring
from swarm.game.world import generate_world, is_feasible
from swarm.prompts import load_prompt_set
from swarm.runner.run import load_experiment
from swarm.runner.spec import build_specs
from tests.runner.conftest import _run

SBM100 = {"topology": "sbm", "communities": 5, "mean_degree": 4, "mixing": 0.1, "min_bridges": 8}


def _game(net: dict, n: int = 100, **score):
    return from_dict({"game": {"n_agents": n, "network": net, **({"score": score} if score else {})}}).game


# ------------------------------------------------------------------ defaults byte-identical
def test_defaults_hash_neutral():
    base = from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm"}}})
    same = from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm", "far_min_hops": 3},
                               "score": {"community_threshold": 0.5, "community_bonus": 0.0}}})
    assert base.config_hash() == same.config_hash()
    plain = from_dict({})
    assert plain.config_hash() == from_dict({"game": {"score": {"community_threshold": 0.7}}}).config_hash()
    assert from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm", "key_placement": "far"}}}).config_hash() \
        != base.config_hash()
    on = from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm"}, "score": {"community_bonus": 0.15}}})
    assert on.config_hash() != base.config_hash()
    assert on.config_hash() != from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm"},
                                                   "score": {"community_bonus": 0.15,
                                                             "community_threshold": 0.8}}}).config_hash()


def test_prompt_lines_only_when_on():
    ps = load_prompt_set("agent/v1")
    off = from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm"}}})
    on = from_dict({"game": {"n_agents": 40, "network": {"topology": "sbm", "communities": 4},
                             "score": {"community_bonus": 0.15}}})
    assert "communities" not in ps.render("system.j2", **system_context(off))
    line = [x for x in ps.render("system.j2", **system_context(on)).splitlines() if "communities" in x]
    assert line == ["- The players are split into 4 communities: at the end of the game, every member of a "
                    "community in which at least 90% of the members have won earns 0.15 extra."]


def test_config_validation():
    with pytest.raises(ValueError, match="community topology"):
        from_dict({"game": {"n_agents": 40, "network": {"topology": "erdos_renyi"}, "score": {"community_bonus": 0.1}}})
    with pytest.raises(ValueError, match="key_placement=far"):
        from_dict({"game": {"network": {"key_placement": "far"}}})


# ------------------------------------------------------------------ far placement
@pytest.mark.parametrize("d", [3, 4])
def test_far_strict_solvable_same_deal(d):
    for seed in range(3):
        r = generate_world(_game(SBM100), seed)
        f = generate_world(_game({**SBM100, "key_placement": "far", "far_min_hops": d}), seed)
        assert is_feasible(f)
        assert [s.keys for s in r.agents.values()] == [s.keys for s in f.agents.values()]       # same deal
        assert [s.contacts for s in r.agents.values()] == [s.contacts for s in f.agents.values()]
        assert min(network.need_hops(f)) >= d                     # strict: every holder >= d hops away
        assert sum(map(network.need_hops(f).count, (d,))) > 0
        for a, s in f.agents.items():                             # every need held by another agent
            for i in s.needs:
                assert any(i in {k.info_id for k in o.keys} for x, o in f.agents.items() if x != a)
        assert generate_world(_game({**SBM100, "key_placement": "far", "far_min_hops": d}), seed).state_hash() \
            == f.state_hash()                                     # deterministic


def test_far_falls_back_when_impossible():
    # community_hubs has diameter <= 3 (hub clique), so 4 hops is impossible: far infos are only preferred
    cfg = _game({"topology": "community_hubs", "communities": 4, "key_placement": "far", "far_min_hops": 4}, n=40)
    st = generate_world(cfg, 1)
    assert is_feasible(st)
    m = network.network_meta(st, cfg, 1)
    assert m["far_min_hops"] == 4 and m["share_needs_far"] == 0.0


def test_far_meta_reports_distances():
    cfg = _game({**SBM100, "key_placement": "far"})
    st = generate_world(cfg, 0)
    m = network.network_meta(st, cfg, 0)
    assert m["share_needs_far"] == 1.0 and min(int(k) for k in m["need_hops_hist"]) >= 3


# ------------------------------------------------------------------ community bonus
def test_community_scores_threshold():
    cfg = _game({"topology": "sbm", "communities": 2}, n=20, community_bonus=0.15, community_threshold=0.9)
    comm = {f"a{i}": (0 if i < 10 else 1) for i in range(20)}
    won = {a: True for a in comm}
    won["a0"] = False                                             # 9/10 of community 0 won: >= 90%
    won["a10"] = won["a11"] = False                               # 8/10 of community 1: below
    sc = scoring.community_scores(won, comm, cfg)
    assert all(sc[f"a{i}"] == 0.15 for i in range(10)) and all(sc[f"a{i}"] == 0.0 for i in range(10, 20))
    assert scoring.community_on(cfg) and not scoring.community_on(_game({"topology": "sbm"}, n=20))


NET = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=12", "game.max_rounds=5",
       "game.network.topology=sbm", "game.network.communities=2", "game.board.scope=neighbors",
       "game.winner_policy=lock_and_stay", "game.score.held_key_bonus=0.05"]


@pytest.fixture(scope="module")
def comm_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("showcase")
    return {"on": _run(tmp, "c_on", [*NET, "game.score.community_bonus=0.15", "game.score.community_threshold=0.5",
                                     "game.score.team_share=0.2", "game.network.key_placement=far",
                                     "game.network.far_min_hops=2"]),
            "full": _run(tmp, "c_full", [*NET, "game.max_rounds=8", "game.score.community_bonus=0.15",
                                         "game.score.held_key_scope=initial", "game.network.key_placement=mixed",
                                         "game.trade.enabled=true",
                                         "game.score.community_threshold_count=4", "game.score.delivery_credit=0.1",
                                         "game.network.connect.enabled=true", "game.network.connect.cost_rounds=0",
                                         "game.network.connect.require_known=true"]),
            "off": _run(tmp, "c_off", NET)}


def test_community_runs_invariants_and_fields(comm_runs):
    for name, run in comm_runs.items():
        assert check_run(run) == [], name
        for d in sorted((run / "games").iterdir()):
            out = json.loads((d / "outcome.json").read_text())
            has = "welfare_community" in out and all("community_bonus" in o for o in out["per_agent"].values())
            assert has == (name != "off")
            assert ("delivery_credit_total" in out) == (name == "full")
            if name != "off":
                n = len(out["per_agent"])
                assert abs(out["welfare"] - sum(o["score"] for o in out["per_agent"].values()) / n) < 1e-9
                calls = [json.loads(x) for x in (d / "llm_calls.jsonl").read_text().splitlines()]
                user = calls[0]["request"]["messages"][-1]["content"]
                assert "Your community has 6 players; your contacts in it:" in user


# ------------------------------------------------------------------ showcase + control configs
def _specs(name: str, overrides: list[str] | None = None):
    cfg, opts = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", overrides or [])
    return [s for g in build_specs(cfg, opts) for s in g]


def test_showcase_and_control_pair_by_world():
    show, ctrl = _specs("showcase100_sim"), _specs("showcase100_control_sim")
    assert [s.branch for s in show] == ["god", "god"]
    assert [s.branch for s in ctrl] == ["nogod", "nogod"]
    nf = _specs("showcase100_control_sim", ["runner.fork.only_branches=[noisefork]"])
    assert [c.branch for c in nf] == ["noisefork"] * 2 and nf[0].cfg.run.resample_from_round == 3
    ctrl = ctrl + nf
    for s in show:
        mates = [c for c in ctrl if c.game_idx == s.game_idx]
        assert {(c.game_id, c.world_seed) for c in mates} == {(s.game_id, s.world_seed)}
    cfg = show[0].cfg
    assert cfg.game.n_agents == 100 and cfg.game.network.key_placement == "mixed" and cfg.game.max_rounds == 25
    assert cfg.game.score.held_key_scope == "initial"
    sc, cn = cfg.game.score, cfg.game.network.connect
    assert (sc.team_share, sc.community_threshold_count, sc.delivery_credit) == (0.0, 16, 0.1)
    assert cn.enabled and cn.cost_rounds == 0 and cn.require_known and cn.max_new_edges_per_agent is None
    assert cfg.god.budget.scope == "round" and cfg.god.budget.k == 2
    assert sorted((sp.round, sp.power) for sp in cfg.god.schedule) == sorted(
        (r, p) for r in range(3, 21) for p in ("message_fraud", "reveal_needs"))
    assert cfg.god.powers.reveal_needs.variant == "mix" and cfg.game.trade.enabled
    assert cfg.game.wrong_defect_penalty == "score_penalty" and cfg.game.defect.win_rule == "all_needs"
    for s in show:                                            # both showcase worlds: exactly one need per bucket
        st = generate_world(cfg.game, s.world_seed)
        hops = network.need_hops(st)
        assert is_feasible(st) and all(sorted(min(h, 3) for h in hops[i:i + 3]) == [1, 2, 3]
                                       for i in range(0, len(hops), 3))


def test_showcase_sim_reach():
    from swarm.analysis.showcase_sim import run_job

    assert run_job(("reach", "far3", "board", 0, None))["welfare"] == 0.0        # 1-hop info cannot reach
    assert run_job(("reach", "far3", "gossipAll", 0, None))["welfare"] > 0.8     # flooding can
    h = run_job(("hoard", "far3", "oracle", 0, "amber_comet"))
    assert set(h) == {"job", "H", "R"} and h["R"]["keys"]["amber_comet"] >= 5


# ------------------------------------------------------------------ delivery credit (anti-farming)
def _ledger(adj: dict[str, list[str]], keys: dict[str, list[tuple[str, int]]], needs: dict[str, list[int]]):
    from types import SimpleNamespace as NS

    led = scoring.DeliveryLedger(0.10)
    led.start({a: NS(contacts=adj[a], needs=needs.get(a, []), burned=[], lost=[],
                     keys=[NS(copy_id=c, info_id=i) for c, i in keys.get(a, [])]) for a in adj})
    return led, NS


def _give(led, NS, cid, info, frm, to, rnd=1):
    led.on_event(NS(type="key_transfer", copy_id=cid, info_id=info, from_agent=frm, to_agent=to, cause="give"), rnd)


def test_delivery_credit_loop_pays_at_most_d():
    # path A - B - ... ; C adjacent to A and to N (needer): d(A, N) = 2
    adj = {"A": ["B", "C"], "B": ["A"], "C": ["A", "N"], "N": ["C"]}
    led, NS = _ledger(adj, {"A": [("k7-0", 7)]}, {"N": [7]})
    for frm, to in (("A", "B"), ("B", "A"), ("A", "C"), ("C", "N")):          # loop A->B->A->C->N
        _give(led, NS, "k7-0", 7, frm, to)
    assert len(led.fills) == 1 and led.fills[0]["d"] == 2
    assert led.fills[0]["paid"] == ["C", "A"]                                   # most recent first, B unpaid
    assert led.credit == {"A": 0.10, "B": 0.0, "C": 0.10, "N": 0.0}
    _give(led, NS, "k7-0", 7, "N", "C"), _give(led, NS, "k7-0", 7, "C", "N")    # hand back and again
    assert len(led.fills) == 1 and abs(led.total() - 0.20) < 1e-12              # once per filled need


def test_delivery_credit_rules():
    adj = {"A": ["N"], "N": ["A", "W"], "W": ["N"]}
    led, NS = _ledger(adj, {"A": [("k1-0", 1), ("k2-0", 2)], "N": [("k3-0", 3)]}, {"N": [1, 2, 3], "W": [3]})
    led.on_event(NS(type="burn", actor="A", target="N", info_id=2, correct=True, lost_info_id=None), 1)
    _give(led, NS, "k2-0", 2, "A", "N")                      # burned need: no fill
    _give(led, NS, "k1-0", 1, "A", "N")                      # direct give: d = 1, pays the holder
    assert [f["paid"] for f in led.fills] == [["A"]]
    led.on_event(NS(type="win", agent="W", score=1.0), 1)
    _give(led, NS, "k3-0", 3, "N", "W")                      # receiver already won: no fill
    assert len(led.fills) == 1


def test_full_run_credit_and_connects(comm_runs):
    run = comm_runs["full"]
    paid, connects, unknown = 0.0, 0, 0
    for d in sorted((run / "games").iterdir()):
        out = json.loads((d / "outcome.json").read_text())
        paid += out["delivery_credit_total"]
        for x in (d / "events.jsonl").read_text().splitlines():
            e = json.loads(x)
            if e["type"] == "edge_added":
                connects += 1
            if e["type"] == "action_result" and e["data"]["action"]["kind"] == "connect" and \
                    e["data"].get("reason") == "unknown_contact":
                unknown += 1
        start = json.loads((d / "events.jsonl").read_text().splitlines()[0])["data"]["agents"]
        assert all(s["heard"] == s["contacts"] for s in start.values())
    assert paid >= 0 and connects + unknown > 0
    sysp = json.loads(next((run / "games").iterdir()).joinpath("llm_calls.jsonl").read_text().splitlines()[0])
    text = sysp["request"]["messages"][0]["content"]
    assert "at least 4 of its 6 members have won" in text and "Delivery credit:" in text
    assert "You can only connect with players you have heard of" in text


# ------------------------------------------------------------------ connect.require_known (contacts mode)
def test_connect_requires_heard_name():
    from swarm.game.observe import mark_observed, observe
    from swarm.game.rules import apply
    from swarm.schema import ConnectAction, SendMessage
    from tests.game.helpers import FOUR, mk_state, run_cfg

    path4 = {"ann": ["bob"], "bob": ["ann", "cat"], "cat": ["bob", "dan"], "dan": ["cat"]}
    g = run_cfg(game={"network": {"topology": "ring", "mode": "contacts",
                                  "connect": {"enabled": True, "cost_rounds": 0, "require_known": True}}}).game
    st = mk_state(FOUR)
    for a, nb in path4.items():
        st.agents[a].contacts = sorted(nb)
        st.agents[a].heard = sorted(nb)
    assert apply(st, "ann", ConnectAction(target="cat"), g)[1][0].data.reason == "unknown_contact"
    st.turn_index = 1
    apply(st, "bob", SendMessage(to="ann", text="cat has key 7, ask cat"), g)       # an introduction
    obs = observe(st, "ann", g, 2)
    network.learn(st, "ann", obs)
    mark_observed(st, "ann", 2)
    assert st.agents["ann"].heard == ["bob", "cat"] and st.agents["ann"].known is None
    st.turn_index = 3
    res = apply(st, "ann", ConnectAction(target="cat"), g)[1]
    assert res[0].data.ok and "cat" in st.agents["ann"].contacts and st.agents["ann"].delayed_until_round < 1
    assert "ann" in st.agents["cat"].heard                                          # a new contact is heard of
    off = run_cfg(game={"network": {"topology": "ring", "connect": {"enabled": True}}})
    assert off.config_hash() == run_cfg(game={"network": {"topology": "ring", "connect": {
        "enabled": True, "require_known": False}}}).config_hash()
    with pytest.raises(ValueError, match="require_known"):
        run_cfg(game={"network": {"topology": "ring", "mode": "known_ids", "connect": {"enabled": True,
                                                                                    "require_known": True}}})


def test_showcase_metrics_compute(comm_runs):
    from swarm.analysis.showcase_metrics import game_metrics, summarize

    games = [game_metrics(d) for d in sorted((comm_runs["full"] / "games").iterdir())]
    s = summarize(games)
    for g in games:
        assert g["chain_completions"] <= g["chain_starts"] and g["stalled_keys"] <= g["chain_starts"]
        out = json.loads((comm_runs["full"] / "games" / g["game"] / "outcome.json").read_text())
        assert abs(g["credited_holders"] * 0.1 - out["delivery_credit_total"]) < 1e-9
        assert g["degree_end"]["mean"] >= g["degree_start"]["mean"] and "0" in g["need_hops_by_round"]
    assert set(s["by_branch"]) == {g["branch"] for g in games}


def test_held_key_scope_initial():
    cfg = _game({"topology": "sbm", "communities": 2}, n=20, held_key_bonus=0.05, held_key_scope="initial")
    st = generate_world(cfg, 3)
    init = scoring.start_infos(st)
    a, b = sorted(st.agents)[:2]
    kc = next(k for k in st.agents[b].keys if k.info_id not in init[a])
    st.agents[b].keys.remove(kc)                       # b passes a key to a (in transit for a)
    st.agents[a].keys.append(kc)
    parts = scoring.score_parts(st, cfg, init)
    assert abs(parts[a].key_bonus - 0.05 * len(init[a])) < 1e-12        # the received key earns nothing
    assert abs(parts[b].key_bonus - 0.05 * (len(init[b]) - 1)) < 1e-12  # the giver loses one
    with pytest.raises(ValueError):
        scoring.score_parts(st, cfg)
    allc = _game({"topology": "sbm", "communities": 2}, n=20, held_key_bonus=0.05)
    assert abs(scoring.score_parts(st, allc)[a].key_bonus - 0.05 * len(st.agents[a].keys)) < 1e-12
    assert from_dict({"game": {"score": {"held_key_bonus": 0.05}}}).config_hash() == from_dict(
        {"game": {"score": {"held_key_bonus": 0.05, "held_key_scope": "all"}}}).config_hash()
    ps = load_prompt_set("agent/v1")
    on = from_dict({"game": {"score": {"held_key_bonus": 0.05, "held_key_scope": "initial"}}})
    assert "keys other players pass to you earn nothing" in ps.render("system.j2", **system_context(on))


@pytest.mark.parametrize("seed", range(4))
def test_mixed_placement_one_need_per_bucket(seed):
    cfg = _game({**SBM100, "key_placement": "mixed"})
    st = generate_world(cfg, seed)
    r = generate_world(_game(SBM100), seed)
    assert is_feasible(st) and [s.keys for s in st.agents.values()] == [s.keys for s in r.agents.values()]
    hops = network.need_hops(st)
    assert all(sorted(min(h, 3) for h in hops[i:i + 3]) == [1, 2, 3] for i in range(0, len(hops), 3))


# ------------------------------------------------------------------ showcase v2: anonymized log, community success
ANON = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=16", "game.max_rounds=6",
        "game.network.topology=erdos_renyi", "game.network.mean_degree=3", "game.network.mode=contacts",
        "game.board.scope=neighbors"]


@pytest.fixture(scope="module")
def anon_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("anon")
    return {"on": _run(tmp, "a_on", [*ANON, "game.view_log.anonymize_noncontacts=true"]),
            "off": _run(tmp, "a_off", ANON)}


def _leaks(run: Path) -> tuple[int, int]:
    """(prompts checked, prompts with a name the agent can only have from the action log): allowed = itself,
    its current contacts, names in message text / senders it received, and its own past actions / notes."""
    import re as _re

    n_checked = n_leaky = 0
    for d in sorted((run / "games").iterdir()):
        evs = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
        names = set(evs[0]["data"]["agents"])
        pat = {n: _re.compile(r"(?<![a-z_])" + n + r"(?![a-z_])") for n in names}
        found = lambda text: {n for n in names if n in text and pat[n].search(text)}  # noqa: E731
        heard: dict[str, set[str]] = {a: set() for a in names}
        turns = {}
        for e in evs:
            if e["type"] != "turn":
                continue
            a, o = e["data"]["agent"], e["data"]["obs"]
            for m in o["inbox"] + o["board"]:
                heard[a] |= found(m["text"]) | {m["sender"]}
            turns[(a, e["round"])] = set(o["pinned"]["peers"]) | heard[a] | {a}
            heard[a] |= found(json.dumps(e["data"]["actions"])) | found(e["data"].get("note") or "")
        for x in (d / "llm_calls.jsonl").read_text().splitlines():
            c = json.loads(x)
            meta = c["request"]["meta"]
            if meta.get("purpose") != "turn":
                continue
            allowed = turns.get((meta["agent"], meta["round"]))
            if allowed is None:
                continue
            n_checked += 1
            if found(c["request"]["messages"][-1]["content"]) - allowed:
                n_leaky += 1
    return n_checked, n_leaky


def test_anonymize_noncontacts_no_name_leaks(anon_runs):
    from swarm.game.observe import OUTSIDE_CONTACTS

    assert check_run(anon_runs["on"]) == [] and check_run(anon_runs["off"]) == []
    checked, leaky = _leaks(anon_runs["on"])
    assert checked > 50 and leaky == 0
    assert _leaks(anon_runs["off"])[1] > 0                      # the check bites: off leaks log names
    text = "".join((d / "llm_calls.jsonl").read_text() for d in (anon_runs["on"] / "games").iterdir())
    assert OUTSIDE_CONTACTS in text
    base = from_dict({"game": {"n_agents": 16, "network": {"topology": "ring"}}})
    assert base.config_hash() == from_dict({"game": {"n_agents": 16, "network": {"topology": "ring"},
                                                     "view_log": {"anonymize_noncontacts": False}}}).config_hash()
    with pytest.raises(ValueError, match="anonymize_noncontacts"):
        from_dict({"game": {"view_log": {"anonymize_noncontacts": True}}})


def test_community_success_needs():
    cfg = _game({"topology": "sbm", "communities": 2}, n=20, community_bonus=0.15, community_threshold_count=3,
                community_success_needs=2)
    comm = {f"a{i}": (0 if i < 10 else 1) for i in range(20)}
    won = {a: False for a in comm}
    held = {a: 1 for a in comm}
    held.update({"a0": 2, "a1": 2, "a2": 3, "a10": 2})
    won["a11"] = True
    sc = scoring.community_scores(scoring.community_success(won, held, cfg), comm, cfg)
    assert sc["a5"] == 0.15 and sc["a15"] == 0.0              # community 0: 3 members >= 2; community 1: 2
    plain = _game({"topology": "sbm", "communities": 2}, n=20, community_bonus=0.15, community_threshold_count=3)
    assert not any(scoring.community_success(won, held, plain)[a] for a in comm if a != "a11")
    ps = load_prompt_set("agent/v1")
    on = from_dict({"game": {"n_agents": 20, "network": {"topology": "sbm", "communities": 2},
                             "score": {"community_bonus": 0.15, "community_threshold_count": 3,
                                       "community_success_needs": 2}}})
    assert "hold keys for at least 2 of their needs (or have won)" in ps.render("system.j2", **system_context(on))
