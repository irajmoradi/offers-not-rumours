"""game.network.connect (dynamic edge formation, CONTRACTS §9e): defaults byte-identical, the connect
action and its cost in both turn modes, every refusal, the whole-turn rule, the simultaneous mutual case,
known_ids, winners-leave, God along new edges, invariants (incl. tampering) and the dynamics analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from swarm.adversary import apply_message_fraud
from swarm.agents.llm_agent import system_context
from swarm.agents.parser import parse_reply
from swarm.analysis import network_dynamics
from swarm.analysis.invariants import check_run
from swarm.config import REPO_ROOT, GodCfg, from_dict
from swarm.game import network
from swarm.game.engine import play_game
from swarm.game.observe import mark_observed, observe
from swarm.game.resolve import resolve_turn
from swarm.game.rules import apply
from swarm.prompts import load_prompt_set
from swarm.runner.run import load_experiment
from swarm.schema import (
    ConnectAction,
    GiveKey,
    GrepLog,
    MemorySink,
    MessageFraud,
    Noop,
    PostBoard,
    PromptView,
    SendMessage,
)
from tests.game.helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg
from tests.runner.conftest import _run

PATH4 = {"ann": ["bob"], "bob": ["ann", "cat"], "cat": ["bob", "dan"], "dan": ["cat"]}   # ann-bob-cat-dan
# config_hash + sha256(system prompt) of the network configs BEFORE game.network.connect (git 4b1a0e7)
BEFORE = {
    "net40_er3_contacts_sim": ("407874f17fb83cdf21fb41c4ca7013f9cfaa1699b3d1998f6a56898cb0642d9a",
                               "92f7e7ad5deeff10a5745b627e2aa89b45c6cd96a1d75f5c2f3ca7bd42901513"),
    "net40_er3_known_sim": ("45354b9d8d1c30568f748c14d13571ecf91d7d851008013f1be1a5411cfc8d10",
                            "aa772fa7789f0de4c126f6d4492660f95308081b2bbed01bdbb61dd6484fb88e"),
    "net20_er3_contacts_sim": ("f60473c4455ed9ed891aa2be6d13a5ab1fb0d274a3fa588c94713f7e940353a3",
                               "d2d20fe060d8e1c742de29c17363168bc9269c59ccfad8ce7cf628ebd88d1562"),
    "network_er3_known_sim": ("65ac7e1847f3cfe52c9b49459f358f46a38f801a1e99e3f33b2c5161f0dd5755",
                              "43f5047f06db88f2b696cc9731f053d6dad0bb7a37d326ee55e41e7c63c4a2f7"),
}


def _sys(cfg) -> str:
    return load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))


def _cfg(mode="contacts", **connect):
    return run_cfg(game={"network": {"topology": "ring", "mode": mode,
                                     "connect": {"enabled": True, **connect}}})


def _state(mode="contacts", spec=FOUR, graph=PATH4):
    st = mk_state(spec)
    for a, nb in graph.items():
        st.agents[a].contacts = sorted(nb)
        if mode == "known_ids":
            st.agents[a].known = sorted(nb)
    return st


def _engine_state(graph=PATH4, mode="contacts"):
    st = fresh_game(FOUR)
    for a, nb in graph.items():
        st.agents[a].contacts = sorted(nb)
        if mode == "known_ids":
            st.agents[a].known = sorted(nb)
    return st


def _evs(sink, typ):
    return [e for e in sink.events if e.type == typ]


# ------------------------------------------------------------------ defaults: byte-identical
@pytest.mark.parametrize("name", sorted(BEFORE))
def test_existing_network_configs_keep_hash_and_prompt(name):
    cfg = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", [])[0]
    assert (cfg.config_hash(), hashlib.sha256(_sys(cfg).encode()).hexdigest()) == BEFORE[name]
    explicit = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml",
                               ["game.network.connect.enabled=false", "game.network.connect.cost_rounds=2"])[0]
    assert explicit.config_hash() == BEFORE[name][0]
    on = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", ["game.network.connect.enabled=true"])[0]
    assert on.config_hash() != BEFORE[name][0]
    assert "connect" not in _sys(cfg) and "connect" in _sys(on)


def test_connect_configs_and_validation():
    for k in (2, 3):
        cfg = load_experiment(REPO_ROOT / "configs" / f"net40_er3_contacts_connect{k}_sim.yaml", [])[0]
        base = load_experiment(REPO_ROOT / "configs" / "net40_er3_contacts_sim.yaml", [])[0]
        c = cfg.game.network.connect
        assert c.enabled and c.cost_rounds == k and c.max_new_edges_per_agent is None
        assert cfg.game.model_copy(update={"network": base.game.network}) == base.game   # only connect differs
        assert f"Costs your next {k} turns" in _sys(cfg)
    with pytest.raises(ValueError, match="connect.enabled needs"):
        from_dict({"game": {"network": {"connect": {"enabled": True}}}})
    one = _cfg(cost_rounds=1, max_new_edges_per_agent=2)
    assert "Costs your next turn " in _sys(one) and "at most 2 new links" in _sys(one)


def test_parser_accepts_connect_only_when_enabled():
    txt = '{"actions": [{"kind": "connect", "target": "@bob"}]}'
    off = parse_reply(txt)
    assert not off.ok and off.errors == ["unknown_kind:0:connect", "all_actions_invalid"]
    assert parse_reply('{"kind": "link", "to": "bob"}').errors == ["no_actions_key"]       # as before
    on = parse_reply(txt, connect=True)
    assert on.ok and on.actions == [ConnectAction(target="bob")]
    assert parse_reply('{"actions": [{"action": "link", "with": "cat"}]}', connect=True).actions == [
        ConnectAction(target="cat")]


# ------------------------------------------------------------------ rules: success and refusals
def test_connect_adds_edge_costs_turns_and_notifies():
    g = _cfg().game
    st = _state()
    st.turn_index = 0
    _, evs = apply(st, "ann", ConnectAction(target="dan"), g)
    assert evs[0].data.ok and st.agents["ann"].contacts == ["bob", "dan"] and st.agents["dan"].contacts == ["ann", "cat"]
    assert st.agents["ann"].delayed_until_round == 1 + 2
    edge = next(e.data for e in evs if e.data.type == "edge_added")
    assert (edge.actor, edge.target, edge.mutual, edge.skip_until_round) == ("ann", "dan", False, 3)
    texts = {(n.agent, n.kind): n for n in st.notices}
    assert texts[("dan", "connected")].text == "ann connected with you; you can now trade and message directly."
    ann = [n for n in st.notices if n.agent == "ann"]
    assert "skip rounds 2 to 3" in ann[0].text and ann[1].deliver_from_turn == 3 * 4      # resume: round 4 slot 0
    assert [h.kind for h in st.history] == ["connect"]
    _, evs = apply(st, "dan", GiveKey(to="ann", info_id=7), g)            # the new edge is live
    assert evs[0].data.ok


@pytest.mark.parametrize("setup,target,reason", [
    ("plain", "bob", "already_connected"),
    ("plain", "zed", "unknown_agent"),
    ("plain", "ann", "self_target"),
    ("left", "dan", "target_left"),
    ("known", "dan", "unknown_contact"),
    ("cap", "cat", "edge_cap_reached"),
    ("nonet", "dan", "not_network"),
    ("disabled", "dan", "connect_disabled"),
])
def test_connect_refusals_change_nothing(setup, target, reason):
    g = _cfg(max_new_edges_per_agent=1).game
    st = _state("known_ids" if setup == "known" else "contacts")
    if setup == "left":
        g = run_cfg(game={"winner_policy": "sit_out", "network": {"topology": "ring", "connect": {"enabled": True}}}).game
        st.agents["dan"].won_round = 1
    if setup == "cap":
        assert apply(st, "ann", ConnectAction(target="dan"), g)[1][0].data.ok
    if setup == "nonet":
        g, st = run_cfg().game, mk_state(FOUR)
    if setup == "disabled":
        g = run_cfg(game={"network": {"topology": "ring"}}).game
    before = st.model_copy(deep=True)
    _, evs = apply(st, "ann", ConnectAction(target=target), g)
    assert not evs[0].data.ok and evs[0].data.reason == reason
    assert not [e for e in evs if e.data.type == "edge_added"]
    assert st.agents["ann"].contacts == before.agents["ann"].contacts
    assert st.agents["ann"].delayed_until_round == before.agents["ann"].delayed_until_round
    assert "You lose no turns." in st.notices[-1].text and st.notices[-1].agent == "ann"


def test_departed_actor_cannot_connect():
    g = run_cfg(game={"winner_policy": "sit_out", "network": {"topology": "ring", "connect": {"enabled": True}}}).game
    st = _state()
    st.agents["ann"].won_round = 1
    assert apply(st, "ann", ConnectAction(target="dan"), g)[1][0].data.reason == "actor_left"


# ------------------------------------------------------------------ whole turn
def test_connect_takes_the_whole_turn():
    g = _cfg().game
    st = _state()
    res = resolve_turn(st, "ann", [GrepLog(pattern="x"), SendMessage(to="bob", text="hi"),
                                   ConnectAction(target="dan"), Noop()], None, g)
    assert res.results == [(False, "grep_unavailable"), (False, "connect_uses_whole_turn"), (True, None), (True, None)]
    assert st.messages == [] and "connecting uses your whole turn" in st.notices[-1].text
    st2 = _state()                          # a refused connect still takes the turn (no cost though)
    res = resolve_turn(st2, "ann", [ConnectAction(target="bob"), GiveKey(to="bob", info_id=1)], None, g)
    assert res.results == [(False, "already_connected"), (False, "connect_uses_whole_turn")]
    assert st2.agents["ann"].delayed_until_round == -1
    both = run_cfg(game={"board": {"post_costs_turn": True},
                         "network": {"topology": "ring", "connect": {"enabled": True}}}).game
    r1 = resolve_turn(_state(), "ann", [PostBoard(text="hi"), ConnectAction(target="dan")], None, both)
    assert r1.results == [(True, None), (False, "board_post_uses_whole_turn")]
    r2 = resolve_turn(_state(), "ann", [ConnectAction(target="dan"), PostBoard(text="hi")], None, both)
    assert r2.results == [(True, None), (False, "connect_uses_whole_turn")]
    off = run_cfg(game={"network": {"topology": "ring"}}).game    # connect off: no whole-turn rule
    r3 = resolve_turn(_state(), "ann", [ConnectAction(target="dan"), SendMessage(to="bob", text="x")], None, off)
    assert r3.results == [(False, "connect_disabled"), (True, None)]


# ------------------------------------------------------------------ engine, both turn modes
@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
def test_engine_cost_resume_and_edge_next_round(mode):
    cfg = run_cfg(game={"turn_mode": mode, "network": {"topology": "ring", "connect": {"enabled": True}}})
    stub = Scripted({(1, "ann"): [ConnectAction(target="dan")],
                     (2, "dan"): [GiveKey(to="ann", info_id=7)],
                     (4, "ann"): [SendMessage(to="dan", text="thanks")]})
    sink = MemorySink(game_id="gt")
    _, eng = play_game(cfg, _engine_state(), stub, sink=sink)
    assert [o.round for o in stub.obs_of("ann")] == [1, 4, 5, 6]        # rounds 2-3 lost
    assert [o.round for o in stub.obs_of("dan")] == [1, 2, 3, 4, 5, 6]
    res = {(e.round, e.data.agent, e.data.action.kind): e.data for e in _evs(sink, "action_result")}
    assert res[(2, "dan", "give_key")].ok and res[(4, "ann", "send_message")].ok
    resumed = stub.obs_of("ann")[1]
    assert any("are over. You can act again." in n.text for n in resumed.notices)
    assert any("skip rounds 2 to 3" in n.text for n in resumed.notices)
    dan2 = stub.obs_of("dan")[1 if mode == "simultaneous" else 0]
    seen = [n for o in stub.obs_of("dan")[:2] for n in o.notices if n.kind == "connected"]
    assert len(seen) == 1 and "ann" in dan2.pinned.peers
    assert eng.state.agents["ann"].contacts == ["bob", "dan"]
    assert [(e.data.actor, e.data.target) for e in _evs(sink, "edge_added")] == [("ann", "dan")]


def test_simultaneous_mutual_connect_is_one_edge_both_pay():
    cfg = run_cfg(game={"turn_mode": "simultaneous", "network": {"topology": "ring", "connect": {"enabled": True}}})
    stub = Scripted({(1, "ann"): [ConnectAction(target="dan")], (1, "dan"): [ConnectAction(target="ann")]})
    sink = MemorySink(game_id="gt")
    _, eng = play_game(cfg, _engine_state(), stub, sink=sink)
    edges = [(e.data.actor, e.data.target, e.data.mutual) for e in _evs(sink, "edge_added")]
    assert edges == [("ann", "dan", False), ("dan", "ann", True)]
    assert eng.state.agents["ann"].contacts == ["bob", "dan"] and eng.state.agents["dan"].contacts == ["ann", "cat"]
    assert [o.round for o in stub.obs_of("ann")] == [o.round for o in stub.obs_of("dan")] == [1, 4, 5, 6]
    assert not [n for n in eng.state.notices if n.agent == "ann" and n.kind == "connected"]   # dan was the second


def test_sequential_second_connect_is_already_connected():
    cfg = run_cfg(game={"network": {"topology": "ring", "connect": {"enabled": True}}})
    stub = Scripted({(1, "ann"): [ConnectAction(target="dan")], (1, "dan"): [ConnectAction(target="ann")]})
    sink = MemorySink(game_id="gt")
    play_game(cfg, _engine_state(), stub, sink=sink)
    res = [(e.data.agent, e.data.ok, e.data.reason) for e in _evs(sink, "action_result")
           if e.data.action.kind == "connect"]
    assert res == [("ann", True, None), ("dan", False, "already_connected")]
    assert [o.round for o in stub.obs_of("dan")] == [1, 2, 3, 4, 5, 6]                 # no cost for dan


def test_simultaneous_connect_is_live_for_later_resolutions_in_the_round():
    cfg = run_cfg(game={"turn_mode": "simultaneous", "network": {"topology": "ring", "connect": {"enabled": True}}})
    stub = Scripted({(1, "ann"): [ConnectAction(target="dan")], (1, "dan"): [GiveKey(to="ann", info_id=7)]})
    sink = MemorySink(game_id="gt")
    play_game(cfg, _engine_state(), stub, sink=sink)
    give = next(e.data for e in _evs(sink, "action_result") if e.data.action.kind == "give_key")
    assert give.ok                                       # ann (slot 0) resolved first: the edge exists


# ------------------------------------------------------------------ known_ids, winners-leave, God
def test_known_ids_requires_known_name_target_learns_actor_and_links_shown():
    g = _cfg("known_ids").game
    st = _state("known_ids")
    assert apply(st, "ann", ConnectAction(target="dan"), g)[1][0].data.reason == "unknown_contact"
    st.agents["ann"].known = ["bob", "dan"]                                 # learned dan somewhere
    st.turn_index = 0
    assert apply(st, "ann", ConnectAction(target="dan"), g)[1][0].data.ok
    obs = observe(st, "dan", g, 3)
    network.learn(st, "dan", obs)
    mark_observed(st, "dan", 3)
    assert "ann" in st.agents["dan"].known and "ann" in obs.pinned.peers
    assert obs.pinned.links == ["ann", "cat"]
    text = load_prompt_set("agent/v1").render("turn.j2", network="known_ids", view=PromptView(
        agent="dan", round=1, pinned=obs.pinned, sections=[]))
    assert "\nYour direct links: ann, cat\n" in text
    plain = observe(_state("known_ids"), "dan", run_cfg(game={"network": {"topology": "ring",
                                                                          "mode": "known_ids"}}).game, 3)
    assert plain.pinned.links is None and "links" not in json.dumps(plain.model_dump(mode="json"))


def test_winners_leave_links_and_peers_drop_departed():
    g = run_cfg(game={"winner_policy": "sit_out",
                      "network": {"topology": "ring", "mode": "known_ids", "connect": {"enabled": True}}}).game
    st = _state("known_ids")
    st.agents["ann"].known = ["bob", "dan"]
    assert apply(st, "ann", ConnectAction(target="dan"), g)[1][0].data.ok
    st.agents["ann"].won_round = 1
    assert observe(st, "dan", g, 3).pinned.links == ["cat"]


def test_god_forges_along_new_edges_only():
    g = _cfg().game
    st = _state()
    st.god_budget_left = 5
    res = apply_message_fraud(st, MessageFraud(sender="dan", receiver="ann", text="give me"), GodCfg())
    assert not res.ok and res.reason == "not_connected"
    apply(st, "ann", ConnectAction(target="dan"), g)
    assert apply_message_fraud(st, MessageFraud(sender="dan", receiver="ann", text="give me"), GodCfg()).ok


# ------------------------------------------------------------------ runner + invariants + analysis
NET = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8", "game.max_rounds=8",
       "game.network.topology=erdos_renyi", "game.network.connect.enabled=true"]


@pytest.fixture(scope="module")
def connect_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("connectruns")
    return {
        "sim": _run(tmp, "conn_sim", [*NET, "game.turn_mode=simultaneous", "game.board.scope=neighbors"]),
        "seq": _run(tmp, "conn_seq", [*NET, "game.network.connect.cost_rounds=3",
                                      "game.network.connect.max_new_edges_per_agent=1"]),
        "known_sit": _run(tmp, "conn_known", [*NET, "game.turn_mode=simultaneous", "game.network.mode=known_ids",
                                              "game.winner_policy=sit_out"]),
    }


def _edge_lines(run: Path) -> list[dict]:
    out = []
    for g in sorted((run / "games").iterdir()):
        out += [json.loads(x) for x in (g / "events.jsonl").read_text().splitlines() if '"edge_added"' in x]
    return out


def test_connect_runs_hold_every_invariant_and_form_edges(connect_runs):
    for name, run in connect_runs.items():
        vs = check_run(run)
        assert vs == [], (name, vs[:5])
    assert _edge_lines(connect_runs["sim"]) and _edge_lines(connect_runs["seq"])


def test_tampered_connect_runs_are_caught(connect_runs, tmp_path):
    for i, how in enumerate(("drop_edge", "snapshot", "cheap")):
        run = tmp_path / f"t{i}"
        shutil.copytree(connect_runs["sim"], run)
        gdir = next(g for g in sorted((run / "games").iterdir()) if '"edge_added"' in (g / "events.jsonl").read_text())
        lines = [json.loads(x) for x in (gdir / "events.jsonl").read_text().splitlines()]
        k = next(j for j, x in enumerate(lines) if x["type"] == "edge_added")
        if how == "drop_edge":
            del lines[k]
            for j, x in enumerate(lines):
                x["seq"] = j
        elif how == "cheap":
            lines[k]["data"]["skip_until_round"] = lines[k]["round"]
        else:
            snaps = [json.loads(x) for x in (gdir / "snapshots.jsonl").read_text().splitlines()]
            a = lines[k]["data"]["actor"]
            snaps[-1]["state"]["agents"][a]["contacts"] = []
            (gdir / "snapshots.jsonl").write_text("".join(json.dumps(x) + "\n" for x in snaps))
        (gdir / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        assert "network" in {v.check for v in check_run(run)}, how


def test_network_dynamics_analysis(connect_runs):
    res = network_dynamics.analyse_run(connect_runs["sim"])
    n_edges = sum(1 for x in _edge_lines(connect_runs["sim"]) if not x["data"]["mutual"])
    p = res["pooled"]
    assert p["n_edges_added"] == n_edges and p["attempts"]["ok"] == len(_edge_lines(connect_runs["sim"]))
    g = next(x for x in res["games"] if x["n_edges_added"])
    assert g["degrees"]["end"]["edges"] == g["degrees"]["start"]["edges"] + g["n_edges_added"]
    assert sum(g["timeline"].values()) == g["n_edges_added"]
    assert {"target_hub", "triadic", "start_dist", "target_holds_actor_need", "gives_after"} <= set(g["edges"][0])
    seq = network_dynamics.analyse_run(connect_runs["seq"])
    for x in seq["games"]:
        actors = [e["actor"] for e in x["edges"]]
        assert len(actors) == len(set(actors))                    # max_new_edges_per_agent = 1
    network_dynamics.main([str(connect_runs["sim"]), "--json", str(connect_runs["sim"] / "dyn.json")])
