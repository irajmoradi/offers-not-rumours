"""game.network (social graph, contacts / known_ids) and game.board.scope: defaults unchanged, graph
generation, rule enforcement, name learning, board scope, God along edges, determinism, invariants."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import pytest

from swarm.adversary import apply_message_fraud
from swarm.adversary.scripted_god import recent_partner
from swarm.agents.llm_agent import system_context
from swarm.analysis import network as net_analysis
from swarm.analysis.invariants import check_run
from swarm.config import GodCfg, NetworkCfg, RunConfig, from_dict
from swarm.game import network
from swarm.game.engine import play_game
from swarm.game.observe import mark_observed, observe
from swarm.game.rules import apply
from swarm.game.world import generate_world
from swarm.prompts import load_prompt_set
from swarm.runner.spec import make_spec
from swarm.schema import DefectAction, GiveKey, MessageFraud, PostBoard, SendMessage
from tests.game.helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg
from tests.runner.conftest import _run

# sha256 of the agent/v1 system prompt before game.network existed (git c8cc707): RunConfig() and G1
SYSTEM_SHA_DEFAULT = "e42f9f8ff387fc965a55aa74ff02b2b9911b8d3f5d37e52e18fcd288812475a7"
SYSTEM_SHA_G1 = "8a9b6ee4477ef8eb4926a06efadb7c1ca747c5c0f9c03447d29ddb2c0bd8da25"
HASH_DEFAULT = "b442ad7af4a87da8bb2ff289c1e794cd7b98f44742c8a7066966921cfad35ee1"


def _sys(cfg: RunConfig) -> str:
    return load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))


def _net_state(contacts: dict[str, list[str]], mode: str = "contacts", spec=FOUR):
    """FOUR with a hand-set graph (mode contacts or known_ids)."""
    st = mk_state(spec)
    for a, nb in contacts.items():
        st.agents[a].contacts = sorted(nb)
        if mode == "known_ids":
            st.agents[a].known = sorted(nb)
    return st


PATH4 = {"ann": ["bob"], "bob": ["ann", "cat"], "cat": ["bob", "dan"], "dan": ["cat"]}   # ann-bob-cat-dan


def _ncfg(mode: str = "contacts", **game):
    return run_cfg(game={"network": {"topology": "ring", "mode": mode}, **game}).game


# ------------------------------------------------------------------ defaults
def test_defaults_off_hash_and_prompt_unchanged():
    cfg = RunConfig()
    assert cfg.game.network == NetworkCfg() and not cfg.game.network.on and cfg.game.board.scope == "global"
    assert cfg.config_hash() == HASH_DEFAULT
    explicit = from_dict({"game": {"network": {"topology": "complete"}, "board": {"scope": "global"}}})
    assert explicit.config_hash() == HASH_DEFAULT
    assert hashlib.sha256(_sys(cfg).encode()).hexdigest() == SYSTEM_SHA_DEFAULT
    from swarm.config import REPO_ROOT
    from swarm.runner.run import load_experiment

    g1 = load_experiment(REPO_ROOT / "configs" / "grid_oss_sim_k2.yaml", [])[0]
    assert hashlib.sha256(_sys(g1).encode()).hexdigest() == SYSTEM_SHA_G1
    on = from_dict({"game": {"network": {"topology": "erdos_renyi"}}})
    assert on.config_hash() != HASH_DEFAULT


def test_default_world_and_state_dump_unchanged():
    cfg = run_cfg(game={"n_agents": 6})
    st = generate_world(cfg.game, 11)
    assert all(s.contacts is None and s.known is None for s in st.agents.values())
    assert "contacts" not in json.dumps(st.model_dump(mode="json"))


def test_turn_template_without_network_is_unchanged():
    from tests.test_prompts import _view

    ps = load_prompt_set("agent/v1")
    plain = ps.render("turn.j2", view=_view("amber_fox"))
    assert ps.render("turn.j2", view=_view("amber_fox"), network="") == plain
    assert "\nPlayers: " in plain
    assert "\nPlayers you know: " in ps.render("turn.j2", view=_view("amber_fox"), network="known_ids")
    assert "\nYour contacts (the only players" in ps.render("turn.j2", view=_view("amber_fox"), network="contacts")


def test_config_validation():
    with pytest.raises(ValueError, match="scope=neighbors"):
        from_dict({"game": {"board": {"scope": "neighbors"}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        from_dict({"game": {"network": {"topology": "ring"}, "board": {"enabled": False, "scope": "neighbors"}}})
    with pytest.raises(ValueError, match="even"):
        from_dict({"game": {"n_agents": 10, "network": {"topology": "watts_strogatz", "k": 3}}})
    with pytest.raises(ValueError, match="m must be"):
        from_dict({"game": {"n_agents": 4, "needs_per_agent": 2, "network": {"topology": "barabasi_albert", "m": 4}}})


# ------------------------------------------------------------------ graph generation
NAMES10 = [f"a{i}_x" for i in range(10)]


@pytest.mark.parametrize("topo", ["erdos_renyi", "watts_strogatz", "barabasi_albert", "ring"])
def test_graphs_connected_symmetric_deterministic(topo):
    cfg = NetworkCfg(topology=topo)
    for seed in range(40):
        g = network.make_graph(cfg, NAMES10, seed)
        assert g == network.make_graph(cfg, list(reversed(NAMES10)), seed)       # order-independent
        assert all(a in g[b] for a in g for b in g[a]) and all(a not in g[a] for a in g)
        edges = [(a, b) for a in g for b in g[a] if a < b]
        assert len(network.components(10, [(NAMES10.index(a), NAMES10.index(b)) for a, b in edges])) == 1
        if topo == "ring":
            assert all(len(v) == 2 for v in g.values())
        if topo == "barabasi_albert":
            assert len(edges) == 2 * (10 - 2)
        if topo == "watts_strogatz":
            assert len(edges) == 10 * 4 // 2
    assert network.make_graph(cfg, NAMES10, 1) != network.make_graph(cfg, NAMES10, 2) or topo == "ring"


def test_erdos_renyi_mean_degree_close_to_target():
    degs = [sum(map(len, network.make_graph(NetworkCfg(topology="erdos_renyi"), NAMES10, s).values())) / 10
            for s in range(300)]
    assert 2.9 < sum(degs) / len(degs) < 3.8          # conditioned on connected: a little above 3


def test_world_deal_unchanged_and_graph_attached():
    off = run_cfg(game={"n_agents": 10, "max_rounds": 3})
    on = run_cfg(game={"n_agents": 10, "max_rounds": 3, "network": {"topology": "erdos_renyi", "mode": "known_ids"}})
    a, b = generate_world(off.game, 5), generate_world(on.game, 5)
    for x in a.agents:
        assert a.agents[x].keys == b.agents[x].keys and a.agents[x].needs == b.agents[x].needs
        assert b.agents[x].contacts and b.agents[x].known == b.agents[x].contacts
    assert generate_world(on.game, 5).model_dump() == b.model_dump()


def test_graph_identical_across_god_nogod_noise_branches():
    cfg = from_dict({"run": {"seed": 1}, "game": {"n_agents": 10, "network": {"topology": "barabasi_albert"}}})
    worlds = []
    for branch in ("god", "nogod", "noise"):
        spec = make_spec(cfg, 3, branch)
        worlds.append({a: s.contacts for a, s in generate_world(cfg.game, spec.world_seed, spec.game_id).agents.items()})
    assert worlds[0] == worlds[1] == worlds[2]


# ------------------------------------------------------------------ contacts mode
def test_contacts_enforced_for_dm_give_defect():
    g = _ncfg()
    st = _net_state(PATH4)
    for act in (SendMessage(to="cat", text="hi"), GiveKey(to="cat", info_id=1), DefectAction(target="cat", info_id=2)):
        _, evs = apply(st, "ann", act, g)
        assert evs[0].data.ok is False and evs[0].data.reason == network.NOT_CONNECTED
        assert evs[1].data.notice.kind == "system" and "not connected to cat" in evs[1].data.notice.text
        assert "(bob)" in evs[1].data.notice.text
    assert st.messages == [] and st.history == [] and [k.info_id for k in st.agents["ann"].keys] == [1, 2]
    _, evs = apply(st, "ann", GiveKey(to="bob", info_id=1), g)
    assert evs[0].data.ok and sorted(k.info_id for k in st.agents["bob"].keys) == [1, 3, 4]


def test_contacts_peers_board_scope_and_local_log():
    g = run_cfg(game={"network": {"topology": "ring"}, "board": {"scope": "neighbors"}}).game
    st = _net_state(PATH4)
    apply(st, "cat", PostBoard(text="cat here"), g)          # ann is not cat's neighbour
    apply(st, "bob", PostBoard(text="bob here"), g)
    st.turn_index = 1
    apply(st, "cat", GiveKey(to="dan", info_id=5), g)          # neither party is ann's contact
    apply(st, "bob", GiveKey(to="cat", info_id=3), g)          # bob is
    st.turn_index = 3
    obs = observe(st, "ann", g)
    assert obs.pinned.peers == ["bob"]
    assert [m.sender for m in obs.board] == ["bob"]
    assert {(ln.kind, ln.actor, ln.target) for ln in obs.log} == {("board_post", "bob", None), ("give", "bob", "cat")}
    sysp = _sys(run_cfg(game={"network": {"topology": "ring"}, "board": {"scope": "neighbors"}}))
    assert "only message, trade with and defect on your contacts" in sysp and "The board is local" in sysp
    assert "board post your contacts can read" in sysp


def test_god_fraud_only_along_edges_in_contacts_mode():
    st = _net_state(PATH4)
    st.god_budget_left = 5
    res = apply_message_fraud(st, MessageFraud(sender="cat", receiver="ann", text="give me"), GodCfg())
    assert not res.ok and res.reason == "not_connected" and st.god_budget_left == 5 and st.messages == []
    res = apply_message_fraud(st, MessageFraud(sender="bob", receiver="ann", text="give me"), GodCfg())
    assert res.ok and st.messages[-1].receiver == "ann"
    known = _net_state(PATH4, "known_ids")
    known.god_budget_left = 5
    assert apply_message_fraud(known, MessageFraud(sender="dan", receiver="ann", text="x"), GodCfg()).ok


def test_scripted_recent_partner_falls_back_to_a_contact():
    st = _net_state(PATH4)
    assert recent_partner(st, "ann") == "bob"                  # ann needs 3 (bob) and 5 (cat): bob first
    st2 = _net_state({"ann": ["dan"], "bob": ["cat"], "cat": ["bob", "dan"], "dan": ["ann", "cat"]})
    assert recent_partner(st2, "ann") == "dan"                 # holders of ann's needs are not contacts
    assert recent_partner(mk_state(FOUR), "ann") == "bob"      # no network: unchanged


# ------------------------------------------------------------------ known_ids mode
def test_unknown_name_fails_then_is_learned_from_a_dm():
    g = _ncfg("known_ids")
    st = _net_state(PATH4, "known_ids")
    _, evs = apply(st, "ann", SendMessage(to="dan", text="hi"), g)
    assert evs[0].data.reason == network.UNKNOWN_CONTACT and "don't know that agent (dan)" in evs[1].data.notice.text
    apply(st, "bob", SendMessage(to="ann", text="dan holds info 3, ask dan_x? no: ask dan"), g)
    st.turn_index = 2
    obs = observe(st, "ann", g)
    assert obs.pinned.peers == ["bob", "dan"]                  # learned in this very observation
    assert network.learn(st, "ann", obs) == ["dan"] and st.agents["ann"].known == ["bob", "dan"]
    mark_observed(st, "ann", 2)
    _, evs = apply(st, "ann", SendMessage(to="dan", text="hi dan"), g)
    assert evs[0].data.ok
    # dan learns ann from her DM; cat learns nothing new
    st.turn_index = 4
    assert network.learn(st, "dan", observe(st, "dan", g)) == ["ann"]


def test_known_ids_learns_from_board_and_gifts_not_from_refusals():
    g = _ncfg("known_ids")
    st = _net_state({"ann": ["bob"], "bob": ["ann"], "cat": ["dan"], "dan": ["cat", "ann"]}, "known_ids")
    apply(st, "cat", PostBoard(text="anyone with info 2?"), g)
    apply(st, "dan", GiveKey(to="ann", info_id=7), g)          # dan knows ann; ann does not know dan
    apply(st, "ann", GiveKey(to="cat", info_id=2), g)          # refused: ann does not know cat (yet)
    st.turn_index = 3
    obs = observe(st, "ann", g)
    assert {n.kind for n in obs.notices} >= {"gift_received", "system"}
    assert network.learn(st, "ann", obs) == ["cat", "dan"]


def test_known_ids_log_shows_only_lines_between_known_players():
    g = _ncfg("known_ids")
    st = _net_state(PATH4, "known_ids")
    apply(st, "cat", GiveKey(to="dan", info_id=6), g)
    apply(st, "cat", GiveKey(to="bob", info_id=5), g)
    st.turn_index = 2
    assert observe(st, "ann", g).log == []                     # ann knows only bob
    st.agents["ann"].known = ["bob", "cat"]
    assert [(ln.actor, ln.target) for ln in observe(st, "ann", g).log] == [("cat", "bob")]
    assert "You only know some players" in _sys(run_cfg(game={"network": {"topology": "ring", "mode": "known_ids"}}))


def test_names_in_text_whole_words_only():
    names = ["amber_fox", "amber_foxglove", "bob_owl"]
    assert network.names_in_text("ask amber_fox, not bob_owlx", names) == {"amber_fox"}
    assert network.names_in_text("amber_foxglove!", names) == {"amber_foxglove"}


# ------------------------------------------------------------------ engine (both turn modes)
@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
@pytest.mark.parametrize("net", ["contacts", "known_ids"])
def test_engine_never_lets_an_action_cross_the_rules(mode, net):
    cfg = run_cfg(game={"turn_mode": mode, "max_rounds": 5, "network": {"topology": "ring", "mode": net}})
    st = _attached(cfg)

    def policy(obs, grep):                                     # also aims at players it may not address
        rng = random.Random(f"{obs.agent}{obs.round}")
        other = rng.choice([a for a in FOUR if a != obs.agent])
        held = [k.info_id for k in obs.pinned.holdings]
        return [SendMessage(to=other, text=f"hi, talk to {rng.choice(list(FOUR))}"),
                GiveKey(to=other, info_id=held[0]) if held else PostBoard(text="hello")]

    stub = Scripted(policy=policy)
    _, eng = play_game(cfg, st, stub)
    _, eng2 = play_game(cfg, _attached(cfg), Scripted(policy=policy))     # deterministic replay
    evs = [e for e in eng.sink.events if e.type == "action_result"]
    refused = [e for e in evs if e.data.reason in network.REASONS]
    assert refused, "the policy should hit the network rule at least once"
    for e in evs:                                              # ok peer actions only on addressable agents
        act = e.data.action
        other = getattr(act, "to", None) or getattr(act, "target", None)
        if e.data.ok and other is not None:
            peers = next(o.pinned.peers for o, _ in stub.calls if o.agent == e.data.agent and o.round == e.round)
            assert other in peers
    assert [e.model_dump() for e in eng.sink.events] == [e.model_dump() for e in eng2.sink.events]


def _attached(cfg):
    st = fresh_game(FOUR)
    network.attach(st, cfg.game, 7)
    return st


# ------------------------------------------------------------------ runner + invariants + analysis
NET = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8", "game.max_rounds=5"]


@pytest.fixture(scope="module")
def net_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("netruns")
    return {
        "contacts_seq": _run(tmp, "net_c_seq", [*NET, "game.network.topology=erdos_renyi"]),
        "known_sim": _run(tmp, "net_k_sim", [*NET, "game.turn_mode=simultaneous", "game.network.topology=erdos_renyi",
                                             "game.network.mode=known_ids"]),
        "known_seq_local": _run(tmp, "net_k_seq", [*NET, "game.network.topology=barabasi_albert",
                                                   "game.network.mode=known_ids", "game.board.scope=neighbors"]),
    }


def test_network_runs_hold_every_invariant(net_runs):
    for name, run in net_runs.items():
        vs = check_run(run)
        assert vs == [], (name, vs[:5])


def test_network_meta_and_game_start_carry_the_graph(net_runs):
    g = net_runs["contacts_seq"] / "games"
    metas = [json.loads((d / "meta.json").read_text()) for d in sorted(g.iterdir())]
    assert all(m["network"]["topology"] == "erdos_renyi" for m in metas)
    assert len({json.dumps(m["network"]["edges"]) for m in metas}) == 1          # god == nogod graph
    start = json.loads((sorted(g.iterdir())[0] / "events.jsonl").read_text().splitlines()[0])
    assert all(s["contacts"] for s in start["data"]["agents"].values())


def test_tampered_network_runs_are_caught(net_runs, tmp_path):
    run = tmp_path / "c"
    shutil.copytree(net_runs["contacts_seq"], run)
    gdir = sorted((run / "games").iterdir())[0]
    lines = [json.loads(x) for x in (gdir / "events.jsonl").read_text().splitlines()]
    contacts = {a: set(s["contacts"]) for a, s in lines[0]["data"]["agents"].items()}
    for e in lines:
        if e["type"] == "message" and e["data"]["message"]["channel"] == "dm":
            m = e["data"]["message"]
            m["receiver"] = next(x for x in sorted(contacts) if x != m["sender_shown"] and x not in contacts[m["sender_shown"]])
            break
    (gdir / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert "network" in {v.check for v in check_run(run)}
    run2 = tmp_path / "k"
    shutil.copytree(net_runs["known_sim"], run2)
    gdir = sorted((run2 / "games").iterdir())[0]
    snaps = [json.loads(x) for x in (gdir / "snapshots.jsonl").read_text().splitlines()]
    a = sorted(snaps[-1]["state"]["agents"])[0]
    snaps[-1]["state"]["agents"][a]["known"] = []
    (gdir / "snapshots.jsonl").write_text("".join(json.dumps(x) + "\n" for x in snaps))
    assert "network" in {v.check for v in check_run(run2)}


def test_network_analysis_on_fake_runs(net_runs):
    res = net_analysis.analyse_run(net_runs["known_sim"])
    s = next(iter(res["graphs"].values()))
    assert s["connected"] and s["n"] == 8 and s["edges"] >= 7
    curve = res["coverage_by_round"]["god"]
    vals = list(curve.values())
    assert vals == sorted(vals) and vals[-1] > vals[0]                       # knowledge only grows
    g = net_analysis.load_run(net_runs["known_sim"])[0]
    d = net_analysis.diffusion(g)
    assert d["learned"] and all(x["hops"] >= 2 for x in d["learned"])        # neighbours were known
    assert set(d["channel_hist"]) <= {"dm_sender", "dm_mention", "board_author", "board_mention", "interaction"}
    c = net_analysis.analyse_run(net_runs["contacts_seq"])
    assert set(c["trades_by_distance"]["god"]) == {1}                        # contacts: gives along edges
    assert c["god"] and all(r["degree"] is not None for r in c["god"])


def test_brokering_counts_multi_hop_and_relays():
    import networkx as nx

    a, b, c, d = "amber_fox", "bold_owl", "coral_elk", "dune_cat"
    game = {"graph": nx.path_graph([a, b, c, d]), "social": {
        "names": [a, b, c, d], "transfers": [],
        "msgs": [dict(seq=1, round=1, ch="dm", s=b, shown=b, r=c, text=f"{a} needs info 3.", forged=False),
                 dict(seq=2, round=2, ch="dm", s=c, shown=c, r=d, text=f"{a} needs info 3.", forged=False)]}}
    br = net_analysis.brokering(game)
    assert br["n"] == 2 and br["relays"] == 1 and br["multi_hop"] == 2 and br["hops_hist"] == {2: 1, 3: 1}
