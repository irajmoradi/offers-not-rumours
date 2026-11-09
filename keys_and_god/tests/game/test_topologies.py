"""Richer game.network topologies (sbm, powerlaw_cluster, community_hubs) and key_placement local_bias:
connected / symmetric / deterministic graphs, planted structure and stats, the original topologies and
hashes byte-identical, local_bias keeps every world solvable, branches share the graph, and the
analysis' community breakdown."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from swarm.analysis import network as net_analysis
from swarm.analysis.invariants import check_run
from swarm.config import NetworkCfg, from_dict
from swarm.game import network
from swarm.game.world import generate_world, is_feasible
from tests.runner.conftest import _run

NAMES10 = [f"a{i}_x" for i in range(10)]
NAMES40 = [f"n{i}_x" for i in range(40)]
NEW = {"sbm": {"mean_degree": 4.0}, "powerlaw_cluster": {}, "community_hubs": {}}

# make_graph for the original topologies over seeds 0-24 on 40 names, and two ER config hashes,
# computed on master (86bb5a3) before these topologies existed
OLD_GRAPHS = {
    ("erdos_renyi", ()): "75a7aa971d966d21dc41a7d571b5fa6138171956f6cac5084aabef099ac02aa5",
    ("watts_strogatz", (("k", 4), ("p", 0.2))): "11dec4c51e3b8e23a96f4a826296de5e4703895df51859ad1ae41f31db8056c6",
    ("barabasi_albert", ()): "29989abf1257758623da4a9187e1ba2d6abddba6140d1a5800bbbf4254ce441c",
    ("ring", ()): "0d0626c373c12d9ddc5a5c9e2705dfcfec2df4eb3eca24ea8d638529f79fe3e9",
}
OLD_HASHES = {"contacts": "c681dd4caa8c066ed288952728836fd789b1614b3e6dc6ffc6cd7d6d0f3fdf41",
              "known_ids": "cdd01cb5229d9748d7444bd0c016d11223131147250a2f5bed1f373997045b0c"}


def _idx_edges(g: dict[str, list[str]], names: list[str]) -> list[tuple[int, int]]:
    return [(names.index(a), names.index(b)) for a in g for b in g[a] if a < b]


# ------------------------------------------------------------------ originals unchanged
@pytest.mark.parametrize("key", list(OLD_GRAPHS))
def test_original_topologies_byte_identical(key):
    topo, kw = key
    g = [network.make_graph(NetworkCfg(topology=topo, **dict(kw)), NAMES40, s) for s in range(25)]
    assert hashlib.sha256(json.dumps(g).encode()).hexdigest() == OLD_GRAPHS[key]


def test_original_network_config_hashes_unchanged():
    for mode, h in OLD_HASHES.items():
        cfg = from_dict({"game": {"n_agents": 40, "network": {"topology": "erdos_renyi", "mode": mode}}})
        assert cfg.config_hash() == h
        explicit = from_dict({"game": {"n_agents": 40, "network": {"topology": "erdos_renyi", "mode": mode,
                                                                   "key_placement": "random", "communities": 4}}})
        assert explicit.config_hash() == h
    moved = from_dict({"game": {"n_agents": 40, "network": {"topology": "erdos_renyi", "key_placement": "local_bias"}}})
    assert moved.config_hash() != OLD_HASHES["contacts"]


def test_original_meta_has_no_new_keys():
    cfg = from_dict({"game": {"n_agents": 12, "network": {"topology": "erdos_renyi"}}}).game
    st = generate_world(cfg, 3)
    assert set(network.network_meta(st, cfg, 3)) == {"topology", "mode", "edges", "degree"}


# ------------------------------------------------------------------ new graphs
@pytest.mark.parametrize("topo", list(NEW))
@pytest.mark.parametrize("names", [NAMES10, NAMES40], ids=["n10", "n40"])
def test_new_graphs_connected_symmetric_deterministic(topo, names):
    cfg = NetworkCfg(topology=topo, **NEW[topo])
    for seed in range(30):
        g, comm = network.make_graph_full(cfg, names, seed)
        assert (g, comm) == network.make_graph_full(cfg, list(reversed(names)), seed)   # order-independent
        assert g == network.make_graph(cfg, names, seed)
        assert all(a in g[b] for a in g for b in g[a]) and all(a not in g[a] for a in g)
        assert len(network.components(len(names), _idx_edges(g, names))) == 1
        if topo == "powerlaw_cluster":
            assert comm is None
        else:
            sizes = sorted(collections_count(comm.values()))
            assert len(sizes) == 4 and sizes[-1] - sizes[0] <= 1
    assert network.make_graph(cfg, names, 1) != network.make_graph(cfg, names, 2)


def collections_count(xs):
    out: dict[int, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return list(out.values())


def test_sbm_bridges_and_modularity():
    cfg = NetworkCfg(topology="sbm", mean_degree=4.0, min_bridges=5)
    mods, degs = [], []
    for seed in range(30):
        g, comm = network.make_graph_full(cfg, NAMES40, seed)
        s = network.graph_summary(g, comm)
        assert s["between_community_edges"] >= 5 and s["connected"] and s["communities_planted"]
        mods.append(s["modularity"])
        degs.append(s["mean_degree"])
    assert statistics.mean(mods) > 0.5 and 3.6 < statistics.mean(degs) < 4.8
    explicit = NetworkCfg(topology="sbm", p_in=0.5, p_out=0.02)
    g, comm = network.make_graph_full(explicit, NAMES40, 0)
    assert network.graph_summary(g, comm)["mean_degree"] > 4.5


def test_powerlaw_cluster_hubs_and_clustering():
    pc = [network.graph_summary(network.make_graph(NetworkCfg(topology="powerlaw_cluster", p_triangle=0.8),
                                                   NAMES40, s), None) for s in range(20)]
    ba = [network.graph_summary(network.make_graph(NetworkCfg(topology="barabasi_albert"), NAMES40, s), None)
          for s in range(20)]
    assert statistics.mean(x["clustering"] for x in pc) > statistics.mean(x["clustering"] for x in ba) + 0.15
    assert statistics.mean(x["max_degree"] for x in pc) >= 10
    assert all(x["edges"] <= 2 * (40 - 2) for x in pc)


def test_community_hubs_structure():
    cfg = NetworkCfg(topology="community_hubs", bridges=3)
    for seed in range(20):
        g, comm = network.make_graph_full(cfg, NAMES40, seed)
        groups: dict[int, list[str]] = {}
        for a, c in comm.items():
            groups.setdefault(c, []).append(a)
        hubs = []
        for ms in groups.values():
            hub = next(h for h in ms if all(m in g[h] for m in ms if m != h))     # adjacent to every member
            hubs.append(hub)
        assert all(b in g[a] for a in hubs for b in hubs if a != b)                # hub clique
        s = network.graph_summary(g, comm)
        assert s["between_community_edges"] == 6 + 3 and s["diameter"] <= 3
        assert set(s["hubs"][:3]) <= set(hubs)


def test_config_validation_new():
    with pytest.raises(ValueError, match="communities"):
        from_dict({"game": {"n_agents": 6, "network": {"topology": "sbm", "communities": 4}}})
    with pytest.raises(ValueError, match="local_bias"):
        from_dict({"game": {"network": {"key_placement": "local_bias"}}})
    with pytest.raises(ValueError, match="m must be"):
        from_dict({"game": {"n_agents": 4, "needs_per_agent": 2, "network": {"topology": "powerlaw_cluster", "m": 4}}})


# ------------------------------------------------------------------ key placement
def _cfg(topo: str, placement: str, **kw):
    return from_dict({"game": {"n_agents": 40, "network": {"topology": topo, "key_placement": placement,
                                                           **NEW.get(topo, {}), **kw}}}).game


@pytest.mark.parametrize("topo", ["erdos_renyi", *NEW])
def test_local_bias_solvable_and_closer(topo):
    near_r, near_b = [], []
    for seed in range(8):
        r = generate_world(_cfg(topo, "random"), seed)
        b = generate_world(_cfg(topo, "local_bias"), seed)
        assert is_feasible(b)
        assert [s.keys for s in r.agents.values()] == [s.keys for s in b.agents.values()]   # same deal
        assert [s.contacts for s in r.agents.values()] == [s.contacts for s in b.agents.values()]  # same graph
        for st in (r, b):                                     # every need held by another agent
            for a, s in st.agents.items():
                for i in s.needs:
                    assert any(i in {k.info_id for k in o.keys} for x, o in st.agents.items() if x != a)
        hr, hb = network.need_hops(r), network.need_hops(b)
        near_r.append(sum(h <= 2 for h in hr) / len(hr))
        near_b.append(sum(h <= 2 for h in hb) / len(hb))
    assert statistics.mean(near_b) > statistics.mean(near_r) + 0.1
    half = [network.need_hops(generate_world(_cfg(topo, "local_bias", local_strength=0.5), s)) for s in range(8)]
    mid = statistics.mean(sum(h <= 2 for h in x) / len(x) for x in half)
    assert statistics.mean(near_r) - 0.05 <= mid <= statistics.mean(near_b) + 0.05


def test_random_placement_needs_unchanged_by_topology():
    base = generate_world(from_dict({"game": {"n_agents": 40}}).game, 5)
    for topo in NEW:
        st = generate_world(_cfg(topo, "random"), 5)
        assert {a: s.needs for a, s in st.agents.items()} == {a: s.needs for a, s in base.agents.items()}


def test_new_meta_carries_stats():
    cfg = _cfg("community_hubs", "local_bias")
    st = generate_world(cfg, 2)
    m = network.network_meta(st, cfg, 2)
    assert m["communities"] and m["stats"]["communities_planted"] and m["stats"]["connected"]
    assert sum(m["need_hops_hist"].values()) == 40 * 3 and 0 <= m["share_needs_within_2_hops"] <= 1


# ------------------------------------------------------------------ runner: branches share the graph
NET = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=12", "game.max_rounds=4"]


@pytest.fixture(scope="module")
def topo_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("toporuns")
    return {
        "sbm": _run(tmp, "t_sbm", [*NET, "game.network.topology=sbm", "game.network.communities=3",
                                   "game.network.key_placement=local_bias"]),
        "hubs": _run(tmp, "t_hubs", [*NET, "game.network.topology=community_hubs", "game.network.communities=3",
                                     "game.network.mode=known_ids", "game.board.scope=neighbors"]),
        "plc": _run(tmp, "t_plc", [*NET, "game.turn_mode=simultaneous", "game.network.topology=powerlaw_cluster",
                                   "game.network.mode=known_ids"]),
    }


def test_topology_runs_invariants_and_shared_graph(topo_runs):
    for name, run in topo_runs.items():
        assert check_run(run) == [], name
        dirs = sorted((run / "games").iterdir())
        metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
        assert len(metas) >= 2
        for key in ("edges", "communities", "stats"):
            assert len({json.dumps(m["network"][key]) for m in metas}) == 1, (name, key)   # god == nogod
        starts = [json.loads((d / "events.jsonl").read_text().splitlines()[0]) for d in dirs]
        assert len({json.dumps({a: s["contacts"] for a, s in x["data"]["agents"].items()}) for x in starts}) == 1


def test_analysis_by_community(topo_runs):
    res = net_analysis.analyse_run(topo_runs["hubs"])
    for br, c in res["communities"].items():
        assert c["planted"] is True
        g = c["gives"]
        assert g["within"] + g["between"] == sum(v for t in res["trades_by_distance"][br].values() for v in [t])
        assert set(c["roles"]) == {"hub", "bridge", "member"}
        assert "names_learned" in c
    plc = net_analysis.analyse_run(topo_runs["plc"])
    assert all(c["planted"] is False for c in plc["communities"].values())       # detected communities
