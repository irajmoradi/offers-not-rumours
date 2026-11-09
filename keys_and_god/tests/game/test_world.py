"""World generation: feasibility (independent max-flow check), determinism, supply, needs."""

from __future__ import annotations

import networkx as nx
import pytest

from swarm.config import GameCfg
from swarm.game.world import generate_world, is_feasible, max_matching


def _flow_feasible(st) -> bool:
    """Independent check: unit-capacity max-flow demand slots -> copies held by others."""
    g = nx.DiGraph()
    demand = 0
    for a, s in st.agents.items():
        for i in s.needs:
            demand += 1
            g.add_edge("S", ("d", a, i), capacity=1)
            for h, hs in st.agents.items():
                for kc in hs.keys:
                    if h != a and kc.info_id == i:
                        g.add_edge(("d", a, i), ("c", kc.copy_id), capacity=1)
    for s in st.agents.values():
        for kc in s.keys:
            g.add_edge(("c", kc.copy_id), "T", capacity=1)
    return nx.maximum_flow_value(g, "S", "T") == demand


@pytest.mark.parametrize("n,k,m,c", [(4, 2, 2, 2), (6, 5, 3, 2), (20, 5, 3, 2), (20, 5, 3, 1), (10, 3, 3, 3),
                                     (7, 5, 3, 2), (3, 2, 1, 2)])
@pytest.mark.parametrize("seed", range(5))
def test_feasible_and_wellformed(n, k, m, c, seed):
    cfg = GameCfg(n_agents=n, keys_per_agent=k, needs_per_agent=m, copies_per_info=c)
    st = generate_world(cfg, seed)
    assert is_feasible(st) and _flow_feasible(st)
    assert len(st.agents) == n and len(set(st.agents)) == n
    copy_ids = [kc.copy_id for s in st.agents.values() for kc in s.keys]
    assert len(copy_ids) == len(set(copy_ids)) == n * k
    for s in st.agents.values():
        held = [kc.info_id for kc in s.keys]
        assert len(held) == k and len(set(held)) == k          # no duplicate info in a hand
        assert len(s.needs) == m == len(set(s.needs))
        assert not set(s.needs) & set(held)                    # D4: needs disjoint from holdings
    counts: dict[int, int] = {}
    for s in st.agents.values():
        for kc in s.keys:
            counts[kc.info_id] = counts.get(kc.info_id, 0) + 1
    assert max(counts.values()) <= c                           # copies_per_info controls supply
    keys = {}
    for s in st.agents.values():
        for kc in s.keys:
            assert keys.setdefault(kc.info_id, kc.key) == kc.key   # all copies share a key value


def test_deterministic():
    cfg = GameCfg(n_agents=20)
    assert generate_world(cfg, 11).state_hash() == generate_world(cfg, 11).state_hash()
    assert generate_world(cfg, 11).state_hash() != generate_world(cfg, 12).state_hash()


def test_names_two_words_and_explicit_names():
    st = generate_world(GameCfg(n_agents=20), 0)
    assert all(len(a.split("_")) == 2 for a in st.agents)
    st2 = generate_world(GameCfg(n_agents=3, keys_per_agent=2, needs_per_agent=1), 0, names=["x", "y", "z"])
    assert list(st2.agents) == ["x", "y", "z"]
    with pytest.raises(ValueError):
        generate_world(GameCfg(n_agents=3, keys_per_agent=2, needs_per_agent=1), 0, names=["x", "x", "z"])


def test_matching_detects_infeasible():
    # two agents need info 9, only one copy held by a third agent
    demands = [("a", 9), ("b", 9)]
    assert max_matching(demands, [("c", 9)]) == 1
    assert max_matching(demands, [("c", 9), ("a", 9)]) == 2      # a's copy can serve b
    assert max_matching([("a", 9)], [("a", 9)]) == 0             # own copy does not count
