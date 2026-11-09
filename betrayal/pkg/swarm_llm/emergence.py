"""
Emergence and scaling analysis: how leaders, roles, and failures appear as the
swarm grows. Everything here is pure measurement on a finished game. No attack.

From the gift/message record we recover:
  * the interaction network (who gave to / messaged whom)
  * leader emergence: concentration of giving, top agent's share, in-degree
    centralization (Freeman), whether a single hub emerged
  * role differentiation: each agent typed as giver / taker / broker / isolate
    from its give-out, take-in, and betweenness
  * failure-with-scale: completion, inequality, betrayal rate vs. N

`scaling_sweep` runs the swarm at a list of sizes and returns one row per
(size, seed) so you can look for phase transitions.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def _gift_graph(st):
    G = nx.DiGraph()
    G.add_nodes_from(range(st.cfg.N))
    for i in range(st.cfg.N):
        for j in range(st.cfg.N):
            w = int(st.gifts[i, j])
            if w:
                G.add_edge(i, j, weight=w)
    return G


def _msg_graph(st):
    G = nx.DiGraph()
    G.add_nodes_from(range(st.cfg.N))
    for i in range(st.cfg.N):
        for h in st.heard[i]:
            if 0 <= h["from_"] < st.cfg.N:
                G.add_edge(h["from_"], i, weight=G.get_edge_data(h["from_"], i, {}).get("weight", 0) + 1)
    return G


def gini(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) == 0 or xs.sum() == 0:
        return 0.0
    n = len(xs)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))


def leader_emergence(st):
    N = st.cfg.N
    out_gifts = st.gifts.sum(axis=1)
    in_gifts = st.gifts.sum(axis=0)
    total = out_gifts.sum()
    G = _gift_graph(st)
    # Freeman in-degree centralization: 0 = flat, 1 = one agent receives everything
    indeg = np.array([G.in_degree(i) for i in range(N)], dtype=float)
    cmax = indeg.max() if N > 1 else 0
    centralization = float((cmax - indeg).sum() / ((N - 1) * (N - 2))) if N > 2 else 0.0
    top = int(np.argmax(out_gifts)) if total else None
    return dict(
        top_giver=top,
        top_giver_share=float(out_gifts.max() / total) if total else 0.0,
        giving_gini=gini(out_gifts),
        receiving_gini=gini(in_gifts),
        indegree_centralization=centralization,
        single_hub=bool(total and out_gifts.max() >= 0.4 * total),
        reciprocity=float(nx.reciprocity(G) or 0.0) if G.number_of_edges() else 0.0,
    )


def role_types(st):
    """Type each agent from standardized give-out, take-in and brokerage."""
    N = st.cfg.N
    out_g = st.gifts.sum(axis=1).astype(float)
    in_g = st.gifts.sum(axis=0).astype(float)
    Gm = _msg_graph(st).to_undirected()
    btw = nx.betweenness_centrality(Gm) if Gm.number_of_edges() else {i: 0.0 for i in range(N)}

    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else a * 0
    zo, zi = z(out_g), z(in_g)
    zb = z(np.array([btw[i] for i in range(N)]))
    roles = {}
    for i in range(N):
        if out_g[i] == 0 and in_g[i] == 0:
            roles[i] = "isolate"
        elif zb[i] > 1.0 and zo[i] > 0:
            roles[i] = "broker"
        elif zo[i] - zi[i] > 0.5:
            roles[i] = "giver"
        elif zi[i] - zo[i] > 0.5:
            roles[i] = "taker"
        else:
            roles[i] = "trader"
    counts = {r: sum(v == r for v in roles.values()) for r in ["giver", "taker", "broker", "trader", "isolate"]}
    return roles, counts


def emergence_profile(st, exclude=()):
    from llm_game import score, betrayal_metrics
    honest = [j for j in range(st.cfg.N) if j not in st.compromised and j not in exclude]
    roles, counts = role_types(st)
    lead = leader_emergence(st)
    scores = [score(st, j) for j in honest]
    bm = betrayal_metrics(st)
    return dict(
        N=st.cfg.N,
        completion=float(np.mean([st.done(j) for j in honest])),
        mean_score=float(np.mean(scores)),
        score_gini=gini(scores),
        role_counts=counts,
        role_entropy=_entropy(list(counts.values())),
        **{k: lead[k] for k in ["top_giver_share", "giving_gini", "indegree_centralization",
                                "single_hub", "reciprocity"]},
        defection_rate=bm["defection_rate"],
        retaliations=bm["retaliations"],
        rounds=st.round,
    )


def _entropy(counts):
    a = np.asarray(counts, dtype=float)
    p = a[a > 0] / a.sum() if a.sum() else a
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def scaling_sweep(run_one, sizes, seeds, **kw):
    """run_one(N, seed, **kw) -> a finished State (or (rows, games) with a 'clean'
    branch). Returns a list of emergence rows, one per (N, seed)."""
    out = []
    for N in sizes:
        for seed in seeds:
            res = run_one(N, seed, **kw)
            st = res if hasattr(res, "cfg") else res[1]["clean"]
            row = dict(seed=seed, **emergence_profile(st))
            out.append(row)
    return out
