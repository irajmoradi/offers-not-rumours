"""
Network-science analysis of a swarm: centralities, community detection, hub and
bridge detection, richer structural roles, and how all of these change over the
rounds (dynamics as the network grows).

Works on a finished game State. Temporal functions need keep_log=True.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------

def gift_graph(st, upto_round=None):
    """Directed, weighted giver -> receiver graph. If upto_round and a log
    exists, only gifts up to that round are included."""
    G = nx.DiGraph()
    G.add_nodes_from(range(st.cfg.N))
    if upto_round is not None and getattr(st, "log", None):
        for ev in st.log:
            if ev.get("type") == "give" and ev.get("round", 0) <= upto_round:
                i, j = ev["agent"], ev["to"]
                G.add_edge(i, j, weight=G.get_edge_data(i, j, {}).get("weight", 0) + 1)
    else:
        for i in range(st.cfg.N):
            for j in range(st.cfg.N):
                if st.gifts[i, j]:
                    G.add_edge(i, j, weight=int(st.gifts[i, j]))
    return G


def message_graph(st):
    G = nx.DiGraph()
    G.add_nodes_from(range(st.cfg.N))
    for i in range(st.cfg.N):
        for h in st.heard[i]:
            s = h["from_"]
            if 0 <= s < st.cfg.N:
                G.add_edge(s, i, weight=G.get_edge_data(s, i, {}).get("weight", 0) + 1)
    return G


# --------------------------------------------------------------------------
# Centralities
# --------------------------------------------------------------------------

def centralities(G):
    """Per-node centralities on a directed weighted graph."""
    N = G.number_of_nodes()
    und = G.to_undirected()
    out = dict(
        in_degree={i: G.in_degree(i, weight="weight") for i in G},
        out_degree={i: G.out_degree(i, weight="weight") for i in G},
        degree_centrality=nx.degree_centrality(G),
        betweenness=nx.betweenness_centrality(und, weight=None) if und.number_of_edges() else {i: 0.0 for i in G},
        closeness=nx.closeness_centrality(G) if G.number_of_edges() else {i: 0.0 for i in G},
        pagerank=nx.pagerank(G, weight="weight") if G.number_of_edges() else {i: 1 / max(N, 1) for i in G},
    )
    try:
        out["eigenvector"] = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
    except Exception:
        out["eigenvector"] = out["pagerank"]
    return out


def freeman_centralization(G):
    """0 = flat, 1 = one node holds all in-edges."""
    N = G.number_of_nodes()
    if N < 3:
        return 0.0
    deg = np.array([G.in_degree(i) for i in range(N)], dtype=float)
    return float((deg.max() - deg).sum() / ((N - 1) * (N - 2)))


# --------------------------------------------------------------------------
# Communities, hubs, bridges
# --------------------------------------------------------------------------

def communities(G):
    """Greedy-modularity communities on the undirected graph. Returns
    (membership dict, list of sets, modularity)."""
    und = G.to_undirected()
    if und.number_of_edges() == 0:
        comms = [{n} for n in und.nodes()]
    else:
        comms = list(nx.community.greedy_modularity_communities(und))
    membership = {n: k for k, c in enumerate(comms) for n in c}
    try:
        mod = nx.community.modularity(und, comms) if und.number_of_edges() else 0.0
    except Exception:
        mod = 0.0
    return membership, [set(c) for c in comms], float(mod)


def hits_hubs(G):
    """HITS hub/authority scores (hubs point to many; authorities are pointed to)."""
    if G.number_of_edges() == 0:
        z = {i: 0.0 for i in G}
        return z, z
    try:
        hubs, auth = nx.hits(G, max_iter=500, normalized=True)
        return hubs, auth
    except Exception:
        z = {i: 0.0 for i in G}
        return z, z


def bridges(G):
    """Bridge edges (removal increases components) and the articulation points,
    on the undirected graph. Also inter-community edges = cross-group links."""
    und = G.to_undirected()
    out = {"bridge_edges": [], "articulation_points": [], "inter_community_edges": []}
    if und.number_of_edges() == 0:
        return out
    out["bridge_edges"] = [list(e) for e in nx.bridges(und)]
    out["articulation_points"] = list(nx.articulation_points(und))
    membership, _, _ = communities(G)
    out["inter_community_edges"] = [list(e) for e in und.edges() if membership.get(e[0]) != membership.get(e[1])]
    return out


# --------------------------------------------------------------------------
# Structural roles (richer than giver/taker/trader)
# --------------------------------------------------------------------------

ROLE_SET = ["leader", "hub", "broker", "connector", "giver", "taker", "peripheral", "isolate"]


def structural_roles(st):
    """Assign one role per agent from centralities + community structure.
    Returns (roles dict, per-agent metrics table)."""
    G = gift_graph(st)
    Gm = message_graph(st)
    N = st.cfg.N
    c = centralities(G)
    cm = centralities(Gm)
    membership, comms, _ = communities(Gm if Gm.number_of_edges() else G)
    br = bridges(Gm if Gm.number_of_edges() else G)
    inter_nodes = {n for e in br["inter_community_edges"] for n in e}
    artic = set(br["articulation_points"])
    out_g = st.gifts.sum(axis=1).astype(float)
    in_g = st.gifts.sum(axis=0).astype(float)
    pr = np.array([c["pagerank"][i] for i in range(N)])
    btw = np.array([cm["betweenness"][i] for i in range(N)])

    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else a * 0
    zpr, zbtw, zog, zig = z(pr), z(btw), z(out_g), z(in_g)
    roles, table = {}, {}
    for i in range(N):
        deg = G.degree(i) + Gm.degree(i)
        if deg == 0:
            role = "isolate"
        elif zpr[i] > 1.0 and in_g[i] >= out_g[i]:
            role = "leader"
        elif zbtw[i] > 1.0 and (i in inter_nodes or i in artic):
            role = "broker"
        elif i in inter_nodes or i in artic:
            role = "connector"
        elif zog[i] > 1.0:
            role = "hub"
        elif zog[i] - zig[i] > 0.5:
            role = "giver"
        elif zig[i] - zog[i] > 0.5:
            role = "taker"
        else:
            role = "peripheral"
        roles[i] = role
        table[i] = dict(community=membership.get(i), pagerank=round(float(pr[i]), 3),
                        betweenness=round(float(btw[i]), 3), gifts_given=int(out_g[i]),
                        gifts_received=int(in_g[i]), is_bridge=i in inter_nodes or i in artic)
    return roles, table


def network_summary(st):
    G = gift_graph(st)
    Gm = message_graph(st)
    membership, comms, mod = communities(Gm if Gm.number_of_edges() else G)
    br = bridges(Gm if Gm.number_of_edges() else G)
    hubs, auth = hits_hubs(G)
    roles, _ = structural_roles(st)
    counts = {r: sum(v == r for v in roles.values()) for r in ROLE_SET}
    return dict(
        N=st.cfg.N,
        n_communities=len(comms),
        modularity=round(mod, 3),
        n_bridge_edges=len(br["bridge_edges"]),
        n_articulation_points=len(br["articulation_points"]),
        n_inter_community_edges=len(br["inter_community_edges"]),
        gift_centralization=round(freeman_centralization(G), 3),
        message_centralization=round(freeman_centralization(Gm), 3),
        top_hub=int(max(hubs, key=hubs.get)) if hubs else None,
        top_authority=int(max(auth, key=auth.get)) if auth else None,
        role_counts=counts,
    )


# --------------------------------------------------------------------------
# Temporal dynamics (needs keep_log=True)
# --------------------------------------------------------------------------

def temporal_metrics(st):
    """One row per round: how structure grows over time."""
    if not getattr(st, "log", None):
        raise ValueError("Build the game with keep_log=True for temporal metrics.")
    rounds = sorted({ev.get("round", 0) for ev in st.log if ev.get("round")})
    rows = []
    for r in rounds:
        G = gift_graph(st, upto_round=r)
        _, comms, mod = communities(G)
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {i: 0 for i in range(st.cfg.N)}
        btw = nx.betweenness_centrality(G.to_undirected()) if G.number_of_edges() else {i: 0 for i in range(st.cfg.N)}
        edges = G.number_of_edges()
        rows.append(dict(
            round=r,
            edges=edges,
            density=round(nx.density(G), 3),
            gift_centralization=round(freeman_centralization(G), 3),
            n_communities=len([c for c in comms if len(c) > 1]) or (1 if edges else 0),
            modularity=round(mod, 3),
            top_leader=int(max(pr, key=pr.get)) if edges else None,
            max_pagerank=round(float(max(pr.values())) if pr else 0.0, 3),
            max_betweenness=round(float(max(btw.values())) if btw else 0.0, 3),
        ))
    return rows


def leader_stability(st):
    """How often the top-pagerank agent changes across rounds; and whether the
    final leader led early (lock-in)."""
    rows = temporal_metrics(st)
    leaders = [r["top_leader"] for r in rows if r["top_leader"] is not None]
    if not leaders:
        return dict(leader_changes=0, final_leader=None, led_from_round=None, lock_in=None)
    final = leaders[-1]
    changes = sum(a != b for a, b in zip(leaders, leaders[1:]))
    led_from = next((rows[k]["round"] for k in range(len(rows))
                     if rows[k]["top_leader"] == final and all(rows[j]["top_leader"] in (final, None)
                     for j in range(k, len(rows)))), None)
    return dict(leader_changes=changes, final_leader=final, led_from_round=led_from,
                lock_in=round(1 - changes / max(len(leaders) - 1, 1), 3))
