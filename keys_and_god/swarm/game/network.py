"""Social network over the agents (game.network; topology "complete" = off = the original rules).

Graph (pure, seeded): `make_graph(net_cfg, agents, seed)` -> {agent: sorted neighbours}. All randomness
comes from make_rng(seed, "network"), where `seed` is the WORLD seed, so the God / no-God / noise
branches of one game get the same graph (the world's own random stream is untouched). Agents are
placed on graph positions by a seeded shuffle, so hubs / ring neighbours are random agents.
  erdos_renyi      G(n, p) with p = mean_degree / (n - 1). Disconnected samples are REJECTED and
                   redrawn from the same stream (so the graph is a sample of G(n, p) conditioned on
                   being connected); after MAX_RESAMPLES rejections the components are chained by
                   one edge each (never reached for n=10, mean degree 3 in practice).
  watts_strogatz   ring lattice with k neighbours (k/2 per side), each edge rewired with prob. p
                   (no self-loops / duplicates); same rejection rule for connectivity.
  barabasi_albert  preferential attachment, m edges per new node from m seed nodes (connected).
  ring             a cycle (degree 2).
  (added later; graphs carry planted communities = blocks of consecutive positions, sizes as equal
   as possible)
  sbm              stochastic block model: `communities` blocks; edge prob. p_in within a block and
                   p_out between blocks, or (p_in / p_out unset) derived from mean_degree and mixing
                   (expected share of between-block edges). Samples that are disconnected or have
                   fewer than min_bridges between-block edges are rejected and redrawn (chain-connect
                   fallback after MAX_RESAMPLES).
  powerlaw_cluster Holme-Kim: preferential attachment with m edges per new node, each edge after
                   the first is, with prob. p_triangle, closed into a triangle with a neighbour of the
                   last target (hubs + high clustering; connected by construction).
  community_hubs   `communities` clusters; each has a hub (its first position) linked to every member,
                   members linked with prob. p_in (default 0.2); the hubs form a clique; plus
                   `bridges` extra member-member edges between random pairs of clusters (so some
                   non-hub members are bridges). Connected by construction.
  LFR is not offered: networkx's LFR generator fails ~1 in 8 seeds at n=40 (community assignment
  does not converge), is not guaranteed connected, and its stream would not be ours.
`attach(state, cfg, seed)` stores the graph on the state: AgentState.contacts (both modes) and, in
mode known_ids, AgentState.known (= the contacts at the start).

Rules (used by game.rules / observe / engine; None everywhere when the network is off):
  contacts   send_message / give_key / defect only along an edge; otherwise the action fails
             "not_connected" (state unchanged) and the actor gets a "system" notice.
  known_ids  any pair may interact, but only on a name the actor KNOWS ("unknown_contact" otherwise,
             with a notice). An agent learns names at each observation (`learn`), from: the sender
             of a DM it receives and every agent named in its text; the author of a board post it
             sees (unless the board is anonymous) and every agent named in its text; agents named in
             its gift_received / burned / defect_attempt notices (a give or defect involving it).
             The learned names are part of that same turn's player list, so it can act on them now.
  view_log   with the network on, an agent sees a give / defect / DM line only when it is local:
             contacts mode: the actor or the target is the agent or one of its contacts;
             known_ids mode: every named party (actor if shown, target) is the agent or known to it.
             Board-post lines follow the board's own scope (below).
  board      game.board.scope = neighbors: an agent sees only posts by its graph neighbours (board
             view, view_log, grep_log), in both modes; global (default): every post, as before.
connect     game.network.connect.enabled (default off): the action connect(target) adds the undirected
             edge actor-target to the live graph (both AgentState.contacts; `add_edge`), so every rule
             above (contacts, view_log locality, board scope, God's forged DMs) uses the CURRENT graph
             from the next action on. Requirements, in order (`connect_error`): network on
             ("not_network"), connect enabled ("connect_disabled"), the target exists
             ("unknown_agent"), is not the actor ("self_target"), has not left (sit_out,
             "target_left"), in mode known_ids is a name the actor knows ("unknown_contact"), the
             actor is under max_new_edges_per_agent ("edge_cap_reached"), and they are not already
             connected ("already_connected"), except the MUTUAL case: the target connected to the
             actor in this round at a slot >= the actor's current observation (so the actor could not
             see it; in practice simultaneous mode), which succeeds as mutual=True (one edge, both pay).
             The cost, the whole-turn rule and the notices are in game/rules.py + game/resolve.py.
             The target's "connected" notice is a learning notice (known_ids: it learns the actor).
God (adversary.powers): in contacts mode a forged DM needs sender -> receiver to be an edge (fails
"not_connected" otherwise, nothing spent). known_ids puts no limit on God; the receiver learns the
shown sender's name like any DM sender's.
Pure (no I/O); `attach` and `learn` mutate the state they are given. Imports: config, rng, schema
(networkx lazily, inside graph_summary, for the meta stats only).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Sequence

from swarm.config import GameCfg, NetworkCfg
from swarm.rng import make_rng
from swarm.schema import AgentId, AgentMessageView, GameState, NoticeView, Observation

Edges = set[tuple[int, int]]
MAX_RESAMPLES = 1000
NOT_CONNECTED = "not_connected"
UNKNOWN_CONTACT = "unknown_contact"
REASONS = (NOT_CONNECTED, UNKNOWN_CONTACT)
LEARN_NOTICE_KINDS = ("gift_received", "burned", "defect_attempt", "connected")   # connected: connect
ANON_SENDER = "(anonymous)"        # = observe.ANON_SENDER (not imported: observe imports this module)


# ------------------------------------------------------------------ graph generation
def _edge(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def components(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the graph on nodes 0..n-1 (each sorted; ordered by smallest node)."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    seen: set[int] = set()
    out: list[list[int]] = []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        out.append(sorted(comp))
    return out


def _erdos_renyi(n: int, mean_degree: float, rng: random.Random) -> Edges:
    p = mean_degree / (n - 1)
    return {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p}


def _ring_lattice(n: int, k: int) -> Edges:
    return {_edge(i, (i + d) % n) for i in range(n) for d in range(1, k // 2 + 1)}


def _watts_strogatz(n: int, k: int, p: float, rng: random.Random) -> Edges:
    edges = _ring_lattice(n, k)
    for d in range(1, k // 2 + 1):                    # the classic rewiring order (by lattice distance)
        for i in range(n):
            e = _edge(i, (i + d) % n)
            if e not in edges or rng.random() >= p:
                continue
            free = [j for j in range(n) if j != i and _edge(i, j) not in edges]
            if free:
                edges.discard(e)
                edges.add(_edge(i, rng.choice(free)))
    return edges


def _barabasi_albert(n: int, m: int, rng: random.Random) -> Edges:
    edges: Edges = set()
    targets = list(range(m))
    repeated: list[int] = []
    for new in range(m, n):
        for t in targets:
            edges.add(_edge(new, t))
        repeated += targets + [new] * m
        chosen: list[int] = []
        while len(chosen) < m and new + 1 < n:
            c = rng.choice(repeated)
            if c not in chosen:
                chosen.append(c)
        targets = chosen
    return edges


def _blocks(n: int, c: int) -> list[int]:
    """Planted community of each position: c blocks of consecutive positions, sizes differ by <= 1."""
    sizes = [n // c + (1 if b < n % c else 0) for b in range(c)]
    return [b for b, sz in enumerate(sizes) for _ in range(sz)]


def _sbm_probs(cfg: NetworkCfg, n: int) -> tuple[float, float]:
    """(p_in, p_out): the configured ones, else from mean_degree d and mixing mu (expected within-
    block degree (1 - mu) d over s - 1 peers, between-block degree mu d over n - s; s = n / c)."""
    if cfg.p_in is not None and cfg.p_out is not None:
        return cfg.p_in, cfg.p_out
    s = n / cfg.communities
    p_in = min(1.0, (1 - cfg.mixing) * cfg.mean_degree / max(s - 1, 1))
    p_out = min(1.0, cfg.mixing * cfg.mean_degree / max(n - s, 1))
    return (cfg.p_in if cfg.p_in is not None else p_in), (cfg.p_out if cfg.p_out is not None else p_out)


def _sbm(n: int, blocks: list[int], p_in: float, p_out: float, rng: random.Random) -> Edges:
    return {(i, j) for i in range(n) for j in range(i + 1, n)
            if rng.random() < (p_in if blocks[i] == blocks[j] else p_out)}


def _powerlaw_cluster(n: int, m: int, p: float, rng: random.Random) -> Edges:
    """Holme-Kim (networkx.powerlaw_cluster_graph's algorithm, on our seeded stream)."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    repeated = list(range(m))

    def add(u: int, v: int) -> None:
        adj[u].add(v)
        adj[v].add(u)

    for src in range(m, n):
        targets: list[int] = []
        while len(targets) < m:                       # m distinct targets, degree-proportional
            t = rng.choice(repeated)
            if t not in targets:
                targets.append(t)
        target = targets.pop()
        add(src, target)
        repeated.append(target)
        count = 1
        while count < m:
            if rng.random() < p:                      # triad formation: a neighbour of the last target
                nbhd = sorted(v for v in adj[target] if v != src and v not in adj[src])
                if nbhd:
                    v = rng.choice(nbhd)
                    add(src, v)
                    repeated.append(v)
                    count += 1
                    continue
            target = targets.pop()
            add(src, target)
            repeated.append(target)
            count += 1
        repeated += [src] * m
    return {_edge(u, v) for u in adj for v in adj[u]}


def _community_hubs(n: int, blocks: list[int], p_in: float, bridges: int, rng: random.Random) -> Edges:
    c = max(blocks) + 1
    members = [[i for i in range(n) if blocks[i] == b] for b in range(c)]
    hubs = [ms[0] for ms in members]
    edges: Edges = {_edge(h, i) for h, ms in zip(hubs, members, strict=True) for i in ms[1:]}
    for ms in members:
        edges |= {(i, j) for x, i in enumerate(ms[1:]) for j in ms[x + 2:] if rng.random() < p_in}
    edges |= {_edge(a, b) for x, a in enumerate(hubs) for b in hubs[x + 1:]}
    added, tries = 0, 0
    while added < bridges and tries < 100 * (bridges + 1):
        tries += 1
        b1, b2 = rng.sample(range(c), 2)
        if len(members[b1]) < 2 or len(members[b2]) < 2:
            continue
        e = _edge(rng.choice(members[b1][1:]), rng.choice(members[b2][1:]))
        if e not in edges:
            edges.add(e)
            added += 1
    return edges


def between_edges(edges: Iterable[tuple[int, int]], blocks: Sequence[int]) -> int:
    """Number of edges joining two different blocks."""
    return sum(1 for i, j in edges if blocks[i] != blocks[j])


def _connect(n: int, edges: Edges) -> Edges:
    """Chain the components (smallest node of each to the next one's): a last-resort fallback."""
    comps = components(n, edges)
    return edges | {_edge(a[0], b[0]) for a, b in zip(comps, comps[1:], strict=False)}


COMMUNITY_TOPOLOGIES = ("sbm", "community_hubs")
NEW_TOPOLOGIES = ("sbm", "powerlaw_cluster", "community_hubs")


def graph_layout(cfg: NetworkCfg, n: int, rng: random.Random) -> tuple[Edges, list[int] | None]:
    """(edges on positions 0..n-1, planted community of each position or None)."""
    if cfg.topology in COMMUNITY_TOPOLOGIES:
        blocks = _blocks(n, cfg.communities)
        if cfg.topology == "community_hubs":
            p_in = 0.2 if cfg.p_in is None else cfg.p_in
            return _community_hubs(n, blocks, p_in, cfg.bridges, rng), blocks
        p_in, p_out = _sbm_probs(cfg, n)
        edges: Edges = set()
        for _ in range(MAX_RESAMPLES):
            edges = _sbm(n, blocks, p_in, p_out, rng)
            if len(components(n, edges)) == 1 and between_edges(edges, blocks) >= cfg.min_bridges:
                return edges, blocks
        return _connect(n, edges), blocks
    if cfg.topology == "powerlaw_cluster":
        return _powerlaw_cluster(n, cfg.m, cfg.p_triangle, rng), None
    return graph_edges(cfg, n, rng), None


def graph_edges(cfg: NetworkCfg, n: int, rng: random.Random) -> Edges:
    """Edges on positions 0..n-1 for cfg.topology (see the module docstring)."""
    if cfg.topology in NEW_TOPOLOGIES:
        return graph_layout(cfg, n, rng)[0]
    if cfg.topology == "complete":
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    if cfg.topology == "ring":
        return _ring_lattice(n, 2)
    if cfg.topology == "barabasi_albert":
        return _barabasi_albert(n, cfg.m, rng)
    edges: Edges = set()
    for _ in range(MAX_RESAMPLES):
        edges = (_erdos_renyi(n, cfg.mean_degree, rng) if cfg.topology == "erdos_renyi"
                 else _watts_strogatz(n, cfg.k, cfg.p, rng))
        if len(components(n, edges)) == 1:
            return edges
    return _connect(n, edges)


def make_graph_full(cfg: NetworkCfg, agents: Sequence[AgentId],
                    seed: int) -> tuple[dict[AgentId, list[AgentId]], dict[AgentId, int] | None]:
    """({agent: sorted neighbours}, {agent: planted community} or None). Deterministic in
    (cfg, set of agents, seed); the same stream as make_graph."""
    rng = make_rng(seed, "network")
    pos = sorted(agents)
    rng.shuffle(pos)                                  # position -> agent
    adj: dict[AgentId, set[AgentId]] = {a: set() for a in pos}
    edges, blocks = graph_layout(cfg, len(pos), rng)
    for i, j in edges:
        adj[pos[i]].add(pos[j])
        adj[pos[j]].add(pos[i])
    comm = None if blocks is None else {pos[i]: b for i, b in enumerate(blocks)}
    return {a: sorted(adj[a]) for a in sorted(adj)}, comm


def make_graph(cfg: NetworkCfg, agents: Sequence[AgentId], seed: int) -> dict[AgentId, list[AgentId]]:
    """{agent: sorted neighbours}. Deterministic in (cfg, set of agents, seed)."""
    return make_graph_full(cfg, agents, seed)[0]


def hop_distances(adj: dict[AgentId, Sequence[AgentId]], src: AgentId) -> dict[AgentId, int]:
    """BFS hop distance from src to every reachable agent."""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def need_hops(state: GameState) -> list[int]:
    """Per need slot (agent order, need order): hop distance to the nearest OTHER holder of a copy."""
    adj = {a: list(s.contacts or []) for a, s in state.agents.items()}
    held: dict[int, list[AgentId]] = {}
    for a, s in state.agents.items():
        for kc in s.keys:
            held.setdefault(kc.info_id, []).append(a)
    out = []
    for a in sorted(state.agents):
        d = hop_distances(adj, a)
        for i in state.agents[a].needs:
            out.append(min(d.get(h, 10**6) for h in held.get(i, []) if h != a))
    return out


def graph_summary(adj: dict[AgentId, Sequence[AgentId]], communities: dict[AgentId, int] | None,
                  top: int = 3) -> dict[str, object]:
    """Network-science stats of one graph: degree distribution, clustering, path lengths, modularity
    of the planted communities (else of greedy-modularity communities, flagged), betweenness, and the
    top hubs (degree) / bridges (betweenness). networkx is imported lazily (stats only)."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(sorted(adj))
    g.add_edges_from((a, b) for a in adj for b in adj[a] if a < b)
    deg = dict(g.degree())
    btw = nx.betweenness_centrality(g)
    planted = communities is not None
    if communities is None:
        parts = nx.algorithms.community.greedy_modularity_communities(g)
        communities = {a: c for c, part in enumerate(sorted(parts, key=lambda p: min(p))) for a in part}
    groups: dict[int, set[AgentId]] = {}
    for a, c in communities.items():
        groups.setdefault(c, set()).add(a)
    hist: dict[int, int] = {}
    for d in deg.values():
        hist[d] = hist.get(d, 0) + 1
    conn = nx.is_connected(g)
    rank = sorted(g, key=lambda a: (-btw[a], a))
    return {"n": len(g), "edges": g.number_of_edges(), "mean_degree": round(2 * g.number_of_edges() / len(g), 3),
            "max_degree": max(deg.values()), "degree_hist": {str(k): hist[k] for k in sorted(hist)},
            "clustering": round(nx.average_clustering(g), 4), "transitivity": round(nx.transitivity(g), 4),
            "avg_path": round(nx.average_shortest_path_length(g), 4) if conn else None,
            "diameter": nx.diameter(g) if conn else None, "connected": conn,
            "communities_planted": planted, "n_communities": len(groups),
            "community_sizes": [len(groups[c]) for c in sorted(groups)],
            "modularity": round(nx.algorithms.community.modularity(g, [groups[c] for c in sorted(groups)]), 4),
            "between_community_edges": sum(1 for a, b in g.edges() if communities[a] != communities[b]),
            "bridge_nodes": sorted(a for a in g if any(communities[b] != communities[a] for b in g[a])),
            "hubs": sorted(g, key=lambda a: (-deg[a], a))[:top],
            "top_betweenness": [[a, round(btw[a], 4)] for a in rank[:top]]}


def attach(state: GameState, cfg: GameCfg, seed: int) -> None:
    """Store the graph on the state (no-op when the network is off). Call once, at world generation."""
    if not cfg.network.on:
        return
    for a, nbrs in make_graph(cfg.network, list(state.agents), seed).items():
        s = state.agents[a]
        s.contacts = list(nbrs)
        if cfg.network.mode == "known_ids":
            s.known = list(nbrs)
        elif cfg.network.connect.require_known:              # showcase: names heard of (connect targets)
            s.heard = list(nbrs)


def edge_list(state: GameState) -> list[tuple[AgentId, AgentId]]:
    """Sorted (a, b) pairs with a < b (empty when the network is off)."""
    return sorted({tuple(sorted((a, b))) for a, s in state.agents.items() for b in (s.contacts or [])})


def network_meta(state: GameState, cfg: GameCfg, seed: int | None = None) -> dict[str, object]:
    """The graph for a game's meta.json (topology, mode, edges, degrees); {} when the network is off.
    For the later topologies (NEW_TOPOLOGIES) or key_placement local_bias, given the world seed, it
    also carries the planted communities, graph_summary stats and the need -> nearest-holder hop
    histogram (the original topologies' meta is unchanged)."""
    if not cfg.network.on:
        return {}
    out: dict[str, object] = {"topology": cfg.network.topology, "mode": cfg.network.mode,
                              "edges": [list(e) for e in edge_list(state)],
                              "degree": {a: len(s.contacts or []) for a, s in sorted(state.agents.items())}}
    if seed is not None and (cfg.network.topology in NEW_TOPOLOGIES or cfg.network.key_placement != "random"):
        adj, comm = make_graph_full(cfg.network, list(state.agents), seed)
        hops = need_hops(state)
        hist: dict[int, int] = {}
        for h in hops:
            hist[h] = hist.get(h, 0) + 1
        out.update({"key_placement": cfg.network.key_placement, "communities": comm,
                    "stats": graph_summary(adj, comm),
                    "need_hops_hist": {str(k): hist[k] for k in sorted(hist)},
                    "share_needs_within_2_hops": round(sum(h <= 2 for h in hops) / max(len(hops), 1), 4)})
        if cfg.network.key_placement == "far":                   # showcase: achieved far placement
            d = cfg.network.far_min_hops
            out.update({"far_min_hops": d,
                        "share_needs_far": round(sum(h >= d for h in hops) / max(len(hops), 1), 4)})
    return out


# ------------------------------------------------------------------ rules
def is_on(state: GameState, agent: AgentId) -> bool:
    """Does the network restrict this agent? (its contacts are set)."""
    return state.agents[agent].contacts is not None


def contacts_mode(state: GameState, agent: AgentId) -> bool:
    """Mode contacts: interactions only along edges (contacts set, known unset)."""
    s = state.agents[agent]
    return s.contacts is not None and s.known is None


def visible_peers(state: GameState, agent: AgentId) -> set[AgentId] | None:
    """Who the agent may address: contacts (mode contacts), known names (known_ids); None = anyone."""
    s = state.agents[agent]
    if s.known is not None:
        return set(s.known)
    return set(s.contacts) if s.contacts is not None else None


def reach_error(state: GameState, actor: AgentId, other: AgentId) -> str | None:
    """Reason an action of `actor` aimed at `other` (a real, different agent) is refused, or None."""
    peers = visible_peers(state, actor)
    if peers is None or other in peers:
        return None
    return NOT_CONNECTED if contacts_mode(state, actor) else UNKNOWN_CONTACT


def reach_text(state: GameState, actor: AgentId, other: AgentId, reason: str, what: str) -> str:
    """The notice an actor gets for a refused action (`what` says what did not happen)."""
    if reason == NOT_CONNECTED:
        mine = ", ".join(state.agents[actor].contacts or []) or "none"
        return (f"You are not connected to {other}: you can only message and trade with your contacts "
                f"({mine}). {what}").rstrip()
    return (f"You don't know that agent ({other}): you can only message, trade with or defect on players "
            f"you know. You learn about other players through messages and the board. {what}").rstrip()


def line_visible(state: GameState, viewer: AgentId, actor: AgentId | None, target: AgentId | None) -> bool:
    """May `viewer` see a give / defect / DM log line between actor and target (None = not shown)?"""
    s = state.agents[viewer]
    if s.contacts is None:
        return True
    named = [x for x in (actor, target) if x is not None]
    if s.known is not None:
        ok = set(s.known) | {viewer}
        return all(x in ok for x in named)
    local = set(s.contacts) | {viewer}
    return any(x in local for x in named)


def sees_post(state: GameState, viewer: AgentId, sender: AgentId, cfg: GameCfg) -> bool:
    """game.board.scope: may `viewer` see a board post by `sender`? (global: always)."""
    if cfg.board.scope != "neighbors":
        return True
    return sender in (state.agents[viewer].contacts or [])


# ------------------------------------------------------------------ learning names (known_ids)
def names_in_text(text: str, names: Iterable[AgentId]) -> set[AgentId]:
    """Agent names occurring in `text` as whole words (names are adjective_noun ids)."""
    return {n for n in names if n in text and re.search(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", text)}


def names_seen(agent: AgentId, inbox: Sequence[AgentMessageView], board: Sequence[AgentMessageView],
               notices: Sequence[NoticeView], names: Iterable[AgentId],
               aliases: Callable[[str], set[AgentId]] | None = None) -> set[AgentId]:
    """Names `agent` learns from one observation's inbox, board and notices (module docstring).
    aliases (game.framing, e.g. facility @handles): text -> the agent ids it names under the framing's
    public names; message text is then also matched against those (None = ids only, the original)."""
    names = list(names)
    real = set(names)
    out: set[AgentId] = set()
    for m in [*inbox, *board]:
        if m.sender in real:
            out.add(m.sender)
        out |= names_in_text(m.text, names)
        if aliases is not None:
            out |= aliases(m.text) & real
    for nt in notices:
        if nt.kind in LEARN_NOTICE_KINDS:
            out |= names_in_text(nt.text, names)
    out.discard(agent)
    return out


def learn(state: GameState, agent: AgentId, obs: Observation,
          aliases: Callable[[str], set[AgentId]] | None = None) -> list[AgentId]:
    """known_ids: add the names seen in `obs` to the agent's known set; returns the new names (sorted).
    aliases: see names_seen (game.framing; connect.require_known learns @handles)."""
    s = state.agents[agent]
    if s.heard is not None:                                  # connect.require_known (mode contacts)
        new_h = names_seen(agent, obs.inbox, obs.board, obs.notices, state.agents, aliases) - set(s.heard)
        if new_h:
            s.heard = sorted(set(s.heard) | new_h)
    if s.known is None:
        return []
    new = names_seen(agent, obs.inbox, obs.board, obs.notices, state.agents, aliases) - set(s.known)
    if new:
        s.known = sorted(set(s.known) | new)
    return sorted(new)


# ------------------------------------------------------------------ dynamic edges (game.network.connect)
ALREADY_CONNECTED = "already_connected"
NOT_NETWORK = "not_network"
CONNECT_DISABLED = "connect_disabled"
EDGE_CAP = "edge_cap_reached"
CONNECT_WHOLE_TURN = "connect_uses_whole_turn"


def connects_made(state: GameState, agent: AgentId) -> int:
    """Successful (paid) connects by `agent` so far (mutual ones included)."""
    return sum(1 for h in state.history if h.kind == "connect" and h.actor == agent)


def is_mutual(state: GameState, actor: AgentId, target: AgentId) -> bool:
    """Did `target` connect to `actor` in this round at a slot the actor could not have seen
    (>= the actor's current observation)? Then the actor's own connect is the mutual half."""
    seen = state.agents[actor].last_obs_turn
    return any(h.kind == "connect" and h.actor == target and h.target == actor and h.round == state.round
               and h.turn_index >= seen for h in state.history)


def connect_error(state: GameState, actor: AgentId, target: AgentId, cfg: GameCfg,
                  departed: bool = False) -> str | None:
    """Reason connect(target) by `actor` fails, or None (module docstring; `departed` = target left)."""
    if not cfg.network.on or state.agents[actor].contacts is None:
        return NOT_NETWORK
    if not cfg.network.connect.enabled:
        return CONNECT_DISABLED
    if target not in state.agents:
        return "unknown_agent"
    if target == actor:
        return "self_target"
    if departed:
        return "target_left"
    s = state.agents[actor]
    if s.known is not None and target not in s.known:
        return UNKNOWN_CONTACT
    if s.heard is not None and target not in s.heard:      # connect.require_known (showcase)
        return UNKNOWN_CONTACT
    cap = cfg.network.connect.max_new_edges_per_agent
    if cap is not None and connects_made(state, actor) >= cap:
        return EDGE_CAP
    if target in (s.contacts or []) and not is_mutual(state, actor, target):
        return ALREADY_CONNECTED
    return None


def add_edge(state: GameState, a: AgentId, b: AgentId) -> bool:
    """Add the undirected edge a-b to the live graph (contacts stay sorted). False if it existed."""
    sa, sb = state.agents[a], state.agents[b]
    if b in (sa.contacts or []):
        return False
    sa.contacts = sorted({*(sa.contacts or []), b})
    sb.contacts = sorted({*(sb.contacts or []), a})
    for x, y in ((sa, b), (sb, a)):                          # connect.require_known: a contact is heard of
        if x.heard is not None and y not in x.heard:
            x.heard = sorted({*x.heard, y})
    return True


def connected_text(actor: AgentId) -> str:
    """The target's notice (kind "connected")."""
    return f"{actor} connected with you; you can now trade and message directly."


def connect_fail_text(target: AgentId, reason: str, cfg: GameCfg) -> str:
    """The actor's notice for a refused connect (nothing changed, no turns lost)."""
    why = {
        ALREADY_CONNECTED: f"you are already connected with {target}",
        "unknown_agent": f"there is no player called {target}",
        "self_target": "you cannot connect with yourself",
        "target_left": f"{target} has won and left the game",
        UNKNOWN_CONTACT: (f"you have not heard of {target}; you can only connect with players named in a "
                          "message, board post or notice you received" if cfg.network.connect.require_known
                          else f"you don't know {target}; you can only connect with players you know"),
        EDGE_CAP: (f"you have already made {cfg.network.connect.max_new_edges_per_agent} new connection(s), "
                   "the most allowed"),
        NOT_NETWORK: "there is no network in this game",
        CONNECT_DISABLED: "connecting is not allowed in this game",
    }.get(reason, reason)
    return f"Your connect with {target} did not happen: {why}. You lose no turns."


def connect_cost_texts(actor: AgentId, target: AgentId, rnd: int, k: int, mutual: bool) -> tuple[str, str | None]:
    """(notice now, resume notice or None when k = 0) for the actor of a successful connect."""
    head = (f"You and {target} both chose to connect in round {rnd}: you are now connected (one link; "
            "you both pay)." if mutual else
            f"You connected with {target} in round {rnd}: you can now trade and message directly.")
    if k <= 0:
        return head, None
    return (f"{head} Connecting costs you your next {k} turns: you skip rounds {rnd + 1} to {rnd + k}. "
            "You can still receive keys and messages.",
            f"Your {k} skipped turns (for connecting with {target} in round {rnd}) are over. You can act again.")
