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
God (adversary.powers): in contacts mode a forged DM needs sender -> receiver to be an edge (fails
"not_connected" otherwise, nothing spent). known_ids puts no limit on God; the receiver learns the
shown sender's name like any DM sender's.
Pure (no I/O); `attach` and `learn` mutate the state they are given. Imports: config, rng, schema.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence

from swarm.config import GameCfg, NetworkCfg
from swarm.rng import make_rng
from swarm.schema import AgentId, AgentMessageView, GameState, NoticeView, Observation

Edges = set[tuple[int, int]]
MAX_RESAMPLES = 1000
NOT_CONNECTED = "not_connected"
UNKNOWN_CONTACT = "unknown_contact"
REASONS = (NOT_CONNECTED, UNKNOWN_CONTACT)
LEARN_NOTICE_KINDS = ("gift_received", "burned", "defect_attempt")
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


def _connect(n: int, edges: Edges) -> Edges:
    """Chain the components (smallest node of each to the next one's): a last-resort fallback."""
    comps = components(n, edges)
    return edges | {_edge(a[0], b[0]) for a, b in zip(comps, comps[1:], strict=False)}


def graph_edges(cfg: NetworkCfg, n: int, rng: random.Random) -> Edges:
    """Edges on positions 0..n-1 for cfg.topology (see the module docstring)."""
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


def make_graph(cfg: NetworkCfg, agents: Sequence[AgentId], seed: int) -> dict[AgentId, list[AgentId]]:
    """{agent: sorted neighbours}. Deterministic in (cfg, set of agents, seed)."""
    rng = make_rng(seed, "network")
    pos = sorted(agents)
    rng.shuffle(pos)                                  # position -> agent
    adj: dict[AgentId, set[AgentId]] = {a: set() for a in pos}
    for i, j in graph_edges(cfg, len(pos), rng):
        adj[pos[i]].add(pos[j])
        adj[pos[j]].add(pos[i])
    return {a: sorted(adj[a]) for a in sorted(adj)}


def attach(state: GameState, cfg: GameCfg, seed: int) -> None:
    """Store the graph on the state (no-op when the network is off). Call once, at world generation."""
    if not cfg.network.on:
        return
    for a, nbrs in make_graph(cfg.network, list(state.agents), seed).items():
        s = state.agents[a]
        s.contacts = list(nbrs)
        if cfg.network.mode == "known_ids":
            s.known = list(nbrs)


def edge_list(state: GameState) -> list[tuple[AgentId, AgentId]]:
    """Sorted (a, b) pairs with a < b (empty when the network is off)."""
    return sorted({tuple(sorted((a, b))) for a, s in state.agents.items() for b in (s.contacts or [])})


def network_meta(state: GameState, cfg: GameCfg) -> dict[str, object]:
    """The graph for a game's meta.json (topology, mode, edges, degrees); {} when the network is off."""
    if not cfg.network.on:
        return {}
    return {"topology": cfg.network.topology, "mode": cfg.network.mode,
            "edges": [list(e) for e in edge_list(state)],
            "degree": {a: len(s.contacts or []) for a, s in sorted(state.agents.items())}}


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
               notices: Sequence[NoticeView], names: Iterable[AgentId]) -> set[AgentId]:
    """Names `agent` learns from one observation's inbox, board and notices (module docstring)."""
    names = list(names)
    real = set(names)
    out: set[AgentId] = set()
    for m in [*inbox, *board]:
        if m.sender in real:
            out.add(m.sender)
        out |= names_in_text(m.text, names)
    for nt in notices:
        if nt.kind in LEARN_NOTICE_KINDS:
            out |= names_in_text(nt.text, names)
    out.discard(agent)
    return out


def learn(state: GameState, agent: AgentId, obs: Observation) -> list[AgentId]:
    """known_ids: add the names seen in `obs` to the agent's known set; returns the new names (sorted)."""
    s = state.agents[agent]
    if s.known is None:
        return []
    new = names_seen(agent, obs.inbox, obs.board, obs.notices, state.agents) - set(s.known)
    if new:
        s.known = sorted(set(s.known) | new)
    return sorted(new)
