"""Score terms of game.score (game-fixes, CONTRACTS §9f). Pure: state + config in, numbers out.

raw       = rules.score(state, agent, cfg): held open needs / needs (locked at the win under
            lock_and_stay). This is what round_end, wins and Outcome.welfare report, unchanged; the
            terms below are reported NEXT to it, never inside welfare.
key_bonus = game.score.held_key_bonus * (key copies the agent holds at game end), for every agent
            (winners included; not locked).  private = raw + key_bonus - penalty.   (`score_parts`)
penalty   = score points taken by wrong defects (wrong_defect_penalty score_penalty / lose_held_key;
            rules.penalty). Only in private (and the agent's shown score), never in raw / welfare.
shared    = raw + game.score.team_share * mean(raw of ALL agents, itself included): the global team
            score.   (`team_scores`)
community = game.score.community_bonus B if at least community_threshold of the agent's planted
            community (game.network sbm / community_hubs, itself included) has won at game end, else
            0.   (`community_scores`; Outcome.welfare_community = mean(raw + community))
delivery  = game.score.delivery_credit phi: DeliveryLedger (fed every key_transfer event by the engine;
            replayed by the invariant checker): a copy that fills an open need pays phi to each DISTINCT
            earlier holder, most recent first, at most d of them (d = contact-graph hops, at fill
            time, from the copy's first holder to the needer); once per (needer, need).
`scoring_on` / `team_on` / `community_on` / `delivery_on` are False at the defaults; the engine then writes the
Outcome exactly as before.
Imports: config, schema, game.rules, game.network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swarm.config import GameCfg
from swarm.schema import AgentId, GameState

from . import rules
from .network import make_graph_full


@dataclass(frozen=True)
class ScoreParts:
    """One agent's held-key decomposition: raw needs score, bonus, raw + bonus."""
    raw: float
    key_bonus: float
    private: float


def penalty_on(cfg: GameCfg) -> bool:
    """Can a wrong defect take score points? (wrong_defect_penalty score_penalty / lose_held_key)"""
    return cfg.wrong_defect_penalty in ("score_penalty", "lose_held_key")


def scoring_on(cfg: GameCfg) -> bool:
    """Does the Outcome report key_bonus / private? (held-key bonus on, or a score-taking penalty)"""
    return cfg.score.held_key_bonus > 0 or penalty_on(cfg)


def start_infos(state: GameState) -> dict[AgentId, set[int]]:
    """agent -> the info ids of the keys it holds now (call at game start: held_key_scope initial)."""
    return {a: {kc.info_id for kc in s.keys} for a, s in state.agents.items()}


def bonus_keys(state: GameState, a: AgentId, cfg: GameCfg, initial: dict[AgentId, set[int]] | None) -> int:
    """Keys that count for held_key_bonus: every copy (scope all), or the starting infos still held
    (scope initial; `initial` = start_infos at game start, required)."""
    if cfg.score.held_key_scope == "initial":
        if initial is None:
            raise ValueError("held_key_scope initial needs the agents' starting infos")
        return len({kc.info_id for kc in state.agents[a].keys} & initial[a])
    return len(state.agents[a].keys)


def score_parts(state: GameState, cfg: GameCfg,
                initial: dict[AgentId, set[int]] | None = None) -> dict[AgentId, ScoreParts]:
    """agent -> ScoreParts for the current state (sorted by agent id). `initial`: start_infos at game
    start (needed for game.score.held_key_scope initial)."""
    b = cfg.score.held_key_bonus
    out = {}
    for a in sorted(state.agents):
        raw = rules.score(state, a, cfg)
        bonus = b * bonus_keys(state, a, cfg, initial) if b else 0.0
        out[a] = ScoreParts(raw=raw, key_bonus=bonus, private=raw + bonus - rules.penalty(state, a))
    return out


def welfare(parts: dict[AgentId, ScoreParts]) -> dict[str, float]:
    """Mean raw / private score."""
    n = max(len(parts), 1)
    return {"raw": sum(p.raw for p in parts.values()) / n, "private": sum(p.private for p in parts.values()) / n}


def team_on(cfg: GameCfg) -> bool:
    """Is the global team score on? (game.score.team_share > 0)"""
    return cfg.score.team_share > 0


def team_scores(state: GameState, cfg: GameCfg) -> dict[AgentId, float]:
    """agent -> shared = raw needs score + team_share * mean raw needs score of every agent."""
    raw = {a: rules.score(state, a, cfg) for a in sorted(state.agents)}
    mean = sum(raw.values()) / max(len(raw), 1)
    return {a: r + cfg.score.team_share * mean for a, r in raw.items()}


def community_on(cfg: GameCfg) -> bool:
    """Is the community threshold bonus on? (game.score.community_bonus > 0)"""
    return cfg.score.community_bonus > 0


def communities(state: GameState, cfg: GameCfg, seed: int) -> dict[AgentId, int]:
    """agent -> planted community (the world seed's graph; connect never changes it)."""
    comm = make_graph_full(cfg.network, list(state.agents), seed)[1]
    if comm is None:
        raise ValueError("game.score.community_bonus needs a community topology")
    return comm


def community_members(comm: dict[AgentId, int]) -> dict[int, list[AgentId]]:
    """community -> sorted members."""
    out: dict[int, list[AgentId]] = {}
    for a in sorted(comm):
        out.setdefault(comm[a], []).append(a)
    return out


def community_success(won: dict[AgentId, bool], held_live: dict[AgentId, int], cfg: GameCfg) -> dict[AgentId, bool]:
    """Does each agent count toward its community threshold? A win, or (community_success_needs k)
    >= k live needs held at game end."""
    k = cfg.score.community_success_needs
    return {a: bool(won[a] or (k is not None and held_live.get(a, 0) >= k)) for a in won}


def held_live_needs(state: GameState) -> dict[AgentId, int]:
    """agent -> number of its live (not burned / lost) needs it holds a key for."""
    out = {}
    for a, s in state.agents.items():
        live = set(s.needs) - set(s.burned) - set(s.lost)
        out[a] = len(live & {kc.info_id for kc in s.keys})
    return out


def community_scores(won: dict[AgentId, bool], comm: dict[AgentId, int], cfg: GameCfg) -> dict[AgentId, float]:
    """agent -> B if >= community_threshold of its community has won, else 0 (pure; won at game end)."""
    b, t, tc = cfg.score.community_bonus, cfg.score.community_threshold, cfg.score.community_threshold_count
    groups = community_members(comm)
    hit = {c: sum(won[a] for a in ms) >= (tc if tc is not None else t * len(ms) - 1e-9) for c, ms in groups.items()}
    return {a: (b if hit[comm[a]] else 0.0) for a in sorted(comm)}


def delivery_on(cfg: GameCfg) -> bool:
    """Is the delivery credit on? (game.score.delivery_credit > 0)"""
    return cfg.score.delivery_credit > 0


def graph_hops(adj: dict[AgentId, Any], src: AgentId, dst: AgentId) -> int | None:
    """BFS hops src -> dst over adj (agent -> neighbours), None if unreachable."""
    if src == dst:
        return 0
    seen, frontier, d = {src}, [src], 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in adj.get(u) or []:
                if v == dst:
                    return d
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return None


def credit_payees(holders: list[AgentId], needer: AgentId, d: int | None) -> list[AgentId]:
    """Distinct earlier holders of a copy (most recent first, the needer excluded), at most d."""
    out: list[AgentId] = []
    for h in reversed(holders):
        if h != needer and h not in out:
            out.append(h)
    return out[: max(d or 0, 0)]


class DeliveryLedger:
    """Delivery credits, driven ONLY by the event stream (engine.emit feeds every body; the invariant
    checker replays events.jsonl through the same class). start(agents) with the game_start agent
    states; then on_event(data, round) per event in emission order. Tracks each copy's holder chain,
    the live contact graph (edge_added), open needs (burn) and wins (win). A key_transfer fills a need
    when the receiver had the info as an open need, had not won, held no other copy of it, and that
    (receiver, info) was never filled before. Pure bookkeeping: never changes play."""

    def __init__(self, phi: float) -> None:
        self.phi = phi
        self.holders: dict[str, list[AgentId]] = {}       # copy -> holders in order (first = origin)
        self.where: dict[str, AgentId] = {}               # copy -> current holder
        self.info: dict[str, int] = {}                    # copy -> info id
        self.adj: dict[AgentId, set[AgentId]] = {}
        self.open: dict[AgentId, set[int]] = {}
        self.won: set[AgentId] = set()
        self.filled: set[tuple[AgentId, int]] = set()
        self.credit: dict[AgentId, float] = {}
        self.fills: list[dict[str, Any]] = []

    def start(self, agents: dict[AgentId, Any]) -> None:
        for a, s in agents.items():
            self.credit[a] = 0.0
            self.adj[a] = set(s.contacts or [])
            self.open[a] = set(s.needs) - set(s.burned) - set(s.lost)
            for kc in s.keys:
                self.holders[kc.copy_id] = [a]
                self.where[kc.copy_id] = a
                self.info[kc.copy_id] = kc.info_id

    def on_event(self, data: Any, rnd: int) -> None:
        t = getattr(data, "type", None)
        if t == "edge_added":
            self.adj.setdefault(data.actor, set()).add(data.target)
            self.adj.setdefault(data.target, set()).add(data.actor)
        elif t == "burn":
            if data.correct:
                self.open[data.target].discard(data.info_id)
            if data.lost_info_id is not None:
                self.open[data.actor].discard(data.lost_info_id)
        elif t == "win":
            self.won.add(data.agent)
        elif t == "key_transfer":
            self._transfer(data.copy_id, data.info_id, data.from_agent, data.to_agent, rnd)

    def _transfer(self, cid: str, info: int, frm: AgentId | None, to: AgentId, rnd: int) -> None:
        chain = self.holders.setdefault(cid, [frm] if frm is not None else [to])   # mint: origin = minter
        self.info[cid] = info
        holds_other = any(h == to and c != cid and self.info.get(c) == info for c, h in self.where.items())
        if (frm is not None and info in self.open.get(to, set()) and to not in self.won and not holds_other
                and (to, info) not in self.filled):
            self.filled.add((to, info))
            d = graph_hops(self.adj, chain[0], to)
            pay = credit_payees(chain, to, d)
            for h in pay:
                self.credit[h] = self.credit.get(h, 0.0) + self.phi
            self.fills.append({"round": rnd, "copy_id": cid, "info_id": info, "needer": to,
                               "first_holder": chain[0], "d": d, "holders": list(chain), "paid": pay})
        if chain[-1] != to:
            chain.append(to)
        self.where[cid] = to

    def total(self) -> float:
        return sum(self.credit.values())
