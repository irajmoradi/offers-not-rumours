"""Game-theory audit of Keys & God as the rules stand now (plan/GAME_THEORY_AUDIT.md). No LLM, no I/O
beyond writing the outputs.

    PYTHONNOUSERSITE=1 $PY -m swarm.analysis.game_theory_audit \
        [--worlds 30] [--procs 16] [--out plan/figs]

Four parts, all on the REAL world generator (swarm.game.world) and the REAL graph builder
(swarm.game.network, attached by generate_world), with world seeds from swarm.runner.spec.world_seed,
so game_idx 0-5 of configs/net40_* are exactly the worlds and graphs the live runs play (checked
against runs/net40_er3_contacts_sim/games/*/meta.json):
  1. world structure: copies and demand per info, contested infos, bilateral swap partners;
  2. network geometry: hop distance from each need to the nearest holder of a copy, and to the copy
     an optimal per-info assignment would use; blind-defect accuracy;
  3. defect EV under defect_gain=mint / wrong_defect_penalty=lose_need (closed form, evaluated);
  4. a scripted-agent simulation: state transitions go through swarm.game.resolve.resolve_turn
     (rules.apply), simultaneous turns with a reshuffled priority order, like configs/net40_*.
     Information is an explicit model (every agent posts its open needs and keys truthfully each
     turn; see INFO_MODES), not the LLM observation pipeline.

Policies (POLICIES): C cooperator (gives a non-needed key to a believed needer it can address;
relays toward far needers in contacts mode), Cn cooperator that never relays, H hoarder (never
gives), D opportunist (correct defect whenever it believes an addressable non-winner shares one of
its own unheld needs, else C), B blind defector (one guessed defect per turn on its own unheld
need, else C), S swapper (gives only to an agent believed to hold one of its unheld needs, or that
gave to it before), K connector (C + a hypothetical connect(target) to the nearest believed holder
of a missing need; costs CONNECT_COST rounds; contacts mode only, models branch connect-edges).
L (game-fixes) is C plus one blind defect in the final round (the "free lottery"). The Sim also
follows the game-fixes switches (CONTRACTS §9f) when a config turns them on: lock_and_stay winners
give any key and need nothing; the result adds private (held-key bonus) and shared (team score)
welfare. With every switch off its outputs are unchanged. swarm.analysis.game_fixes_sim drives the
before/after runs.

Outputs: <out>/game_theory_audit.json, <out>/gt_audit_distance.png, <out>/gt_audit_sim.png, and a
text summary on stdout. Deterministic in (--worlds, --seed-base).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, deque
from collections.abc import Iterable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from swarm.config import GameCfg, RunConfig, from_dict, load_raw
from swarm.game import resolve, rules, scoring
from swarm.game.world import generate_world
from swarm.runner.spec import world_seed
from swarm.schema import AgentId, DefectAction, GameState, GiveKey, Noop

CFG_DIR = Path("configs")
INFO_MODES = {
    # name: (hop horizon for knowing another agent's needs/keys, lag in rounds per hop)
    "oracle": (math.inf, 0),     # everyone knows the true current state (upper bound)
    "board": (1, 1),             # truthful posts every turn, visible to readers one round later
    "gossip2": (2, 1),           # + every agent forwards its neighbours' posts once (2 hops, lag = d)
    "gossipAll": (math.inf, 1),  # perfect flooding: anyone's post reaches distance d after d rounds
}
POLICIES = ("C", "Cn", "H", "D", "B", "S", "K", "L")   # L (game-fixes): C + one blind defect in the last round
CONNECT_COST = 2


# ------------------------------------------------------------------ configs and worlds
def load_cfg(name: str, overrides: Iterable[str] = ()) -> RunConfig:
    """configs/<name>.yaml with the runner block stripped (it is not part of RunConfig)."""
    raw = load_raw(CFG_DIR / f"{name}.yaml")
    raw.pop("runner", None)
    return from_dict(raw, list(overrides))


def world(cfg: RunConfig, game_idx: int) -> GameState:
    """The world (and graph) game `game_idx` of this config plays."""
    return generate_world(cfg.game, world_seed(cfg, game_idx))


def holders(state: GameState) -> dict[int, list[AgentId]]:
    """info -> agents holding a copy (one entry per copy)."""
    out: dict[int, list[AgentId]] = {}
    for a, s in state.agents.items():
        for kc in s.keys:
            out.setdefault(kc.info_id, []).append(a)
    return out


def needers(state: GameState) -> dict[int, list[AgentId]]:
    out: dict[int, list[AgentId]] = {}
    for a, s in state.agents.items():
        for i in s.needs:
            out.setdefault(i, []).append(a)
    return out


def world_structure(state: GameState) -> dict[str, Any]:
    """Supply/demand per info, contested infos, swap partners (part 1)."""
    hold, need = holders(state), needers(state)
    infos = sorted(set(hold) | set(need))
    copies = Counter(len(hold.get(i, [])) for i in infos)
    demand = Counter(len(need.get(i, [])) for i in infos)
    slack = Counter(len(hold.get(i, [])) - len(need.get(i, [])) for i in infos)
    n_need = sum(len(v) for v in need.values())
    # a need (a, X) has a bilateral swap partner if some holder h of X needs a key a holds
    swap = 0
    for a, s in state.agents.items():
        mine = {kc.info_id for kc in s.keys}
        for x in s.needs:
            if any(set(state.agents[h].needs) & mine for h in hold.get(x, [])):
                swap += 1
    contested_needs = sum(len(need[i]) for i in need if len(need[i]) == len(hold.get(i, [])))
    useful = sum(min(len(hold.get(i, [])), len(need.get(i, []))) for i in infos)
    return {"n_infos": len(infos), "copies_hist": dict(copies), "demand_hist": dict(demand),
            "slack_hist": dict(slack), "need_slots": n_need,
            "copies": sum(len(v) for v in hold.values()),
            "useful_copies": useful,
            "share_needs_contested": contested_needs / n_need,
            "share_needs_co_needed": sum(len(v) for v in need.values() if len(v) >= 2) / n_need,
            "share_needs_with_swap_partner": swap / n_need,
            "max_demand_minus_supply": max(len(need.get(i, [])) - len(hold.get(i, [])) for i in infos)}


# ------------------------------------------------------------------ graph helpers
def adjacency(state: GameState, active: set[AgentId] | None = None) -> dict[AgentId, list[AgentId]]:
    """Graph neighbours (complete graph when the network is off), restricted to `active`."""
    agents = list(state.agents)
    act = set(agents) if active is None else active
    out = {}
    for a in agents:
        c = state.agents[a].contacts
        nb = [b for b in agents if b != a] if c is None else c
        out[a] = [b for b in nb if b in act]
    return out


def bfs(adj: dict[AgentId, list[AgentId]], src: AgentId) -> tuple[dict[AgentId, int], dict[AgentId, AgentId]]:
    """Hop distances and parents from src."""
    dist, par = {src: 0}, {}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                par[v] = u
                q.append(v)
    return dist, par


def first_hop(par: dict[AgentId, AgentId], src: AgentId, dst: AgentId) -> AgentId:
    v = dst
    while par[v] != src:
        v = par[v]
    return v


def need_distances(state: GameState) -> dict[str, list[int]]:
    """Per need slot: hop distance to the nearest holder, and to the copy chosen by a per-info
    assignment minimising the total distance (infos have <= 2 copies and <= 2 needers)."""
    adj = adjacency(state)
    dist = {a: bfs(adj, a)[0] for a in state.agents}
    hold, need = holders(state), needers(state)
    nearest, matched = [], []
    for i, ns in need.items():
        hs = hold.get(i, [])
        for a in ns:
            nearest.append(min(dist[a][h] for h in hs))
        if len(ns) == 1:
            matched.append(min(dist[ns[0]][h] for h in hs))
        else:                                     # 2 needers, 2 copies: the cheaper of the 2 pairings
            a, b = ns
            h1, h2 = hs
            p = (dist[a][h1] + dist[b][h2], [dist[a][h1], dist[b][h2]])
            q = (dist[a][h2] + dist[b][h1], [dist[a][h2], dist[b][h1]])
            matched += min(p, q)[1]
    degs = [len(adj[a]) for a in adj]
    return {"nearest": nearest, "matched": matched, "degree": degs}


def blind_accuracy(state: GameState) -> dict[str, float]:
    """P(correct) of a blind defect naming one's own unheld need X: on a uniformly random addressable
    agent (contacts) / any agent; and P(the co-needer of X is a neighbour)."""
    need = needers(state)
    adj = adjacency(state)
    p_nb, p_any, co_nb = [], [], []
    n = len(state.agents)
    for a, s in state.agents.items():
        for x in s.needs:
            others = [b for b in need[x] if b != a]
            p_any.append(len(others) / (n - 1))
            nb = adj[a]
            p_nb.append(sum(1 for b in others if b in nb) / max(len(nb), 1))
            co_nb.append(1.0 if any(b in nb for b in others) else 0.0)
    return {"p_blind_any": statistics.mean(p_any), "p_blind_neighbour": statistics.mean(p_nb),
            "p_co_needer_is_neighbour": statistics.mean(co_nb)}


# ------------------------------------------------------------------ defect EV (closed form)
def defect_ev(p: float, q_gain: float, h_lost_held: float, q_lost: float) -> float:
    """EV in need units of defect(t, X) with X the actor's own unheld open need, defect_gain=mint,
    wrong_defect_penalty=lose_need (the actor's FIRST open need, held or not, is lost):
      p           P(t really has X as an open need and has not won)
      q_gain      P(the actor would have got X anyway by game end)
      h_lost_held P(the first open need is currently held)
      q_lost      P(an unheld first open need would have been filled anyway)"""
    return p * (1 - q_gain) - (1 - p) * (h_lost_held + (1 - h_lost_held) * q_lost)


def defect_table() -> list[dict[str, Any]]:
    rows = []
    for label, q, h, ql in [
        ("cooperative world, early (q=1, first need held 1/3)", 1.0, 1 / 3, 1.0),
        ("LLM-like world (q=0.63, GAME_THEORY.md B3)", 0.63, 1 / 3, 0.63),
        ("far need in ER3 contacts, no gossip (q=0.1)", 0.1, 1 / 3, 0.1),
        ("last round, nothing held (q=0, h=0)", 0.0, 0.0, 0.0),
        ("last round, first need held (q=0, h=1)", 0.0, 1.0, 0.0),
    ]:
        for p in (0.02, 0.5, 0.8, 1.0):
            rows.append({"case": label, "p": p, "ev": round(defect_ev(p, q, h, ql), 3)})
    return rows


def break_even_p(q: float, h: float, ql: float) -> float:
    loss = h + (1 - h) * ql
    gain = 1 - q
    return 1.0 if gain + loss == 0 else loss / (gain + loss)


# ------------------------------------------------------------------ simulation
class Sim:
    """One scripted game. Transitions via resolve.resolve_turn (the real rules); information via
    INFO_MODES; policies per agent."""

    def __init__(self, cfg: RunConfig, game_idx: int, types: dict[AgentId, str] | str, info: str,
                 order_seed: int, connect_cost: int = CONNECT_COST, blind_rate: float = 1.0) -> None:
        self.g: GameCfg = cfg.game
        self.state = world(cfg, game_idx)
        self.start_infos = scoring.start_infos(self.state)   # game.score.held_key_scope initial
        self.agents = sorted(self.state.agents)
        self.types = types if isinstance(types, dict) else {a: types for a in self.agents}
        self.horizon, self.lag = INFO_MODES[info]
        self.rng = random.Random(order_seed)
        self.net_mode = (None if self.state.agents[self.agents[0]].contacts is None else self.g.network.mode)
        self.snaps: list[dict[AgentId, tuple[frozenset, frozenset, bool]]] = []
        self.gave_to: dict[AgentId, set[AgentId]] = {a: set() for a in self.agents}
        self.connect_cost = connect_cost
        self.blind_rate = blind_rate
        self.connected: set[AgentId] = set()
        self.stats = Counter()
        self.welfare_by_round: list[float] = []
        self.first_done: int | None = None

    # ---------------------------------------------------------- truth helpers
    def departed(self, a: AgentId) -> bool:
        return rules.departed(self.state, a, self.g)

    def active(self, a: AgentId, r: int) -> bool:
        return not self.departed(a) and r > self.state.agents[a].delayed_until_round

    def snapshot(self) -> dict[AgentId, tuple[frozenset, frozenset, bool]]:
        st = self.state
        lock = rules.score_locked(self.g)                 # game-fixes: a locked winner needs nothing
        return {a: (frozenset() if (lock and st.agents[a].won_round is not None) else frozenset(rules.open_needs(st, a)),
                    frozenset(rules.held_infos(st, a)), st.agents[a].won_round is not None) for a in self.agents}

    def addressable(self, a: AgentId) -> set[AgentId]:
        s = self.state.agents[a]
        if s.known is not None:
            base = set(s.known)
        elif s.contacts is not None:
            base = set(s.contacts)
        else:
            base = set(self.agents) - {a}
        return {b for b in base if not self.departed(b)}

    def comm_adj(self, live: set[AgentId]) -> dict[AgentId, list[AgentId]]:
        """Who can pass information to whom: graph edges; in known_ids mode also every known pair
        (a known name can be DMed directly, and the receiver learns the sender), symmetrised."""
        adj = {a: set(v) for a, v in adjacency(self.state, live).items()}
        if self.net_mode == "known_ids":
            for a in live:
                for b in self.state.agents[a].known or []:
                    if b in live:
                        adj[a].add(b)
                        adj[b].add(a)
        return {a: sorted(v) for a, v in adj.items()}

    # ---------------------------------------------------------- beliefs
    def beliefs(self, a: AgentId, r: int, dist: dict[AgentId, int]) -> dict[AgentId, tuple[frozenset, frozenset, bool]]:
        """What `a` believes about others at round r: (open needs, held infos, won) from the
        snapshot taken at the start of round r - lag*d (unknown if that is before round 1)."""
        out = {}
        for b, d in dist.items():
            if b == a or d > self.horizon or self.departed(b):
                continue
            t = r - self.lag * d
            if t < 1:
                continue
            out[b] = self.snaps[t - 1][b]
        return out

    # ---------------------------------------------------------- policy
    def decide(self, a: AgentId, r: int) -> list:
        st, typ = self.state, self.types[a]
        if typ == "H":
            return [Noop()]
        live = {b for b in self.agents if not self.departed(b)}
        adj = adjacency(st, live)
        dist, par = bfs(adj, a)                 # graph distance: relays (contacts mode)
        bel = self.beliefs(a, r, bfs(self.comm_adj(live), a)[0])
        if self.net_mode == "known_ids":        # names learned from forwarded posts are addressable
            s = st.agents[a]
            s.known = sorted(set(s.known or []) | set(bel))
        addr = self.addressable(a)
        mine_open = set(rules.open_needs(st, a))
        held = Counter(kc.info_id for kc in st.agents[a].keys)
        unheld = sorted(x for x in mine_open if held[x] == 0)
        acts: list = []
        if typ == "K" and self.net_mode == "contacts" and unheld and a not in self.connected:
            tgt = self.connect_target(a, unheld, bel, addr, dist)
            if tgt is not None:                 # connect uses this turn and the next cost-1 rounds
                self.connected.add(a)
                st.agents[a].contacts = sorted(set(st.agents[a].contacts) | {tgt})
                st.agents[tgt].contacts = sorted(set(st.agents[tgt].contacts) | {a})
                st.agents[a].delayed_until_round = max(st.agents[a].delayed_until_round,
                                                       r + self.connect_cost - 1)
                self.stats["connects"] += 1
                return [Noop()]
        if typ == "D":
            for b in sorted(addr & set(bel)):
                o, _h, won = bel[b]
                hit = sorted(set(unheld) & o)
                if hit and not won:
                    acts.append(DefectAction(target=b, info_id=hit[0]))
                    self.stats["defects_D"] += 1
                    break
        if typ == "L" and unheld and addr and r == self.g.max_rounds:   # game-fixes: last-round lottery
            acts.append(DefectAction(target=self.rng.choice(sorted(addr)), info_id=unheld[0]))
            self.stats["defects_L"] += 1
        if typ == "B" and unheld and addr and self.rng.random() < self.blind_rate:
            acts.append(DefectAction(target=self.rng.choice(sorted(addr)), info_id=unheld[0]))
            self.stats["defects_B"] += 1
        locked = st.agents[a].won_round is not None and rules.score_locked(self.g)
        spare = Counter({x: c - (0 if locked else 1 if x in mine_open else 0) for x, c in held.items()})
        cands: list[tuple[int, float, AgentId, int, bool]] = []
        for b, (o, h, _won) in bel.items():
            for x in o - h:
                if spare[x] <= 0:
                    continue
                if b in addr:
                    if typ == "S" and not (h & set(unheld)) and b not in self.gave_to[a]:
                        continue
                    cands.append((1, self.rng.random(), b, x, False))
                elif typ in ("C", "D", "B", "K", "L") and self.net_mode == "contacts" and b in dist:
                    cands.append((dist[b], self.rng.random(), b, x, True))
        cands.sort()
        for d, _, b, x, relay in cands:
            if len(acts) >= self.g.actions_per_turn:
                break
            if spare[x] <= 0:
                continue
            to = first_hop(par, a, b) if relay else b
            acts.append(GiveKey(to=to, info_id=x))
            spare[x] -= 1
            self.stats["relay_gives" if relay else "direct_gives"] += 1
        return acts or [Noop()]

    def connect_target(self, a, unheld, bel, addr, dist) -> AgentId | None:
        best = None
        for b, (_o, h, _w) in bel.items():
            if b in addr:
                continue
            if set(unheld) & h:
                if best is None or dist[b] < dist[best]:
                    best = b
        if best is not None and dist[best] >= 2:
            return best
        return None

    # ---------------------------------------------------------- loop
    def play(self) -> dict[str, Any]:
        st, n = self.state, len(self.agents)
        for r in range(1, self.g.max_rounds + 1):
            st.round = r
            self.snaps.append(self.snapshot())
            order = list(self.agents)
            self.rng.shuffle(order)
            actors = [a for a in order if self.active(a, r)]
            chosen = {a: self.decide(a, r) for a in actors}      # simultaneous: all decide first
            for p, a in enumerate(order):
                if a not in chosen:
                    continue
                st.turn_index = (r - 1) * n + p
                res = resolve.resolve_turn(st, a, chosen[a], None, self.g)
                for act, (ok, reason) in zip(chosen[a], res.results, strict=True):
                    if isinstance(act, GiveKey) and ok:
                        self.gave_to[act.to].add(a)
                    if isinstance(act, DefectAction):
                        self.stats["defect_ok" if ok and act.info_id in st.agents[act.target].burned
                                   else "defect_other"] += 1
                    if not ok:
                        self.stats[f"fail_{reason}"] += 1
            self.welfare_by_round.append(statistics.mean(rules.score(st, a, self.g) for a in self.agents))
            if all(rules.is_done(st, a) for a in self.agents):
                self.first_done = r
                break
        return self.result()

    def unmet_classes(self) -> Counter:
        """Why each open, unheld need at the end is unmet."""
        st = self.state
        active = {b for b in self.agents if not self.departed(b)}
        adj = self.comm_adj(active)            # contacts: the graph; known_ids: graph + known pairs
        out = Counter()
        for a in self.agents:
            s = st.agents[a]
            out["burned"] += len(s.burned)
            out["lost"] += len(s.lost)
            locked = s.won_round is not None and rules.score_locked(self.g)   # game-fixes: score locked
            for x in rules.open_needs(st, a):
                if locked or x in rules.held_infos(st, a):
                    out["held"] += 1
                    continue
                hs = [b for b in self.agents if b != a and x in rules.held_infos(st, b)]
                live = [b for b in hs if b in active]
                if not live:
                    out["stranded_with_departed" if hs else "no_copy"] += 1
                elif a in active and not any(b in bfs(adj, a)[0] for b in live):
                    out["disconnected"] += 1
                else:
                    out["reachable_unmet"] += 1
        return out

    def result(self) -> dict[str, Any]:
        st = self.state
        sc = {a: rules.score(st, a, self.g) for a in self.agents}
        parts = scoring.score_parts(st, self.g, self.start_infos)   # game-fixes: = raw when game.score is off
        team = scoring.team_scores(st, self.g)
        return {"welfare": statistics.mean(sc.values()),
                "welfare_private": statistics.mean(p.private for p in parts.values()),
                "welfare_shared": statistics.mean(team.values()),
                "private_scores": {a: p.private for a, p in parts.items()}, "shared_scores": team,
                "win_rate": statistics.mean(1.0 if st.agents[a].won_round is not None else 0.0
                                            for a in self.agents),
                "rounds": self.first_done or self.g.max_rounds, "finished": self.first_done is not None,
                "scores": sc, "types": self.types, "curve": self.welfare_by_round,
                "unmet": dict(self.unmet_classes()), "stats": dict(self.stats)}


# ------------------------------------------------------------------ experiment grid
TOPOS = {"complete": "net40_complete_sim", "er3_contacts": "net40_er3_contacts_sim",
         "er3_known": "net40_er3_known_sim", "complete10": "grid_oss_sim_k2"}
WINNER = ("continue", "sit_out")


def _cfg(topo: str, winner: str, extra: tuple[str, ...] = ()) -> RunConfig:
    # all N=40 topologies at 30 rounds (the live A cell now runs 20; B/C run 30); N=10 keeps its 20
    rounds = 20 if topo == "complete10" else 30
    return load_cfg(TOPOS[topo], [f"game.winner_policy={winner}", f"game.max_rounds={rounds}", *extra])


def bridge_seat(state: GameState) -> int:
    """Index (in sorted agents) of the agent that is interior to the most needer -> nearest-holder
    shortest paths: the relay a God note-overwrite ("never give keys") hurts most."""
    adj = adjacency(state)
    hold = holders(state)
    count = Counter()
    for a, s in state.agents.items():
        dist, par = bfs(adj, a)
        for x in s.needs:
            h = min(hold[x], key=lambda b: (dist[b], b))
            v = h
            while par.get(v, a) != a:
                v = par[v]
                count[v] += 1
    agents = sorted(state.agents)
    return agents.index(max(agents, key=lambda b: (count[b], b)))


def run_one(job: tuple) -> dict[str, Any]:
    topo, winner, info, pop, gi, focal, extra = job
    cfg = _cfg(topo, winner, extra)
    st = world(cfg, gi)
    agents = sorted(st.agents)
    if focal is None and pop.startswith("C+H"):        # all C except one hoarding/blocked seat
        seat_i = bridge_seat(st) if pop == "C+Hbridge" else random.Random(gi).randrange(len(agents))
        types: dict[AgentId, str] | str = {a: ("H" if i == seat_i else "C") for i, a in enumerate(agents)}
    elif focal is None:
        types = pop
    else:
        seat = agents[focal]
        types = {a: (focal_type(pop) if a == seat else resident_type(pop)) for a in agents}
    sim = Sim(cfg, gi, types, info, order_seed=hash_job(job))
    res = sim.play()
    res.update({"topo": topo, "winner": winner, "info": info, "pop": pop, "world": gi, "focal": focal,
                "extra": list(extra)})
    if focal is not None:
        res["focal_score"] = res["scores"][agents[focal]]
        res["resident_score"] = statistics.mean(v for a, v in res["scores"].items() if a != agents[focal])
    del res["scores"], res["types"]
    return res


def focal_type(pop: str) -> str:
    return pop.split("_in_")[0]


def resident_type(pop: str) -> str:
    return pop.split("_in_")[1]


def hash_job(job: tuple) -> int:
    s = repr(job)
    h = 0
    for ch in s:
        h = (h * 1000003 + ord(ch)) % (1 << 61)
    return h


def jobs(n_worlds: int) -> list[tuple]:
    out = []
    homog = {"complete": ["oracle", "board"], "er3_contacts": ["board", "gossip2", "gossipAll"],
             "er3_known": ["board", "gossip2", "gossipAll"]}
    for topo, infos in homog.items():
        for winner in WINNER:
            for info in infos:
                for pop in ("C", "Cn", "D", "B", "S") + (("K",) if topo == "er3_contacts" else ()):
                    if pop == "Cn" and topo != "er3_contacts":
                        continue
                    for gi in range(n_worlds):
                        out.append((topo, winner, info, pop, gi, None, ()))
    # one blocked relay: the max-betweenness agent vs a random agent never gives (a God note-overwrite)
    for topo, info in (("er3_contacts", "gossipAll"), ("er3_contacts", "gossip2"), ("complete", "board")):
        for winner in WINNER:
            for pop in ("C+Hbridge", "C+Hrandom"):
                for gi in range(n_worlds):
                    out.append((topo, winner, info, pop, gi, None, ()))
    # N=10 complete graph (the GAME_THEORY.md setting, simultaneous, 20 rounds): re-derivation
    for winner in WINNER:
        for pop in ("C", "D", "B", "S"):
            for gi in range(16):
                out.append(("complete10", winner, "board", pop, gi, None, ()))
    # connect cost 3 (the other end of the 2-3 turn range)
    for winner in WINNER:
        for info in ("gossip2", "gossipAll"):
            for gi in range(n_worlds):
                out.append(("er3_contacts", winner, info, "K", gi, None, ("_connect3",)))
    # invasion: one focal agent among residents (4 seats per world, a third of the worlds)
    inv_cells = [("complete", "board"), ("er3_contacts", "gossipAll"), ("er3_known", "gossipAll")]
    types = ("C", "H", "D", "B", "S")
    for topo, info in inv_cells:
        for winner in WINNER:
            for res_t in types:
                for foc_t in types:
                    for gi in range(max(4, n_worlds // 3)):
                        for seat in (0, 10, 20, 30):
                            out.append((topo, winner, info, f"{foc_t}_in_{res_t}", gi, seat, ()))
    # relaying / connecting as an individual choice: one focal non-relayer or connector among C
    for winner in WINNER:
        for info in ("gossip2", "gossipAll"):
            for foc_t in ("Cn", "K", "C"):
                for gi in range(max(4, n_worlds // 3)):
                    for seat in (0, 10, 20, 30):
                        out.append(("er3_contacts", winner, info, f"{foc_t}_in_C", gi, seat, ()))
    return out


def run_job(job: tuple) -> dict[str, Any]:
    topo, winner, info, pop, gi, focal, extra = job
    if "_connect3" in extra:
        cfg = _cfg(topo, winner)
        sim = Sim(cfg, gi, pop, info, order_seed=hash_job(job), connect_cost=3)
        res = sim.play()
        res.update({"topo": topo, "winner": winner, "info": info, "pop": "K3", "world": gi, "focal": None,
                    "extra": []})
        del res["scores"], res["types"]
        return res
    return run_one(job)


# ------------------------------------------------------------------ aggregation
def mean_ci(xs: list[float], boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 95% percentile bootstrap CI (resampling worlds/games)."""
    if not xs:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(boot))
    return statistics.mean(xs), ms[int(0.025 * boot)], ms[int(0.975 * boot) - 1]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    homog: dict[str, Any] = {}
    inv: dict[str, Any] = {}
    for r in results:
        key = f"{r['topo']}|{r['winner']}|{r['info']}|{r['pop']}"
        (homog if r["focal"] is None else inv).setdefault(key, []).append(r)
    out_h = {}
    for k, rs in sorted(homog.items()):
        w = mean_ci([r["welfare"] for r in rs])
        unmet = Counter()
        stats = Counter()
        for r in rs:
            unmet.update(r["unmet"])
            stats.update(r["stats"])
        slots = sum(unmet.values())
        out_h[k] = {"n": len(rs), "welfare": w, "win_rate": statistics.mean(r["win_rate"] for r in rs),
                    "rounds": statistics.mean(r["rounds"] for r in rs),
                    "finished": statistics.mean(1.0 if r["finished"] else 0.0 for r in rs),
                    "need_outcomes": {c: v / slots for c, v in sorted(unmet.items())},
                    "per_game": {c: v / len(rs) for c, v in sorted(stats.items())},
                    "curve": [statistics.mean(r["curve"][t] if t < len(r["curve"]) else r["curve"][-1]
                                              for r in rs) for t in range(30)]}
    out_i = {}
    for k, rs in sorted(inv.items()):
        out_i[k] = {"n": len(rs), "focal": mean_ci([r["focal_score"] for r in rs]),
                    "resident": statistics.mean(r["resident_score"] for r in rs)}
    return {"homogeneous": out_h, "invasion": out_i}


# ------------------------------------------------------------------ figures
def fig_distance(dist_data: dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for lab, key, col in (("nearest copy", "nearest", "#4C72B0"), ("assigned copy", "matched", "#DD8452")):
        xs = dist_data["er3_" + key]
        hist = Counter(xs)
        ks = sorted(hist)
        tot = len(xs)
        off = -0.2 if key == "nearest" else 0.2
        ax.bar([k + off for k in ks], [hist[k] / tot for k in ks], width=0.4, label=lab, color=col)
    ax.set_xlabel("hops from needer to key holder")
    ax.set_ylabel("share of need slots")
    ax.set_title("ER mean degree 3, N=40 (net40 worlds + extra seeds)")
    ax.legend(frameon=False)
    ax = axes[1]
    labels = ["d=1 (direct)", "d=2", "d>=3"]
    near = dist_data["er3_nearest"]
    mt = dist_data["er3_matched"]
    vals_n = [sum(1 for d in near if d == 1) / len(near), sum(1 for d in near if d == 2) / len(near),
              sum(1 for d in near if d >= 3) / len(near)]
    vals_m = [sum(1 for d in mt if d == 1) / len(mt), sum(1 for d in mt if d == 2) / len(mt),
              sum(1 for d in mt if d >= 3) / len(mt)]
    x = range(3)
    ax.bar([i - 0.2 for i in x], vals_n, width=0.4, color="#4C72B0", label="nearest copy")
    ax.bar([i + 0.2 for i in x], vals_m, width=0.4, color="#DD8452", label="assigned copy")
    for i, v in enumerate(vals_n):
        ax.text(i - 0.2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    for i, v in enumerate(vals_m):
        ax.text(i + 0.2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(list(x), labels)
    ax.set_title("Only needs at d=1 are solvable without relays/gossip")
    ax.set_ylim(0, max(vals_n + vals_m) + 0.1)
    fig.tight_layout()
    fig.savefig(out / "gt_audit_distance.png", dpi=150)
    plt.close(fig)


def fig_sim(summary: dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = summary["homogeneous"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    lines = [("complete|{w}|board|C", "complete, global board", "#222222", "-"),
             ("er3_contacts|{w}|board|C", "ER3 contacts, neighbour board", "#C44E52", "-"),
             ("er3_contacts|{w}|gossip2|C", "ER3 contacts, 2-hop gossip", "#C44E52", "--"),
             ("er3_contacts|{w}|gossipAll|C", "ER3 contacts, full flooding", "#C44E52", ":"),
             ("er3_known|{w}|board|C", "ER3 known_ids, neighbour board", "#4C72B0", "-"),
             ("er3_known|{w}|gossipAll|C", "ER3 known_ids, full flooding", "#4C72B0", ":")]
    for ax, w in zip(axes[:2], WINNER, strict=True):
        for key, lab, col, ls in lines:
            k = key.format(w=w)
            if k in h:
                ax.plot(range(1, 31), h[k]["curve"], color=col, ls=ls, label=lab)
        ax.set_title(f"all cooperators, winner_policy={w}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("welfare (mean score)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[2]
    cells = [("complete|continue|board|C", "complete\ncontinue"), ("complete|sit_out|board|C", "complete\nsit_out"),
             ("er3_contacts|continue|gossipAll|C", "ER3 contacts\nflood, continue"),
             ("er3_contacts|sit_out|gossipAll|C", "ER3 contacts\nflood, sit_out"),
             ("er3_known|sit_out|gossipAll|C", "ER3 known\nflood, sit_out")]
    cats = [("held", "#55A868"), ("reachable_unmet", "#CCB974"), ("stranded_with_departed", "#C44E52"),
            ("disconnected", "#8172B2"), ("burned", "#937860"), ("lost", "#DA8BC3")]
    bottom = [0.0] * len(cells)
    for c, col in cats:
        vals = [h[k]["need_outcomes"].get(c, 0.0) if k in h else 0 for k, _ in cells]
        ax.bar(range(len(cells)), vals, bottom=bottom, color=col, label=c)
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    ax.set_xticks(range(len(cells)), [lab for _, lab in cells], fontsize=7)
    ax.set_title("where need slots end (all cooperators)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "gt_audit_sim.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def geometry(n_worlds: int) -> dict[str, Any]:
    """Parts 1-2 on the net40 worlds (game_idx 0-5 = the live runs) plus extra seeds."""
    cfg_er = load_cfg(TOPOS["er3_contacts"])
    cfg_n10 = load_cfg("grid_oss_sim_k2") if (CFG_DIR / "grid_oss_sim_k2.yaml").exists() else None
    out: dict[str, Any] = {"er3_nearest": [], "er3_matched": [], "er3_degree": [], "per_world": []}
    struct = []
    blind = []
    for gi in range(n_worlds):
        st = world(cfg_er, gi)
        struct.append(world_structure(st))
        d = need_distances(st)
        blind.append(blind_accuracy(st))
        out["er3_nearest"] += d["nearest"]
        out["er3_matched"] += d["matched"]
        out["er3_degree"] += d["degree"]
        if gi < 6:
            out["per_world"].append({"game_idx": gi, "mean_degree": statistics.mean(d["degree"]),
                                     "share_nearest_ge2": sum(x >= 2 for x in d["nearest"]) / len(d["nearest"]),
                                     "share_nearest_ge3": sum(x >= 3 for x in d["nearest"]) / len(d["nearest"]),
                                     "mean_nearest": statistics.mean(d["nearest"]),
                                     "mean_matched": statistics.mean(d["matched"])})
    out["structure_n40"] = {k: statistics.mean(s[k] for s in struct) for k in struct[0]
                            if isinstance(struct[0][k], (int, float))}
    out["structure_n40"]["max_demand_minus_supply_any_world"] = max(s["max_demand_minus_supply"] for s in struct)
    out["structure_n40"]["demand_hist_total"] = dict(sum((Counter(s["demand_hist"]) for s in struct), Counter()))
    out["blind_n40_er3"] = {k: statistics.mean(b[k] for b in blind) for k in blind[0]}
    if cfg_n10 is not None:
        s10 = [world_structure(world(cfg_n10, gi)) for gi in range(16)]
        out["structure_n10"] = {k: statistics.mean(s[k] for s in s10) for k in s10[0]
                                if isinstance(s10[0][k], (int, float))}
    near, mt = out["er3_nearest"], out["er3_matched"]
    out["er3_summary"] = {
        "mean_degree": statistics.mean(out["er3_degree"]),
        "share_isolated_degree1": sum(1 for x in out["er3_degree"] if x == 1) / len(out["er3_degree"]),
        "mean_nearest": statistics.mean(near), "mean_matched": statistics.mean(mt),
        "share_nearest_ge2": sum(x >= 2 for x in near) / len(near),
        "share_nearest_ge3": sum(x >= 3 for x in near) / len(near),
        "share_matched_ge2": sum(x >= 2 for x in mt) / len(mt),
        "share_matched_ge3": sum(x >= 3 for x in mt) / len(mt),
        "agents_all3_needs_far": None}
    # agents with no need at d=1
    far_agents = []
    for gi in range(n_worlds):
        st = world(cfg_er, gi)
        adj = adjacency(st)
        hold = holders(st)
        for a, s in st.agents.items():
            nb = set(adj[a])
            far_agents.append(all(not (set(hold[x]) & nb) for x in s.needs))
    out["er3_summary"]["agents_all3_needs_far"] = statistics.mean(1.0 if f else 0.0 for f in far_agents)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="plan/figs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = geometry(args.worlds)
    js = jobs(args.worlds)
    with Pool(args.procs) as pool:
        results = pool.map(run_job, js, chunksize=4)
    summ = summarize(results)
    defect = {"table": defect_table(),
              "break_even": {"q=1,h=1/3": break_even_p(1, 1 / 3, 1), "q=0.63,h=1/3": break_even_p(0.63, 1 / 3, 0.63),
                             "q=0.1,h=1/3": break_even_p(0.1, 1 / 3, 0.1), "q=0,h=0": break_even_p(0, 0, 0)}}
    report = {"geometry": {k: v for k, v in geo.items() if k not in ("er3_nearest", "er3_matched", "er3_degree")},
              "defect_ev": defect, "sim": summ, "n_jobs": len(js), "worlds": args.worlds}
    (out / "game_theory_audit.json").write_text(json.dumps(report, indent=1, default=float))
    fig_distance(geo, out)
    fig_sim(summ, out)
    print(json.dumps(report["geometry"], indent=1, default=float))
    print(json.dumps(defect["break_even"], indent=1))
    for k, v in summ["homogeneous"].items():
        no = {c: round(x, 3) for c, x in v["need_outcomes"].items()}
        print(f"{k:42s} n={v['n']:3d} W={v['welfare'][0]:.3f} [{v['welfare'][1]:.3f},{v['welfare'][2]:.3f}] "
              f"win={v['win_rate']:.2f} rounds={v['rounds']:.1f} fin={v['finished']:.2f} {no} "
              f"{ {c: round(x, 1) for c, x in v['per_game'].items()} }")
    for k, v in summ["invasion"].items():
        print(f"INV {k:40s} focal={v['focal'][0]:.3f} [{v['focal'][1]:.3f},{v['focal'][2]:.3f}] res={v['resident']:.3f}")


if __name__ == "__main__":
    main()
