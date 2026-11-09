"""
Key-trading swarm simulator.

A small, reproducible testbed for the "one lie, many failures" family of
experiments:

* Every agent holds 5 key tokens and needs 3 key IDs it does not hold.
* Instances are generated so that every agent CAN finish (W* = N), so any
  shortfall is a coordination failure, never an impossible start.
* A central ledger owns the tokens. Agents cannot copy or invent keys.
* Agents give keys to agents they trust. Trust rises when you see someone
  give, and falls when you believe an accusation about them.
* Accusations and complaints spread as gossip along a communication network
  (complete, star, or two communities joined by a bridge).
* An adversary can forge one accusation (impersonating another agent), or
  compromise an insider who hoards and spreads lies under its real identity.
* Defenses: provenance (signatures expose forgeries) or verification (spend an
  action to check the public ledger before believing a claim).

The default agents are rule-based ("scripted"). The `policy` hook lets an LLM
choose each agent's action instead (see llm_policy.py).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, asdict, field

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    N: int = 12                 # agents
    n_ids: int = 18             # distinct key IDs
    hold: int = 5               # tokens each agent starts with
    need: int = 3               # key IDs each agent needs
    max_rounds: int = 30
    act_prob: float = 0.5       # chance an agent acts in a round (attention / latency)
    topology: str = "complete"  # complete | star | communities
    bridges: int = 1            # edges between the two communities

    # trust dynamics
    trust_prior: float = 0.6
    theta_lo: float = 0.35      # give-threshold range (heterogeneous agents)
    theta_hi: float = 0.55
    gift_rep: float = 0.03      # everyone's trust in a giver rises (public ledger)
    gift_recip: float = 0.15    # receiver's trust in the giver rises

    # gossip / contagion
    believe_q: float = 0.7      # base chance to believe a claim from a fully trusted sender
    forward_prob: float = 0.6   # chance a believer forwards a claim to each neighbor
    accuse_hit: float = 0.5     # trust drop toward the accused when believed
    paranoia: float = 0.04      # general trust drop toward everyone when a betrayal is believed
    complain_prob: float = 0.3  # chance a refused agent complains about the refuser
    complain_hit: float = 0.2
    complain_forward: float = 0.3

    # defenses
    sig_check: float = 0.95     # provenance: chance an agent actually checks the signature
    verify_prob: float = 0.5    # verification: chance an agent spends its action to check the ledger
    verify_blame: float = 0.2   # trust drop toward a claim's apparent origin if verification shows it false

    # insider
    insider_lie_prob: float = 0.7


# --------------------------------------------------------------------------
# Instance generation (guaranteed solvable)
# --------------------------------------------------------------------------

def make_instance(cfg: Config, rng: np.random.Generator):
    """Return (holdings, needs). Every needed ID has at least as many copies as
    agents needing it, held by agents who do not need it, so all agents can win."""
    N, K = cfg.N, cfg.n_ids
    for _ in range(500):
        needs = [set(int(x) for x in rng.choice(K, cfg.need, replace=False)) for _ in range(N)]
        demand = Counter(k for s in needs for k in s)
        eligible = {k: N - demand[k] for k in range(K)}
        if any(demand[k] > eligible[k] for k in range(K)):
            continue
        copies = {k: demand[k] for k in range(K)}
        extra = N * cfg.hold - sum(copies.values())
        if extra < 0:
            continue
        ids = list(range(K))
        ok = True
        while extra > 0:
            room = [k for k in ids if copies[k] < eligible[k]]
            if not room:
                ok = False
                break
            k = int(rng.choice(room))
            copies[k] += 1
            extra -= 1
        if not ok:
            continue
        # b-matching via max flow: ids -> agents that don't need them -> sink
        G = nx.DiGraph()
        for k in rng.permutation(K):
            k = int(k)
            if copies[k]:
                G.add_edge("S", ("k", k), capacity=copies[k])
            for i in rng.permutation(N):
                i = int(i)
                if k not in needs[i]:
                    G.add_edge(("k", k), ("a", i), capacity=1)
        for i in range(N):
            G.add_edge(("a", i), "T", capacity=cfg.hold)
        val, flow = nx.maximum_flow(G, "S", "T")
        if val != N * cfg.hold:
            continue
        holdings = [set() for _ in range(N)]
        for k in range(K):
            for node, f in flow.get(("k", k), {}).items():
                if f > 0:
                    holdings[node[1]].add(k)
        return holdings, needs
    raise RuntimeError("could not build a solvable instance")


def make_graph(cfg: Config, rng: np.random.Generator) -> nx.Graph:
    N = cfg.N
    if cfg.topology == "complete":
        return nx.complete_graph(N)
    if cfg.topology == "star":
        return nx.star_graph(N - 1)  # node 0 is the hub
    if cfg.topology == "communities":
        a = list(range(N // 2))
        b = list(range(N // 2, N))
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from((x, y) for i, x in enumerate(a) for y in a[i + 1:])
        G.add_edges_from((x, y) for i, x in enumerate(b) for y in b[i + 1:])
        for _ in range(cfg.bridges):
            G.add_edge(int(rng.choice(a)), int(rng.choice(b)))
        return G
    raise ValueError(cfg.topology)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass
class State:
    cfg: Config
    rng: np.random.Generator
    G: nx.Graph
    holdings: list
    needs_left: list
    needs0: list
    theta: np.ndarray
    trust: np.ndarray
    gifts: np.ndarray
    round: int = 0
    defense: str = "none"            # none | provenance | verify
    compromised: set = field(default_factory=set)
    inbox: list = field(default_factory=list)
    claims_sources: list = field(default_factory=list)  # per agent: claim_id -> set(senders)
    believed: list = field(default_factory=list)        # per agent: set(claim_id)
    exposed: dict = field(default_factory=dict)         # claim_id -> set(agents who received it)
    believers: dict = field(default_factory=dict)       # claim_id -> set(agents who believed it)
    refusals: list = field(default_factory=list)        # (round, refuser, refused)
    done_round: list = field(default_factory=list)
    heard: list = field(default_factory=list)           # per agent: messages received (for LLM agents)
    next_claim: int = 0
    log: list = field(default_factory=list)
    keep_log: bool = False

    def emit(self, **ev):
        if self.keep_log:
            ev["round"] = self.round
            self.log.append(ev)

    def done(self, i):
        return not self.needs_left[i]


def new_state(cfg: Config, seed: int, keep_log=False) -> State:
    rng = np.random.default_rng(seed)
    holdings, needs = make_instance(cfg, rng)
    G = make_graph(cfg, rng)
    N = cfg.N
    trust = np.full((N, N), cfg.trust_prior)
    np.fill_diagonal(trust, 1.0)
    st = State(
        cfg=cfg, rng=rng, G=G,
        holdings=holdings,
        needs_left=[set(s) for s in needs],
        needs0=[set(s) for s in needs],
        theta=rng.uniform(cfg.theta_lo, cfg.theta_hi, N),
        trust=trust,
        gifts=np.zeros((N, N), dtype=int),
        inbox=[[] for _ in range(N)],
        claims_sources=[{} for _ in range(N)],
        believed=[set() for _ in range(N)],
        done_round=[None] * N,
        heard=[[] for _ in range(N)],
        keep_log=keep_log,
    )
    st.emit(type="init", holdings=[sorted(h) for h in holdings], needs=[sorted(n) for n in needs],
            topology=cfg.topology, edges=[list(e) for e in G.edges()])
    return st


def fork(st: State, branch_seed: int) -> State:
    """Copy the whole world and give the copy a fresh RNG, so branches that share
    branch_seed differ only by the intervention applied after the fork."""
    st2 = copy.deepcopy(st)
    st2.rng = np.random.default_rng(branch_seed)
    return st2


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def _claim(st: State, kind, target, origin, forged, falsifiable_false):
    cid = st.next_claim
    st.next_claim += 1
    st.exposed[cid] = set()
    st.believers[cid] = set()
    return dict(cid=cid, kind=kind, target=target, origin=origin,
                forged=forged, false=falsifiable_false)


def send(st: State, claim: dict, sender: int, to: int):
    msg = dict(claim)
    msg["sender"] = sender
    st.inbox[to].append(msg)
    st.emit(type="msg", kind=claim["kind"], cid=claim["cid"], sender=sender, to=to,
            target=claim["target"], origin=claim["origin"], forged=claim["forged"])


def _clip(st):
    np.clip(st.trust, 0.0, 1.0, out=st.trust)
    np.fill_diagonal(st.trust, 1.0)


def process_inbox(st: State, i: int) -> bool:
    """Handle agent i's messages. Returns True if i used its action on verification."""
    cfg, rng = st.cfg, st.rng
    used_action = False
    msgs, st.inbox[i] = st.inbox[i], []
    for m in msgs:
        cid, T, S, O = m["cid"], m["target"], m["sender"], m["origin"]
        st.exposed[cid].add(i)
        st.heard[i].append(dict(round=st.round, kind=m["kind"], about=T, from_=S, origin=O,
                                signature_ok=(not m["forged"]) if st.defense == "provenance" else None))
        if T == i or cid in st.believed[i]:
            continue
        # provenance: forged claims carry no valid signature
        if st.defense == "provenance" and m["forged"] and rng.random() < cfg.sig_check:
            st.emit(type="reject_sig", agent=i, cid=cid)
            continue
        srcs = st.claims_sources[i].setdefault(cid, set())
        if S in srcs:
            continue
        srcs.add(S)
        # verification: spend the action to check the public ledger
        if st.defense == "verify" and not used_action and rng.random() < cfg.verify_prob:
            used_action = True
            if m["false"]:
                st.trust[i, O] -= cfg.verify_blame   # the apparent origin gets blamed
                st.believed[i].add(cid)              # settled: won't reconsider
                st.believers[cid].discard(i)
                st.emit(type="verify_false", agent=i, cid=cid, blamed=O)
                _clip(st)
                continue
            p = 1.0
        else:
            p = cfg.believe_q * st.trust[i, S]
        if rng.random() >= p:
            continue
        # believe
        st.believed[i].add(cid)
        st.believers[cid].add(i)
        hit = cfg.accuse_hit if m["kind"] == "accuse" else cfg.complain_hit
        st.trust[i, T] -= hit
        if m["kind"] == "accuse":
            for j in range(cfg.N):
                if j != i and j != T:
                    st.trust[i, j] -= cfg.paranoia
        _clip(st)
        st.emit(type="believe", agent=i, cid=cid, target=T)
        fwd = cfg.forward_prob if m["kind"] == "accuse" else cfg.complain_forward
        for nb in st.G.neighbors(i):
            if nb != S and nb != T and rng.random() < fwd:
                send(st, m, i, nb)
    return used_action


# --------------------------------------------------------------------------
# Agent policy (scripted)
# --------------------------------------------------------------------------

def candidates(st: State, i: int):
    """(j, k) pairs where i holds key k and active agent j still needs it."""
    out = []
    for j in range(st.cfg.N):
        if j == i or st.done(j):
            continue
        for k in st.holdings[i] & st.needs_left[j]:
            out.append((j, k))
    return out


def scripted_policy(st: State, i: int):
    """Give one key to the most trusted agent who needs one you hold.
    Returns ("give", j, k) or ("idle",). Also records refusals."""
    cands = candidates(st, i)
    if not cands:
        return ("idle",)
    trusted = [(j, k) for j, k in cands if st.trust[i, j] >= st.theta[i]]
    refused = sorted({j for j, k in cands if st.trust[i, j] < st.theta[i]})
    for j in refused:
        st.refusals.append((st.round, i, j))
        st.emit(type="refuse", agent=i, refused=j)
        if st.rng.random() < st.cfg.complain_prob:
            c = _claim(st, "complain", target=i, origin=j, forged=False, falsifiable_false=False)
            for nb in st.G.neighbors(j):
                if nb != i:
                    send(st, c, j, nb)
    if not trusted:
        return ("idle",)
    noise = st.rng.random(len(trusted)) * 0.05
    best = int(np.argmax([st.trust[i, j] + n for (j, _), n in zip(trusted, noise)]))
    j, k = trusted[best]
    return ("give", j, k)


def apply_give(st: State, i: int, j: int, k: int):
    if k not in st.holdings[i] or k not in st.needs_left[j]:
        st.emit(type="invalid", agent=i, to=j, key=k)
        return False
    st.holdings[i].remove(k)
    st.needs_left[j].remove(k)
    st.gifts[i, j] += 1
    st.trust[j, i] += st.cfg.gift_recip
    st.trust[:, i] += st.cfg.gift_rep
    _clip(st)
    st.emit(type="give", agent=i, to=j, key=k)
    if st.done(j) and st.done_round[j] is None:
        st.done_round[j] = st.round
        st.emit(type="win", agent=j)
    return True


# --------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------

def forge_accusation(st: State, victim=None):
    """One forged message: tell the agent holding the most keys the victim still
    needs that the victim betrayed a third agent (impersonating that agent)."""
    N, rng = st.cfg.N, st.rng
    if victim is None:
        open_ = [j for j in range(N) if not st.done(j)]
        if not open_:
            return None
        victim = max(open_, key=lambda j: (len(st.needs_left[j]), rng.random()))
    holders = [i for i in range(N) if i != victim]
    recipient = max(holders, key=lambda i: (len(st.holdings[i] & st.needs_left[victim]), rng.random()))
    others = [o for o in range(N) if o not in (victim, recipient)]
    origin = max(others, key=lambda o: (st.trust[recipient, o], rng.random()))
    c = _claim(st, "accuse", target=victim, origin=origin, forged=True, falsifiable_false=True)
    msg = dict(c, sender=origin)
    st.inbox[recipient].append(msg)
    st.exposed[c["cid"]]  # created
    st.emit(type="forge", cid=c["cid"], victim=victim, recipient=recipient, impersonated=origin)
    return dict(cid=c["cid"], victim=victim, recipient=recipient, origin=origin)


def pick_target(st: State, how: str) -> int:
    N, rng = st.cfg.N, st.rng
    pool = list(range(N))
    if how == "random":
        return int(rng.choice(pool))
    if how == "comm_hub":
        deg = dict(st.G.degree())
        top = max(deg.values())
        return int(rng.choice([i for i in pool if deg[i] == top]))
    if how == "top_giver":
        out = st.gifts.sum(axis=1)
        top = out.max()
        return int(rng.choice([i for i in pool if out[i] == top]))
    if how == "bottleneck":
        demand = Counter(k for j in range(N) for k in st.needs_left[j])
        supply = Counter(k for j in range(N) for k in st.holdings[j])
        def shortfall(x):
            s = 0
            for k in st.holdings[x]:
                s += max(0, demand[k] - (supply[k] - 1))
            return s
        scores = {i: shortfall(i) for i in pool}
        top = max(scores.values())
        return int(rng.choice([i for i in pool if scores[i] == top]))
    raise ValueError(how)


def compromise(st: State, x: int):
    st.compromised.add(x)
    st.emit(type="compromise", agent=x)


def insider_turn(st: State, x: int):
    """A compromised insider hoards, and lies about a random agent to its
    neighbors under its own (validly signed) identity."""
    if st.rng.random() >= st.cfg.insider_lie_prob:
        return
    others = [j for j in range(st.cfg.N) if j != x]
    T = int(st.rng.choice(others))
    c = _claim(st, "accuse", target=T, origin=x, forged=False, falsifiable_false=True)
    for nb in st.G.neighbors(x):
        if nb != T:
            send(st, c, x, nb)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def step(st: State, policy=None):
    policy = policy or scripted_policy
    st.round += 1
    N = st.cfg.N
    order = st.rng.permutation(N)
    for i in order:
        i = int(i)
        if i in st.compromised:
            insider_turn(st, i)
            st.inbox[i] = []
            continue
        used = process_inbox(st, i)
        if used or st.rng.random() >= st.cfg.act_prob:
            continue
        act = policy(st, i)
        if act[0] == "give":
            apply_give(st, i, act[1], act[2])


def run(st: State, until=None, policy=None):
    until = st.cfg.max_rounds if until is None else until
    while st.round < until and not all(st.done(j) for j in range(st.cfg.N) if j not in st.compromised):
        step(st, policy)
    return st


def metrics(st: State, exclude=()):
    N = st.cfg.N
    honest = [j for j in range(N) if j not in st.compromised and j not in exclude]
    done = [j for j in honest if st.done(j)]
    rounds = [st.done_round[j] for j in done]
    keys_got = sum(len(st.needs0[j]) - len(st.needs_left[j]) for j in honest)
    return dict(
        completion=len(done) / len(honest),
        keys_frac=keys_got / (len(honest) * st.cfg.need),
        mean_done_round=float(np.mean(rounds)) if rounds else None,
        rounds=st.round,
        refusals=len(st.refusals),
    )


def write_log(st: State, path):
    with open(path, "w") as f:
        f.write(json.dumps({"type": "config", **asdict(st.cfg), "defense": st.defense}) + "\n")
        for ev in st.log:
            f.write(json.dumps(ev, default=int) + "\n")
