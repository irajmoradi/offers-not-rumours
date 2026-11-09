"""World generation (WP1a): deal key copies and needs so that every world is FEASIBLE.

A world is feasible when every (agent, need) pair can be served by a DISTINCT key copy held by some
other agent at the start (D4). That is a bipartite matching (= unit-capacity max-flow) between
demand slots and copies; `is_feasible` checks it with Kuhn's augmenting-path algorithm.

Construction (all randomness from one private `random.Random(seed)`):
  1. n_infos = ceil(n_agents * keys_per_agent / copies_per_info); info j gets copies_per_info copies
     (the last infos get fewer when the total does not divide evenly).
  2. Copies are dealt keys_per_agent per agent with no agent holding two copies of the same info
     (shuffle, then repair duplicates by swaps).
  3. Needs: needs_per_agent distinct infos the agent does NOT hold, each backed by a distinct copy
     reserved from another agent (constructive), then verified by max matching.
     With game.network.key_placement = local_bias (added later), step 3 first tries the copies held
     within game.network.local_hops graph hops of the needer, then the rest (same shuffled order
     otherwise; per agent with prob. local_strength), so more needs sit near a holder; copy
     counts and the feasibility guarantee are unchanged.
     The graph is built from the same world seed on its own stream, exactly as attach builds it.
     With key_placement = far (showcase), step 3 may only pick infos whose EVERY holder is at least
     game.network.far_min_hops hops from the needer (strict); if no strict assignment is found in
     FAR_STRICT_ATTEMPTS tries, those infos are only tried first. Feasibility is unchanged.
     With key_placement = mixed (showcase), an agent's needs cycle over nearest-holder distance buckets
     1 hop / 2 hops / >= 3 hops (one each for 3 needs; strict, else preferred), for a dose-response by
     distance with every world still solvable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from swarm.config import GameCfg
from swarm.schema import AgentId, AgentState, GameState, InfoId, KeyCopy

from .network import attach as attach_network
from .network import hop_distances, make_graph

ADJECTIVES = (
    "amber", "brisk", "cedar", "dusty", "ember", "fern", "gilded", "hazel", "ivory", "jade",
    "keen", "lunar", "maple", "noble", "olive", "pale", "quiet", "rusty", "sandy", "tidal",
    "umber", "vivid", "windy", "young", "zesty", "azure", "bold", "coral", "dawn", "early",
    "frost", "green", "honey", "inky", "jolly", "kind", "lemon", "misty", "north", "opal",
)
NOUNS = (
    "badger", "beacon", "canyon", "delta", "falcon", "garnet", "harbor", "island", "jasper", "kettle",
    "lantern", "meadow", "nectar", "orchid", "pebble", "quarry", "raven", "staple", "thistle", "upland",
    "valley", "willow", "yarrow", "zephyr", "anchor", "battery", "comet", "dune", "otter", "fjord",
    "grove", "heron", "ingot", "juniper", "kestrel", "lynx", "marten", "nimbus", "osprey", "piper",
)

MAX_ATTEMPTS = 2000


def make_names(n: int, rng: random.Random) -> list[AgentId]:
    """n unique two-word names ("battery_staple" style), seeded."""
    if n > len(ADJECTIVES) * len(NOUNS):
        raise ValueError("not enough name combinations")
    pool = [f"{a}_{b}" for a in ADJECTIVES for b in NOUNS]
    return rng.sample(pool, n)


def _deal(n_agents: int, k: int, copies: int, rng: random.Random) -> list[list[InfoId]]:
    total = n_agents * k
    n_infos = math.ceil(total / copies)
    if n_infos < k:
        raise ValueError("keys_per_agent exceeds the number of distinct infos")
    pool = [i for c in range(copies) for i in range(n_infos)][:total]
    for _ in range(MAX_ATTEMPTS):
        rng.shuffle(pool)
        for _sweep in range(50):
            bad = [p for p in range(total) if pool[p] in pool[(p // k) * k: p]]
            if not bad:
                return [pool[a * k:(a + 1) * k] for a in range(n_agents)]
            for p in bad:
                q = rng.randrange(total)
                pool[p], pool[q] = pool[q], pool[p]
    raise RuntimeError("could not deal keys without duplicates")


def max_matching(demands: Sequence[tuple[AgentId, InfoId]],
                 copies: Sequence[tuple[AgentId, InfoId]]) -> int:
    """Size of a maximum matching demand -> copy, where demand (a, i) may use copy (h, i) iff h != a."""
    by_info: dict[InfoId, list[int]] = {}
    for ci, (_h, i) in enumerate(copies):
        by_info.setdefault(i, []).append(ci)
    owner_of_copy: dict[int, int] = {}

    def augment(d: int, seen: set[int]) -> bool:
        a, i = demands[d]
        for ci in by_info.get(i, []):
            if copies[ci][0] == a or ci in seen:
                continue
            seen.add(ci)
            if ci not in owner_of_copy or augment(owner_of_copy[ci], seen):
                owner_of_copy[ci] = d
                return True
        return False

    return sum(1 for d in range(len(demands)) if augment(d, set()))


def is_feasible(state: GameState) -> bool:
    """True iff every agent's needs can be met simultaneously from the other agents' current keys."""
    demands = [(a, i) for a, s in state.agents.items() for i in s.needs
               if i not in s.burned and i not in s.lost]
    copies = [(a, kc.info_id) for a, s in state.agents.items() for kc in s.keys]
    return max_matching(demands, copies) == len(demands)


def _assign_needs(agents: Sequence[AgentId], holdings: dict[AgentId, list[KeyCopy]],
                  copies: Sequence[tuple[AgentId, InfoId]], m: int,
                  rng: random.Random,
                  near: dict[AgentId, set[AgentId]] | None = None,
                  strength: float = 1.0,
                  far: dict[AgentId, set[InfoId]] | None = None, strict: bool = False,
                  attempts: int = MAX_ATTEMPTS) -> dict[AgentId, list[InfoId]]:
    """Constructive: reserve a DISTINCT copy held by someone else for every (agent, need) slot, so
    the matching (and hence feasibility) exists by construction. Greedy over a shuffled agent order,
    retried on dead ends; the result is re-verified with max_matching. `near` (key_placement
    local_bias): agent -> agents whose copies it tries first, for each agent with prob. `strength`.
    `far` (key_placement far): agent -> the infos it may need (every holder far enough); `strict`
    scans only those, else they are tried first."""
    for _ in range(attempts):
        order = list(agents)
        rng.shuffle(order)
        pool = list(range(len(copies)))
        rng.shuffle(pool)
        used: set[int] = set()
        needs: dict[AgentId, list[InfoId]] = {}
        ok = True
        for a in order:
            held = {kc.info_id for kc in holdings[a]}
            picked: list[InfoId] = []
            scan = pool
            if near is not None and (strength >= 1.0 or rng.random() < strength):
                scan = ([ci for ci in pool if copies[ci][0] in near[a]]
                        + [ci for ci in pool if copies[ci][0] not in near[a]])
            if far is not None:
                scan = ([ci for ci in pool if copies[ci][1] in far[a]]
                        + ([] if strict else [ci for ci in pool if copies[ci][1] not in far[a]]))
            for ci in scan:
                h, i = copies[ci]
                if ci in used or h == a or i in held or i in picked:
                    continue
                used.add(ci)
                picked.append(i)
                if len(picked) == m:
                    break
            if len(picked) < m:
                ok = False
                break
            needs[a] = sorted(picked)
        if ok:
            demands = [(a, i) for a in agents for i in needs[a]]
            if max_matching(demands, copies) == len(demands):
                return needs
    raise RuntimeError("could not generate a feasible world; raise copies_per_info or lower needs")


FAR_STRICT_ATTEMPTS = 300


def far_infos(adj: dict[AgentId, Sequence[AgentId]], agents: Sequence[AgentId],
              copies: Sequence[tuple[AgentId, InfoId]], d_min: int) -> dict[AgentId, set[InfoId]]:
    """agent -> infos whose every holder is >= d_min graph hops away (and that it does not hold)."""
    holders: dict[InfoId, set[AgentId]] = {}
    for h, i in copies:
        holders.setdefault(i, set()).add(h)
    out = {}
    for a in agents:
        close = {b for b, d in hop_distances(adj, a).items() if d < d_min}     # includes a itself
        out[a] = {i for i, hs in holders.items() if not (hs & close)}
    return out


def _far_needs(cfg: GameCfg, agents: Sequence[AgentId], holdings: dict[AgentId, list[KeyCopy]],
               copies: Sequence[tuple[AgentId, InfoId]], m: int, rng: random.Random,
               seed: int) -> dict[AgentId, list[InfoId]]:
    """key_placement far: strict (every need's holders >= far_min_hops away) when a strict assignment
    is found in FAR_STRICT_ATTEMPTS tries, else far infos first (always feasible, as random)."""
    adj = make_graph(cfg.network, agents, seed)
    far = far_infos(adj, agents, copies, cfg.network.far_min_hops)
    try:
        return _assign_needs(agents, holdings, copies, m, rng, far=far, strict=True,
                             attempts=FAR_STRICT_ATTEMPTS)
    except RuntimeError:
        return _assign_needs(agents, holdings, copies, m, rng, far=far, strict=False)


def info_buckets(adj: dict[AgentId, Sequence[AgentId]], agents: Sequence[AgentId],
                 copies: Sequence[tuple[AgentId, InfoId]], top: int = 3) -> dict[AgentId, dict[InfoId, int]]:
    """agent -> info -> min(hops to the nearest OTHER holder, top) (infos it holds are left out)."""
    holders: dict[InfoId, set[AgentId]] = {}
    for h, i in copies:
        holders.setdefault(i, set()).add(h)
    out = {}
    for a in agents:
        dist = hop_distances(adj, a)
        out[a] = {i: min(min(dist.get(h, 10**6) for h in hs), top) for i, hs in holders.items() if a not in hs}
    return out


def _mixed_assign(agents: Sequence[AgentId], holdings: dict[AgentId, list[KeyCopy]],
                  copies: Sequence[tuple[AgentId, InfoId]], m: int, rng: random.Random,
                  bucket: dict[AgentId, dict[InfoId, int]], strict: bool, attempts: int) -> dict[AgentId, list[InfoId]]:
    """_assign_needs for key_placement mixed: slot j of every agent wants an info in distance bucket
    (j % 3) + 1 (1 hop, 2 hops, >= 3 hops); strict scans only that bucket, else it is tried first."""
    for _ in range(attempts):
        order = list(agents)
        rng.shuffle(order)
        pool = list(range(len(copies)))
        rng.shuffle(pool)
        used: set[int] = set()
        needs: dict[AgentId, list[InfoId]] = {}
        ok = True
        for a in order:
            held = {kc.info_id for kc in holdings[a]}
            picked: list[InfoId] = []
            slots = [(j % 3) + 1 for j in range(m)]
            rng.shuffle(slots)
            for want in slots:
                scan = [ci for ci in pool if bucket[a].get(copies[ci][1]) == want]
                if not strict:
                    scan += [ci for ci in pool if bucket[a].get(copies[ci][1]) != want]
                got = None
                for ci in scan:
                    h, i = copies[ci]
                    if ci in used or h == a or i in held or i in picked:
                        continue
                    got = ci
                    break
                if got is None:
                    break
                used.add(got)
                picked.append(copies[got][1])
            if len(picked) < m:
                ok = False
                break
            needs[a] = sorted(picked)
        if ok:
            demands = [(a, i) for a in agents for i in needs[a]]
            if max_matching(demands, copies) == len(demands):
                return needs
    raise RuntimeError("could not generate a feasible world; raise copies_per_info or lower needs")


def _mixed_needs(cfg: GameCfg, agents: Sequence[AgentId], holdings: dict[AgentId, list[KeyCopy]],
                 copies: Sequence[tuple[AgentId, InfoId]], m: int, rng: random.Random,
                 seed: int) -> dict[AgentId, list[InfoId]]:
    """key_placement mixed (showcase): each agent's needs spread over nearest-holder distance 1 / 2 / >= 3
    (one per slot, cycling); strict when found in FAR_STRICT_ATTEMPTS tries, else preferred."""
    adj = make_graph(cfg.network, agents, seed)
    bucket = info_buckets(adj, agents, copies)
    try:
        return _mixed_assign(agents, holdings, copies, m, rng, bucket, True, FAR_STRICT_ATTEMPTS)
    except RuntimeError:
        return _mixed_assign(agents, holdings, copies, m, rng, bucket, False, MAX_ATTEMPTS)


def generate_world(cfg: GameCfg, seed: int, game_id: str = "g0",
                   names: Sequence[AgentId] | None = None) -> GameState:
    """Build the initial GameState (round 0). Deterministic in (cfg, seed, names). Always feasible.
    With game.network on, the agents' contacts (and known names) come from game.network.make_graph,
    seeded by the same world seed but on its own stream (the deal and needs are unchanged)."""
    rng = random.Random(seed)
    n, k, m = cfg.n_agents, cfg.keys_per_agent, cfg.needs_per_agent
    agents = list(names) if names is not None else make_names(n, rng)
    if len(agents) != n or len(set(agents)) != n:
        raise ValueError("names must be n_agents unique ids")
    hands = _deal(n, k, cfg.copies_per_info, rng)
    n_infos = math.ceil(n * k / cfg.copies_per_info)
    key_of = {i: rng.randint(1000, 9999) for i in range(n_infos)}
    copy_no: dict[InfoId, int] = {}
    holdings: dict[AgentId, list[KeyCopy]] = {}
    for a, hand in zip(agents, hands, strict=True):
        kcs = []
        for i in sorted(hand):
            j = copy_no.get(i, 0)
            copy_no[i] = j + 1
            kcs.append(KeyCopy(copy_id=f"k{i}-{j}", info_id=i, key=key_of[i]))
        holdings[a] = kcs
    copies = [(a, kc.info_id) for a in agents for kc in holdings[a]]
    near = None
    if cfg.network.on and cfg.network.key_placement == "local_bias":
        adj = make_graph(cfg.network, agents, seed)
        near = {a: {b for b, d in hop_distances(adj, a).items() if 0 < d <= cfg.network.local_hops}
                for a in agents}
    if cfg.network.on and cfg.network.key_placement == "far":
        needs = _far_needs(cfg, agents, holdings, copies, m, rng, seed)
    elif cfg.network.on and cfg.network.key_placement == "mixed":
        needs = _mixed_needs(cfg, agents, holdings, copies, m, rng, seed)
    else:
        needs = _assign_needs(agents, holdings, copies, m, rng, near, cfg.network.local_strength)
    state = GameState(
        game_id=game_id, round=0, turn_index=0, order=list(agents),
        agents={a: AgentState(agent=a, keys=holdings[a], needs=needs[a]) for a in agents},
    )
    attach_network(state, cfg, seed)
    return state
