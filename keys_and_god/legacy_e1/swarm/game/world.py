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
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from swarm.config import GameCfg
from swarm.schema import AgentId, AgentState, GameState, InfoId, KeyCopy

from .network import attach as attach_network

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
                  rng: random.Random) -> dict[AgentId, list[InfoId]]:
    """Constructive: reserve a DISTINCT copy held by someone else for every (agent, need) slot, so
    the matching (and hence feasibility) exists by construction. Greedy over a shuffled agent order,
    retried on dead ends; the result is re-verified with max_matching."""
    for _ in range(MAX_ATTEMPTS):
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
            for ci in pool:
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
    needs = _assign_needs(agents, holdings, copies, m, rng)
    state = GameState(
        game_id=game_id, round=0, turn_index=0, order=list(agents),
        agents={a: AgentState(agent=a, keys=holdings[a], needs=needs[a]) for a in agents},
    )
    attach_network(state, cfg, seed)
    return state
