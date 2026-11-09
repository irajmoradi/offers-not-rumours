"""Ripple metric (WP6): per-agent divergence between paired fork branches over time.

Given branch A (God) and control B (no God) of the same game_idx (same world, same agent seeds):
  div[agent][round] = 1 if the agent's actions in that round differ between A and B
                      (also 1 if it acted in one branch and not the other), else 0.
  Action signatures: mode="coarse" (kind, to/target, info_id; ignores message wording; default)
                     or mode="strict" (adds message text).
Derived series (index = round 1..R):
  frac[r]        fraction of agents whose round-r actions differ
  cum[r]         fraction of agents that have diverged at any round <= r ("infected")
  score_gap[r]   mean |score_A - score_B| over agents at round end
  welfare_a/b[r] welfare per branch at round end
Plus fork_round / fork_turn (God's first successful power in A), first_div[agent], victims,
hops[agent] (shortest interaction-graph distance from a victim, control branch) and
pre_fork_frac (mean frac before fork_round; ~0 means the pairing worked).
With a noise replica B', ripple(B, B') is the sampling-noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from swarm.rng import canonical_json
from swarm.runlog.trace import GameTrace
from swarm.schema import CotControl, MessageFraud


def action_sig(action: Any, mode: str = "coarse") -> str:
    d = action.model_dump(mode="json") if hasattr(action, "model_dump") else dict(action)
    if mode == "coarse":
        d.pop("text", None)
        d.pop("pattern", None)
    return canonical_json(d)


def turn_actions(g: GameTrace, mode: str = "coarse") -> dict[tuple[str, int], tuple[str, ...]]:
    out: dict[tuple[str, int], tuple[str, ...]] = {}
    for e in g.of_type("turn"):
        d = e.data
        key = (d.agent, e.round)  # type: ignore[union-attr]
        out[key] = out.get(key, ()) + tuple(action_sig(a, mode) for a in d.actions)  # type: ignore[union-attr]
    return out


def first_intervention(g: GameTrace) -> tuple[int, int | None] | None:
    for e in g.of_type("god_power"):
        if e.data.ok:  # type: ignore[union-attr]
            return e.round, e.turn_index
    return None


def victims(g: GameTrace) -> list[str]:
    out: list[str] = []
    for e in g.of_type("god_power"):
        d = e.data
        if not d.ok:  # type: ignore[union-attr]
            continue
        a = d.action  # type: ignore[union-attr]
        t = a.receiver if isinstance(a, MessageFraud) else a.agent if isinstance(a, CotControl) else None
        if t and t not in out:
            out.append(t)
    return out


def interaction_graph(g: GameTrace) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(g.agents)
    for e in g.of_type("message"):
        m = e.data.message  # type: ignore[union-attr]
        if m.channel == "dm" and m.receiver and not m.forged:
            G.add_edge(m.sender_shown, m.receiver)
    for e in g.of_type("key_transfer"):
        d = e.data
        if d.from_agent:  # type: ignore[union-attr]
            G.add_edge(d.from_agent, d.to_agent)  # type: ignore[union-attr]
    for e in g.of_type("burn"):
        G.add_edge(e.data.actor, e.data.target)  # type: ignore[union-attr]
    return G


def _round_scores(g: GameTrace) -> dict[int, dict[str, float]]:
    return {e.round: dict(e.data.scores) for e in g.of_type("round_end")}  # type: ignore[union-attr]


@dataclass
class RippleResult:
    agents: list[str]
    rounds: list[int]
    div: dict[str, list[int]]
    frac: list[float]
    cum: list[float]
    score_gap: list[float]
    welfare_a: list[float]
    welfare_b: list[float]
    fork_round: int | None
    fork_turn: int | None
    first_div: dict[str, int | None]
    victims: list[str]
    hops: dict[str, int | None] = field(default_factory=dict)
    pre_fork_frac: float = 0.0
    mode: str = "coarse"

    def summary(self) -> dict[str, Any]:
        post = [f for r, f in zip(self.rounds, self.frac, strict=True) if self.fork_round and r >= self.fork_round]
        return {"fork_round": self.fork_round, "fork_turn": self.fork_turn, "victims": self.victims,
                "pre_fork_frac": round(self.pre_fork_frac, 4),
                "post_fork_mean_frac": round(sum(post) / len(post), 4) if post else None,
                "final_cum": round(self.cum[-1], 4) if self.cum else None,
                "n_diverged": sum(1 for v in self.first_div.values() if v is not None),
                "n_agents": len(self.agents),
                "final_welfare_gap": round(self.welfare_a[-1] - self.welfare_b[-1], 4) if self.rounds else None}

    def by_hop(self) -> dict[int | None, float]:
        """Fraction of agents at each hop distance from a victim that ever diverged."""
        groups: dict[int | None, list[int]] = {}
        for a in self.agents:
            groups.setdefault(self.hops.get(a), []).append(int(self.first_div[a] is not None))
        return {h: sum(v) / len(v) for h, v in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}


def ripple(a: GameTrace, b: GameTrace, mode: str = "coarse", fork_from: GameTrace | None = None) -> RippleResult:
    """Divergence of B from A. `fork_from` (default A) supplies fork point and victims."""
    src = fork_from or a
    acts_a, acts_b = turn_actions(a, mode), turn_actions(b, mode)
    agents = list(a.agents) or sorted({k[0] for k in acts_a} | {k[0] for k in acts_b})
    max_r = max([k[1] for k in acts_a] + [k[1] for k in acts_b] + [0])
    rounds = list(range(1, max_r + 1))
    div = {ag: [int(acts_a.get((ag, r)) != acts_b.get((ag, r))) for r in rounds] for ag in agents}
    n = max(1, len(agents))
    frac = [sum(div[ag][i] for ag in agents) / n for i in range(len(rounds))]
    first = {ag: next((r for r, d in zip(rounds, div[ag], strict=True) if d), None) for ag in agents}
    cum = [sum(1 for ag in agents if first[ag] is not None and first[ag] <= r) / n for r in rounds]
    sa, sb = _round_scores(a), _round_scores(b)

    def _last(s: dict[int, dict[str, float]], r: int) -> dict[str, float]:
        ks = [k for k in s if k <= r]
        return s[max(ks)] if ks else {}

    gap, wa, wb = [], [], []
    for r in rounds:
        x, y = _last(sa, r), _last(sb, r)
        gap.append(sum(abs(x.get(ag, 0.0) - y.get(ag, 0.0)) for ag in agents) / n)
        wa.append(sum(x.values()) / n if x else 0.0)
        wb.append(sum(y.values()) / n if y else 0.0)
    fi = first_intervention(src)
    fork_round, fork_turn = (fi if fi else (None, None))
    pre = [f for r, f in zip(rounds, frac, strict=True) if fork_round is not None and r < fork_round]
    vic = victims(src)
    hops: dict[str, int | None] = {}
    G = interaction_graph(b)
    for ag in agents:
        ds = [nx.shortest_path_length(G, v, ag) for v in vic if v in G and ag in G and nx.has_path(G, v, ag)]
        hops[ag] = min(ds) if ds else None
    return RippleResult(agents, rounds, div, frac, cum, gap, wa, wb, fork_round, fork_turn, first, vic, hops,
                        sum(pre) / len(pre) if pre else 0.0, mode)
