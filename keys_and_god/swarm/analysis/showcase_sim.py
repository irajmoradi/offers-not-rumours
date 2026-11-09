"""Scripted-agent check of the showcase incentives (configs/showcase100_sim.yaml, plan/SHOWCASE.md). No LLM.

    PYTHONNOUSERSITE=1 PYTHONPATH=. python -m swarm.analysis.showcase_sim [--worlds 8] [--procs 8] \
        [--out plan/figs/showcase_sim.json]

Uses swarm.analysis.game_theory_audit.Sim (the REAL world generator, graph and rules; simultaneous turns)
on the showcase config: N=100 SBM (5x20), contacts, key_placement mixed (1 / 2 / >= 3 hops), lock_and_stay,
held_key_bonus 0.05 on starting infos (held_key_scope initial), team_share 0. Policies are scripted (C relays toward far needers,
H never gives), so the score terms do not change play and the community bonus B is applied post hoc:
one game gives the payoff for every B; delivery credits (phi) are tracked by scoring.DeliveryLedger
over the resolved events and scaled post hoc.
  reach    all-C welfare by key placement (random / far d_min 2 / far d_min 3) and information mode
           (board 1 hop, gossip2, gossipAll flooding): what far placement demands of the agents.
  hoard    focal H among C (6 seats per world: 3 bridge nodes, 3 interior), vs the same seat playing C;
           also focal R (ShowSim: plays C but never gives a copy it held at the start, i.e. keeps its
           own spares and still relays transit keys) = the own-spares dilemma without the transit sink.
           Payoff = needs + 0.05 * keys held + 0.2 * mean needs + B * [>= 90% of its community won].
           Reports, per B, mean (C - H) and the share of seats where giving pays, plus the two terms:
           the held-key gain of hoarding (0.05 * extra keys kept) and the pivotal loss (did the seat's
           hoarding drop its community below the threshold?).
Deterministic in (--worlds).
"""

from __future__ import annotations

import argparse
import json
import statistics
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from swarm.analysis.game_theory_audit import Sim, load_cfg, mean_ci
from swarm.game import resolve, rules, scoring
from swarm.runner.spec import world_seed
from swarm.schema import GiveKey, Noop

CONFIG = "showcase100_sim"
PLACEMENTS = {"random": ["game.network.key_placement=random"],
              "far2": ["game.network.key_placement=far", "game.network.far_min_hops=2"],
              "far3": ["game.network.key_placement=far"], "mixed": []}
INFOS = ("board", "gossip2", "gossipAll")
B_VALUES = (0.0, 0.15, 0.30)
PHI_VALUES = (0.0, 0.10)
SEATS = 6


class ShowSim(Sim):
    """Sim + policy R ("keeps its own spares"): plays C but never gives a copy it held at the start,
    so it still relays keys passing through it (the held-key dilemma without the transit sink)."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.initial = {x: {kc.copy_id for kc in s.keys} for x, s in self.state.agents.items()}
        self.ledger = scoring.DeliveryLedger(self.g.score.delivery_credit or 0.10)   # phi always tracked
        self.ledger.start(self.state.agents)

    def play(self) -> dict[str, Any]:
        """Sim.play with every resolved turn's events fed to the delivery ledger."""
        real = resolve.resolve_turn

        def tracked(*a: Any, **kw: Any) -> Any:
            res = real(*a, **kw)
            for b in res.events:
                self.ledger.on_event(b.data, b.round)
            return res
        resolve.resolve_turn = tracked
        try:
            return super().play()
        finally:
            resolve.resolve_turn = real

    def decide(self, a: str, r: int) -> list:
        if self.types[a] != "R":
            return super().decide(a, r)
        self.types[a] = "C"
        try:
            acts = super().decide(a, r)
        finally:
            self.types[a] = "R"
        transit = {kc.info_id for kc in self.state.agents[a].keys if kc.copy_id not in self.initial[a]}
        keep = [x for x in acts if not (isinstance(x, GiveKey) and x.info_id not in transit)]
        return keep or [Noop()]


def _cfg(overrides: list[str]):
    return load_cfg(CONFIG, overrides)


def _bridge_seats(cfg, gi: int) -> list[str]:
    """3 bridge nodes (an edge into another community) and 3 interior nodes, by sorted name."""
    st = Sim(cfg, gi, "C", "oracle", order_seed=0).state
    comm = scoring.communities(st, cfg.game, world_seed(cfg, gi))
    agents = sorted(st.agents)
    bridge = [a for a in agents if any(comm[b] != comm[a] for b in st.agents[a].contacts or [])]
    inner = [a for a in agents if a not in bridge]
    return bridge[:: max(len(bridge) // 3, 1)][:3] + inner[:: max(len(inner) // 3, 1)][:3]


def _payoffs(sim: Sim, cfg, gi: int) -> dict[str, Any]:
    st, g = sim.state, cfg.game
    comm = scoring.communities(st, g, world_seed(cfg, gi))
    won = {a: st.agents[a].won_round is not None for a in sim.agents}
    groups = scoring.community_members(comm)
    won = scoring.community_success(won, scoring.held_live_needs(st), g)
    hit = scoring.community_scores(won, comm, g.model_copy(update={"score": g.score.model_copy(update={"community_bonus": 1.0})}))
    cross = {c: bool(hit[ms[0]]) for c, ms in groups.items()}
    raw = {a: rules.score(st, a, g) for a in sim.agents}
    mean = statistics.mean(raw.values())
    nk = {a: scoring.bonus_keys(st, a, g, sim.start_infos) for a in sim.agents}   # held_key_scope
    base = {a: raw[a] + g.score.held_key_bonus * nk[a] + g.score.team_share * mean for a in sim.agents}
    credit = sim.ledger.credit if isinstance(sim, ShowSim) else {}
    return {"base": base, "keys": nk, "raw": raw,
            "credit": {a: credit.get(a, 0.0) for a in sim.agents},
            "cross": {a: cross[comm[a]] for a in sim.agents},
            "community_win_share": {str(c): sum(won[a] for a in ms) / len(ms) for c, ms in groups.items()},
            "welfare": mean, "win_rate": statistics.mean(won.values())}


def run_job(job: tuple) -> dict[str, Any]:
    kind, placement, info, gi, seat = job
    cfg = _cfg(PLACEMENTS[placement])
    if kind == "reach":
        r = Sim(cfg, gi, "C", info, order_seed=gi).play()
        return {"job": list(job), "welfare": r["welfare"], "win_rate": r["win_rate"],
                "relays": r["stats"].get("relay_gives", 0)}
    agents = sorted(Sim(cfg, gi, "C", info, order_seed=gi).state.agents)
    out: dict[str, Any] = {"job": list(job)}
    for pol in ("H", "R"):
        sim = ShowSim(cfg, gi, {a: (pol if a == seat else "C") for a in agents}, info, order_seed=gi)
        sim.play()
        out[pol] = _payoffs(sim, cfg, gi)
    return out


def jobs(n_worlds: int) -> list[tuple]:
    out: list[tuple] = [("reach", p, i, gi, None) for p in PLACEMENTS for i in INFOS for gi in range(n_worlds)]
    for info in ("gossipAll", "oracle"):
        cfg = _cfg([])
        for gi in range(n_worlds):
            out += [("hoard", "mixed", info, gi, s) for s in _bridge_seats(cfg, gi)]
    return out


def summarize(results: list[dict[str, Any]], n_worlds: int) -> dict[str, Any]:
    reach: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["job"][0] != "reach":
            continue
        reach.setdefault(f"{r['job'][1]}/{r['job'][2]}", []).append(r)
    reach_s = {k: {"welfare": mean_ci([x["welfare"] for x in v]), "win_rate": statistics.mean(x["win_rate"] for x in v),
                   "relays": statistics.mean(x["relays"] for x in v)} for k, v in reach.items()}
    cfg = _cfg([])
    allc = {}
    for info in ("gossipAll", "oracle"):
        for gi in range(n_worlds):
            sim = ShowSim(cfg, gi, "C", info, order_seed=gi)
            sim.play()
            allc[(info, gi)] = _payoffs(sim, cfg, gi)
    hoard: dict[str, Any] = {}
    for info in ("gossipAll", "oracle"):
        for pol in ("H", "R"):
            rows = []
            for r in results:
                if r["job"][0] != "hoard" or r["job"][2] != info:
                    continue
                gi, seat = r["job"][3], r["job"][4]
                c, h = allc[(info, gi)], r[pol]
                rows.append({"world": gi, "seat": seat,
                             "keys_gain": h["keys"][seat] - c["keys"][seat],
                             "raw_C": c["raw"][seat], "raw_H": h["raw"][seat],
                             "cross_C": c["cross"][seat], "cross_H": h["cross"][seat],
                             "base_C_minus_H": c["base"][seat] - h["base"][seat],
                             "credit_C_minus_H": c["credit"][seat] - h["credit"][seat],
                             "others_welfare_loss": c["welfare"] - h["welfare"]})
            per_b = {}
            piv = [x for x in rows if x["cross_C"] and not x["cross_H"]]
            for b in B_VALUES:
                for phi in PHI_VALUES:
                    k = phi / 0.10                       # credits were tracked at phi = 0.10
                    d = [x["base_C_minus_H"] + b * (x["cross_C"] - x["cross_H"]) + k * x["credit_C_minus_H"]
                         for x in rows]
                    dp = [x["base_C_minus_H"] + b + k * x["credit_C_minus_H"] for x in piv]
                    per_b[f"B={b:.2f} phi={phi:.2f}"] = {
                        "C_minus_H": mean_ci(d), "share_giving_pays": sum(v > 0 for v in d) / max(len(d), 1),
                        "pivotal_C_minus_H": statistics.mean(dp) if dp else None}
            hoard[f"{pol}/{info}"] = {
                "n": len(rows), "per_B": per_b,
                "held_key_gain_of_hoarding": statistics.mean(0.05 * x["keys_gain"] for x in rows),
                "share_pivotal": len(piv) / max(len(rows), 1),
                "share_community_crossed_allC": statistics.mean(float(x["cross_C"]) for x in rows),
                "community_win_share_allC": statistics.mean(
                    v for gi in range(n_worlds) for v in allc[(info, gi)]["community_win_share"].values()),
                "others_welfare_loss": statistics.mean(x["others_welfare_loss"] for x in rows),
                "credit_C": statistics.mean(allc[(info, x["world"])]["credit"][x["seat"]] for x in rows),
                "rows": rows}
    return {"reach": reach_s, "hoard": hoard}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="plan/figs/showcase_sim.json")
    a = ap.parse_args()
    js = jobs(a.worlds)
    with Pool(min(a.procs, 8)) as pool:
        results = pool.map(run_job, js)
    s = summarize(results, a.worlds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=1, default=str))
    print("reach (all-C):  placement/info  welfare [95% CI]  win  relays")
    for k, v in sorted(s["reach"].items()):
        m, lo, hi = v["welfare"]
        print(f"  {k:18s} {m:.3f} [{lo:.3f}, {hi:.3f}]  {v['win_rate']:.3f}  {v['relays']:.0f}")
    for key, h in s["hoard"].items():
        print(f"hoard {key} (n={h['n']} seats): held-key gain {h['held_key_gain_of_hoarding']:+.3f}, "
              f"pivotal {h['share_pivotal']:.2f}, community crossed (all-C) {h['share_community_crossed_allC']:.2f}, "
              f"community win share {h['community_win_share_allC']:.2f}, others' welfare loss {h['others_welfare_loss']:.4f}, "
              f"seat credit as C {h['credit_C']:.3f}")
        for b, v in h["per_B"].items():
            m, lo, hi = v["C_minus_H"]
            pv = v["pivotal_C_minus_H"]
            print(f"  {b}: payoff(C) - payoff({key[0]}) {m:+.3f} [{lo:+.3f}, {hi:+.3f}], giving pays at "
                  f"{v['share_giving_pays']:.2f}; when pivotal {'n/a' if pv is None else f'{pv:+.3f}'}")


if __name__ == "__main__":
    main()
