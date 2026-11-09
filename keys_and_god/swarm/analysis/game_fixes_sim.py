"""Before/after simulation of the kept game-fixes switches (plan/GAME_FIXES.md, CONTRACTS §9f). No LLM.

    PYTHONNOUSERSITE=1 PYTHONPATH=. $PY -m swarm.analysis.game_fixes_sim \
        [--worlds 30] [--procs 20] [--out plan/figs/game_fixes_sim.json]

Re-runs the scripted-agent simulation of swarm.analysis.game_theory_audit (its Sim: the REAL world
generator, graph builder and rules via game.resolve.resolve_turn, simultaneous turns) with each kept
switch on, next to the same cell with it off. Topologies: g1 = configs/grid_oss_sim_k2 (N=10, 20
rounds, the E4 worlds; 16 worlds), complete = net40_complete_sim and er3 = net40_er3_contacts_sim
(N=40, 30 rounds; the E3 worlds are er3 0-3).
  winners  continue / sit_out / lock_and_stay: all-C welfare, stranded needs; one focal D among C.
  defect   lose_need vs lose_held_need: all-B, focal B among C; the last-round lottery L vs C (ER3
           neighbour board, where many needs are hopeless at the end).
  bonus    held_key_bonus 0.05: all-C / all-H / all-S, focal H / S among C (private = raw + bonus).
  e4       the E4 bundle (lock_and_stay + lose_held_need + held_key_bonus 0.05) vs v1 on g1.
  team     team_share 0.3 on ER3 contacts (E3): C (relays) vs Cn (never relays) by information mode,
           needs-only and team welfare; one focal Cn among C (does the team term make relaying pay?).
Each job is (cell, topology, config overrides, info mode, population, world, focal seat). Invasion
cells use 4 seats x max(4, worlds/3) worlds (g1: 4 seats x 8 worlds). Output: one JSON with per-cell
means and 95% bootstrap CIs over worlds; a text table on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from swarm.analysis.game_theory_audit import Sim, hash_job, load_cfg, mean_ci, world
from swarm.config import RunConfig

# name -> (config, rounds, fixed number of worlds or None = --worlds, focal seats)
TOPOS = {"g1": ("grid_oss_sim_k2", 20, 16, (0, 3, 6, 9)),
         "complete": ("net40_complete_sim", 30, None, (0, 10, 20, 30)),
         "er3": ("net40_er3_contacts_sim", 30, None, (0, 10, 20, 30))}
WINNERS = {w: [f"game.winner_policy={w}"] for w in ("continue", "sit_out", "lock_and_stay")}
PENALTY = {"lose_need": [], "lose_held_need": ["game.wrong_defect_penalty=lose_held_need"]}
BONUS = {"no bonus": [], "bonus 0.05": ["game.score.held_key_bonus=0.05"]}
E4 = {"v1": [], "E4": ["game.winner_policy=lock_and_stay", "game.wrong_defect_penalty=lose_held_need",
                       "game.score.held_key_bonus=0.05"]}
TEAM = {"no team": [], "team 0.3": ["game.score.team_share=0.3"]}


def cfg_for(topo: str, overrides: list[str]) -> RunConfig:
    """The topology's config at its round count, with the cell's overrides."""
    name, rounds, _, _ = TOPOS[topo]
    return load_cfg(name, [f"game.max_rounds={rounds}", *overrides])


def run_job(job: tuple) -> dict[str, Any]:
    """One scripted game: needs-only / private / team welfare, need outcomes, focal scores."""
    cell, topo, overrides, info, pop, gi, seat = job
    cfg = cfg_for(topo, list(overrides))
    agents = sorted(world(cfg, gi).agents)
    if seat is None:
        types: dict[str, str] | str = pop
    else:
        foc, res = pop.split("_in_")
        types = {a: (foc if i == seat else res) for i, a in enumerate(agents)}
    r = Sim(cfg, gi, types, info, order_seed=hash_job(job)).play()
    out = {"cell": cell, "pop": pop, "world": gi, "seat": seat, "welfare": r["welfare"],
           "welfare_private": r["welfare_private"], "welfare_shared": r["welfare_shared"],
           "rounds": r["rounds"], "finished": r["finished"], "unmet": r["unmet"], "stats": r["stats"]}
    if seat is not None:
        a = agents[seat]
        out.update(focal_raw=r["scores"][a], focal_private=r["private_scores"][a],
                   focal_shared=r["shared_scores"][a])
    return out


def jobs(n_worlds: int) -> list[tuple]:
    """Every job of the five experiments."""
    out: list[tuple] = []

    def worlds(topo: str, inv: bool) -> range:
        fixed = TOPOS[topo][2]
        if fixed is not None:
            return range(fixed // 2 if inv else fixed)
        return range(max(4, n_worlds // 3) if inv else n_worlds)

    def homog(cell, topo, ov, info, pops):
        for pop in pops:
            for gi in worlds(topo, False):
                out.append((cell, topo, tuple(ov), info, pop, gi, None))

    def invade(cell, topo, ov, info, pops):
        for pop in pops:
            for gi in worlds(topo, True):
                for seat in TOPOS[topo][3]:
                    out.append((cell, topo, tuple(ov), info, pop, gi, seat))

    for w, ov in WINNERS.items():                                     # 1. winners
        for topo in ("g1", "complete"):
            homog(f"winners|{topo} board|{w}", topo, ov, "board", ["C"])
            invade(f"winners|{topo} board|{w}", topo, ov, "board", ["D_in_C", "C_in_C"])
        homog(f"winners|er3 gossipAll|{w}", "er3", ov, "gossipAll", ["C"])
    for pen, ov in PENALTY.items():                                   # 2. wrong-defect penalty
        for topo in ("g1", "complete"):
            homog(f"defect|{topo} board|{pen}", topo, ov, "board", ["B"])
            invade(f"defect|{topo} board|{pen}", topo, ov, "board", ["B_in_C", "C_in_C"])
        homog(f"defect|er3 board|{pen}", "er3", ov, "board", ["L", "C"])
        invade(f"defect|er3 board|{pen}", "er3", ov, "board", ["L_in_C", "C_in_C"])
    for b, ov in BONUS.items():                                       # 3. held-key bonus
        for topo in ("g1", "complete"):
            homog(f"bonus|{topo} board|{b}", topo, ov, "board", ["C", "H", "S"])
            invade(f"bonus|{topo} board|{b}", topo, ov, "board", ["H_in_C", "S_in_C", "C_in_C"])
    for e, ov in E4.items():                                          # 4. the E4 bundle on G1
        homog(f"e4|g1 board|{e}", "g1", ov, "board", ["C", "H", "D", "B", "S"])
        invade(f"e4|g1 board|{e}", "g1", ov, "board", ["H_in_C", "D_in_C", "B_in_C", "C_in_C"])
    for t, ov in TEAM.items():                                        # 5. team score on ER3 (E3)
        for info in ("board", "gossip2", "gossipAll"):
            homog(f"team|er3 {info}|{t}", "er3", ov, info, ["C", "Cn"])
        for info in ("gossip2", "gossipAll"):
            invade(f"team|er3 {info}|{t}", "er3", ov, info, ["Cn_in_C", "C_in_C"])
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (cell, population): welfare means / CIs, need outcomes, focal scores (invasion cells)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        key = f"{r['cell']}|{r['pop']}" + ("|focal" if r["seat"] is not None else "")
        groups.setdefault(key, []).append(r)
    out: dict[str, Any] = {}
    for k, rs in sorted(groups.items()):
        unmet: Counter = Counter()
        stats: Counter = Counter()
        for r in rs:
            unmet.update(r["unmet"])
            stats.update(r["stats"])
        slots = sum(unmet.values()) or 1
        row = {"n": len(rs), "welfare": mean_ci([r["welfare"] for r in rs]),
               "welfare_private": mean_ci([r["welfare_private"] for r in rs]),
               "welfare_shared": mean_ci([r["welfare_shared"] for r in rs]),
               "rounds": statistics.mean(r["rounds"] for r in rs),
               "finished": statistics.mean(1.0 if r["finished"] else 0.0 for r in rs),
               "need_outcomes": {c: v / slots for c, v in sorted(unmet.items())},
               "per_game": {c: v / len(rs) for c, v in sorted(stats.items())}}
        if rs[0]["seat"] is not None:
            for f in ("focal_raw", "focal_private", "focal_shared"):
                row[f] = mean_ci([r[f] for r in rs])
        out[k] = row
    return out


def _fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.3f} [{ci[1]:.3f},{ci[2]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--out", default="plan/figs/game_fixes_sim.json")
    args = ap.parse_args()
    js = jobs(args.worlds)
    with Pool(args.procs) as pool:
        results = pool.map(run_job, js, chunksize=2)
    summ = summarize(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"worlds": args.worlds, "n_jobs": len(js), "cells": summ},
                                         indent=1, default=float))
    for k, v in summ.items():
        no = {c: round(x, 3) for c, x in v["need_outcomes"].items() if x}
        foc = (f" focal raw={_fmt(v['focal_raw'])} private={_fmt(v['focal_private'])} "
               f"shared={_fmt(v['focal_shared'])}") if "focal_raw" in v else ""
        print(f"{k:52s} n={v['n']:3d} W={_fmt(v['welfare'])} Wp={v['welfare_private'][0]:.3f} "
              f"Ws={v['welfare_shared'][0]:.3f} rounds={v['rounds']:.1f} {no}{foc}")


if __name__ == "__main__":
    main()
