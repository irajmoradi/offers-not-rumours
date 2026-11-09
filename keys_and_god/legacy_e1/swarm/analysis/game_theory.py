"""Game theory and balance of the Keys & God game, from finished run dirs (plan/GAME_THEORY.md).

    PYTHONNOUSERSITE=1 conda run -n swarm python -m swarm.analysis.game_theory [--out plan/figs] [--boot 5000]

Reads only runs/*/games/*/{events,outcome,meta}.json[l]: no LLM, no network. Every game is re-played
from its game_start state through the key_transfer / burn / win events, so each give and defect is
scored against the exact state at that moment (who held what, which needs were still open). The replay
is checked against outcome.json (final score of every agent must match).

Cells (all N=10, 20 rounds, gpt-oss-120b; health filter = swarm.analysis.grid_stats.game_health):
  base   = board on, God-free: G1 nogod + G1 noise replica (grid_oss_sim_k2), G3 nogod (grid_oss_sim_k6),
           G2 nogod (r2_grid_oss_seq_k2, sequential turns, worlds 0-7)
  boardoff = c16_sweep_board_off nogod (worlds 0-15)
  god    = the God branches of G1, G3, G2 (used only for the "does God create an advantage" check)
All cells play the same 16 worlds (run.seed 1, game_idx 0-15).

Outputs: <out>/game_theory.json (every number in plan/GAME_THEORY.md), <out>/gt_score_by_type.png,
<out>/gt_defect_ev.png, and a text summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm.analysis.grid_stats import game_health

RUNS = Path("runs")
CELLS: dict[str, list[tuple[str, str]]] = {        # cell -> [(run dir, branch)]
    "base": [("grid_oss_sim_k2", "nogod"), ("grid_oss_sim_k2", "noise"), ("grid_oss_sim_k6", "nogod"),
             ("r2_grid_oss_seq_k2", "nogod")],
    "boardoff": [("c16_sweep_board_off", "nogod")],
    "god": [("grid_oss_sim_k2", "god"), ("grid_oss_sim_k6", "god"), ("r2_grid_oss_seq_k2", "god")],
}
TYPES = ("generous", "reciprocator", "hoarder", "defector")
NUM = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


# ------------------------------------------------------------------ replay
@dataclass
class Agent:
    needs: list[int]
    hold: Counter = field(default_factory=Counter)
    burned: set[int] = field(default_factory=set)
    lost: set[int] = field(default_factory=set)
    won: int | None = None
    ever: set[int] = field(default_factory=set)

    def open(self) -> list[int]:
        return [i for i in self.needs if i not in self.burned and i not in self.lost]

    def unheld(self) -> list[int]:
        return [i for i in self.open() if self.hold[i] <= 0]

    def score(self) -> float:
        return sum(1 for i in self.open() if self.hold[i] > 0) / len(self.needs)


@dataclass
class Game:
    cell: str
    src: str            # run/branch
    world: int
    rounds: int
    agents: dict[str, Agent]
    init: dict[str, dict[str, Any]]                     # agent -> {needs, keys}
    gives: list[dict[str, Any]] = field(default_factory=list)
    defects: list[dict[str, Any]] = field(default_factory=list)
    wins: list[dict[str, Any]] = field(default_factory=list)
    acts: dict[str, Counter] = field(default_factory=dict)
    round_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    final: dict[str, float] = field(default_factory=dict)
    powers: list[dict[str, Any]] = field(default_factory=list)
    disclosed: dict[tuple[str, int], bool] = field(default_factory=dict)


def replay(gdir: Path, cell: str, src: str) -> Game:
    ev = [json.loads(ln) for ln in open(gdir / "events.jsonl", encoding="utf-8")]
    meta = json.loads((gdir / "meta.json").read_text())
    out = json.loads((gdir / "outcome.json").read_text())
    start = next(e for e in ev if e["type"] == "game_start")["data"]
    ag = {a: Agent(list(s["needs"]), Counter(k["info_id"] for k in s["keys"])) for a, s in start["agents"].items()}
    init = {a: {"needs": list(s["needs"]), "keys": [k["info_id"] for k in s["keys"]]} for a, s in start["agents"].items()}
    g = Game(cell, src, int(meta["game_idx"]), int(out["rounds_played"]), ag, init,
             acts={a: Counter() for a in ag})
    said: dict[str, list[tuple[int, str | None, set[int]]]] = defaultdict(list)  # sender -> (seq, receiver|None, nums)
    last_change: tuple[str, set[str]] | None = None   # (cause, agents touched) of the latest state change
    pending_mint: dict[str, int] = {}
    for e in ev:
        t, d, r, seq = e["type"], e["data"], e["round"], e["seq"]
        if t == "turn":
            g.acts[d["agent"]]["turns"] += 1
        elif t == "action_result":
            k = d["action"]["kind"]
            a = d["agent"]
            if d.get("ok"):
                g.acts[a][k] += 1
            elif k == "defect" and d.get("reason") == "target_won":
                g.acts[a]["defect_vs_winner"] += 1
        elif t == "message":
            m = d["message"]
            if not m["forged"]:
                nums = {int(x) for x in NUM.findall(m["text"])}
                said[m["sender_shown"]].append((seq, m["receiver"], nums))
                if m["channel"] == "board" and m["sender_shown"] in ag:
                    for i in ag[m["sender_shown"]].needs:
                        if i in nums:
                            g.disclosed[(m["sender_shown"], i)] = True
        elif t == "key_transfer":
            info = int(d["info_id"])
            if d["cause"] == "give":
                fa, ta = d["from_agent"], d["to_agent"]
                giver, recv = ag[fa], ag[ta]
                g.gives.append({
                    "seq": seq, "round": r, "from": fa, "to": ta, "info": info,
                    "giver_needed": info in giver.open() and giver.hold[info] == 1,   # gave away its last copy of a need
                    "recv_needed": info in recv.unheld(),
                    "giver_won": giver.won is not None, "recv_won": recv.won is not None,
                    "prior_from_recv": any(x["from"] == ta and x["to"] == fa for x in g.gives),
                })
                giver.hold[info] -= 1
                recv.hold[info] += 1
                if info in recv.needs:
                    recv.ever.add(info)
                last_change = ("give", {fa, ta})
            elif d["cause"] == "defect_mint":
                pending_mint[d["to_agent"]] = info
                ag[d["to_agent"]].hold[info] += 1
                if info in ag[d["to_agent"]].needs:
                    ag[d["to_agent"]].ever.add(info)
            else:
                raise ValueError(f"unexpected key_transfer cause {d['cause']}")
        elif t == "burn":
            a, tg, x, ok = d["actor"], d["target"], int(d["info_id"]), bool(d["correct"])
            me, vic = ag[a], ag[tg]
            minted = pending_mint.pop(a, None)
            held_x_before = me.hold[x] - (1 if minted == x else 0)
            rec = {"seq": seq, "round": r, "actor": a, "target": tg, "info": x, "correct": ok,
                   "actor_won": me.won is not None, "actor_needs_x": x in me.open(),
                   "useful_mint": ok and x in me.open() and held_x_before <= 0,
                   "victim_held_x": ok and vic.hold[x] > 0,
                   "target_held_x": vic.hold[x] > 0,
                   "lost": d.get("lost_info_id"), "lost_held": False,
                   "informed": any(s < seq and (rv is None or rv == a) and x in nums for s, rv, nums in said[tg])}
            if ok:
                vic.burned.add(x)
            else:
                lost = d.get("lost_info_id")
                if lost is not None:
                    rec["lost_held"] = me.hold[int(lost)] > 0
                    me.lost.add(int(lost))
            g.defects.append(rec)
            last_change = ("burn", {a, tg})
        elif t == "win":
            a = d["agent"]
            ag[a].won = r
            how = "other"
            if last_change is not None and a in last_change[1]:
                how = "receive" if last_change[0] == "give" else (
                    "own_wrong_defect" if g.defects[-1]["actor"] == a and not g.defects[-1]["correct"] else
                    "own_correct_defect" if g.defects[-1]["actor"] == a else "burned_by_other")
            g.wins.append({"agent": a, "round": r, "score": float(d["score"]), "how": how})
        elif t == "round_end":
            g.round_scores[r] = dict(d["scores"])
        elif t == "god_power" and d.get("ok"):
            act = d["action"]
            g.powers.append({"round": r, "power": d["power"],
                             "target": act.get("receiver") if d["power"] == "message_fraud" else act.get("agent"),
                             "spoofed": act.get("sender")})
    g.final = {a: float(o["score"]) for a, o in out["per_agent"].items()}
    for a, s in ag.items():                       # replay must reproduce the logged outcome exactly
        if abs(s.score() - g.final[a]) > 1e-9 or (s.won is None) != (out["per_agent"][a]["won_round"] is None):
            raise AssertionError(f"replay mismatch {gdir} {a}: {s.score()} vs {g.final[a]}")
    return g


def load_cells() -> dict[str, list[Game]]:
    out: dict[str, list[Game]] = {}
    for cell, srcs in CELLS.items():
        out[cell] = []
        for run, br in srcs:
            for gd in sorted((RUNS / run / "games").glob(f"g*-{br}")):
                meta = json.loads((gd / "meta.json").read_text())
                if meta.get("status") != "done" or game_health(gd)["reasons"]:
                    continue
                out[cell].append(replay(gd, cell, f"{run}/{br}"))
    return out


# ------------------------------------------------------------------ stats helpers
def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def cboot(groups: list[list[float]], n_boot: int, rng: random.Random) -> tuple[float, float, float, int]:
    """Mean over pooled items, 95% percentile CI from a bootstrap that resamples games (clusters)."""
    groups = [g for g in groups if g]
    flat = [x for g in groups for x in g]
    if len(groups) < 2:
        return mean(flat), float("nan"), float("nan"), len(flat)
    ms = []
    for _ in range(n_boot):
        s = [x for g in rng.choices(groups, k=len(groups)) for x in g]
        ms.append(mean(s))
    ms.sort()
    return mean(flat), ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot) - 1], len(flat)


def ratio_boot(groups: list[list[tuple[float, float]]], n_boot: int, rng: random.Random) -> tuple[float, float, float]:
    """sum(num)/sum(den) with a game-clustered bootstrap CI."""
    groups = [g for g in groups if g]

    def r(gs):
        n = sum(a for g in gs for a, _ in g)
        dd = sum(b for g in gs for _, b in g)
        return n / dd if dd else float("nan")
    ms = sorted(r(rng.choices(groups, k=len(groups))) for _ in range(n_boot)) if len(groups) > 1 else [float("nan")]
    return r(groups), ms[int(0.025 * len(ms))], ms[int(0.975 * len(ms)) - 1]


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2
        i = j + 1
    return rk


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def icc1(groups: list[list[float]]) -> float:
    """One-way random-effects ICC(1) for equal group size k."""
    k = len(groups[0])
    assert all(len(g) == k for g in groups)
    n = len(groups)
    grand = mean([x for g in groups for x in g])
    msb = k * sum((mean(g) - grand) ** 2 for g in groups) / (n - 1)
    msw = sum((x - mean(g)) ** 2 for g in groups for x in g) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) else float("nan")


def icc_boot(groups: list[list[float]], n_boot: int, rng: random.Random) -> tuple[float, float, float]:
    ms = sorted(icc1(rng.choices(groups, k=len(groups))) for _ in range(n_boot))
    return icc1(groups), ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot) - 1]


def gini(xs: list[float]) -> float:
    n, s = len(xs), sum(xs)
    if not s:
        return 0.0
    return sum(abs(a - b) for a in xs for b in xs) / (2 * n * n * (s / n))


# ------------------------------------------------------------------ per-agent table
def agent_rows(g: Game) -> list[dict[str, Any]]:
    rows = []
    for a in g.agents:
        out_g = [x for x in g.gives if x["from"] == a]
        in_g = [x for x in g.gives if x["to"] == a]
        dfs = [x for x in g.defects if x["actor"] == a]
        recip = sum(x["prior_from_recv"] for x in out_g)
        if dfs:
            typ = "defector"
        elif not out_g:
            typ = "hoarder"
        elif recip >= len(out_g) / 2:
            typ = "reciprocator"
        else:
            typ = "generous"
        won = g.agents[a].won
        rows.append({
            "game": g.src, "world": g.world, "agent": a, "type": typ, "score": g.final[a], "won": won is not None,
            "won_round": won, "gives_out": len(out_g), "gives_in": len(in_g),
            "gives_useful": sum(x["recv_needed"] for x in out_g), "gives_selfharm": sum(x["giver_needed"] for x in out_g),
            "gives_postwin": sum(x["giver_won"] for x in out_g),
            "defects": len(dfs), "defects_correct": sum(x["correct"] for x in dfs),
            "defect_vs_winner": g.acts[a]["defect_vs_winner"],
            "posts": g.acts[a]["post_board"], "dms": g.acts[a]["send_message"], "turns": g.acts[a]["turns"],
            "burned_in": len(g.agents[a].burned), "lost": len(g.agents[a].lost),
        })
    return rows


# ------------------------------------------------------------------ world structure
def world_features(g: Game) -> dict[str, Any]:
    """From the initial deal only. demand[i] = agents needing info i; copies[i] = copies of i in the world."""
    copies: Counter = Counter(i for s in g.init.values() for i in s["keys"])
    demand: Counter = Counter(i for s in g.init.values() for i in s["needs"])
    per_agent = {}
    for a, s in g.init.items():
        contested = sum(1 for i in s["needs"] if demand[i] >= copies[i])
        swaps = 0          # needs for which some holder of the key needs something this agent holds
        for i in s["needs"]:
            holders = [h for h, t in g.init.items() if h != a and i in t["keys"]]
            if any(set(g.init[h]["needs"]) & set(s["keys"]) for h in holders):
                swaps += 1
        wanted = sum(1 for i in s["keys"] if demand[i] > 0)    # own keys someone else needs (bargaining chips)
        per_agent[a] = {"contested": contested, "swaps": swaps, "wanted": wanted}
    return {"n_contested_infos": sum(1 for i in demand if demand[i] >= copies[i]),
            "n_needed_infos": len(demand), "n_dead_keys": sum(c for i, c in copies.items() if demand[i] == 0),
            "per_agent": per_agent}


# ------------------------------------------------------------------ analyses
def analyse(cells: dict[str, list[Game]], n_boot: int, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    R: dict[str, Any] = {"n_games": {c: len(gs) for c, gs in cells.items()},
                         "sources": {c: dict(Counter(g.src for g in gs)) for c, gs in cells.items()}}

    for cell in ("base", "boardoff"):
        gs = cells[cell]
        rows = {g.src + str(g.world): agent_rows(g) for g in gs}
        allrows = [r for rs in rows.values() for r in rs]
        C: dict[str, Any] = {"n_agents": len(allrows)}
        # --- outcome headline
        C["welfare"] = cboot([[r["score"] for r in rs] for rs in rows.values()], n_boot, rng)
        C["win_rate"] = cboot([[float(r["won"]) for r in rs] for rs in rows.values()], n_boot, rng)
        C["gini"] = cboot([[gini([r["score"] for r in rs])] for rs in rows.values()], n_boot, rng)
        C["score_dist"] = dict(sorted(Counter(round(r["score"] * 3) for r in allrows).items()))
        # --- per-agent activity means
        C["per_agent_means"] = {k: mean([r[k] for r in allrows]) for k in
                                ("gives_out", "gives_useful", "gives_selfharm", "gives_postwin", "defects",
                                 "defects_correct", "defect_vs_winner", "posts", "dms", "burned_in", "lost")}
        # --- strategy types
        T: dict[str, Any] = {}
        for typ in TYPES:
            grp = [[r["score"] for r in rs if r["type"] == typ] for rs in rows.values()]
            wgrp = [[float(r["won"]) for r in rs if r["type"] == typ] for rs in rows.values()]
            tr = [r for r in allrows if r["type"] == typ]
            T[typ] = {"n": len(tr), "score": cboot(grp, n_boot, rng), "win": cboot(wgrp, n_boot, rng),
                      "gives_out": mean([r["gives_out"] for r in tr]), "gives_in": mean([r["gives_in"] for r in tr]),
                      "posts": mean([r["posts"] for r in tr]), "dms": mean([r["dms"] for r in tr]),
                      "defects": mean([r["defects"] for r in tr]), "lost": mean([r["lost"] for r in tr])}
        C["types"] = T
        # 2x2 gave x defected
        C["gave_x_defected"] = {f"gave={gv},defected={df}": {
            "n": len(sel := [r for r in allrows if (r["gives_out"] > 0) == gv and (r["defects"] > 0) == df]),
            "score": mean([r["score"] for r in sel]), "win": mean([float(r["won"]) for r in sel])}
            for gv in (True, False) for df in (True, False)}
        # defectors split by whether all their defects were correct
        for lab, f in (("defector_all_correct", lambda r: r["defects"] and r["defects_correct"] == r["defects"]),
                       ("defector_any_wrong", lambda r: r["defects"] and r["defects_correct"] < r["defects"])):
            grp = [[r["score"] for r in rs if f(r)] for rs in rows.values()]
            C[lab] = {"n": sum(map(len, grp)), "score": cboot(grp, n_boot, rng)}
        # --- defects
        D = [d | {"game": g.src + str(g.world)} for g in gs for d in g.defects]
        dg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in D:
            dg[d["game"]].append(d)
        groups = list(dg.values()) + [[] for _ in range(len(gs) - len(dg))]
        pot = [[(1.0 if d["useful_mint"] else 0.0) - (0.0 if d["correct"] else 1.0) for d in gg] for gg in groups]
        imm = [[(1.0 if d["useful_mint"] else 0.0) - (1.0 if (not d["correct"] and d["lost_held"]) else 0.0)
                for d in gg] for gg in groups]
        victim = [[-1.0 if d["correct"] else 0.0 for d in gg] for gg in groups]
        C["defects"] = {
            "n": len(D), "n_games_with_defect": len(dg),
            "p_correct": cboot([[float(d["correct"]) for d in gg] for gg in groups], n_boot, rng),
            "p_useful_mint": cboot([[float(d["useful_mint"]) for d in gg] for gg in groups], n_boot, rng),
            "share_x_is_own_need": mean([float(d["actor_needs_x"]) for d in D]),
            "p_correct_if_x_own_need": mean([float(d["correct"]) for d in D if d["actor_needs_x"]]),
            "n_x_own_need": sum(d["actor_needs_x"] for d in D),
            "share_by_winner": mean([float(d["actor_won"]) for d in D]),
            "share_informed": mean([float(d["informed"]) for d in D]),
            "p_correct_informed": mean([float(d["correct"]) for d in D if d["informed"]]),
            "n_informed": sum(d["informed"] for d in D),
            "p_correct_uninformed": mean([float(d["correct"]) for d in D if not d["informed"]]),
            "wrong_lost_held": mean([float(d["lost_held"]) for d in D if not d["correct"]]),
            "victim_held_x": mean([float(d["victim_held_x"]) for d in D if d["correct"]]),
            "ev_potential": cboot(pot, n_boot, rng),       # need units: +1 useful mint, -1 wrong (need gone)
            "ev_immediate": cboot(imm, n_boot, rng),       # +1 useful mint, -1 wrong AND lost need was held
            "victim_potential": cboot(victim, n_boot, rng),
            "wrong_target_held_x": mean([float(d["target_held_x"]) for d in D if not d["correct"]]),
            "ev_potential_informed": cboot([[(1.0 if d["useful_mint"] else 0.0) - (0.0 if d["correct"] else 1.0)
                                             for d in gg if d["informed"]] for gg in groups], n_boot, rng),
            "ev_potential_uninformed": cboot([[(1.0 if d["useful_mint"] else 0.0) - (0.0 if d["correct"] else 1.0)
                                               for d in gg if not d["informed"]] for gg in groups], n_boot, rng),
            "attempts_vs_winners": sum(r["defect_vs_winner"] for r in allrows),
            "first_round_median": statistics.median([d["round"] for d in D]) if D else None,
        }
        # q = P(an open need is held at game end), the "would have got it anyway" rate
        held = [[float(g.agents[a].hold[i] > 0) for a in g.agents for i in g.agents[a].open()] for g in gs]
        C["q_open_need_held_at_end"] = cboot(held, n_boot, rng)
        q = C["q_open_need_held_at_end"][0]
        disc = [[(1 - q if d["useful_mint"] else 0.0) - (0.0 if d["correct"] else (1.0 if d["lost_held"] else q))
                 for d in gg] for gg in groups]
        C["defects"]["ev_discounted"] = cboot(disc, n_boot, rng)   # gain/loss weighted by "would not / would have got it"
        # disclosure exposure: a need the owner named on the public board before, vs never named
        expo: dict[str, list[list[float]]] = {"named_burned": [], "unnamed_burned": []}
        for g in gs:
            nb, ub = [], []
            for a, s in g.agents.items():
                for i in s.needs:
                    (nb if g.disclosed.get((a, i)) else ub).append(float(i in s.burned))
            expo["named_burned"].append(nb)
            expo["unnamed_burned"].append(ub)
        C["disclosure"] = {"p_burned_if_named_on_board": cboot(expo["named_burned"], n_boot, rng),
                           "p_burned_if_never_named": cboot(expo["unnamed_burned"], n_boot, rng)}
        # --- reciprocity
        pairs_after, pairs_base = [], []
        for g in gs:
            first: dict[tuple[str, str], int] = {}
            for x in g.gives:
                first.setdefault((x["from"], x["to"]), x["seq"])
            ags = list(g.agents)
            pa, pb = [], []
            for i in ags:
                for j in ags:
                    if i == j:
                        continue
                    back = first.get((j, i))
                    if (i, j) in first:
                        pa.append(float(back is not None and back > first[(i, j)]))
                    else:
                        pb.append(float(back is not None))
            pairs_after.append(pa)
            pairs_base.append(pb)
        C["reciprocity"] = {
            "p_back_after_gift": cboot(pairs_after, n_boot, rng),
            "p_gift_no_prior": cboot(pairs_base, n_boot, rng),
            "spearman_out_in": spearman([r["gives_out"] for r in allrows], [r["gives_in"] for r in allrows]),
            "share_gives_reciprocal": mean([float(x["prior_from_recv"]) for g in gs for x in g.gives]),
            "share_gives_useful": mean([float(x["recv_needed"]) for g in gs for x in g.gives]),
            "share_gives_selfharm": mean([float(x["giver_needed"]) for g in gs for x in g.gives]),
            "share_selfharm_prewin": mean([float(x["giver_needed"]) for g in gs for x in g.gives if not x["giver_won"]]),
            "share_selfharm_postwin": mean([float(x["giver_needed"]) for g in gs for x in g.gives if x["giver_won"]]),
            "share_gives_by_winners": mean([float(x["giver_won"]) for g in gs for x in g.gives]),
            "n_gives": sum(len(g.gives) for g in gs),
        }
        # --- winners
        W = []
        for g in gs:
            winsc = {w["agent"]: w["score"] for w in g.wins}
            for a, s in g.agents.items():
                if s.won is None:
                    continue
                pre = [x for x in g.gives if x["from"] == a and not x["giver_won"]]
                post = [x for x in g.gives if x["from"] == a and x["giver_won"]]
                pre_r, post_r = max(1, s.won), max(1, g.rounds - s.won)
                W.append({"game": g.src + str(g.world), "won_round": s.won, "score_at_win": winsc[a],
                          "final": g.final[a], "pre_rate": len(pre) / pre_r, "post_rate": len(post) / post_r,
                          "post": len(post), "post_useful": sum(x["recv_needed"] for x in post),
                          "post_selfharm": sum(x["giver_needed"] for x in post), "post_rounds": g.rounds - s.won,
                          "defects_after": sum(1 for d in g.defects if d["actor"] == a and d["actor_won"])})
        early = [w for w in W if w["won_round"] <= 5]
        C["winners"] = {
            "n": len(W), "n_early": len(early),
            "how": dict(Counter(w["how"] for g in gs for w in g.wins)),
            "score_at_win": mean([w["score_at_win"] for w in W]), "final": mean([w["final"] for w in W]),
            "share_final_below_win": mean([float(w["final"] < w["score_at_win"] - 1e-9) for w in W]),
            "share_final_zero": mean([float(w["final"] == 0) for w in W]),
            "share_win_at_full_3_3": mean([float(abs(w["score_at_win"] - 1) < 1e-9) for w in W]),
            "post_gives_per_winner": mean([w["post"] for w in W]),
            "post_useful_share": (sum(w["post_useful"] for w in W) / max(1, sum(w["post"] for w in W))),
            "post_selfharm_share": (sum(w["post_selfharm"] for w in W) / max(1, sum(w["post"] for w in W))),
            "share_winners_selfharm": mean([float(w["post_selfharm"] > 0) for w in W]),
            "early_pre_rate": mean([w["pre_rate"] for w in early]), "early_post_rate": mean([w["post_rate"] for w in early]),
            "early_share_keep_giving": mean([float(w["post"] > 0) for w in early]),
            "winner_defects_after_win": sum(w["defects_after"] for w in W),
        }
        # welfare decomposition: where did need-slots go at game end
        slots = Counter()
        for g in gs:
            for s in g.agents.values():
                for i in s.needs:
                    slots["burned" if i in s.burned else "lost" if i in s.lost else
                          "held" if s.hold[i] > 0 else "open_unheld_once_held" if i in s.ever
                          else "open_never_held"] += 1
        tot = sum(slots.values())
        C["need_slots_at_end"] = {k: v / tot for k, v in slots.items()} | {"n": tot}
        # --- world structure (initial deal) vs outcome
        feats = {g.src + str(g.world): world_features(g) for g in gs}
        ar_f = []
        for key, rs in rows.items():
            for r in rs:
                ar_f.append(r | feats[key]["per_agent"][r["agent"]])
        C["world"] = {
            "contested_infos_mean": mean([f["n_contested_infos"] for f in feats.values()]),
            "dead_keys_mean": mean([f["n_dead_keys"] for f in feats.values()]),
            "agent_score_by_contested": {k: (len(v), mean(v)) for k, v in sorted(
                _group(ar_f, "contested", "score").items())},
            "agent_score_by_swaps": {k: (len(v), mean(v)) for k, v in sorted(_group(ar_f, "swaps", "score").items())},
            "spearman_score_swaps": spearman([r["swaps"] for r in ar_f], [r["score"] for r in ar_f]),
            "spearman_score_contested": spearman([r["contested"] for r in ar_f], [r["score"] for r in ar_f]),
            "spearman_score_wanted": spearman([r["wanted"] for r in ar_f], [r["score"] for r in ar_f]),
            "spearman_welfare_contested": spearman(
                [feats[k]["n_contested_infos"] for k in rows], [mean([r["score"] for r in rows[k]]) for k in rows]),
        }
        if cell == "base":
            # replicas of the same world (simultaneous cells only: G1 nogod, G1 noise, G3 nogod; k=3 x 16 worlds)
            sim = [g for g in gs if not g.src.startswith("r2_")]
            byw: dict[int, list[Game]] = defaultdict(list)
            for g in sim:
                byw[g.world].append(g)
            wk = sorted(w for w, v in byw.items() if len(v) == 3)
            C["world"]["icc_welfare"] = icc_boot([[mean(list(g.final.values())) for g in byw[w]] for w in wk], n_boot, rng)
            ag_groups = [[g.final[a] for g in byw[w]] for w in wk for a in byw[w][0].agents]
            C["world"]["icc_agent_score"] = icc_boot(ag_groups, n_boot, rng)
            C["world"]["icc_agent_won"] = icc_boot([[float(g.agents[a].won is not None) for g in byw[w]]
                                                    for w in wk for a in byw[w][0].agents], n_boot, rng)
            C["world"]["icc_n_worlds"] = len(wk)
        R[cell] = C

    # ------------------------------------------------------------------ God: targets vs same agent without God
    paired = {}
    for g in cells["base"]:
        if g.src.endswith("/nogod"):
            paired[(g.src.split("/")[0], g.world)] = g
    gd: dict[str, list[float]] = defaultdict(list)
    comply = Counter()
    for g in cells["god"]:
        ng = paired.get((g.src.split("/")[0], g.world))
        if ng is None:
            continue
        for pi, p in enumerate(g.powers):
            role = "fraud_receiver" if p["power"] == "message_fraud" else "cot_target"
            gd[role].append(g.final[p["target"]] - ng.final[p["target"]])
            if pi == 0:      # the r3 fraud: target picked on a state identical to the no-God branch (rounds 1-2 shared)
                gd["first_fraud_receiver_r3"].append(g.final[p["target"]] - ng.final[p["target"]])
                gd["first_fraud_spoofed_r3"].append(g.final[p["spoofed"]] - ng.final[p["spoofed"]])
            if p["power"] == "message_fraud":
                gd["fraud_spoofed_sender"].append(g.final[p["spoofed"]] - ng.final[p["spoofed"]])
                sent = [x for x in g.gives if x["from"] == p["target"] and x["to"] == p["spoofed"]
                        and p["round"] <= x["round"] <= p["round"] + 2]
                comply["fraud"] += 1
                comply["complied"] += bool(sent)
                comply["complied_selfharm"] += any(x["giver_needed"] for x in sent)
                comply["complied_useful_to_spoofed"] += any(x["recv_needed"] for x in sent)
        others = {p["target"] for p in g.powers} | {p["spoofed"] for p in g.powers if p["spoofed"]}
        gd["untouched"].extend(g.final[a] - ng.final[a] for a in g.agents if a not in others)
    R["god"] = {k: {"n": len(v), "mean_diff": cboot([[x] for x in v], n_boot, rng)} for k, v in gd.items()}
    R["god"]["fraud_compliance"] = dict(comply)
    return R


# ------------------------------------------------------------------ scripted-policy simulation (no LLM)
# Idealized play on the real rules (swarm.game.rules.apply) and the same 16 worlds (regenerated from the
# logged world seeds). Every policy sees the full state, i.e. as if every agent posted its needs truthfully
# on the board. Sequential turns, reshuffled order, 2 actions per turn, 20 rounds, N=10.
SIM_POLICIES = ("C", "R", "H", "D", "Dp", "Cw")
SIM_DESC = {
    "C": "cooperator: gives a spare key (not one of its own open needs) to an agent that needs it",
    "R": "grim reciprocator: as C, but never gives to an agent that defected on it or whose defect is public",
    "H": "hoarder: never gives, never defects",
    "D": "opportunist: defects (correctly) whenever a non-winner shares one of its unheld needs, else as C",
    "Dp": "pure defector: as D but never gives",
    "Cw": "naive cooperator: as C, but once it has won it also gives away keys it needs (the LLM pattern)",
}


def _sim_cfg():
    from swarm.config import load
    return load("configs/base.yaml", ["game.n_agents=10", "game.max_rounds=20"]).game


def _sim_world_seeds() -> list[int]:
    seeds = []
    for i in range(16):
        meta = json.loads((RUNS / "grid_oss_sim_k2" / "games" / f"g{i:02d}-nogod" / "meta.json").read_text())
        seeds.append(int(meta["seed"]))
    return seeds


def _sim_game(cfg, wseed: int, policy: dict[str, str], rng: random.Random, p_acc: float = 1.0) -> dict[str, float]:
    from swarm.game import rules
    from swarm.game.world import generate_world
    from swarm.schema import DefectAction, GiveKey, Noop

    st = generate_world(cfg, wseed)
    names = sorted(st.agents)

    def unheld(a):
        held = rules.held_infos(st, a)
        return [i for i in rules.open_needs(st, a) if i not in held]

    def shunned(me, j):
        for h in st.history:
            if h.kind == "defect" and h.actor == j and (h.target == me or h.round <= st.round - 3):
                return True
        return False

    def act(me):
        pol = policy[me]
        s = st.agents[me]
        if pol == "H":
            return Noop()
        if pol in ("D", "Dp"):
            mine = unheld(me)
            opts = [(t, x) for t in names if t != me and st.agents[t].won_round is None
                    for x in mine if x in unheld(t)]
            if opts:
                t, x = rng.choice(opts)
                if rng.random() >= p_acc:                       # a wrong guess: name an agent that does not need x
                    wrong = [u for u in names if u != me and st.agents[u].won_round is None
                             and x not in rules.open_needs(st, u)]
                    if wrong:
                        t = rng.choice(wrong)
                return DefectAction(target=t, info_id=x)
            if pol == "Dp":
                return Noop()
        own = set(rules.open_needs(st, me))
        give_needed = pol == "Cw" and s.won_round is not None
        cands = [(j, kc.info_id) for kc in s.keys for j in names
                 if j != me and (kc.info_id not in own or give_needed) and kc.info_id in unheld(j)
                 and not (pol == "R" and shunned(me, j))]
        if cands:
            j, i = rng.choice(cands)
            return GiveKey(to=j, info_id=i)
        return Noop()

    t = 0
    for r in range(1, cfg.max_rounds + 1):
        st.round = r
        order = names[:]
        rng.shuffle(order)
        for me in order:
            for _ in range(cfg.actions_per_turn):
                st.turn_index = t
                rules.apply(st, me, act(me), cfg)
            t += 1
        if all(rules.is_done(st, a) for a in names):
            break
    return {a: rules.score(st, a) for a in names} | {"_won": {a: st.agents[a].won_round is not None for a in names},
                                                     "_rounds": st.round}


def simulate(n_seeds: int = 10, seed: int = 0) -> dict[str, Any]:
    cfg = _sim_cfg()
    ws = _sim_world_seeds()
    out: dict[str, Any] = {"desc": SIM_DESC, "n_games_per_cell": len(ws) * n_seeds}
    # (1) monomorphic populations: welfare and win rate
    mono = {}
    for pol in SIM_POLICIES:
        rng = random.Random(seed)
        sc, wn, rd = [], [], []
        for w in ws:
            for _ in range(n_seeds):
                res = _sim_game(cfg, w, defaultdict(lambda p=pol: p), rng)
                won = res.pop("_won")
                rd.append(res.pop("_rounds"))
                sc.append(mean(list(res.values())))
                wn.append(mean([float(v) for v in won.values()]))
        mono[pol] = {"welfare": mean(sc), "win_rate": mean(wn), "rounds": mean(rd)}
    out["mono"] = mono
    # (2) invasion: one focal agent of policy B among 9 residents of policy A; focal score, resident mean
    inv: dict[str, dict[str, dict[str, float]]] = {}
    for A in ("C", "R", "H", "D", "Dp"):
        inv[A] = {}
        for B in ("C", "R", "H", "D", "Dp"):
            rng = random.Random(seed)
            fs, rs = [], []
            for w in ws:
                names = sorted(generate_names(cfg, w))
                for k in range(n_seeds):
                    focal = names[k % len(names)]
                    pol = defaultdict(lambda a=A: a)
                    pol[focal] = B
                    res = _sim_game(cfg, w, pol, rng)
                    res.pop("_won")
                    res.pop("_rounds")
                    fs.append(res[focal])
                    rs.append(mean([v for a, v in res.items() if a != focal]))
            inv[A][B] = {"focal": mean(fs), "residents": mean(rs), "focal_sd": statistics.pstdev(fs)}
    out["invasion"] = inv
    # (3) defect accuracy sweep: focal opportunist D with accuracy p, residents C (generous) or H (hoarders)
    sweep: dict[str, dict[str, float]] = {}
    for A in ("C", "H"):
        sweep[A] = {}
        for p in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            rng = random.Random(seed)
            fs = []
            for w in ws:
                names = sorted(generate_names(cfg, w))
                for k in range(n_seeds):
                    focal = names[k % len(names)]
                    pol = defaultdict(lambda a=A: a)
                    pol[focal] = "D"
                    res = _sim_game(cfg, w, pol, rng, p_acc=p)
                    res.pop("_won")
                    res.pop("_rounds")
                    fs.append(res[focal])
            sweep[A][f"{p:.2f}"] = mean(fs)
        sweep[A]["baseline_" + A] = inv[A][A]["focal"]
    out["accuracy_sweep"] = sweep
    return out


def generate_names(cfg, wseed: int) -> list[str]:
    from swarm.game.world import generate_world
    return list(generate_world(cfg, wseed).agents)


def _group(rows: list[dict[str, Any]], key: str, val: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r[val])
    return out


# ------------------------------------------------------------------ figures
def figures(R: dict[str, Any], out: Path) -> None:
    from swarm.viz.style import INK2, plt
    from swarm.viz.style import C as COL

    # 1. score and win rate by strategy type (base and board-off)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    colors = {"generous": COL["give"], "reciprocator": COL["a"], "hoarder": COL["dm"], "defector": COL["defect"]}
    for ax, (metric, lab) in zip(axes, (("score", "final score (share of 3 needs held)"),
                                        ("win", "win rate (ever held all open needs)")), strict=True):
        for ci, cell in enumerate(("base", "boardoff")):
            for ti, typ in enumerate(TYPES):
                t = R[cell]["types"][typ]
                if not t["n"]:
                    continue
                m, lo, hi, n = t[metric]
                x = ti + (ci - 0.5) * 0.36
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]] if lo == lo else None, fmt="o" if cell == "base" else "s",
                            color=colors[typ], mfc=colors[typ] if cell == "base" else "white", ms=7, capsize=3, lw=1.5)
                ax.annotate(f"n={n}", (x, 0.02), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(TYPES)), TYPES)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab, fontsize=10)
    axes[0].plot([], [], "o", color=INK2, label="board on (base, God-free)")
    axes[0].plot([], [], "s", color=INK2, mfc="white", label="board off")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Outcome by behavioural type (association, not causal; 95% CI, bootstrap over games)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gt_score_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. defect EV vs accuracy p (need units): theory + realized (left), scripted simulation (right)
    q = R["base"]["q_open_need_held_at_end"][0]
    ps = [i / 100 for i in range(101)]
    sim = R.get("sim")
    fig, axes = plt.subplots(1, 2 if sim else 1, figsize=(11.5 if sim else 6.4, 4.2), squeeze=False)
    ax = axes[0][0]
    ax.plot(ps, [2 * p - 1 for p in ps], color=INK2, lw=1.6, label="mint covers your unheld need (g=1, c=1): p*=0.50")
    ax.plot(ps, [p * (1 - q) - (1 - p) * q for p in ps], color=COL["a"], lw=1.6, ls="--",
            label=f"discounted by q={q:.2f} (you'd get it anyway): p*={q:.2f}")
    ax.plot(ps, [-(1 - p) for p in ps], color=COL["defect"], lw=1.6, ls=":",
            label="mint is not a need of yours (g=0): never +EV")
    ax.axhline(0, color=INK2, lw=0.8)
    for cell, mk, lab in (("base", "o", "board on"), ("boardoff", "s", "board off")):
        d = R[cell]["defects"]
        p, plo, phi, n = d["p_correct"]
        e, elo, ehi, _ = d["ev_discounted"]
        ax.errorbar(p, e, xerr=[[p - plo], [phi - p]], yerr=[[e - elo], [ehi - e]], fmt=mk, color=COL["defect"],
                    mfc=COL["defect"] if cell == "base" else "white", ms=8, capsize=3, lw=1.4)
        ax.annotate(f"realized, {lab} (n={n})", (p, e), xytext=(8, -14), textcoords="offset points", fontsize=8)
    ax.set_xlabel("p = probability the defect is correct (target really needs the info)")
    ax.set_ylabel("expected change for the defector, in needs\n(+1 = one need gained, -1 = one lost)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Theory lines; realized points use the q-discounted EV", fontsize=10)
    if sim:
        ax = axes[0][1]
        sw = sim["accuracy_sweep"]
        for A, col, lab in (("C", COL["give"], "residents give spares (C)"), ("H", COL["dm"], "residents hoard (H)")):
            xs = sorted(float(k) for k in sw[A] if not k.startswith("baseline"))
            ax.plot(xs, [sw[A][f"{x:.2f}"] for x in xs], "o-", color=col, ms=5, label=f"defector D among {lab}")
            ax.axhline(sw[A]["baseline_" + A], color=col, lw=1, ls="--")
            ax.annotate(f"same seat playing {A}", (0.02, sw[A]["baseline_" + A]), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xlabel("defect accuracy p (scripted, full information)")
        ax.set_ylabel("focal agent's final score")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title(f"Scripted simulation, 16 worlds x {sim['n_games_per_cell'] // 16} orders", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gt_defect_ev.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt(x: Any) -> str:
    if isinstance(x, (tuple, list)) and len(x) == 4 and all(isinstance(v, (int, float)) for v in x):
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}] n={x[3]}"
    if isinstance(x, (tuple, list)) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def dump(d: Any, ind: str = "") -> None:
    for k, v in d.items():
        if isinstance(v, dict) and not all(isinstance(x, (int, float)) for x in v.values()):
            print(f"{ind}{k}:")
            dump(v, ind + "  ")
        else:
            print(f"{ind}{k}: {_fmt(v) if not isinstance(v, dict) else {kk: _fmt(vv) for kk, vv in v.items()}}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="plan/figs")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--sim-seeds", type=int, default=10, help="order seeds per world in the scripted simulation (0 = skip)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    R = analyse(cells, a.boot)
    if a.sim_seeds:
        R["sim"] = simulate(a.sim_seeds)
    (out / "game_theory.json").write_text(json.dumps(R, indent=1, default=str))
    figures(R, out)
    dump(R)


if __name__ == "__main__":
    main()
