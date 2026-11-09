"""Dynamic edges (game.network.connect) in finished Keys & God runs. No LLM calls, read-only.

    PYTHONNOUSERSITE=1 python -m swarm.analysis.network_dynamics runs/<run_id> [more runs] [--json out.json]

Inputs: runs/<id>/games/<label>/events.jsonl (+ meta.json), loaded with swarm.analysis.network.load_game
(its `graph` is the START graph, read off game_start). Edges formed during the game come from the
`edge_added` events (actor, target, mutual; round on the event); the graph at any time is the start
graph plus the edges added so far. A run without connect gives empty numbers (n_edges_added = 0).
Outputs per game (JSON-able dict) and pooled per run:
  attempts     connect action_results: ok, refused by reason; turns lost to the cost.
  timeline     new edges per round.
  edges        per new (non-mutual) edge: round, actor, target, degree of each at that time, whether the
               target was a hub (degree in the top 10% of the current graph), triadic closure (they
               already shared a neighbour), the start-graph hop distance, and the key motive: did the
               target hold a key for one of the actor's missing needs (and the reverse) at that time.
  degrees      edges / mean / max degree, degree histogram, clustering, mean shortest path: start vs end.
  payoff       per new edge: gives and DMs along it afterwards (either way), gives that filled a need of
               the actor, rounds to the first give, whether the actor won after connecting.
  agents       connectors vs non-connectors (n, win rate, mean score); relay (a key passed on by an
               agent that did not need it, swarm.analysis.network_compare.relays) vs connect per agent:
               only relayed / only connected / both / neither.
Imports: networkx, swarm.analysis.network (loading, graph stats), swarm.analysis.network_compare (relays).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import networkx as nx

from swarm.analysis.network import graph_stats, load_run
from swarm.analysis.network_compare import relays

HUB_QUANTILE = 0.9


def _degree_summary(g: nx.Graph) -> dict[str, Any]:
    st = graph_stats(g)
    return {k: st[k] for k in ("edges", "mean_degree", "degree_hist", "clustering", "avg_path")} | {
        "max_degree": max((d for _, d in g.degree()), default=0)}


def _is_hub(g: nx.Graph, node: str) -> bool:
    degs = sorted(d for _, d in g.degree())
    return bool(degs) and g.degree(node) >= degs[min(len(degs) - 1, int(HUB_QUANTILE * len(degs)))]


def analyse_game(game: dict[str, Any]) -> dict[str, Any]:
    """Connect attempts, edge timeline, who connects to whom, degrees before/after, payoff, agents."""
    events, names = game["events"], game["names"]
    start = events[0]["data"]["agents"]
    g0: nx.Graph = game["graph"]
    g = g0.copy()
    dist0 = {a: dict(d) for a, d in nx.all_pairs_shortest_path_length(g0)} if not game["complete"] else {}
    hold = {a: {k["copy_id"]: k["info_id"] for k in s["keys"]} for a, s in start.items()}
    closed: dict[str, set[int]] = {a: set() for a in names}            # burned + lost needs
    needs = {a: list(s["needs"]) for a, s in start.items()}

    def missing(a: str) -> set[int]:
        return {i for i in needs[a] if i not in closed[a]} - set(hold[a].values())

    attempts: collections.Counter[str] = collections.Counter()
    turns_lost, mutual = 0, 0
    edges: list[dict[str, Any]] = []
    per_round: collections.Counter[int] = collections.Counter()
    gives: list[tuple[int, str, str, int]] = []                        # (round, from, to, info)
    dms: list[tuple[int, str, str]] = []
    for e in events:
        typ, d, r = e["type"], e["data"], e["round"]
        if typ == "action_result" and d["action"]["kind"] == "connect":
            attempts["ok" if d["ok"] else (d.get("reason") or "invalid")] += 1
        elif typ == "edge_added":
            a, b = d["actor"], d["target"]
            turns_lost += max(0, d["skip_until_round"] - r)
            if d.get("mutual"):
                mutual += 1
                continue
            per_round[r] += 1
            edges.append({
                "round": r, "actor": a, "target": b, "deg_actor": g.degree(a), "deg_target": g.degree(b),
                "target_hub": _is_hub(g, b), "triadic": bool(set(g[a]) & set(g[b])),
                "start_dist": dist0.get(a, {}).get(b),
                "target_holds_actor_need": bool(missing(a) & set(hold[b].values())),
                "actor_holds_target_need": bool(missing(b) & set(hold[a].values()))})
            g.add_edge(a, b)
        elif typ == "key_transfer":
            f, t = d.get("from_agent"), d["to_agent"]
            if f in hold:
                hold[f].pop(d["copy_id"], None)
            if d["cause"] == "give":
                gives.append((r, f, t, d["info_id"]))
            hold[t][d["copy_id"]] = d["info_id"]
        elif typ == "burn":
            if d["correct"]:
                closed[d["target"]].add(d["info_id"])
            if d.get("lost_info_id") is not None:
                closed[d["actor"]].add(d["lost_info_id"])
        elif typ == "message" and d["message"]["channel"] == "dm":
            m = d["message"]
            dms.append((r, m["sender_shown"], m["receiver"]))
    out = game["outcome"] or {"per_agent": {}}
    won = {a: o.get("won_round") for a, o in out["per_agent"].items()}
    score = {a: o.get("score", 0.0) for a, o in out["per_agent"].items()}
    for ed in edges:
        a, r0, pair = ed["actor"], ed["round"], {ed["actor"], ed["target"]}
        gv = [x for x in gives if x[0] >= r0 and {x[1], x[2]} == pair]
        ed["gives_after"] = len(gv)
        ed["gives_filling_actor_need"] = sum(1 for x in gv if x[2] == a and x[3] in needs[a])
        ed["dms_after"] = sum(1 for x in dms if x[0] >= r0 and {x[1], x[2]} == pair)
        ed["rounds_to_first_give"] = (gv[0][0] - r0) if gv else None
        ed["actor_won_after"] = won.get(a) is not None and won[a] >= r0
    connectors = {ed["actor"] for ed in edges}
    rel = relays({"events": events}, dist0)
    relayers = {x["relay"] for x in rel["relay_events"]}

    def grp(xs: set[str]) -> dict[str, Any]:
        xs = {x for x in xs if x in score}
        return {"n": len(xs), "win_rate": (sum(won[x] is not None for x in xs) / len(xs)) if xs else None,
                "mean_score": (sum(score[x] for x in xs) / len(xs)) if xs else None}

    return {
        "label": game["label"], "branch": game["branch"], "game_idx": game["game_idx"],
        "attempts": dict(attempts), "turns_lost_to_cost": turns_lost, "n_edges_added": len(edges),
        "n_mutual": mutual, "timeline": {r: per_round[r] for r in sorted(per_round)}, "edges": edges,
        "degrees": {"start": _degree_summary(g0), "end": _degree_summary(g)},
        "agents": {"connectors": grp(connectors), "non_connectors": grp(set(names) - connectors),
                   "relay_vs_connect": {"only_relayed": len(relayers - connectors),
                                        "only_connected": len(connectors - relayers),
                                        "both": len(relayers & connectors),
                                        "neither": len(set(names) - relayers - connectors)},
                   "n_relay_events": rel["n_relay_events"]},
    }


def _share(edges: list[dict[str, Any]], key: str) -> float | None:
    vals = [bool(e[key]) for e in edges if e.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def pool(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level summary over every game."""
    edges = [e for g in games for e in g["edges"]]
    attempts: collections.Counter[str] = collections.Counter()
    for g in games:
        attempts.update(g["attempts"])
    dist = collections.Counter(str(e["start_dist"]) for e in edges)

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None
    return {"games": len(games), "attempts": dict(attempts), "n_edges_added": len(edges),
            "n_mutual": sum(g["n_mutual"] for g in games),
            "turns_lost_to_cost": sum(g["turns_lost_to_cost"] for g in games),
            "share_target_hub": _share(edges, "target_hub"), "share_triadic": _share(edges, "triadic"),
            "share_target_holds_actor_need": _share(edges, "target_holds_actor_need"),
            "share_with_give_after": _share(edges, "gives_after"),
            "share_filled_actor_need": _share(edges, "gives_filling_actor_need"),
            "share_actor_won_after": _share(edges, "actor_won_after"),
            "start_dist_hist": dict(sorted(dist.items())),
            "mean_degree_start": mean([g["degrees"]["start"]["mean_degree"] for g in games]),
            "mean_degree_end": mean([g["degrees"]["end"]["mean_degree"] for g in games])}


def analyse_run(run_dir: Path) -> dict[str, Any]:
    """Every game of a run plus the pooled summary."""
    games = [analyse_game(g) for g in load_run(run_dir)]
    return {"run": str(run_dir), "pooled": pool(games), "games": games}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Dynamic edges (game.network.connect) in finished runs")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write the full result here")
    args = ap.parse_args(argv)
    res = [analyse_run(r) for r in args.runs]
    for x in res:
        p = x["pooled"]
        print(f"== {x['run']}: {p['games']} games, {p['n_edges_added']} new edges (+{p['n_mutual']} mutual), "
              f"attempts {p['attempts']}, turns lost {p['turns_lost_to_cost']}")
        print(f"   mean degree {p['mean_degree_start']} -> {p['mean_degree_end']}; target hub {p['share_target_hub']}, "
              f"triadic {p['share_triadic']}, target holds an actor need {p['share_target_holds_actor_need']}, "
              f"give after {p['share_with_give_after']}, actor won after {p['share_actor_won_after']}; "
              f"start distance {p['start_dist_hist']}")
        for g in x["games"]:
            print(f"   {g['label']}: timeline {g['timeline']} connectors {g['agents']['connectors']} "
                  f"others {g['agents']['non_connectors']} relay/connect {g['agents']['relay_vs_connect']}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
