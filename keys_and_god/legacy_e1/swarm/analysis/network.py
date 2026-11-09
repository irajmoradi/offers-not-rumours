"""Network analysis of finished Keys & God runs (game.network; Track 2). No LLM calls, read-only.

    PYTHONNOUSERSITE=1 python -m swarm.analysis.network runs/<run_id> [more runs] [--json out.json]

Inputs: runs/<id>/games/<label>/{events,snapshots}.jsonl + meta.json (branch, game_idx). The graph
is read off game_start (AgentState.contacts); a run without game.network is analysed as a complete
graph, so the same numbers exist for the baseline. Works on FakeLLM dry runs.
Outputs (JSON-able dict, printed as a summary):
  graph        per world (game_idx): n, edges, mean degree, degree histogram, average clustering,
               average shortest path, diameter; per agent degree and betweenness.
  diffusion    (known_ids) per branch: mean share of the other agents that know each name, per round
               (the S-curve), and per learned (agent, name): the round, the graph hop distance between
               them at the start, and the channel (dm_sender, dm_mention, board_author,
               board_mention, interaction = a gift / burn / defect notice).
  brokering    messages naming a third agent's keys or needs (swarm.analysis.social claims
               needs_claim / holds_claim): count, the hop distance subject -> receiver (>= 2 = the
               information travelled beyond the subject's neighbourhood), and relays (the sender had
               itself been told about that subject by someone other than the subject).
  trades       gives (key_transfer cause=give) by graph distance giver -> receiver.
  centrality   per (game, agent): degree, betweenness, broker messages sent, won, score; Spearman
               correlations of degree / betweenness with brokering, wins and score.
  god          per God intervention: victim, its degree / betweenness, and the victim's score and the
               welfare in the god branch minus the paired nogod branch (same game_idx).
Imports: networkx, swarm.analysis.social (claim rules), swarm.game.network (name matching).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import networkx as nx

from swarm.analysis import social
from swarm.game.network import LEARN_NOTICE_KINDS, names_in_text

BROKER_CATS = ("needs_claim", "holds_claim")


# ------------------------------------------------------------------ loading
def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_game(gdir: Path) -> dict[str, Any]:
    """One game: names, graph, events, per-round known sets, outcome, branch / game_idx."""
    events = _jsonl(gdir / "events.jsonl")
    meta = json.loads((gdir / "meta.json").read_text()) if (gdir / "meta.json").exists() else {}
    start = events[0]["data"]
    names = sorted(start["agents"])
    contacts = {a: s.get("contacts") for a, s in start["agents"].items()}
    complete = all(c is None for c in contacts.values())
    graph = nx.complete_graph(names) if complete else nx.Graph()
    if not complete:
        graph.add_nodes_from(names)
        graph.add_edges_from((a, b) for a, nb in contacts.items() for b in nb or [])
    known = {s["round"]: {a: set(x["known"]) for a, x in s["state"]["agents"].items() if x.get("known") is not None}
             for s in _jsonl(gdir / "snapshots.jsonl")}
    end = next((e["data"]["outcome"] for e in reversed(events) if e["type"] == "game_end"), None)
    return {"dir": gdir, "label": gdir.name, "branch": meta.get("branch", gdir.name.split("-")[-1]),
            "game_idx": meta.get("game_idx", gdir.name.split("-")[0]), "names": names, "graph": graph,
            "complete": complete, "events": events, "known": known, "outcome": end,
            "mode": "known_ids" if any(start["agents"][a].get("known") is not None for a in names)
            else ("contacts" if not complete else "none"),
            "social": social.load_game(str(gdir / "events.jsonl"))}


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    """Every game dir of a run (sorted by label)."""
    return [load_game(g) for g in sorted((run_dir / "games").iterdir()) if (g / "events.jsonl").exists()]


# ------------------------------------------------------------------ graph stats
def graph_stats(g: nx.Graph) -> dict[str, Any]:
    """Degree distribution, clustering, path lengths, per-agent degree and betweenness."""
    deg = dict(g.degree())
    conn = nx.is_connected(g) if len(g) else False
    return {"n": g.number_of_nodes(), "edges": g.number_of_edges(),
            "mean_degree": round(2 * g.number_of_edges() / max(len(g), 1), 3),
            "degree_hist": dict(sorted(collections.Counter(deg.values()).items())),
            "clustering": round(nx.average_clustering(g), 3),
            "avg_path": round(nx.average_shortest_path_length(g), 3) if conn else None,
            "diameter": nx.diameter(g) if conn else None, "connected": conn,
            "degree": deg, "betweenness": {a: round(b, 4) for a, b in nx.betweenness_centrality(g).items()}}


def _dist(g: nx.Graph) -> dict[str, dict[str, int]]:
    return {a: dict(d) for a, d in nx.all_pairs_shortest_path_length(g)}


# ------------------------------------------------------------------ diffusion (known_ids)
def _channels(agent: str, obs: dict[str, Any], names: list[str]) -> dict[str, str]:
    """name -> first channel it appears through in one observation (the learning rule's sources)."""
    out: dict[str, str] = {}
    for box, tag in (("inbox", "dm"), ("board", "board")):
        for m in obs.get(box, []):
            if m["sender"] in names:
                out.setdefault(m["sender"], f"{tag}_{'sender' if tag == 'dm' else 'author'}")
            for n in sorted(names_in_text(m["text"], names)):
                out.setdefault(n, f"{tag}_mention")
    for nt in obs.get("notices", []):
        if nt["kind"] in LEARN_NOTICE_KINDS:
            for n in sorted(names_in_text(nt["text"], names)):
                out.setdefault(n, "interaction")
    out.pop(agent, None)
    return out


def diffusion(game: dict[str, Any]) -> dict[str, Any] | None:
    """S-curve of name knowledge and per-learning hop distance / channel (None unless known_ids)."""
    if game["mode"] != "known_ids":
        return None
    names, n = game["names"], len(game["names"])
    curve = {r: round(sum(len(k) for k in ks.values()) / (n * (n - 1)), 4) for r, ks in sorted(game["known"].items())}
    dist = _dist(game["graph"])
    known = {a: set(k) for a, k in game["known"].get(0, {}).items()}
    learned: list[dict[str, Any]] = []
    for e in game["events"]:
        if e["type"] != "turn":
            continue
        a = e["data"]["agent"]
        for name, ch in sorted(_channels(a, e["data"]["obs"], names).items()):
            if name not in known[a]:
                known[a].add(name)
                learned.append({"agent": a, "name": name, "round": e["round"], "hops": dist[a].get(name),
                                "channel": ch})
    return {"coverage_by_round": curve, "learned": learned,
            "hops_hist": dict(sorted(collections.Counter(x["hops"] for x in learned).items())),
            "channel_hist": dict(collections.Counter(x["channel"] for x in learned).most_common())}


# ------------------------------------------------------------------ brokering / trades / centrality
def brokering(game: dict[str, Any]) -> dict[str, Any]:
    """Messages about a third agent's keys / needs: hop distance subject -> receiver, relays."""
    dist = _dist(game["graph"])
    told: dict[str, set[str]] = collections.defaultdict(set)    # agent -> subjects it was told about
    items = []
    for c in sorted(social.claims_of(game["social"]), key=lambda c: c["seq"]):
        if c["cat"] not in BROKER_CATS or c["m"].get("forged"):
            continue
        s, r, z = c["sender"], c["r"], c["subj"]
        relay = z in told[s]
        hops = dist[z].get(r) if r else None                    # board posts: no single receiver
        items.append({"round": c["round"], "sender": s, "receiver": r, "subject": z, "cat": c["cat"],
                      "channel": c["ch"], "hops_subject_receiver": hops, "relay": relay})
        if r:
            told[r].add(z)
    return {"n": len(items), "relays": sum(i["relay"] for i in items),
            "multi_hop": sum(1 for i in items if (i["hops_subject_receiver"] or 0) >= 2),
            "hops_hist": dict(sorted(collections.Counter(i["hops_subject_receiver"] for i in items
                                                         if i["receiver"]).items())),
            "by_sender": dict(collections.Counter(i["sender"] for i in items)), "items": items}


def trades_by_distance(game: dict[str, Any]) -> dict[int, int]:
    """Gives by graph distance between giver and receiver."""
    dist = _dist(game["graph"])
    return dict(sorted(collections.Counter(dist[t["f"]].get(t["to"]) for t in game["social"]["transfers"]
                                           if t["cause"] == "give").items()))


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    import pandas as pd  # Pearson on average ranks = Spearman (no scipy)

    return round(float(pd.Series(xs).rank().corr(pd.Series(ys).rank())), 3)


def centrality_rows(game: dict[str, Any], stats: dict[str, Any], broker: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per agent: centrality, brokering, outcome."""
    per = (game["outcome"] or {}).get("per_agent", {})
    return [{"game": game["label"], "agent": a, "degree": stats["degree"][a], "betweenness": stats["betweenness"][a],
             "broker_msgs": broker["by_sender"].get(a, 0), "won": per.get(a, {}).get("won_round") is not None,
             "score": per.get(a, {}).get("score", 0.0)} for a in game["names"]]


def centrality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman correlations of degree / betweenness with brokering, wins and score."""
    out: dict[str, Any] = {"n_rows": len(rows)}
    for c in ("degree", "betweenness"):
        for y in ("broker_msgs", "won", "score"):
            out[f"{c}~{y}"] = _spearman([r[c] for r in rows], [float(r[y]) for r in rows])
    return out


# ------------------------------------------------------------------ God
def god_effects(games: list[dict[str, Any]], stats_by_idx: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ok God power: the victim's centrality and its score / the welfare, god minus nogod."""
    by = {(g["game_idx"], g["branch"]): g for g in games}
    out = []
    for g in games:
        if g["branch"] != "god":
            continue
        ctrl = by.get((g["game_idx"], "nogod"))
        st = stats_by_idx[g["game_idx"]]
        for e in g["events"]:
            if e["type"] != "god_power" or not e["data"]["ok"]:
                continue
            act = e["data"]["action"]
            victim = act.get("receiver") or act.get("agent")
            row = {"game": g["label"], "round": e["round"], "power": e["data"]["power"], "victim": victim,
                   "impersonated": act.get("sender"), "degree": st["degree"].get(victim),
                   "betweenness": st["betweenness"].get(victim)}
            if ctrl is not None and g["outcome"] and ctrl["outcome"]:
                row["victim_score_delta"] = round(g["outcome"]["per_agent"][victim]["score"]
                                                  - ctrl["outcome"]["per_agent"][victim]["score"], 4)
                row["welfare_delta"] = round(g["outcome"]["welfare"] - ctrl["outcome"]["welfare"], 4)
            out.append(row)
    return out


# ------------------------------------------------------------------ run summary
def analyse_run(run_dir: Path) -> dict[str, Any]:
    """Everything above for one run dir."""
    games = load_run(run_dir)
    stats_by_idx: dict[Any, dict[str, Any]] = {}
    for g in games:
        stats_by_idx.setdefault(g["game_idx"], graph_stats(g["graph"]))
    rows: list[dict[str, Any]] = []
    per_game: dict[str, Any] = {}
    curves: dict[str, list[dict[int, float]]] = collections.defaultdict(list)
    trades: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for g in games:
        b = brokering(g)
        d = diffusion(g)
        t = trades_by_distance(g)
        trades[g["branch"]].update(t)
        rows += centrality_rows(g, stats_by_idx[g["game_idx"]], b)
        if d:
            curves[g["branch"]].append(d["coverage_by_round"])
        per_game[g["label"]] = {"mode": g["mode"], "brokering": {k: v for k, v in b.items() if k != "items"},
                                "trades_by_distance": t,
                                "diffusion": None if d is None else {k: v for k, v in d.items() if k != "learned"}}
    mean_curve = {br: {r: round(sum(c.get(r, c[max(c)]) for c in cs) / len(cs), 4)
                       for r in sorted({r for c in cs for r in c})} for br, cs in curves.items()}
    graphs = {str(i): {k: v for k, v in s.items() if k not in ("degree", "betweenness")} for i, s in stats_by_idx.items()}
    return {"run": str(run_dir), "n_games": len(games), "graphs": graphs, "coverage_by_round": mean_curve,
            "trades_by_distance": {br: dict(sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))
                                   for br, c in trades.items()},
            "centrality": centrality_summary(rows), "god": god_effects(games, stats_by_idx), "games": per_game}


def _print(res: dict[str, Any]) -> None:
    print(f"== {res['run']} ({res['n_games']} games)")
    for i, s in res["graphs"].items():
        print(f"  world {i}: n={s['n']} m={s['edges']} <k>={s['mean_degree']} hist={s['degree_hist']} "
              f"C={s['clustering']} L={s['avg_path']} D={s['diameter']}")
    for br, c in res["coverage_by_round"].items():
        print(f"  name coverage ({br}): " + " ".join(f"r{r}={v:.2f}" for r, v in c.items()))
    print(f"  trades by distance: {res['trades_by_distance']}")
    tot = collections.Counter()
    for gm in res["games"].values():
        tot.update({k: gm["brokering"][k] for k in ("n", "relays", "multi_hop")})
    print(f"  brokering msgs: {dict(tot)}")
    print(f"  centrality: {res['centrality']}")
    for row in res["god"]:
        print(f"  god {row['game']} r{row['round']} {row['power']} victim={row['victim']} deg={row['degree']} "
              f"btw={row['betweenness']} d_score={row.get('victim_score_delta')} d_welfare={row.get('welfare_delta')}")


def main(argv: list[str] | None = None) -> None:
    """CLI: print a summary per run; --json writes all results."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = [analyse_run(Path(r)) for r in a.runs]
    for r in res:
        _print(r)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
