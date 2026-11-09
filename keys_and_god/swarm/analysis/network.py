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
  communities  (by_community) planted communities from meta.json["network"]["communities"]
               (sbm / community_hubs), else greedy-modularity communities of the graph (flagged
               planted=False; a complete graph is one community). Gives, true-sender DMs, brokering
               claims (subject's community vs the receiver's) and learned names (known_ids), each
               split within / between communities; per-agent roles: hub (the top-degree agent of its
               community), bridge (any other agent with a neighbour in another community), member;
               per role: agents, cross-community gives / DMs / introductions sent, key relays across
               communities (a copy received from community A given on to community B != A, B != the
               any A != B), score; and the bridge share of all cross-community
               gives and introductions (brokerage).
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
    planted = (meta.get("network") or {}).get("communities")
    return {"communities": planted, "dir": gdir, "label": gdir.name, "branch": meta.get("branch", gdir.name.split("-")[-1]),
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


# ------------------------------------------------------------------ communities
def communities_of(game: dict[str, Any]) -> tuple[dict[str, int], bool]:
    """(agent -> community, planted?): the planted blocks from meta.json, else greedy modularity."""
    if game.get("communities"):
        return {a: int(c) for a, c in game["communities"].items()}, True
    if game["complete"]:
        return {a: 0 for a in game["names"]}, False
    parts = nx.algorithms.community.greedy_modularity_communities(game["graph"])
    return {a: c for c, part in enumerate(sorted(parts, key=min)) for a in part}, False


def roles(game: dict[str, Any], comm: dict[str, int]) -> dict[str, str]:
    """hub = top-degree agent of its community (ties: name); bridge = another agent with a neighbour
    in a different community; member = the rest."""
    g = game["graph"]
    hubs = {}
    for a in sorted(g, key=lambda a: (-g.degree(a), a)):
        hubs.setdefault(comm[a], a)
    hub_set = set(hubs.values())
    return {a: ("hub" if a in hub_set else
                "bridge" if any(comm[b] != comm[a] for b in g[a]) else "member") for a in game["names"]}


def _split(pairs: list[tuple[str, str]], comm: dict[str, int]) -> dict[str, int]:
    w = sum(1 for a, b in pairs if comm[a] == comm[b])
    return {"within": w, "between": len(pairs) - w}


def by_community(game: dict[str, Any], broker: dict[str, Any], diff: dict[str, Any] | None) -> dict[str, Any]:
    """Within- vs between-community trades, DMs, brokering and name learning; bridge brokerage."""
    comm, planted = communities_of(game)
    role = roles(game, comm)
    soc = game["social"]
    gives = [t for t in soc["transfers"] if t["cause"] == "give" and t["f"]]
    dms = [m for m in soc["msgs"] if m["ch"] == "dm" and not m["forged"] and m["r"]]
    intro = [i for i in broker["items"] if i["receiver"]]
    # key relays: X received copy c from Y, later gave c to Z (comm(Y) != comm(Z))
    got_from: dict[tuple[str, str], str] = {}
    relays = []
    for e in game["events"]:
        if e["type"] != "key_transfer" or e["data"]["cause"] != "give":
            continue
        d = e["data"]
        src = got_from.get((d["from_agent"], d["copy_id"]))
        if src is not None:
            relays.append({"relayer": d["from_agent"], "from": src, "to": d["to_agent"],
                           "cross": comm[src] != comm[d["to_agent"]]})
        got_from[(d["to_agent"], d["copy_id"])] = d["from_agent"]
    per = (game["outcome"] or {}).get("per_agent", {})
    rows: dict[str, dict[str, float]] = {}
    for r in ("hub", "bridge", "member"):
        ags = [a for a in game["names"] if role[a] == r]
        rows[r] = {"agents": len(ags),
                   "cross_gives": sum(1 for t in gives if role[t["f"]] == r and comm[t["f"]] != comm[t["to"]]),
                   "cross_dms": sum(1 for m in dms if role[m["s"]] == r and comm[m["s"]] != comm[m["r"]]),
                   "cross_intros": sum(1 for i in intro if role[i["sender"]] == r
                                       and comm[i["subject"]] != comm[i["receiver"]]),
                   "cross_relays": sum(1 for x in relays if x["cross"] and role[x["relayer"]] == r),
                   "mean_score": round(sum(per.get(a, {}).get("score", 0.0) for a in ags) / len(ags), 4) if ags else None}
    cross_g = sum(v["cross_gives"] for v in rows.values())
    cross_i = sum(v["cross_intros"] for v in rows.values())
    nonmember = ("hub", "bridge")
    out = {"planted": planted, "n_communities": len(set(comm.values())),
           "gives": _split([(t["f"], t["to"]) for t in gives], comm),
           "dms": _split([(m["s"], m["r"]) for m in dms], comm),
           "intros": _split([(i["subject"], i["receiver"]) for i in intro], comm),
           "key_relays": {"n": len(relays), "cross": sum(x["cross"] for x in relays)},
           "roles": rows,
           "bridge_share_cross_gives": round(sum(rows[r]["cross_gives"] for r in nonmember) / cross_g, 4) if cross_g else None,
           "bridge_share_cross_intros": round(sum(rows[r]["cross_intros"] for r in nonmember) / cross_i, 4) if cross_i else None}
    if diff is not None:
        out["names_learned"] = _split([(x["agent"], x["name"]) for x in diff["learned"]], comm)
        n = len(game["names"])
        fin = game["known"].get(max(game["known"]), {}) if game["known"] else {}
        pw = sum(1 for a in game["names"] for b in game["names"] if a != b and comm[a] == comm[b])
        pb = n * (n - 1) - pw
        kw = sum(1 for a, ks in fin.items() for b in ks if comm[a] == comm[b])
        kb = sum(1 for a, ks in fin.items() for b in ks if comm[a] != comm[b])
        out["final_coverage"] = {"within": round(kw / pw, 4) if pw else None, "between": round(kb / pb, 4) if pb else None}
    return out


def _sum_community(per_game: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals of by_community over the games of a run."""
    tot: dict[str, Any] = {}
    for k in ("gives", "dms", "intros", "names_learned"):
        xs = [g[k] for g in per_game if k in g]
        if xs:
            tot[k] = {s: sum(x[s] for x in xs) for s in ("within", "between")}
    tot["key_relays"] = {s: sum(g["key_relays"][s] for g in per_game) for s in ("n", "cross")}
    tot["roles"] = {r: {c: sum(g["roles"][r][c] for g in per_game)
                        for c in ("agents", "cross_gives", "cross_dms", "cross_intros", "cross_relays")}
                    for r in ("hub", "bridge", "member")}
    cg = sum(v["cross_gives"] for v in tot["roles"].values())
    tot["bridge_share_cross_gives"] = (round((tot["roles"]["hub"]["cross_gives"] + tot["roles"]["bridge"]["cross_gives"]) / cg, 4)
                                       if cg else None)
    tot["planted"] = all(g["planted"] for g in per_game) if per_game else None
    return tot


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
    comms: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for g in games:
        b = brokering(g)
        d = diffusion(g)
        t = trades_by_distance(g)
        c = by_community(g, b, d)
        comms[g["branch"]].append(c)
        trades[g["branch"]].update(t)
        rows += centrality_rows(g, stats_by_idx[g["game_idx"]], b)
        if d:
            curves[g["branch"]].append(d["coverage_by_round"])
        per_game[g["label"]] = {"mode": g["mode"], "brokering": {k: v for k, v in b.items() if k != "items"},
                                "trades_by_distance": t, "communities": c,
                                "diffusion": None if d is None else {k: v for k, v in d.items() if k != "learned"}}
    mean_curve = {br: {r: round(sum(c.get(r, c[max(c)]) for c in cs) / len(cs), 4)
                       for r in sorted({r for c in cs for r in c})} for br, cs in curves.items()}
    graphs = {str(i): {k: v for k, v in s.items() if k not in ("degree", "betweenness")} for i, s in stats_by_idx.items()}
    return {"run": str(run_dir), "n_games": len(games), "graphs": graphs, "coverage_by_round": mean_curve,
            "trades_by_distance": {br: dict(sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))
                                   for br, c in trades.items()},
            "communities": {br: _sum_community(cs) for br, cs in comms.items()},
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
    for br, c in res["communities"].items():
        print(f"  communities ({br}, planted={c['planted']}): gives {c['gives']} dms {c['dms']} "
              f"intros {c['intros']} names {c.get('names_learned')} key relays {c['key_relays']} "
              f"bridge share of cross gives {c['bridge_share_cross_gives']}")
        for r, v in c["roles"].items():
            print(f"    {r:6s} {v}")
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
