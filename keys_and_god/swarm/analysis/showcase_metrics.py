"""Showcase metrics (plan/SHOWCASE.md, plan/SHOWCASE_RESULTS_TEMPLATE.md): relaying requires trust; does an
adversary break it? Reads one or more run dirs (games/*/events.jsonl, snapshots.jsonl, meta.json,
outcome.json); no LLM.

    PYTHONNOUSERSITE=1 PYTHONPATH=. python -m swarm.analysis.showcase_metrics runs/showcase100_sim \
        runs/showcase100_control_sim [--out plan/figs/showcase_metrics.json]

Per game (= one world x branch), all from the event stream (copy ids make every key traceable):
  relay / trust game ("transit" give = a give to an agent that does not have the info as an open, unheld
  need; a chain = a copy that made >= 1 transit give):
    chain_starts            copies given in transit at least once
    chain_completions       of those, copies that later filled a need (scoring.DeliveryLedger fills)
    transit_gives           all transit gives;  hops_per_completion
    stalled_keys            transit copies held by a non-needer at game end
    misdirected_gives       transit gives that did not move the copy closer (live graph) to any agent that
                            then had it as an open unheld need, or with no such agent at all
    orphaned_by_burn        transit copies whose info had an open needer at the copy's first transit give,
                            and every such needer had that need burned / lost by the end
    god_extracted           gives by a forged-DM receiver to the shown sender within 2 rounds of the
                            forgery (god_power ok), split by whether the forged text asks to relay /
                            forward / pass a key ("relay_ask") or not
    defects_on_carriers     burn events whose target held >= 1 in-transit copy at that moment (and correct)
    delivery_credit_total, fills, fills_via_relay, mean_payees, unpaid_padding (earlier holders beyond d)
  connect (game.network.connect):
    edges_per_round, cross_community_edges, connect_attempts / refused (by reason)
    learned_from            per new edge: how the actor had heard of the target before connecting:
                            "intro_dm" (a DM from a third party naming it), "board" (a board post naming it or
                            by it), "notice", "contact" (already a contact = mutual race), "unknown"
    need_hops_by_round      mean hops (live graph) from each open, unheld need to its nearest holder
    degree_start / degree_end (mean, max); mean_path_start / mean_path_end (networkx, connected graph)
Summary: the same per branch (summed / averaged over worlds), plus per-world showcase - control deltas.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from swarm.game.scoring import DeliveryLedger

RELAY_ASK = re.compile(r"\b(relay|forward|pass (it|this|that|them|the key)|pass on|hand (it )?on)\b", re.I)
EXTRACT_WINDOW = 2


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _ns(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{"lost_info_id": None, "correct": False, **d})


def _bfs_all(adj: dict[str, set[str]], src: str) -> dict[str, int]:
    dist, front = {src: 0}, [src]
    while front:
        nxt = []
        for u in front:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        front = nxt
    return dist


def _mean_path(adj: dict[str, set[str]]) -> float | None:
    tot, n = 0, 0
    for a in adj:
        d = _bfs_all(adj, a)
        if len(d) < len(adj):
            return None
        tot += sum(d.values())
        n += len(d) - 1
    return round(tot / max(n, 1), 4)


def _deg(adj: dict[str, set[str]]) -> dict[str, float]:
    ds = [len(v) for v in adj.values()]
    return {"mean": round(statistics.mean(ds), 3), "max": max(ds)}


def game_metrics(gdir: Path, upto: int | None = None) -> dict[str, Any]:
    """All showcase metrics of one game dir (upto: only events / snapshots of rounds <= upto; a live
    run's interim)."""
    events = _jsonl(gdir / "events.jsonl")
    if upto is not None:
        events = [e for e in events if e["round"] <= upto]
    meta = json.loads((gdir / "meta.json").read_text())
    out_path = gdir / "outcome.json"
    outcome = json.loads(out_path.read_text()) if (out_path.exists() and upto is None) else {}
    comm = (meta.get("network") or {}).get("communities") or {}
    start = events[0]["data"]["agents"]
    adj = {a: set(s.get("contacts") or []) for a, s in start.items()}
    adj0 = {a: set(v) for a, v in adj.items()}
    open_ = {a: set(s["needs"]) - set(s.get("burned", [])) - set(s.get("lost", [])) for a, s in start.items()}
    where: dict[str, str] = {}
    info: dict[str, int] = {}
    for a, s in start.items():
        for k in s["keys"]:
            where[k["copy_id"]], info[k["copy_id"]] = a, k["info_id"]
    won: set[str] = set()
    led = DeliveryLedger(1.0)
    led.start({a: SimpleNamespace(contacts=s.get("contacts"), needs=s["needs"], burned=s.get("burned", []),
                                  lost=s.get("lost", []), keys=[SimpleNamespace(**k) for k in s["keys"]])
               for a, s in start.items()})

    def needs_it(a: str, i: int, cid: str) -> bool:
        return (i in open_[a] and a not in won
                and not any(h == a and c != cid and info.get(c) == i for c, h in where.items()))

    transit: dict[str, int] = collections.Counter()           # copy -> transit gives
    first_transit_needers: dict[str, set[tuple[str, int]]] = {}
    m = collections.Counter()
    forged: list[dict[str, Any]] = []
    edges_by_round: collections.Counter[int] = collections.Counter()
    cross_edges = 0
    refused: collections.Counter[str] = collections.Counter()
    attempts = 0
    heard_from: dict[tuple[str, str], str] = {}               # (agent, name) -> first source kind
    names = set(start)
    learned: collections.Counter[str] = collections.Counter()
    name_re = {n: re.compile(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])") for n in names}

    def hear(agent: str, text: str, kind: str, sender: str | None) -> None:
        cands = {n for n in names if n in text and name_re[n].search(text)}
        if sender in names:
            cands.add(sender)
        for n in cands - {agent}:
            heard_from.setdefault((agent, n), kind if (kind != "intro_dm" or n != sender) else "dm")

    for e in events[1:]:
        d, t, r = e["data"], e["type"], e["round"]
        led.on_event(_ns(d), r)
        if t == "message":
            msg = d["message"]
            if msg["channel"] == "dm":
                hear(msg["receiver"], msg["text"], "intro_dm", msg["sender_shown"])
            else:
                for b in adj.get(msg["sender_shown"], ()):
                    hear(b, msg["text"], "board", msg["sender_shown"])
        elif t == "notice":
            nt = d["notice"]
            hear(nt["agent"], nt["text"], "notice", None)
        elif t == "win":
            won.add(d["agent"])
        elif t == "burn":
            if d["correct"]:
                carrying = [c for c, h in where.items() if h == d["target"] and transit.get(c)]
                if carrying:
                    m["defects_on_carriers"] += 1
                open_[d["target"]].discard(d["info_id"])
            if d.get("lost_info_id") is not None:
                open_[d["actor"]].discard(d["lost_info_id"])
        elif t == "edge_added":
            a, b = d["actor"], d["target"]
            src = "contact" if b in adj[a] else heard_from.get((a, b), "unknown")
            learned[src] += 1
            if b not in adj[a]:
                edges_by_round[r] += 1
                if comm and comm.get(a) != comm.get(b):
                    cross_edges += 1
            adj[a].add(b)
            adj[b].add(a)
        elif t == "action_result" and d["action"]["kind"] == "connect":
            attempts += 1
            if not d["ok"]:
                refused[d.get("reason") or "?"] += 1
        elif t == "god_power" and d["power"] == "message_fraud" and d["ok"]:
            act = d["action"]
            forged.append({"round": r, "receiver": act["receiver"], "sender": act["sender"],
                           "relay_ask": bool(RELAY_ASK.search(act.get("text", "")))})
        elif t == "key_transfer":
            cid, i, frm, to = d["copy_id"], d["info_id"], d["from_agent"], d["to_agent"]
            info[cid] = i
            if d["cause"] == "give" and frm is not None:
                for f in forged:
                    if f["receiver"] == frm and f["sender"] == to and 0 <= r - f["round"] <= EXTRACT_WINDOW:
                        m["god_extracted_relay_ask" if f["relay_ask"] else "god_extracted_other"] += 1
                        break
                if not needs_it(to, i, cid):
                    transit[cid] += 1
                    m["transit_gives"] += 1
                    needers = {a for a in open_ if needs_it(a, i, cid) and a not in (frm, to)}
                    if cid not in first_transit_needers:
                        first_transit_needers[cid] = {(a, i) for a in needers}
                    closer = False
                    for a in needers:
                        dist = _bfs_all(adj, a)
                        if dist.get(to, 10**6) < dist.get(frm, 10**6):
                            closer = True
                            break
                    if not closer:
                        m["misdirected_gives"] += 1
            where[cid] = to
    filled = {f["copy_id"] for f in led.fills}
    starts = set(transit)
    stalled = [c for c in starts if not needs_it(where[c], info[c], c) and c not in filled]
    orphan = [c for c, ns in first_transit_needers.items()
              if ns and c not in filled and all(i not in open_[a] for a, i in ns)]
    via_relay = [f for f in led.fills if len(set(f["holders"])) >= 2 and f["copy_id"] in starts]
    hops_by_round: dict[str, float | None] = {}
    snaps = _jsonl(gdir / "snapshots.jsonl") if (gdir / "snapshots.jsonl").exists() else []
    snaps = [x for x in snaps if upto is None or x["round"] <= upto]
    for sn in snaps:
        ag = sn["state"]["agents"]
        sadj = {a: set(s.get("contacts") or []) for a, s in ag.items()}
        hold: dict[int, list[str]] = collections.defaultdict(list)
        for a, s in ag.items():
            for k in s["keys"]:
                hold[k["info_id"]].append(a)
        hs = []
        for a, s in ag.items():
            if s.get("won_round") is not None:
                continue
            mine = {k["info_id"] for k in s["keys"]}
            for i in set(s["needs"]) - set(s.get("burned", [])) - set(s.get("lost", [])) - mine:
                dist = _bfs_all(sadj, a)
                hs.append(min((dist.get(h, 10**6) for h in hold.get(i, []) if h != a), default=10**6))
        fin = [h for h in hs if h < 10**6]
        hops_by_round[str(sn["round"])] = round(statistics.mean(fin), 3) if fin else None
    ends = [e for e in events if e["type"] == "round_end"]
    return {
        "game": gdir.name, "branch": meta.get("branch"), "game_idx": meta.get("game_idx"),
        "welfare": outcome.get("welfare", ends[-1]["data"]["welfare"] if ends else None),
        "last_round": ends[-1]["round"] if ends else 0, "win_rate": outcome.get("win_rate"),
        "wins": len(won),
        "chain_starts": len(starts), "chain_completions": len(starts & filled),
        "transit_gives": m["transit_gives"],
        "hops_per_completion": (round(statistics.mean(len(f["holders"]) for f in via_relay), 3) if via_relay else None),
        "stalled_keys": len(stalled), "misdirected_gives": m["misdirected_gives"], "orphaned_by_burn": len(orphan),
        "god_forgeries": len(forged), "god_extracted_relay_ask": m["god_extracted_relay_ask"],
        "god_extracted_other": m["god_extracted_other"], "defects_on_carriers": m["defects_on_carriers"],
        "fills": len(led.fills), "fills_via_relay": len(via_relay),
        "delivery_credit_total": outcome.get("delivery_credit_total"),
        "credited_holders": sum(len(f["paid"]) for f in led.fills),
        "mean_payees": (round(statistics.mean(len(f["paid"]) for f in led.fills), 3) if led.fills else None),
        "unpaid_padding": sum(len(set(f["holders"]) - {f["needer"]}) - len(f["paid"]) for f in led.fills),
        "connect_attempts": attempts, "connect_refused": dict(refused),
        "edges_added": sum(edges_by_round.values()), "edges_per_round": {str(k): v for k, v in sorted(edges_by_round.items())},
        "cross_community_edges": cross_edges, "learned_from": dict(learned),
        "need_hops_by_round": hops_by_round,
        "degree_start": _deg(adj0), "degree_end": _deg(adj),
        "mean_path_start": _mean_path(adj0), "mean_path_end": _mean_path(adj),
    }


SUM_KEYS = ("chain_starts", "chain_completions", "transit_gives", "stalled_keys", "misdirected_gives",
            "orphaned_by_burn", "god_forgeries", "god_extracted_relay_ask", "god_extracted_other",
            "defects_on_carriers", "fills", "fills_via_relay", "delivery_credit_total", "connect_attempts",
            "edges_added", "cross_community_edges")


def summarize(games: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for g in games:
        by[g["branch"]].append(g)
    out: dict[str, Any] = {}
    for br, gs in sorted(by.items()):
        s: dict[str, Any] = {"worlds": len(gs)}
        for k in SUM_KEYS:
            vals = [g[k] for g in gs if g.get(k) is not None]
            s[k] = round(sum(vals), 4) if vals else None
        s["welfare"] = statistics.mean(g["welfare"] for g in gs if g["welfare"] is not None) if gs else None
        s["mean_path_end"] = [g["mean_path_end"] for g in gs]
        s["degree_end"] = [g["degree_end"] for g in gs]
        s["learned_from"] = dict(sum((collections.Counter(g["learned_from"]) for g in gs), collections.Counter()))
        out[br] = s
    deltas = {}
    god = {g["game_idx"]: g for g in by.get("god", [])}
    for ctrl in ("nogod", "noisefork"):
        for g in by.get(ctrl, []):
            a = god.get(g["game_idx"])
            if a is not None:
                deltas[f"god-{ctrl}/w{g['game_idx']}"] = {k: (None if a.get(k) is None or g.get(k) is None
                                                             else round(a[k] - g[k], 4))
                                                         for k in ("welfare", *SUM_KEYS)}
    return {"by_branch": out, "deltas": deltas}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--upto", type=int, default=None, help="only rounds <= this (interim of a live run)")
    a = ap.parse_args()
    games = [game_metrics(d, a.upto) for r in a.runs for d in sorted((Path(r) / "games").iterdir()) if d.is_dir()]
    res = {"games": games, **summarize(games)}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    for br, s in res["by_branch"].items():
        print(br, {k: v for k, v in s.items() if k not in ("degree_end",)})
    for k, v in res["deltas"].items():
        print(k, v)


if __name__ == "__main__":
    main()
