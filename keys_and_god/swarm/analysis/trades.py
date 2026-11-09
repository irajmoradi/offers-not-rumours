"""game.trade metrics (plan/TRADE_DESIGN.md): offers, accept rate, keys moved, need-completing trades
(executed trades split by how many sides got an open need: need_both / need_one / need_none; under
held_key_bonus a 1-for-1 trade is bonus-neutral, so a need is the only private gain),
and (with a network) trades touching a bridge (top-decile betweenness) or hub (top-decile degree) node.

    summarize(events, contacts=None) -> dict          # events: dicts of one game's events.jsonl
    summarize_run(run_dir) -> {"games": {gid: ...}, "total": ...}
    python -m swarm.analysis.trades RUN_DIR

contacts: agent -> neighbour list; None = read it from game_start (AgentState.contacts), absent when the
network is off. Imports: networkx (centrality only), stdlib.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

COUNTS = ("offers", "offers_refused", "replies", "accepted", "rejected", "executed", "failed",
          "keys_moved", "need_completing", "needs_completed", "need_both", "need_one", "need_none",
          "bridge_trades", "hub_trades")


def _top_decile(score: Mapping[str, float]) -> set[str]:
    if not score:
        return set()
    vals = sorted(score.values(), reverse=True)
    cut = vals[max(0, len(vals) // 10 - 1)] if len(vals) >= 10 else vals[0]
    return {a for a, v in score.items() if v >= cut and v > 0}


def central_nodes(contacts: Mapping[str, Sequence[str]]) -> tuple[set[str], set[str]]:
    """(bridges, hubs): top-decile betweenness / degree agents (at least the top one)."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(contacts)
    g.add_edges_from((a, b) for a, nb in contacts.items() for b in nb)
    return _top_decile(nx.betweenness_centrality(g)), _top_decile(dict(g.degree()))


def summarize(events: Iterable[Mapping[str, Any]], contacts: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Trade counts for one game's events (dicts as in events.jsonl)."""
    evs = list(events)
    if contacts is None:
        start = next((e for e in evs if e.get("type") == "game_start"), None)
        ags = (start or {}).get("data", {}).get("agents", {})
        if ags and all(s.get("contacts") is not None for s in ags.values()):
            contacts = {a: list(s["contacts"]) for a, s in ags.items()}
    bridges, hubs = central_nodes(contacts) if contacts else (set(), set())
    c: Counter = Counter()
    reasons: Counter = Counter()
    for e in evs:
        t, d = e.get("type"), e.get("data", {})
        if t == "action_result" and d.get("action", {}).get("kind") == "offer_trade" and not d.get("ok"):
            c["offers_refused"] += 1
            reasons[d.get("reason") or "invalid"] += 1
        elif t == "trade_offer":
            c["offers"] += 1
        elif t == "trade_reply":
            c["replies"] += 1
            c["accepted" if d["accept"] else "rejected"] += 1
        elif t == "trade_failed":
            c["failed"] += 1
            reasons[f"failed:{d['reason']}"] += 1
        elif t == "trade_executed":
            c["executed"] += 1
            c["keys_moved"] += len(d["give"]) + len(d["want"])
            done = len(d.get("completes_need_proposer", [])) + len(d.get("completes_need_target", []))
            c["needs_completed"] += done
            c["need_completing"] += bool(done)
            sides = bool(d.get("completes_need_proposer")) + bool(d.get("completes_need_target"))
            c[("need_none", "need_one", "need_both")[sides]] += 1
            pair = {d["proposer"], d["target"]}
            c["bridge_trades"] += bool(pair & bridges)
            c["hub_trades"] += bool(pair & hubs)
    out: dict[str, Any] = {k: c[k] for k in COUNTS}
    out["accept_rate"] = round(c["accepted"] / c["replies"], 4) if c["replies"] else None
    out["refusal_reasons"] = dict(sorted(reasons.items()))
    out["network"] = bool(contacts)
    return out


def summarize_run(run_dir: Path | str) -> dict[str, Any]:
    """summarize() per game under run_dir/**/events.jsonl, plus the totals."""
    games: dict[str, Any] = {}
    for p in sorted(Path(run_dir).rglob("events.jsonl")):
        evs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        games[str(p.parent.relative_to(run_dir))] = summarize(evs)
    tot: Counter = Counter()
    for g in games.values():
        tot.update({k: g[k] for k in COUNTS})
    total: dict[str, Any] = {k: tot[k] for k in COUNTS}
    total["accept_rate"] = round(tot["accepted"] / tot["replies"], 4) if tot["replies"] else None
    return {"games": games, "total": total}


if __name__ == "__main__":
    print(json.dumps(summarize_run(sys.argv[1])["total"], indent=1))
