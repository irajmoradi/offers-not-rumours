"""Final showcase tables (plan/SHOWCASE_RESULTS.md): per game outcomes, fill by hop bucket, payoff terms,
trust-game + connect metrics (showcase_metrics), trades (analysis.trades), God reveals (god_reveal),
calls / providers / ledger cost.  python -m swarm.analysis.showcase_report runs/showcase100_sim
"""
import os
import collections
import json
import sys
from pathlib import Path

from swarm.analysis.god_reveal import analyse
from swarm.analysis.showcase_metrics import game_metrics
from swarm.analysis.trades import summarize
from swarm.game import network


def main() -> None:
    run = Path(sys.argv[1])
    out: dict = {}
    for gd in sorted((run / "games").iterdir()):
        ev = [json.loads(x) for x in (gd / "events.jsonl").read_text().splitlines() if x.strip()]
        o = json.loads((gd / "outcome.json").read_text()); meta = json.loads((gd / "meta.json").read_text())
        comm = meta["network"]["communities"]; start = ev[0]["data"]["agents"]
        # hop bucket per need at start
        adj = {a: s["contacts"] for a, s in start.items()}
        hold = collections.defaultdict(list)
        for a, s in start.items():
            for k in s["keys"]: hold[k["info_id"]].append(a)
        bucket = {}
        for a, s in start.items():
            d = network.hop_distances(adj, a)
            for i in s["needs"]: bucket[(a, i)] = min(min(d.get(h, 99) for h in hold[i] if h != a), 3)
        snaps = [json.loads(x) for x in (gd / "snapshots.jsonl").read_text().splitlines() if x.strip()]
        fin = snaps[-1]["state"]["agents"]
        filled = collections.Counter(); tot = collections.Counter()
        for (a, i), b in bucket.items():
            tot[b] += 1
            s = fin[a]
            if i not in s.get("burned", []) and i not in s.get("lost", []) and i in {k["info_id"] for k in s["keys"]}:
                filled[b] += 1
        per = o["per_agent"]; n = len(per)
        cw = collections.Counter(comm[a] for a, x in per.items() if x["won_round"] is not None)
        pen = sum(x["score"] + x.get("key_bonus", 0) - x["private"] for x in per.values()) / n
        tr = summarize(ev)
        cross = within = 0
        for e in ev:
            if e["type"] == "key_transfer" and e["data"]["from_agent"]:
                if comm[e["data"]["from_agent"]] != comm[e["data"]["to_agent"]]: cross += 1
                else: within += 1
        burns = [e["data"] for e in ev if e["type"] == "burn"]
        turns = [e["data"] for e in ev if e["type"] == "turn"]
        gm = game_metrics(gd)
        out[gd.name] = {"welfare": o["welfare"], "win_rate": o["win_rate"], "wins": sum(cw.values()),
                        "communities_ge16": sum(v >= 16 for v in cw.values()), "winners_by_comm": dict(cw),
                        "fill_by_hops": {b: f"{filled[b]}/{tot[b]}" for b in (1, 2, 3)},
                        "mean_key_bonus": sum(x.get("key_bonus", 0) for x in per.values()) / n,
                        "mean_community_bonus": sum(x.get("community_bonus", 0) for x in per.values()) / n,
                        "credit_total": o.get("delivery_credit_total"), "mean_penalty": pen,
                        "welfare_private": o.get("welfare_private"),
                        "transfers_within_cross": [within, cross],
                        "defects": len(burns), "correct": sum(b["correct"] for b in burns),
                        "parse_fail": sum(not t["parse_ok"] for t in turns), "turns": len(turns),
                        "trades": {k: tr.get(k) for k in ("offers", "accepted", "rejected", "executed", "need_completing", "need_both", "need_one", "need_none", "bridge_trades")},
                        **{k: gm[k] for k in ("chain_starts", "chain_completions", "transit_gives", "hops_per_completion", "stalled_keys", "misdirected_gives", "orphaned_by_burn", "god_forgeries", "god_extracted_relay_ask", "god_extracted_other", "defects_on_carriers", "fills", "fills_via_relay", "credited_holders", "unpaid_padding", "connect_attempts", "connect_refused", "edges_added", "cross_community_edges", "learned_from", "degree_start", "degree_end", "mean_path_start", "mean_path_end")},
                        "need_hops": {r: gm["need_hops_by_round"].get(r) for r in ("0", "5", "10", "15", "20", "25")}}
    rv = analyse(run, 3)
    out["reveal"] = {"n": rv["n_reveals"], "totals": rv["totals"], "accuracy": rv["defect_accuracy"]}
    # per-call provider/cost
    prov = collections.Counter(); cost = 0.0; calls = 0; fresh = 0; length = 0
    for gd in sorted((run / "games").iterdir()):
        for x in (gd / "llm_calls.jsonl").read_text().splitlines():
            c = json.loads(x); r = c.get("response") or {}; calls += 1
            if r.get("finish_reason") == "length": length += 1
            if not r.get("cache_hit"): fresh += 1; prov[r.get("provider")] += 1; cost += r.get("cost_usd") or 0
    out["calls"] = {"total": calls, "fresh": fresh, "providers": dict(prov), "cost_calls": cost, "length": length}
    led = 0.0
    for x in open(f"{os.environ.get('SWARM_ROOT', '.')}/runs/openrouter_ledger.jsonl"):
        r = json.loads(x)
        if r.get("run_id") == run.name: led += r.get("cost_usd") or 0
    out["ledger_cost"] = led
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
