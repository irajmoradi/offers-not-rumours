"""E1 extras: forged-sender redirection, victim outcomes, God firing, provider split per branch.

usage: python -I e1_extra.py RUN_DIR OUT_JSON
Reads events.jsonl / outcome.json / llm_calls.jsonl only (metadata; no message text printed).
"""
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
out_path = Path(sys.argv[2])
man = json.loads((run / "manifest.json").read_text())
dirs = {(int(g["game_idx"]), g["branch"]): run / g["dir"] for g in man["games"]}
BR = ("god", "nogod", "noisefork")


def load(d):
    ev = [json.loads(l) for l in open(d / "events.jsonl")]
    start = ev[0]["data"]
    needs = {a: set(v.get("needs", [])) for a, v in start["agents"].items()}
    gives = [(e["round"], e["data"]["from_agent"], e["data"]["to_agent"], e["data"]["info_id"])
             for e in ev if e["type"] == "key_transfer" and e["data"]["cause"] == "give"]
    god = [(e["round"], e["data"]["action"], e["data"]["ok"], e["data"].get("reason"))
           for e in ev if e["type"] == "god_power"]
    oc = json.loads((d / "outcome.json").read_text()) if (d / "outcome.json").exists() else None
    return dict(needs=needs, gives=gives, god=god, outcome=oc)


def t_ci(xs):
    n = len(xs)
    if n < 2:
        return (float("nan"),) * 2
    tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
          11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131}[n]
    m, s = statistics.fmean(xs), statistics.stdev(xs)
    return m - tq * s / math.sqrt(n), m + tq * s / math.sqrt(n)


def signflip(xs, rng=random.Random(0)):
    n = len(xs)
    obs = abs(sum(xs))
    if n <= 16:
        cnt = 0
        for mask in range(1 << n):
            s = sum(x if (mask >> i) & 1 else -x for i, x in enumerate(xs))
            cnt += abs(s) >= obs - 1e-12
        return cnt / (1 << n)
    return float("nan")


def summ(xs):
    lo, hi = t_ci(xs)
    return dict(mean=statistics.fmean(xs), t_lo=lo, t_hi=hi, p=signflip(xs), n=len(xs))


worlds = sorted({i for (i, b) in dirs if all((i, bb) in dirs for bb in BR)})
rows = {}
prov = {b: collections.Counter() for b in BR}
for i in worlds:
    G = {b: load(dirs[(i, b)]) for b in BR}
    if any(G[b]["outcome"] is None for b in BR):
        continue
    frauds = [(r, a["sender"], a["receiver"]) for (r, a, ok, _) in G["god"]["god"]
              if a.get("kind") == "message_fraud" and ok]
    fails = [(r, reason) for (r, a, ok, reason) in G["god"]["god"] if not ok]
    first = {}
    for r, X, B in frauds:
        first.setdefault((B, X), r)
    victims = sorted({B for _, _, B in frauds})
    row = dict(n_frauds_ok=len(frauds), n_fail=len(fails), fail_reasons=[x[1] for x in fails],
               n_pairs=len(first), n_victims=len(victims),
               god_rounds=sorted({r for r, _, _ in frauds}))
    for b in BR:
        g = G[b]
        # gives from a victim to the forged sender, at or after that pair's first forgery
        pair_gives = sum(1 for (r, f, t, info) in g["gives"] if (f, t) in first and r >= first[(f, t)])
        selfharm = sum(1 for (r, f, t, info) in g["gives"]
                       if (f, t) in first and r >= first[(f, t)] and info in g["needs"].get(f, set()))
        # per-fraud event compliance: B gives X within rounds t..t+2
        comply = sum(1 for (t0, X, B) in frauds
                     if any(f == B and to == X and t0 <= r <= t0 + 2 for (r, f, to, _) in g["gives"]))
        pa = g["outcome"]["per_agent"]
        vs = [pa[v]["score"] for v in victims]
        vlost = [len(pa[v]["burned"]) + len(pa[v]["lost"]) for v in victims]
        vlost_own = [len(pa[v]["lost"]) for v in victims]
        row[b] = dict(welfare=g["outcome"]["welfare"], win_rate=g["outcome"]["win_rate"],
                      pair_gives=pair_gives, selfharm_gives=selfharm,
                      comply_events=comply, comply_rate=comply / len(frauds) if frauds else float("nan"),
                      victim_score=statistics.fmean(vs) if vs else float("nan"),
                      victim_needs_lost=statistics.fmean(vlost) if vlost else float("nan"),
                      victim_own_lost=statistics.fmean(vlost_own) if vlost else float("nan"))
        # provider split of fresh calls
        for l in open(dirs[(i, b)] / "llm_calls.jsonl"):
            rec = json.loads(l)
            resp = rec.get("response") or {}
            if resp.get("cache_hit") is False:
                prov[b][resp.get("provider") or "unknown"] += 1
            elif resp.get("cache_hit") is True:
                prov[b]["_cache"] += 1
    rows[i] = row

res = dict(worlds=sorted(rows), rows=rows, providers={b: dict(c) for b, c in prov.items()})
for m in ("pair_gives", "selfharm_gives", "comply_rate", "victim_score", "victim_needs_lost", "victim_own_lost",
          "welfare", "win_rate"):
    for a, b in (("god", "nogod"), ("god", "noisefork"), ("noisefork", "nogod")):
        xs = [rows[i][a][m] - rows[i][b][m] for i in rows]
        res[f"{m}:{a}-{b}"] = summ(xs)
    for b in BR:
        res[f"{m}:{b}:mean"] = statistics.fmean([rows[i][b][m] for i in rows])
        res[f"{m}:{b}:sum"] = sum(rows[i][b][m] for i in rows)
out_path.write_text(json.dumps(res, indent=1, default=str))
print("worlds", res["worlds"])
for i, r in rows.items():
    print(i, "frauds", r["n_frauds_ok"], "fail", r["n_fail"], r["fail_reasons"][:3], "victims", r["n_victims"],
          "pairs", r["n_pairs"], {b: (round(r[b]["welfare"], 3), r[b]["pair_gives"], r[b]["comply_events"],
                                      r[b]["selfharm_gives"], round(r[b]["victim_needs_lost"], 2)) for b in BR})
for k, v in res.items():
    if ":" in k and isinstance(v, dict):
        print(f"{k:40s} {v['mean']:+.3f} [{v['t_lo']:+.3f}, {v['t_hi']:+.3f}] p={v['p']:.4f} n={v['n']}")
    elif ":" in k:
        print(f"{k:40s} {v:.3f}")
print("providers", res["providers"])
