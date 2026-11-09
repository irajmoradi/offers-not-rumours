"""Arm T ledger detection by round with ties at the minimum split evenly (expected top-1), replacing analyze_t.py's single
random tie-break draw. Same rule otherwise: cumulative LLM gift events + delivered payments up to round R; only games still
running at R. Bootstrap 95% CI over games. Written by p1-critic-plan; read-only on raw data."""
import os
import json, csv, numpy as np
O = f"{os.environ.get('PLANC_ROOT', '.')}/out"
G = [json.loads(l) for l in open(f"{O}/arm_T_paytiming/games.jsonl")]
G = [g for g in G if g["arm"] == "T" and g.get("cond", g["branch"]) != "clean"]
rng = np.random.default_rng(5); rows = []
for c in ["never_pays", "pays_now", "pays_later3", "pays_at_end"]:
    gs = [g for g in G if g.get("cond") == c]
    for R in range(2, 13):
        v = []
        for g in gs:
            if g["rounds"] < R: continue
            giv = np.zeros(g["N"])
            for rd, a, t in g["gift_events"]:
                if rd <= R: giv[a] += 1
            for pr, br, t in (g.get("payments") or []):
                if pr is not None and pr <= R: giv[br] += 1
            t_ = np.flatnonzero(giv == giv.min()); v.append(1 / len(t_) if g["insider"] in t_ else 0.0)
        if not v: continue
        v = np.array(v); bs = rng.choice(v, (5000, len(v))).mean(1)
        rows.append(dict(cell=c, round=R, games=len(v), ledger_top1=round(v.mean(), 3),
                         lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3)))
    v = []
    for g in gs:
        led = np.array(g["ledger_gives"], float); t_ = np.flatnonzero(led == led.min()); v.append(1 / len(t_) if g["insider"] in t_ else 0.0)
    rows.append(dict(cell=c, round="final", games=len(v), ledger_top1=round(float(np.mean(v)), 3), lo="", hi=""))
with open(f"{O}/arm_T_v2_2/figs/figT_detection_over_rounds_tiesplit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
