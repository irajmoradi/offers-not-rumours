"""Independent recompute of arm J 'never gives' ledger detection (p1-critic-plan; read-only). Insider branches vs clean twin,
all-Gemma N=8, 20 seeds per schedule (arm_J_gemma). Ties: expected top-1 (1/#tied if the insider is among the minima),
plus the share of games where the insider is the UNIQUE minimum, plus the spread of Fisher p over 2000 random tie-breaks."""
import os
import json, numpy as np, itertools
from scipy.stats import fisher_exact
L = [json.loads(l) for l in open(f"{os.environ.get('PLANC_ROOT', '.')}/out/arm_J_gemma/games.jsonl")]
clean = {g["cell"]: g for g in L if g["attacker"] == "clean"}
def gives(g, after):
    c = np.zeros(g["N"])
    for e in g["gift_events"]:
        if e[0] > after: c[e[1]] += 1
    return c
def score(g, rule, pay):
    x = int(g["insider"]); p = g.get("bribes_paid") or 0
    if rule == "raw":   v = gives(g, 0)
    elif rule == "raw_post": v = gives(g, 1)
    else:               v = gives(g, 1) - gives(clean[g["cell"]], 1)
    if pay: v[x] += p if p == p else 0
    m = v.min(); t = np.flatnonzero(v == m)
    return (1 / len(t) if x in t else 0.0), int(len(t) == 1 and t[0] == x)
rng = np.random.default_rng(0)
print("rule        pay | never: E[top1] unique | always: E[top1] unique | Fisher p (expected counts) | p range over random tie-breaks")
for rule, pay in itertools.product(["raw", "raw_post", "twin_diff"], [True, False]):
    res = {}
    for s in ("never", "always"):
        gs = [g for g in L if g["attacker"] == "insider" and g.get("fulfillment") == s]
        res[s] = [score(g, rule, pay) for g in gs]
    e = {s: sum(a for a, _ in res[s]) for s in res}; u = {s: sum(b for _, b in res[s]) for s in res}
    n = len(res["never"])
    ps = []
    for _ in range(2000):
        k = {s: sum(rng.random() < a for a, _ in res[s]) for s in res}
        ps.append(fisher_exact([[k["never"], n - k["never"]], [k["always"], n - k["always"]]])[1])
    pe = fisher_exact([[round(e["never"]), n - round(e["never"])], [round(e["always"]), n - round(e["always"])]])[1]
    print(f"{rule:10s} {str(pay):5s} | {e['never']/n:.2f} {u['never']/n:.2f} | {e['always']/n:.2f} {u['always']/n:.2f} | {pe:.3g} | "
          f"{np.percentile(ps, 5):.3g}-{np.percentile(ps, 95):.3g} (median {np.median(ps):.3g})")
