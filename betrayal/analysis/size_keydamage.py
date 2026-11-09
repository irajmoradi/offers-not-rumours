"""Swarm-size insider damage on the KEY metric, honest agents only, recomputed from raw games.jsonl rows.
Key score per agent = (needs met - keys burned)/3 (llm_game.score), so a burned need counts as LOST.
Attacked branch keys_frac already excludes the insider. The clean twin's keys_frac includes the agent x that becomes the
insider, so we remove x: s_x is known exactly when x completed in the clean twin (s_x = 1 if the twin had no burns),
otherwise imputed as the mean score of the twin's incomplete agents; bounds use s_x in [0, 2/3] (incomplete) or
[2/3, 1] (complete, twin had burns). Completion (honest-only, completion_ex_insider) is recomputed for comparison.
Sources: all-Qwen = v2.1 arm C; Llama / gpt-oss / Gemma = arm I (main + pilot); mixed = arm H (main + pilot) + v2.1 arm D
(N=8, defense none). Written by p1-critic-plan; independent of analyze_p1.py / analyze_i.py."""
import json, os, numpy as np, csv
O = f"{os.environ.get('PLANC_ROOT', '.')}/out"
def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []
R = []
for g in load(f"{O}/final_v2_1/games.jsonl"):
    if g["arm"] == "C": R.append(("all-Qwen", g))
    if g["arm"] == "D" and g["branch"] in ("clean|none", "insider|none"): R.append(("mixed", g))
for d in ("arm_I_main", "arm_I_pilot"):
    for g in load(f"{O}/{d}/games.jsonl"):
        if g["arm"] == "I": R.append(("all-" + g["model"].split("-")[0].replace("gpt", "gpt-oss").replace("gpt-oss", "gpt-oss") if False else "all-" + g["model"], g))
for d in ("arm_H_main", "arm_H_pilot"):
    for g in load(f"{O}/{d}/games.jsonl"):
        if g["arm"] == "H": R.append(("mixed", g))
pairs = {}
for fleet, g in R:
    k = (fleet, g["N"], g["arm"], g["seed"], g.get("topology"))
    pairs.setdefault(k, {})["clean" if g["branch"].startswith("clean") else "att"] = g
rows = {}
for (fleet, N, arm, seed, _), p in pairs.items():
    if "clean" not in p or "att" not in p: continue
    c, a = p["clean"], p["att"]
    n_complete = round(c["completion"] * N)
    x_complete = round(c["completion"] * N - c["completion_ex_insider"] * (N - 1)) >= 1
    tot = c["keys_frac"] * N
    burns = c.get("keys_burned", 0) or 0
    if x_complete:
        sx, lo, hi = 1.0, (2 / 3 if burns else 1.0), 1.0
    else:
        inc = N - n_complete
        sx = min(max((tot - n_complete) / inc, 0.0), 2 / 3) if inc else 0.0
        lo, hi = 0.0, 2 / 3
    kc = lambda s: (tot - s) / (N - 1)
    rows.setdefault((fleet, N), []).append(dict(
        key=kc(sx) - a["keys_frac"], key_lo=kc(hi) - a["keys_frac"], key_hi=kc(lo) - a["keys_frac"],
        key_allagents=c["keys_frac"] - a["keys_frac"], comp=c["completion_ex_insider"] - a["completion_ex_insider"]))
rng = np.random.default_rng(7)
def ci(x):
    x = np.asarray(x); m = rng.choice(x, (10000, len(x))).mean(1); return np.percentile(m, [2.5, 97.5])
out = []
for (fleet, N) in sorted(rows, key=lambda k: (k[0], k[1])):
    v = rows[(fleet, N)]; f = lambda k: np.array([r[k] for r in v]) * 100
    k = f("key"); lo_, hi_ = ci(k)
    out.append(dict(fleet=fleet, N=N, seeds=len(v), key_damage_honest_pp=round(k.mean(), 1), ci_lo=round(lo_, 1), ci_hi=round(hi_, 1),
                    bound_lo=round(f("key_lo").mean(), 1), bound_hi=round(f("key_hi").mean(), 1),
                    key_damage_allagents_pp=round(f("key_allagents").mean(), 1), completion_damage_honest_pp=round(f("comp").mean(), 1)))
with open(f"{O}/size_keydamage_honest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out: print(r)
