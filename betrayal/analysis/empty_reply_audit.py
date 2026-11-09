"""Read-only audit: forced-idle moves (agent reply empty or unparseable -> action "" -> idle) per branch, clean twin vs
attacked, for every Gemma-3-12B Plan C arm with an actions.jsonl. Paired by (cell, seed). Written by p1-critic-plan."""
import os
import json, collections, numpy as np, sys
O = f"{os.environ.get('PLANC_ROOT', '.')}/out"
ARMS = ["arm_U_pilot", "arm_U_main", "arm_UF_main", "arm_T_paytiming", "arm_C_contagion", "arm_C16_contagion",
        "arm_C2K_contagion", "arm_DB_disclosure", "arm_SE_levers", "arm_DC_dual", "arm_K_gemma2"]
rng = np.random.default_rng(3)
out = {}
for arm in ARMS:
    rate = {}   # (cell, seed, branch) -> (blank, turns)
    for l in open(f"{O}/{arm}/actions.jsonl"):
        g = json.loads(l)
        acts = g["actions"]
        rate[(g["cell"], g["seed"], g["branch"])] = (sum(a[2] == "" for a in acts), len(acts))
    clean = {k: v for k, v in rate.items() if k[2].startswith("clean")}
    att = {k: v for k, v in rate.items() if not k[2].startswith("clean")}
    def twin(k):   # same cell+seed; prefer clean|<defense> when the branch names a defense
        cand = [c for c in clean if c[0] == k[0] and c[1] == k[1]]
        if not cand: return None
        if k[2].startswith("defense="):
            d = k[2].split("=", 1)[1]
            m = [c for c in cand if c[2].split("|", 1)[-1] == d or (d == "ALL" and c[2].count("+") >= 6)]
            if m: return m[0]
        m = [c for c in cand if c[2] in ("clean|none", "clean")] or cand
        return m[0]
    diffs, by_branch = [], collections.defaultdict(list)
    for k, (b, n) in att.items():
        t = twin(k)
        if t is None or n == 0 or clean[t][1] == 0: continue
        d = b / n - clean[t][0] / clean[t][1]
        diffs.append(d); by_branch[k[2]].append(d)
    cb = sum(v[0] for v in clean.values()); cn = sum(v[1] for v in clean.values())
    ab = sum(v[0] for v in att.values()); an = sum(v[1] for v in att.values())
    x = np.array(diffs)
    ci = np.percentile(rng.choice(x, (5000, len(x))).mean(1), [2.5, 97.5]) if len(x) else [np.nan, np.nan]
    worst = sorted(((np.mean(v), br, len(v)) for br, v in by_branch.items()), key=lambda z: -abs(z[0]))[:3]
    out[arm] = dict(clean_rate=cb / cn, attacked_rate=ab / an, paired_diff=float(x.mean()), ci=[float(ci[0]), float(ci[1])],
                    pairs=len(x), turns=cn + an, worst=[(round(w[0], 4), w[1], w[2]) for w in worst])
    print(f"{arm:20s} clean {cb/cn:.3f}  attacked {ab/an:.3f}  paired diff {x.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] "
          f"pairs={len(x)} | largest per-branch: " + "; ".join(f"{w[1]} {w[0]:+.3f} (n={w[2]})" for w in worst))
json.dump(out, open(f"{O}/empty_reply_audit.json", "w"), indent=1)
