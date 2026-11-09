"""Arm DC (dual control) analysis, as pre-registered in prespec_dual_control.md. Usage: analyze_dc.py OUT_DIR SRC_DIR [...]
Primary: extra defections/game vs the matched clean twin (dual cells vs the clean twin WITH dual control; no-dual cells vs
the plain twin), paired by seed: H-DC1 BASE dual vs no dual; H-DC2 bribe-only dual vs no dual; H-DC3 welfare cost
(completion of clean|dual vs clean|none). Holm over the three. Sign-flip, seeded Monte Carlo (200k); seed-cluster
bootstrap CIs. Secondary (descriptive): dual_false_veto, blocks by reason, attempted burns, approver = insider, key damage,
acceptance (executed) and attempted acceptance (blocked request against a bribe's named victim by its target)."""
import sys, os, json, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[G.arm == "DC"]
rng = np.random.default_rng(41)
def ci(x):
    x = np.asarray(x, float); bs = rng.choice(x, (5000, len(x))).mean(1)
    return round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
def sf(d, name):
    d = np.asarray(d, float); r = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    null = np.abs((r.choice([1, -1], (200000, len(d))) * d).mean(1)); return float(((null >= abs(d.mean()) - 1e-12).sum() + 1) / 200001)
rows = []
for c in ("DC0_base", "DC1_base_dual", "DC0_bribe", "DC1_bribe_dual", "clean", "clean_dual"):
    g = G[G.cond == c]
    if not len(g): continue
    row = dict(cond=c, seeds=int(g.seed.nunique()), completion=round(float(g.completion.mean()), 3), keys_frac=round(float(g.keys_frac.mean()), 3))
    if c not in ("clean", "clean_dual"):
        de = ci(g.defection_effect); row.update(extra_defections=de[0], ed_lo=de[1], ed_hi=de[2], key_damage=round(float(g.key_effect.mean()), 3),
                                                bribe_offers=int(g.bribe_offers.sum()), bribes_accepted_executed=int(g.bribes_accepted.sum()))
    if "dual" in c:
        row.update(dual_requests=int(g.dual_requests.sum()), dual_approvals=int(g.dual_approvals.sum()), dual_blocks=int(g.dual_blocks.sum()),
                   dual_false_veto=int(g.dual_false_veto.sum()), approver_is_insider=int(g.dual_approver_is_insider.sum()),
                   insider_approved=int(g.dual_insider_approved.sum()),
                   block_reasons={k: int(sum(d.get(k, 0) for d in g.dual_block_reasons)) for k in ("approver_no", "unparsed", "self_approval", "no_approver")})
    rows.append(row)
W = G.pivot_table(index="seed", columns="cond", values="defection_effect"); C = G.pivot_table(index="seed", columns="cond", values="completion")
tests = []
for name, a, b, M in (("H-DC1 base: dual - no dual (extra defections)", "DC1_base_dual", "DC0_base", W),
                      ("H-DC2 bribe-only: dual - no dual (extra defections)", "DC1_bribe_dual", "DC0_bribe", W),
                      ("H-DC3 welfare cost: clean_dual - clean (completion)", "clean_dual", "clean", C)):
    if a not in M or b not in M: continue
    d = (M[a] - M[b]).dropna().values.astype(float)
    lo, hi = np.percentile(rng.choice(d, (5000, len(d))).mean(1), [2.5, 97.5])
    tests.append(dict(test=name, n=len(d), diff=round(float(d.mean()), 3), ci95=[round(float(lo), 3), round(float(hi), 3)], p=round(sf(d, name), 4)))
o = np.argsort([t["p"] for t in tests]); k = len(tests); run = 0
for j, i in enumerate(o): run = max(run, min(1.0, (k - j) * tests[i]["p"])); tests[i]["p_holm"] = round(run, 4)
json.dump(dict(label="PRE-REGISTERED arm DC (dual control); see prespec_dual_control.md", cells=rows, tests=tests),
          open(os.path.join(OUT, "results_arm_DC.json"), "w"), indent=1, default=str)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "figs", "figDC_cells.csv"), index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).T.to_string()); print(pd.DataFrame(tests).to_string())
