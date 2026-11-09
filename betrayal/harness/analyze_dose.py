"""Outsider bribe-size dose-response at N=16 (1/2/3 keys), with and without contagion. EXPLORATORY.
Usage: analyze_dose.py OUT_DIR C16_SRC C2K_SRC
1-key and 3-key cells come from the C16 run, 2-key cells from the C2K run. Each run has its own clean twin per seed
(same seed and world setup, fresh LLM draws), so each cell's effect is vs ITS OWN twin; effects are paired across
sizes by seed. Trend tests per contagion flag: (a) per-seed OLS slope over sizes 1,2,3 (= (e3-e1)/2), sign-flip;
(b) Page's L test (within-seed ranks, ties averaged), seeded Monte Carlo permutation of ranks within seed.
Holm over the 4 trend tests."""
import sys, os, json, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, S16, S2K = sys.argv[1:4]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
rd = lambda s: pd.DataFrame([json.loads(l) for l in open(os.path.join(s, "games.jsonl"))])
A = rd(S16); A = A[A.arm == "C16"]; B = rd(S2K); B = B[B.arm == "C2K"]
G = pd.concat([A, B]); G = G[G.cond != "clean"]
SZ = {"outsider_only": 1, "outsider_only_2key": 2, "outsider_only_3key": 3,
      "outsider+contagion": 1, "outsider+contagion_2key": 2, "outsider+contagion_3key": 3}
G = G[G.cond.isin(SZ)]; G["size"] = G.cond.map(SZ); G["contagion"] = G.cond.str.contains("contagion")
rng = np.random.default_rng(29)
def ci(x):
    x = np.asarray(x, float); bs = rng.choice(x, (5000, len(x))).mean(1)
    return round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
def nrng(name): return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
rows, tests = [], []
for ct in (False, True):
    g = G[G.contagion == ct]
    W = g.pivot_table(index="seed", columns="size", values="defection_effect").dropna()
    for z in (1, 2, 3):
        gz = g[g["size"] == z]
        m, lo, hi = ci(gz.defection_effect)
        rec = sum(len(L) for L in gz.contagion_log if isinstance(L, list))
        rows.append(dict(contagion=ct, size=z, seeds=int(gz.seed.nunique()), extra_defections=m, ed_lo=lo, ed_hi=hi,
                         accept_rate=round(float(gz.bribes_accepted.sum() / max(1, gz.bribe_offers.sum())), 3),
                         key_damage=round(float(gz.key_effect.mean()), 3), recruits_per_game=round(rec / len(gz), 3),
                         source="C2K" if z == 2 else "C16"))
    X = W[[1, 2, 3]].values; n = len(X)
    # (a) per-seed slope
    sl = (X[:, 2] - X[:, 0]) / 2.0; r = nrng(f"slope_{ct}")
    null = np.abs((r.choice([1, -1], (200000, n)) * sl).mean(1)); p = float(((null >= abs(sl.mean()) - 1e-12).sum() + 1) / 200001)
    lo, hi = np.percentile(rng.choice(sl, (5000, n)).mean(1), [2.5, 97.5])
    tests.append(dict(test=f"slope {'contagion' if ct else 'outsider-only'}", n=n, estimate=round(float(sl.mean()), 3),
                      ci95=[round(float(lo), 3), round(float(hi), 3)], p=round(p, 4), unit="extra defections per extra key"))
    # (b) Page's L
    R = np.vstack([rankdata(x) for x in X]); L = float((R * np.array([1, 2, 3])).sum()); r = nrng(f"page_{ct}")
    perm = np.array([[r.permutation(row) for row in R] for _ in range(20000)])
    Ln = (perm * np.array([1, 2, 3])).sum(axis=(1, 2)); EL = n * 2 * 6.0
    p = float(((np.abs(Ln - EL) >= abs(L - EL) - 1e-9).sum() + 1) / (len(Ln) + 1))
    tests.append(dict(test=f"Page L {'contagion' if ct else 'outsider-only'}", n=n, estimate=L, expected_under_null=EL, p=round(p, 4),
                      unit="two-sided"))
order = np.argsort([t["p"] for t in tests]); k = len(tests); run = 0
for j, i in enumerate(order): run = max(run, min(1.0, (k - j) * tests[i]["p"])); tests[i]["p_holm"] = round(run, 4)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "figs", "figDose_N16.csv"), index=False)
json.dump(dict(label="EXPLORATORY outsider bribe-size dose-response at N=16 (C16 1/3-key + C2K 2-key); recruits always 1-key",
               cells=rows, trend_tests=tests,
               caveat="2-key cells come from a separate run (C2K) with its own clean twins; paired by seed across runs, not by shared prefix."),
          open(os.path.join(OUT, "results_dose_N16.json"), "w"), indent=1)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for ct, col, lab in ((False, "#2a78d6", "outsider only"), (True, "#eb6834", "outsider + contagion")):
    d = T[T.contagion == ct].sort_values("size"); off = -0.04 if not ct else 0.04
    ax.errorbar(d["size"] + off, d.extra_defections, yerr=[d.extra_defections - d.ed_lo, d.ed_hi - d.extra_defections], color=col,
                marker="o", ms=7, lw=2, capsize=4, label=lab)
ax.axhline(0, color="#c3c2b7", lw=1); ax.set_xticks([1, 2, 3]); ax.set_xlabel("outsider's bribe size (keys offered; recruits always offer 1)")
ax.set_ylabel("extra defections per game vs clean twin"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
ax.set_title("Bribe-size dose-response, all-Gemma-3 N=16, 40 seeds (95% CI)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figDose_N16.png"), dpi=200); plt.close(fig)
pd.set_option("display.width", 250); print(T.to_string(index=False)); print(pd.DataFrame(tests).to_string())
