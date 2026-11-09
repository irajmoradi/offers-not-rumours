"""Arm UF: 2^(7-3) resolution-IV fractional factorial. EXPLORATORY.
Usage: analyze_uf.py OUT_DIR SRC_DIR [...]
Response per run x seed: extra defections vs the clean twin (and key damage). Effect of factor f = mean(+) - mean(-)
over the 16 runs, computed per seed and then averaged (seed-bootstrap CI; two-sided sign-flip p). 2fi chains are
estimated the same way, using the product column."""
import sys, os, json, hashlib, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[(G.arm == "UF") & (G.cond != "clean")]
F = ["A_position", "B_method", "C_timing", "D_N", "E_defense", "F_models", "G_records"]
for f in F: G[f] = G.spec.map(lambda s: s["uf_coded"][f])
rng = np.random.default_rng(9)
def eff(col, resp):
    per = G.groupby("seed").apply(lambda d: d[d[col] > 0][resp].mean() - d[d[col] < 0][resp].mean()).dropna().values
    bs = rng.choice(per, (5000, len(per))).mean(1)
    r = np.random.default_rng(int(hashlib.sha256((col + resp).encode()).hexdigest()[:16], 16)); s = r.choice([-1, 1], (20000, len(per)))
    p = ((np.abs((s * per).mean(1)) >= abs(per.mean()) - 1e-12).sum() + 1) / 20001
    return round(float(per.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), round(float(p), 4), len(per)
rows = []
LAB = {"A_position": "position: outsider(+) vs insider(-)", "B_method": "method: bribe(+) vs forge(-)", "C_timing": "timing: late(+) vs early(-)",
       "D_N": "N: 16(+) vs 4(-)", "E_defense": "defense: all-7(+) vs none(-)", "F_models": "models: mixed(+) vs Gemma-3(-)", "G_records": "records: open-shows(+) vs closed(-)"}
for f in F:
    for resp in ("defection_effect", "key_effect"):
        e, lo, hi, p, n = eff(f, resp); rows.append(dict(term=f, label=LAB[f], kind="main", response=resp, effect=e, lo=lo, hi=hi, p=p, seeds=n))
chains = ["AB=CE=FG", "AC=BE=DG", "AD=CG=EF", "AE=BC=DF", "AF=BG=DE", "AG=BF=CD", "BD=CF=EG"]
code = {c[0]: c for c in F}
for ch in chains:
    a, b = ch.split("=")[0]; col = f"_{a}{b}"; G[col] = G[code[a]] * G[code[b]]
    e, lo, hi, p, n = eff(col, "defection_effect"); rows.append(dict(term=ch, label="2fi alias chain", kind="2fi", response="defection_effect", effect=e, lo=lo, hi=hi, p=p, seeds=n))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "figs", "figUF_effects.csv"), index=False)
runs = G.groupby("cond").agg(seeds=("seed", "nunique"), extra_defections=("defection_effect", "mean"), key_damage=("key_effect", "mean"), turned=("turned", "sum")).round(3).reset_index()
runs.to_csv(os.path.join(OUT, "figs", "figUF_runs.csv"), index=False)
json.dump(dict(label="EXPLORATORY arm UF, 2^(7-3) resolution IV", defining_relation="I=ABCE=ABFG=ACDG=ADEF=BCDF=BDEG=CEFG", alias_chains=chains,
               effects=rows), open(os.path.join(OUT, "results_arm_UF.json"), "w"), indent=1)
d = T[(T.response == "defection_effect")]
fig, ax = plt.subplots(figsize=(9, 5)); y = np.arange(len(d))[::-1]
ax.errorbar(d.effect, y, xerr=[np.clip(d.effect - d.lo, 0, None), np.clip(d.hi - d.effect, 0, None)], fmt="o", color="#2a78d6", capsize=3)
ax.axvline(0, color="#52514e", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([f"{r.label if r.kind == 'main' else r.term} (p={r.p})" for r in d.itertuples()], fontsize=8)
ax.set_xlabel("effect on extra defections per game (high − low level)")
ax.set_title(f"Arm UF (exploratory): 2^(7-3) res-IV factorial, {int(G.seed.nunique())} seeds; main effects clear of 2fi", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figUF_effects.png"), dpi=200); plt.close(fig)
print(T.to_string()); print(runs.to_string())
