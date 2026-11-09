"""Arm I (homogeneous fleets), EXPLORATORY. Usage: analyze_i.py OUT_DIR I_SRC_DIR [...] -- H_SRC_DIR [...]
Insider vs clean twin, no defense, by fleet and N:
  all-Qwen = v2.1 arm C; all-Llama / all-gpt-oss / all-Gemma = arm I; mixed = arm H (+ v2.1 arm D at N=8)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; k = sys.argv.index("--"); IS, HS = sys.argv[2:k], sys.argv[k + 1:]
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); rng = np.random.default_rng(0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load = lambda ds: pd.DataFrame([json.loads(l) for d in ds for l in open(os.path.join(d, "games.jsonl"))]) if ds else pd.DataFrame()
I, H = load(IS), load(HS)
V = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "out", "final_v2_1", "games.jsonl"))])
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0, mean=None, ci95=[None, None])
    bs = rng.choice(x, (B, len(x))).mean(1)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
fleets = {"all-Qwen-2.5-7B": V[V.arm == "C"]}
if len(I):
    for m, g in I[I.arm == "I"].groupby("model"): fleets[f"all-{m}"] = g
mixed = pd.concat([H[H.arm == "H"] if len(H) else pd.DataFrame(), V[(V.arm == "D") & (V.defense == "none") & V.attacker.isin(["insider", "clean"])]])
fleets["mixed (qwen/llama/gemma-3/gpt-oss)"] = mixed
rows, R = [], {"label": "EXPLORATORY (arm I added after pre-registration)", "fleets": {}}
for name, df in fleets.items():
    R["fleets"][name] = {}
    for n, g in df.groupby("N"):
        ins, cl = g[g.attacker == "insider"], g[g.attacker == "clean"]
        e = dict(seeds=int(ins.seed.nunique()), insider_damage=ci(ins.attack_effect.values), defection_effect=ci(ins.defection_effect.values),
                 clean_completion=ci(cl.completion.values), clean_defections=ci(cl.defections.values),
                 turned=int(ins.turned.sum()), honest=int(ins.honest.sum()), invalid_per_game=ci(g.invalid.values))
        R["fleets"][name][int(n)] = e
        rows.append(dict(fleet=name, N=int(n), seeds=e["seeds"], insider_damage=e["insider_damage"]["mean"], dmg_lo=e["insider_damage"]["ci95"][0],
                         dmg_hi=e["insider_damage"]["ci95"][1], defection_effect=e["defection_effect"]["mean"], de_lo=e["defection_effect"]["ci95"][0],
                         de_hi=e["defection_effect"]["ci95"][1], clean_completion=e["clean_completion"]["mean"], clean_defections=e["clean_defections"]["mean"],
                         turned=e["turned"], honest=e["honest"], invalid_per_game=e["invalid_per_game"]["mean"]))
json.dump(R, open(os.path.join(OUT, "results_arm_I.json"), "w"), indent=1)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "figs", "figI1_fleets_damage_vs_N.csv"), index=False)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]; INK = "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for j, (name, col) in enumerate(zip(fleets, COL)):
    d = df[df.fleet == name].sort_values("N")
    if not len(d): continue
    xs = np.log2(d.N) + (j - 2) * 0.03
    ax[0].errorbar(xs, 100 * d.insider_damage, yerr=[100 * (d.insider_damage - d.dmg_lo), 100 * (d.dmg_hi - d.insider_damage)], fmt="o-", color=col, lw=2, ms=6, capsize=3, label=name)
    ax[1].errorbar(xs, d.defection_effect, yerr=[d.defection_effect - d.de_lo, d.de_hi - d.defection_effect], fmt="o-", color=col, lw=2, ms=6, capsize=3, label=name)
for a in ax:
    a.set_xticks(np.log2([4, 8, 16, 32])); a.set_xticklabels(["4", "8", "16", "32"]); a.set_xlabel("swarm size N"); a.axhline(0, color=INK, lw=0.8)
ax[0].set_ylabel("insider damage: completion drop vs clean twin (pp)"); ax[0].set_title("Insider damage by fleet")
ax[1].set_ylabel("extra defections per game vs clean twin"); ax[1].set_title("Insider-induced defections by fleet")
ax[0].legend(frameon=False, fontsize=7)
fig.suptitle("Arm I (exploratory): homogeneous fleets vs mixed swarm" + os.environ.get("LABEL", ""), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figI1_fleets_damage_vs_N.png"), dpi=200); plt.close(fig)
print(df.to_string())
