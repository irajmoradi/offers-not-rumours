"""Growth condition (arm GR), EXPLORATORY: N grows 6->8 at round 4, outsider bribe at mid timing (round 4),
flagged/disguised, vs its own clean twin (which also grows). It is set beside arm G mid (no growth, N=8 from the start).
Usage: analyze_gr.py OUT_DIR GR_SRC_DIR [GR_SRC_DIR ...] -- G_SRC_DIR [G_SRC_DIR ...]"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; k = sys.argv.index("--"); GRS, GS = sys.argv[2:k], sys.argv[k + 1:]
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); rng = np.random.default_rng(0)
load = lambda ds: pd.DataFrame([json.loads(l) for d in ds for l in open(os.path.join(d, "games.jsonl"))])
GR, G = load(GRS), load(GS)
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
R = {"label": "EXPLORATORY growth condition (6->8 agents at round 4)", "rows": []}
for src, df, lab in ((GR, GR, "growth 6->8"), (G, G, "no growth (N=8)")):
    for r in ("flagged", "disguised"):
        d = df[df.attacker == f"obribe:mid:{r}"]
        if not len(d): continue
        e = dict(condition=lab, recognition=r, seeds=int(d.seed.nunique()), defection_effect=ci(d.defection_effect.values),
                 damage=ci(d.attack_effect.values), turned=int(d.turned.sum()), honest=int(d.honest.sum()),
                 bribes_seen=int(d.bribes_seen.sum()), bribes_accepted=int(d.bribes_accepted_strict_seen.sum()))
        R["rows"].append(e)
cl = GR[GR.attacker == "clean"]
R["growth_clean_twin"] = dict(completion=ci(cl.completion.values), defections=ci(cl.defections.values))
json.dump(R, open(os.path.join(OUT, "results_growth.json"), "w"), indent=1)
df = pd.DataFrame([dict(condition=e["condition"], recognition=e["recognition"], seeds=e["seeds"], defection_effect=e["defection_effect"]["mean"],
                        de_lo=e["defection_effect"]["ci95"][0], de_hi=e["defection_effect"]["ci95"][1], damage_pp=100 * e["damage"]["mean"],
                        turned=e["turned"], honest=e["honest"], bribes_seen=e["bribes_seen"], bribes_accepted=e["bribes_accepted"]) for e in R["rows"]])
df.to_csv(os.path.join(OUT, "figs", "figG3_growth_vs_nogrowth.csv"), index=False)
INK = "#52514e"; C = {"flagged": "#2a78d6", "disguised": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(7, 4)); conds = ["no growth (N=8)", "growth 6->8"]; x = np.arange(2)
for j, r in enumerate(("flagged", "disguised")):
    d = df[df.recognition == r].set_index("condition").reindex(conds)
    ax.bar(x + (j - 0.5) * 0.36, d.defection_effect, 0.34, color=C[r], label=r,
           yerr=[d.defection_effect - d.de_lo, d.de_hi - d.defection_effect], error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(seeds: {int(df[df.condition == c].seeds.max()) if (df.condition == c).any() else 0})" for c in conds])
ax.axhline(0, color=INK, lw=0.8); ax.set_ylabel("defections per game vs clean twin")
ax.set_title("Outsider bribe at mid timing: growing vs fixed swarm (exploratory)" + os.environ.get("LABEL", "")); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figG3_growth_vs_nogrowth.png"), dpi=200); plt.close(fig)
print(df.to_string())
