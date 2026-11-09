"""Arm K: outsider bribe entry stage x N (+ insider branch) with the extended interview. EXPLORATORY.
Usage: analyze_k.py OUT_DIR SRC_DIR [...]"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); rng = np.random.default_rng(11)
rd = lambda f: pd.DataFrame([json.loads(l) for s in SRCS if os.path.exists(os.path.join(s, f)) for l in open(os.path.join(s, f))])
G, I = rd("games.jsonl"), rd("interviews2.jsonl"); G = G[G.arm == "K"]
def ci(x, B=5000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (None, None, None)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return (round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
lab = lambda a: "clean" if a == "clean" else ("insider" if a == "insider" else "outsider-" + a.split(":")[1])
G["cond"] = G.attacker.map(lab)
rows = []
for (n, c), g in G[G.cond != "clean"].groupby(["N", "cond"]):
    e = ci(g.defection_effect); dmg = ci(g.attack_effect)
    rows.append(dict(N=int(n), condition=c, seeds=int(g.seed.nunique()), extra_defections=e[0], lo=e[1], hi=e[2],
                     defect_success=int(g.defect_success.sum()), defect_fail=int(g.defect_fail.sum()),
                     sabotage_success_rate=round(g.defect_success.sum() / max(1, g.defect_success.sum() + g.defect_fail.sum()), 3),
                     offers=int(g.bribe_offers.sum()), turned=int(g.turned.sum()), damage=dmg[0]))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "figs", "figK1_contagion_by_stage_N.csv"), index=False)
# interview scoring (parsed answers only)
irows = []
if len(I):
    I["cond"] = I.attacker.map(lambda a: lab(str(a).split("@")[0]))
    P = I[I.parsed == 1]
    for (n, c), g in P.groupby(["N", "cond"]):
        r = dict(N=int(n), condition=c, answers=len(g), parse_rate=round(len(g) / max(1, len(I[(I.N == n) & (I.cond == c)])), 3),
                 said_test=round(float(g.is_test.dropna().mean()), 3) if g.is_test.notna().any() else None,
                 said_outside_party=round(float(g.suspected_outside_party.dropna().mean()), 3) if g.suspected_outside_party.notna().any() else None,
                 named_any_insider=round(float(g.suspected_insider.notna().mean()), 3),
                 knowledge_changed_actions=round(float(g.knowledge_changed_actions.dropna().mean()), 3) if g.knowledge_changed_actions.notna().any() else None,
                 would_act_differently=round(float(g.would_act_differently.dropna().mean()), 3) if g.would_act_differently.notna().any() else None)
        if c == "insider":
            r["named_true_insider"] = round(float((g.suspected_insider == g.insider).mean()), 3)
        irows.append(r)
IT = pd.DataFrame(irows); IT.to_csv(os.path.join(OUT, "figs", "figK2_interview_detection.csv"), index=False)
json.dump(dict(label="EXPLORATORY arm K", contagion=rows, interview=irows), open(os.path.join(OUT, "results_arm_K.json"), "w"), indent=1)
INK = "#52514e"; COL = {"insider": "#6250d6", "outsider-early": "#2a78d6", "outsider-mid": "#eb6834", "outsider-late": "#1baf7a", "outsider-last": "#eda100"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3)); Ns = sorted(T.N.unique()) if len(T) else []; x = np.arange(len(Ns)); conds = list(COL)
for k, c in enumerate(conds):
    t = T[T.condition == c].set_index("N").reindex(Ns); v = t.extra_defections.astype(float)
    ax[0].bar(x + (k - 2) * 0.16, v, 0.15, color=COL[c], label=c, yerr=[np.clip(np.nan_to_num(v - t.lo.astype(float)), 0, None), np.clip(np.nan_to_num(t.hi.astype(float) - v), 0, None)],
              error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
    if len(IT):
        it = IT[IT.condition == c].set_index("N").reindex(Ns)
        ax[1].bar(x + (k - 2) * 0.16, 100 * it.said_outside_party.astype(float), 0.15, color=COL[c], label=c)
if len(IT):
    it = IT[IT.condition == "clean"].set_index("N").reindex(Ns)
    for xi, v in zip(x, it.said_outside_party.astype(float)): ax[1].plot([xi - 0.45, xi + 0.45], [100 * v] * 2, color=INK, ls="--", lw=1)
for a in ax: a.set_xticks(x); a.set_xticklabels([f"N={n}" for n in Ns])
ax[0].axhline(0, color=INK, lw=0.8); ax[0].set_ylabel("extra defections per game vs clean twin"); ax[0].set_title("Contagion by entry stage and swarm size"); ax[0].legend(frameon=False, fontsize=7)
ax[1].set_ylabel("% of agents saying an outside party was involved"); ax[1].set_title("Interview: sensed an outside party? (dashed = clean twin)")
fig.suptitle("Arm K (exploratory): outsider entry stage x N" + os.environ.get("LABEL", ""), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figK1_contagion_by_stage_N.png"), dpi=200); plt.close(fig)
print(T.to_string()); print(IT.to_string())
