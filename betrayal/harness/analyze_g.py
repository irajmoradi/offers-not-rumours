"""Arm G (outsider briber) analysis. Exploratory: this arm was added after pre-registration, so there is no Holm family
and every result is labelled exploratory.
Usage: analyze_g.py OUT_DIR SRC_DIR [SRC_DIR ...]   (SRC_DIRs hold games.jsonl; v2.1 reference read from out/final_v2_1)"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; SRCS = sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))])
G = G[G.arm == "G"]
V = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "out", "final_v2_1", "games.jsonl"))])
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
def signflip(x, B=20000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    s = rng.choice([-1, 1], (B, len(x))); return float((((s * x).mean(1) >= x.mean() - 1e-12).sum() + 1) / (B + 1))
def ratio_ci(df, num, den, B=10000):
    seeds = df.seed.unique(); g = df.groupby("seed")[[num, den]].sum()
    est = g[num].sum() / max(1, g[den].sum()); bs = []
    for _ in range(B):
        p = g.loc[rng.choice(seeds, len(seeds))]; d = p[den].sum()
        if d: bs.append(p[num].sum() / d)
    return dict(rate=round(float(est), 4), k=int(g[num].sum()), n=int(g[den].sum()),
                ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)] if bs else None)
R = {"label": "EXPLORATORY (arm G added after pre-registration; no Holm family)", "seeds": int(G.seed.nunique()),
     "design": "outsider (D-Evil) bribe-and-renege; budget 4 offers/game (<=2/round); timing start round early=2, mid=4, late=6 "
               "(thirds of the effective horizon); recognition flagged (agent -1 + 'unknown sender' prefix) vs disguised "
               "(impersonated neighbour); Qwen-2.5-7B N=8 complete, defense none; shared prefix + clean twin per seed",
     "conditions": {}}
rows = []
for t in ("early", "mid", "late"):
    for r in ("flagged", "disguised"):
        d = G[G.attacker == f"obribe:{t}:{r}"]
        e = dict(defection_effect=ci(d.defection_effect.values), p_one_sided=signflip(d.defection_effect.values),
                 damage=ci(d.attack_effect.values), offers_per_game=ci(d.bribe_offers.values),
                 acceptance_per_seen=ratio_ci(d, "bribes_accepted_strict_seen", "bribes_seen"),
                 turned_per_honest_agent=ratio_ci(d, "turned", "honest"))
        R["conditions"][f"{t}|{r}"] = e
        rows.append(dict(timing=t, recognition=r, n_seeds=e["defection_effect"]["n"],
                         defection_effect=e["defection_effect"]["mean"], de_lo=e["defection_effect"]["ci95"][0], de_hi=e["defection_effect"]["ci95"][1],
                         p_one_sided=round(e["p_one_sided"], 4), damage_pp=100 * e["damage"]["mean"],
                         acceptance_per_seen=e["acceptance_per_seen"]["rate"], acc_k=e["acceptance_per_seen"]["k"], acc_n=e["acceptance_per_seen"]["n"],
                         acc_lo=(e["acceptance_per_seen"]["ci95"] or [None, None])[0], acc_hi=(e["acceptance_per_seen"]["ci95"] or [None, None])[1],
                         turned_rate=e["turned_per_honest_agent"]["rate"], turned_k=e["turned_per_honest_agent"]["k"], offers_per_game=e["offers_per_game"]["mean"]))
# factor contrasts, paired by seed
att = G[G.attacker.str.startswith("obribe")].copy()
att["timing"] = att.attacker.str.split(":").str[1]; att["recog"] = att.attacker.str.split(":").str[2]
p = att.pivot_table(index="seed", columns="recog", values="defection_effect", aggfunc="mean")
R["contrast_disguised_minus_flagged"] = dict(ci(p["disguised"] - p["flagged"]), p_two_sided=min(1, 2 * min(signflip((p["disguised"] - p["flagged"]).values), signflip((p["flagged"] - p["disguised"]).values))))
q = att.pivot_table(index="seed", columns="timing", values="defection_effect", aggfunc="mean")
R["contrast_early_minus_late"] = dict(ci(q["early"] - q["late"]), p_two_sided=min(1, 2 * min(signflip((q["early"] - q["late"]).values), signflip((q["late"] - q["early"]).values))))
R["pooled_outsider_bribe"] = dict(defection_effect=ci(att.groupby("seed").defection_effect.mean().values),
                                  p_one_sided=signflip(att.groupby("seed").defection_effect.mean().values),
                                  acceptance_per_seen=ratio_ci(att, "bribes_accepted_strict_seen", "bribes_seen"),
                                  turned_per_honest_agent=ratio_ci(att, "turned", "honest"))
# v2.1 references (main grid A, defense none; complete topology to match G, and all topologies)
A = V[(V.arm == "A") & (V.defense == "none")]
for lab, sub in [("complete", A[A.topology == "complete"]), ("all_topologies", A)]:
    for a in ("insider", "d_evil"):
        d = sub[sub.attacker == a]
        R[f"v2_1_ref_{a}_{lab}"] = dict(defection_effect=ci(d.groupby("seed").defection_effect.mean().values),
                                        turned_per_honest_agent=dict(rate=round(float(d.turned.sum() / max(1, d.honest.sum())), 4),
                                                                     k=int(d.turned.sum()), n=int(d.honest.sum())))
cost = {}
for s in SRCS:
    p_ = os.path.join(s, "cost_tally.json")
    if os.path.exists(p_):
        c = json.load(open(p_)); cost[os.path.basename(s)] = round(c.get("by_arm", {}).get("G", {}).get("usd", 0), 4)
R["cost_usd"] = dict(cost, total=round(sum(cost.values()), 4))
json.dump(R, open(os.path.join(OUT, "results_arm_G.json"), "w"), indent=1)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "figs", "figG1_defection_effect.csv"), index=False)
# figures
C = {"flagged": "#2a78d6", "disguised": "#eb6834"}; INK = "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(3)
for k, r in enumerate(("flagged", "disguised")):
    d = df[df.recognition == r]
    ax.bar(x + (k - 0.5) * 0.36, d.defection_effect, 0.34, color=C[r], label=f"outsider bribe, {r}",
           yerr=[d.defection_effect - d.de_lo, d.de_hi - d.defection_effect], error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
refs = [("v2.1 insider (H2b)", R["v2_1_ref_insider_complete"]["defection_effect"], "#1baf7a"), ("v2.1 D-Evil messages (H2a)", R["v2_1_ref_d_evil_complete"]["defection_effect"], "#b8b7b0")]
for j, (lab, e, col) in enumerate(refs):
    xx = 3.3 + j * 0.45
    ax.bar(xx, e["mean"], 0.4, color=col, label=lab, yerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]], error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
ax.set_xticks(list(x) + [3.525]); ax.set_xticklabels(["early (round 2)", "mid (round 4)", "late (round 6)", "references\n(v2.1, complete, no defense)"], fontsize=8)
ax.axhline(0, color=INK, lw=0.8); ax.set_ylabel("defections per game vs clean twin")
ax.set_title(f"Arm G (exploratory): outsider bribe-and-renege, {R['seeds']} seeds" + os.environ.get("LABEL", "")); ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figG1_defection_effect.png"), dpi=200); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, r in enumerate(("flagged", "disguised")):
    d = df[df.recognition == r]
    ax[0].bar(x + (k - 0.5) * 0.36, 100 * d.acceptance_per_seen, 0.34, color=C[r], label=r,
              yerr=[100 * (d.acceptance_per_seen - d.acc_lo), 100 * (d.acc_hi - d.acceptance_per_seen)], error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
    ax[1].bar(x + (k - 0.5) * 0.36, 100 * d.turned_rate, 0.34, color=C[r], label=r)
for a, t in [(ax[0], "% of seen bribes followed by the asked-for defection"), (ax[1], "% of honest agent-games turned")]:
    a.set_xticks(x); a.set_xticklabels(["early", "mid", "late"]); a.set_ylabel(t, fontsize=8); a.legend(frameon=False, fontsize=8)
ax[0].set_title("Bribe acceptance (strict)"); ax[1].set_title("Turned agents")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figG2_acceptance_turned.png"), dpi=200); plt.close(fig)
df[["timing", "recognition", "acceptance_per_seen", "acc_k", "acc_n", "acc_lo", "acc_hi", "turned_rate", "turned_k", "offers_per_game"]].to_csv(
    os.path.join(OUT, "figs", "figG2_acceptance_turned.csv"), index=False)
print(json.dumps({k: v for k, v in R.items() if k != "design"}, indent=1, default=str)[:5000])
