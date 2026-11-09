"""Arm H (heterogeneous scaling) analysis. EXPLORATORY (added after pre-registration).
Usage: analyze_h.py OUT_DIR SRC_DIR [SRC_DIR ...]
Compares, per N, insider vs clean twin (no defense):
  homogeneous Qwen-2.5-7B = v2.1 arm C (N 4/8/16/32, 10 seeds)
  heterogeneous mixed     = arm H (N 4/16/32) + v2.1 arm D at N=8 (insider|none vs clean|none, 40 seeds; D also had
                            instruments/defense branches, but the insider|none vs clean|none contrast is the same design)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; SRCS = sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
H = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); H = H[H.arm == "H"]
V = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "out", "final_v2_1", "games.jsonl"))])
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
def summarize(df):
    ins, cl = df[df.attacker == "insider"], df[df.attacker == "clean"]
    return dict(insider_damage=ci(ins.attack_effect.values), defection_effect=ci(ins.defection_effect.values),
                clean_completion=ci(cl.completion.values), clean_defections=ci(cl.defections.values),
                turned_rate=dict(rate=round(float(ins.turned.sum() / max(1, ins.honest.sum())), 4), k=int(ins.turned.sum()), n=int(ins.honest.sum())),
                seeds=int(ins.seed.nunique()))
R = {"label": "EXPLORATORY (arm H added after pre-registration)", "homogeneous_qwen_v2_1_C": {}, "heterogeneous_mixed": {}}
C = V[V.arm == "C"]
for n, g in C.groupby("N"): R["homogeneous_qwen_v2_1_C"][int(n)] = summarize(g)
for n, g in H.groupby("N"): R["heterogeneous_mixed"][int(n)] = dict(summarize(g), source="arm H")
D = V[(V.arm == "D") & (V.defense == "none") & (V.attacker.isin(["insider", "clean"]))]
R["heterogeneous_mixed"][8] = dict(summarize(D), source="v2.1 arm D (insider|none vs clean|none)")
# heterogeneous - homogeneous damage difference per N (unpaired; different seeds/instances are fine: same seeds 0..k share instances)
diff = {}
for n in sorted(set(R["homogeneous_qwen_v2_1_C"]) & set(R["heterogeneous_mixed"])):
    a = H[(H.N == n) & (H.attacker == "insider")] if n != 8 else D[D.attacker == "insider"]
    b = C[(C.N == n) & (C.attacker == "insider")]
    m = a.groupby("seed").attack_effect.mean().rename("het").to_frame().join(b.groupby("seed").attack_effect.mean().rename("hom"), how="inner")
    diff[int(n)] = dict(paired_by_seed_n=int(len(m)), het_minus_hom=ci((m.het - m.hom).values) if len(m) else None)
R["het_minus_hom_damage_same_seeds"] = diff
cost = {}
for s in SRCS:
    p = os.path.join(s, "cost_tally.json")
    if os.path.exists(p):
        for k, v in json.load(open(p)).get("by_arm", {}).items():
            if k.startswith("H_"): cost[k] = round(cost.get(k, 0) + v["usd"], 4)
R["cost_usd"] = dict(cost, total=round(sum(cost.values()), 4))
json.dump(R, open(os.path.join(OUT, "results_arm_H.json"), "w"), indent=1)
rows = []
for kind, dd in (("homogeneous_qwen", R["homogeneous_qwen_v2_1_C"]), ("heterogeneous_mixed", R["heterogeneous_mixed"])):
    for n, e in sorted(dd.items()):
        rows.append(dict(swarm=kind, N=n, seeds=e["seeds"], insider_damage=e["insider_damage"]["mean"], dmg_lo=e["insider_damage"]["ci95"][0],
                         dmg_hi=e["insider_damage"]["ci95"][1], defection_effect=e["defection_effect"]["mean"], de_lo=e["defection_effect"]["ci95"][0],
                         de_hi=e["defection_effect"]["ci95"][1], clean_completion=e["clean_completion"]["mean"], turned_rate=e["turned_rate"]["rate"],
                         turned_k=e["turned_rate"]["k"], source=e.get("source", "v2.1 arm C")))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "figs", "figH1_damage_vs_N_hom_vs_het.csv"), index=False)
INK = "#52514e"; COL = {"homogeneous_qwen": "#2a78d6", "heterogeneous_mixed": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for kind, lab in (("homogeneous_qwen", "homogeneous (Qwen-2.5-7B)"), ("heterogeneous_mixed", "heterogeneous (qwen/llama/gemma-3/gpt-oss)")):
    d = df[df.swarm == kind].sort_values("N"); off = -0.04 if kind.startswith("hom") else 0.04
    xs = np.log2(d.N) + off
    ax[0].errorbar(xs, 100 * d.insider_damage, yerr=[100 * (d.insider_damage - d.dmg_lo), 100 * (d.dmg_hi - d.insider_damage)], fmt="o-", color=COL[kind],
                   lw=2, ms=7, capsize=3, label=lab)
    ax[1].errorbar(xs, d.defection_effect, yerr=[d.defection_effect - d.de_lo, d.de_hi - d.defection_effect], fmt="o-", color=COL[kind], lw=2, ms=7, capsize=3, label=lab)
for a in ax:
    a.set_xticks(np.log2([4, 8, 16, 32])); a.set_xticklabels(["4", "8", "16", "32"]); a.set_xlabel("swarm size N"); a.axhline(0, color=INK, lw=0.8)
ax[0].set_ylabel("insider damage: completion drop vs clean twin (pp)"); ax[0].set_title("Insider damage vs N")
ax[1].set_ylabel("extra defections per game vs clean twin"); ax[1].set_title("Insider-induced defections vs N")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Arm H (exploratory): homogeneous vs heterogeneous swarms across sizes (N=8 mixed point = v2.1 arm D)" + os.environ.get("LABEL", ""), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figH1_damage_vs_N_hom_vs_het.png"), dpi=200); plt.close(fig)
print(df.to_string()); print(json.dumps(diff, indent=1)); print("cost", R["cost_usd"])
