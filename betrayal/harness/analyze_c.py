"""Arm C (contagion) analysis. EXPLORATORY. Usage: analyze_c.py OUT_DIR SRC_DIR [...]"""
import sys, os, json, hashlib, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); ARM = os.environ.get("ARM", "C"); G = G[G.arm == ARM]; NN = int(G.N.iloc[0]) if len(G) else 8
A = G[G.cond != "clean"]; rng = np.random.default_rng(13)
def ci(x):
    x = np.asarray(x, float); bs = rng.choice(x, (5000, len(x))).mean(1)
    return round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
rows = []
for c, g in A.groupby("cond"):
    gens, recruits = [], []
    for log in g.contagion_log:
        if isinstance(log, list):
            gens += [e["generation"] for e in log]
            recruits += list(collections.Counter(e["recruited_by"] for e in log).values())
    de, kd, cd = ci(g.defection_effect), ci(g.key_effect), ci(g.attack_effect)
    rows.append(dict(cell=c, seeds=int(g.seed.nunique()), extra_defections=de[0], ed_lo=de[1], ed_hi=de[2], key_damage=kd[0],
                     completion_damage=cd[0], turned=int(g.turned.sum()), honest_agent_games=int(g.honest.sum()),
                     bribe_offers=int(g.bribe_offers.sum()), bribes_accepted=int(g.bribes_accepted.sum()),
                     recruited_total=len(gens), games_with_recruit=int(sum(1 for l in g.contagion_log if isinstance(l, list) and l)),
                     max_hop=max(gens) if gens else 0, hop_counts=dict(collections.Counter(gens)),
                     mean_recruits_per_recruiter=round(float(np.mean(recruits)), 3) if recruits else None))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "figs", "figC_contagion.csv"), index=False)
P = A.pivot_table(index="seed", columns="cond", values="defection_effect")
def paired(a, b):
    if a not in P or b not in P: return None
    d = (P[a] - P[b]).dropna().values
    r = np.random.default_rng(int(hashlib.sha256(f"{a}-{b}".encode()).hexdigest()[:16], 16)); s_ = r.choice([-1, 1], (20000, len(d)))
    p_ = float(((np.abs((s_ * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / 20001); c_ = ci(d)
    return dict(diff=c_[0], ci95=[c_[1], c_[2]], p_two_sided=round(p_, 4), n=len(d))
comps = {"contagion_effect@1key": paired("outsider+contagion", "outsider_only"),
         "contagion_effect@3key": paired("outsider+contagion_3key", "outsider_only_3key"),
         "outsider_bribe_size_effect(3-1)@no_contagion": paired("outsider_only_3key", "outsider_only"),
         "outsider_bribe_size_effect(3-1)@contagion": paired("outsider+contagion_3key", "outsider+contagion")}
sizes = {c: dict(__import__("collections").Counter(x for l in g.outsider_offer_sizes if isinstance(l, list) for x in l)) for c, g in A.groupby("cond")}
res = dict(label=f"EXPLORATORY arm {ARM} N={NN} (contagion x outsider bribe size). Recruits always bribe with Jennifer's 1-key template; only the OUTSIDER's offer size varies.",
           cells=rows, paired=comps, outsider_offer_size_counts=sizes)
json.dump(res, open(os.path.join(OUT, "results_arm_C.json"), "w"), indent=1, default=int)
dd = (comps["contagion_effect@1key"] or {}).get("diff", 0); p = (comps["contagion_effect@1key"] or {}).get("p_two_sided", 1)
INK = "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
v = T.extra_defections.astype(float)
ax[0].bar([c.replace("outsider", "out").replace("_", " ") for c in T.cell], v, color=["#2a78d6", "#6250d6", "#eb6834", "#eda100"][:len(T)],
          yerr=[np.clip(v - T.ed_lo, 0, None), np.clip(T.ed_hi - v, 0, None)], error_kw=dict(ecolor=INK, capsize=3))
ax[0].tick_params(axis="x", labelsize=8)
ax[0].set_ylabel("extra defections per game vs clean twin"); ax[0].set_title("Contagion x outsider bribe size (recruits: 1-key)")
hc = {}
for h_ in T[T.cell.str.contains("contagion")].hop_counts:
    for k_, v_ in (h_ or {}).items(): hc[k_] = hc.get(k_, 0) + v_
ks = sorted(hc) if hc else []
ax[1].bar([str(k) for k in ks], [hc[k] for k in ks], color="#1baf7a"); ax[1].set_xlabel("hop depth (generation)"); ax[1].set_ylabel("agents recruited")
ax[1].set_title("Chain depth of recruits (both contagion cells, 1- and 3-key)")
fig.suptitle(f"Arm C (exploratory): outsider bribe with vs without contagion, all-Gemma-3 N={NN}, {int(A.seed.nunique())} seeds", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figC_contagion.png"), dpi=200); plt.close(fig)
pd.set_option("display.width", 250); print(T.to_string()); print(json.dumps(comps, indent=1)); print(sizes)
