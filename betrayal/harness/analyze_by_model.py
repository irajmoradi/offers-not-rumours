"""Turned agents by each agent's OWN model in a mixed swarm (does mixing matter, or is it one model?).
Usage: analyze_by_model.py OUT_DIR ARM MODELS(comma, agent i -> MODELS[i % k]) SRC_DIR [SRC_DIR ...]
Per (attacker, model): honest agent-games, exposed, turned, turned/exposed, and post-fork defections per agent in the
attacked branch vs the SAME agent in its clean twin (same seed, same defense), with seed-bootstrap CIs."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, ARM, MODELS, SRCS = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); rng = np.random.default_rng(0)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[G.arm == ARM]
A = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "agents.jsonl"))]); A = A[(A.arm == ARM) & (A.insider == 0)]
short = lambda m: m.split("/")[-1]
A["model"] = A.agent.map(lambda i: short(MODELS[i % len(MODELS)]))
FORK = 1
def defs_per_agent(row):
    c = np.zeros(row["N"])
    for (r, a, t, s) in row["defection_events"]:
        if r > FORK: c[a] += 1
    return c
G["_defs"] = G.apply(defs_per_agent, axis=1)
key = lambda r: (r["cell"], r["defense"])
clean = {key(r): r["_defs"] for _, r in G[G.attacker == "clean"].iterrows()}
rows = []
for _, r in G[G.attacker.isin(["d_evil", "insider"])].iterrows():
    cd = clean.get(key(r))
    if cd is None: continue
    for i in range(r["N"]):
        if r["insider"] is not None and r["attacker"] == "insider" and i == r["insider"]: continue
        rows.append(dict(seed=r["seed"], cell=r["cell"], defense=r["defense"], attacker=r["attacker"], agent=i,
                         model=short(MODELS[i % len(MODELS)]), def_attacked=r["_defs"][i], def_clean=cd[i]))
D = pd.DataFrame(rows)
D = D.merge(A[["cell", "defense", "attacker", "agent", "exposed", "turned"]], on=["cell", "defense", "attacker", "agent"], how="left")
def ci(x, B=5000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (None, None, None)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return (round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
out = []
for (att, m), g in D.groupby(["attacker", "model"]):
    per_seed = g.groupby("seed").apply(lambda s: (s.def_attacked - s.def_clean).mean())
    da, dlo, dhi = ci(per_seed.values)
    ex, tu = int(g.exposed.sum()), int(g.turned.sum())
    out.append(dict(arm=ARM, attacker=att, model=m, agent_games=len(g), exposed=ex, turned=tu,
                    turned_per_exposed=round(tu / ex, 4) if ex else None, turned_per_agent_game=round(tu / len(g), 4),
                    defections_per_agent_attacked=round(g.def_attacked.mean(), 3), defections_per_agent_clean_twin=round(g.def_clean.mean(), 3),
                    extra_defections_per_agent=da, extra_lo=dlo, extra_hi=dhi,
                    turned_who_defected_no_more_than_in_clean_twin=int(((g.turned == 1) & (g.def_attacked <= g.def_clean)).sum())))
df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, "figs", f"fig_by_model_{ARM}.csv"), index=False)
INK = "#52514e"; C = {"d_evil": "#2a78d6", "insider": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
models = sorted(df.model.unique()); x = np.arange(len(models))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for k, att in enumerate(("d_evil", "insider")):
    d = df[df.attacker == att].set_index("model").reindex(models)
    ax[0].bar(x + (k - 0.5) * 0.38, 100 * d.turned_per_exposed.astype(float), 0.36, color=C[att], label={"d_evil": "D-Evil", "insider": "insider"}[att])
    for xi, (t, e) in zip(x + (k - 0.5) * 0.38, zip(d.turned, d.exposed)):
        ax[0].text(xi, 0.5, f"{int(t) if t == t else 0}/{int(e) if e == e else 0}", ha="center", fontsize=7, color=INK, rotation=90, va="bottom")
    v = d.extra_defections_per_agent.astype(float)
    ax[1].bar(x + (k - 0.5) * 0.38, v, 0.36, color=C[att], label={"d_evil": "D-Evil", "insider": "insider"}[att],
              yerr=[np.nan_to_num(v - d.extra_lo.astype(float)), np.nan_to_num(d.extra_hi.astype(float) - v)], error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
for a in ax:
    a.set_xticks(x); a.set_xticklabels(models, fontsize=8, rotation=10); a.legend(frameon=False, fontsize=8)
ax[0].set_ylabel("% of exposed agents turned (labels: turned/exposed)"); ax[0].set_title(f"Arm {ARM}: who gets turned, by the agent's own model")
ax[1].axhline(0, color=INK, lw=0.8); ax[1].set_ylabel("extra defections per agent vs its clean twin"); ax[1].set_title("Defections caused, by own model")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", f"fig_by_model_{ARM}.png"), dpi=200); plt.close(fig)
print(df.to_string())
