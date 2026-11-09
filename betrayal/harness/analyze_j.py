"""Arm J: bribe fulfillment schedule x briber type. EXPLORATORY.
Usage: analyze_j.py OUT_DIR SRC_DIR [...]
Per (briber, schedule): extra defections per honest agent vs its clean twin (post-fork, same seed), bribes accepted,
paid and pay_failed, damage; plus, for insider branches, ledger "never gives" detection: top-1 = the insider has the
lowest post-fork gives relative to its clean-twin self (ties broken uniformly at random, with a fixed RNG)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[G.arm == "J"]
rng = np.random.default_rng(7); FORK = 1
def per_agent(row, kind):
    c = np.zeros(row["N"])
    ev = row["defection_events"] if kind == "def" else row["gift_events"]
    for e in ev:
        if e[0] > FORK: c[e[1]] += 1
    return c
G["defv"] = G.apply(lambda r: per_agent(r, "def"), axis=1); G["givv"] = G.apply(lambda r: per_agent(r, "giv"), axis=1)
clean = {r.cell: r for r in G[G.attacker == "clean"].itertuples()}
def briber(r):
    return "insider" if r.attacker == "insider" else ("outsider-early (r2)" if ":early:" in r.attacker else "outsider-late (r6)")
rows = []
for r in G[G.attacker != "clean"].itertuples():
    c = clean.get(r.cell)
    if c is None: continue
    honest = [i for i in range(r.N) if not (r.attacker == "insider" and i == r.insider)]
    extra = float(np.mean([r.defv[i] - c.defv[i] for i in honest]))
    det = None
    if r.attacker == "insider":
        ga = r.givv.copy(); ga[int(r.insider)] += (r.bribes_paid if r.bribes_paid == r.bribes_paid else 0)   # payments are ledger gives
        dg = ga - c.givv; m = dg.min(); cands = np.flatnonzero(dg == m)
        det = int(rng.choice(cands) == r.insider)
    rows.append(dict(seed=r.seed, briber=briber(r), schedule=r.fulfillment, extra_def_per_agent=extra, damage=r.attack_effect,
                     accepted=r.bribes_accepted, paid=r.bribes_paid, pay_failed=r.pay_failed, offers=r.bribe_offers,
                     turned=r.turned, ledger_top1=det))
D = pd.DataFrame(rows)
def ci(x, B=5000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (None, None, None)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return (round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
SCH = ["never", "initial_only", "initial_sporadic", "sporadic", "always"]; BR = ["insider", "outsider-early (r2)", "outsider-late (r6)"]
out = []
for b in BR:
    for s in SCH:
        d = D[(D.briber == b) & (D.schedule == s)]
        if not len(d): continue
        e, lo, hi = ci(d.extra_def_per_agent); dm = ci(d.damage)
        lt = ci(d.ledger_top1.dropna()) if b == "insider" else (None, None, None)
        out.append(dict(briber=b, schedule=s, seeds=len(d), extra_def_per_agent=e, lo=lo, hi=hi, damage=dm[0],
                        accepted=int(d.accepted.sum()), paid=int(d.paid.sum()), pay_failed=int(d.pay_failed.sum()), offers=int(d.offers.sum()),
                        turned=int(d.turned.sum()), ledger_top1=lt[0], ledger_lo=lt[1], ledger_hi=lt[2]))
T = pd.DataFrame(out); T.to_csv(os.path.join(OUT, "figs", "figJ1_fulfillment.csv"), index=False)
json.dump(dict(label="EXPLORATORY arm J (scripted bribe fulfillment)", model=os.environ.get("J_MODEL_LABEL", "see games"),
               seeds=int(D.seed.nunique()), rows=out), open(os.path.join(OUT, "results_arm_J.json"), "w"), indent=1)
INK = "#52514e"; C = {"insider": "#2a78d6", "outsider-early (r2)": "#eb6834", "outsider-late (r6)": "#1baf7a"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3)); x = np.arange(len(SCH))
for k, b in enumerate(BR):
    t = T[T.briber == b].set_index("schedule").reindex(SCH); v = t.extra_def_per_agent.astype(float)
    ax[0].bar(x + (k - 1) * 0.27, v, 0.25, color=C[b], label=b, yerr=[np.clip(np.nan_to_num(v - t.lo.astype(float)), 0, None), np.clip(np.nan_to_num(t.hi.astype(float) - v), 0, None)],
              error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
t = T[T.briber == "insider"].set_index("schedule").reindex(SCH); v = 100 * t.ledger_top1.astype(float)
ax[1].bar(x, v, 0.5, color=C["insider"], yerr=[np.clip(np.nan_to_num(v - 100 * t.ledger_lo.astype(float)), 0, None), np.clip(np.nan_to_num(100 * t.ledger_hi.astype(float) - v), 0, None)],
          error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
for xi, (pd_, ac) in zip(x, zip(t.paid, t.accepted)):
    ax[1].text(xi, 2, f"paid {int(pd_) if pd_ == pd_ else 0}/{int(ac) if ac == ac else 0}", ha="center", fontsize=7, color="white")
ax[1].axhline(12.5, color=INK, lw=0.8, ls="--"); ax[1].text(4.4, 13.5, "chance", fontsize=7, color=INK)
for a in ax: a.set_xticks(x); a.set_xticklabels([s.replace("_", "\n") for s in SCH], fontsize=8)
ax[0].axhline(0, color=INK, lw=0.8); ax[0].set_ylabel("extra defections per honest agent vs clean twin"); ax[0].set_title("Does paying the bribe change betrayal?"); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_ylabel("% games: ledger ('never gives') finds the insider"); ax[1].set_title("Insider detection by the ledger, by payment schedule")
fig.suptitle("Arm J (exploratory): bribe fulfillment x briber type" + os.environ.get("LABEL", ""), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figJ1_fulfillment.png"), dpi=200); plt.close(fig)
print(T.to_string())
