"""Arm T (pay timing) analysis. EXPLORATORY. Usage: analyze_t.py OUT_DIR SRC_DIR [...]
Ledger detection over time: at round r, the twin-free rule flags argmin(cumulative gives up to r), where gives = LLM gift
events plus delivered bribe payments (payments go through her give_key, so they are on the ledger). Ties are broken at
random with a fixed RNG. Only games still running at round r are counted."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[G.arm == "T"]
A = G[G.cond != "clean"]; rng = np.random.default_rng(17)
def ci(x):
    x = np.asarray(x, float); bs = rng.choice(x, (5000, len(x))).mean(1)
    return round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
ORDER = ["never_pays", "pays_now", "pays_later3", "pays_at_end"]
rows, curves, PER = [], [], []
for c in ORDER:
    g = A[A.cond == c]
    if not len(g): continue
    de = ci(g.defection_effect)
    det_by_r = {r: [] for r in range(2, 13)}
    final = []
    for _, r_ in g.iterrows():
        _mid = []
        x = r_["insider"]; N = r_["N"]
        for R in range(2, 13):
            if r_["rounds"] < R: continue
            giv = np.zeros(N)
            for (rd, a, t) in r_["gift_events"]:
                if rd <= R: giv[a] += 1
            for (pr, br, t) in (r_.get("payments") or []):
                if pr is not None and pr <= R: giv[br] += 1
            cand = np.flatnonzero(giv == giv.min()); _f = int(rng.choice(cand) == x); det_by_r[R].append(_f)
            if 6 <= R <= 10: _mid.append(_f)
        led = np.array(r_["ledger_gives"]); cand = np.flatnonzero(led == led.min()); final.append(int(rng.choice(cand) == x))
        PER.append(dict(cell=c, seed=int(r_["seed"]), final=final[-1], mid=float(np.mean(_mid)) if _mid else np.nan))
    for R, v in det_by_r.items():
        if v: curves.append(dict(cell=c, round=R, games=len(v), ledger_top1=round(float(np.mean(v)), 3)))
    rows.append(dict(cell=c, seeds=int(g.seed.nunique()), extra_defections=de[0], ed_lo=de[1], ed_hi=de[2],
                     key_damage=round(float(g.key_effect.mean()), 3), bribe_offers=int(g.bribe_offers.sum()), bribes_accepted=int(g.bribes_accepted.sum()),
                     payments_delivered=int(g.bribes_paid.sum()), payments_due_undelivered=int(g.pay_pending_undelivered.sum()), pay_failed=int(g.pay_failed.sum()),
                     ledger_top1_final=round(float(np.mean(final)), 3)))
T = pd.DataFrame(rows); C = pd.DataFrame(curves)
T.to_csv(os.path.join(OUT, "figs", "figT_paytiming.csv"), index=False); C.to_csv(os.path.join(OUT, "figs", "figT_detection_over_rounds.csv"), index=False)
json.dump(dict(label="EXPLORATORY arm T (pay timing)", cells=rows, detection_over_rounds=curves), open(os.path.join(OUT, "results_arm_T.json"), "w"), indent=1)
COL = {"never_pays": "#2a78d6", "pays_now": "#eb6834", "pays_later3": "#1baf7a", "pays_at_end": "#6250d6"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c in ORDER:
    d = C[C.cell == c]
    if len(d): ax[0].plot(d["round"], 100 * d.ledger_top1, "o-", color=COL[c], label=c, lw=2, ms=4)
ax[0].axhline(12.5, color="#52514e", ls=":", lw=1); ax[0].set_xlabel("round"); ax[0].set_ylabel("% games: ledger flags the true insider")
ax[0].set_title("Does a late payer hide longer? (ledger 'never gives', twin-free)"); ax[0].legend(frameon=False, fontsize=8)
v = T.extra_defections.astype(float)
ax[1].bar(T.cell, v, color=[COL[c] for c in T.cell], yerr=[np.clip(v - T.ed_lo, 0, None), np.clip(T.ed_hi - v, 0, None)], error_kw=dict(ecolor="#52514e", capsize=3))
for i_, r_ in T.iterrows(): ax[1].text(i_, 0.05, f"paid {r_.payments_delivered}\nundeliv. {r_.payments_due_undelivered}", ha="center", fontsize=7, color="white")
ax[1].set_ylabel("extra defections per game vs clean twin"); ax[1].set_title("Betrayal by pay timing"); ax[1].tick_params(axis="x", labelsize=8)
fig.suptitle(f"Arm T (exploratory): insider bribe+lie that always pays, by WHEN it pays; all-Gemma-3 N=8, {int(A.seed.nunique())} seeds", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figT_paytiming.png"), dpi=200); plt.close(fig)
pd.set_option("display.width", 250); print(T.to_string()); print(C.pivot_table(index="round", columns="cell", values="ledger_top1").to_string())

# Paired tests vs never_pays (same seed), EXPLORATORY: final ledger top-1 and mid-game (rounds 6-10) top-1.
import hashlib
P = pd.DataFrame(PER); tests = []
def signflip(d, name, B=200000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; obs = abs(d.mean())
    r = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    if len(d) <= 22:
        import itertools; m = np.array(list(itertools.product([1, -1], repeat=len(d)))); null = np.abs((m * d).mean(1))
    else: null = np.abs((r.choice([1, -1], (B, len(d))) * d).mean(1))
    return float(((null >= obs - 1e-12).sum() + (0 if len(d) <= 22 else 1)) / (len(null) + (0 if len(d) <= 22 else 1))), len(d)
for c in ORDER[1:]:
    for met in ("final", "mid"):
        m = P[P.cell == c].merge(P[P.cell == "never_pays"], on="seed", suffixes=("", "_np"))
        d = m[met] - m[met + "_np"]; name = f"T_{c}_vs_never_{met}"; p, n = signflip(d, name)
        tests.append(dict(test=name, n=n, diff=round(float(np.nanmean(d)), 3), p=round(p, 4)))
tests.sort(key=lambda t: t["p"]); k = len(tests); run = 0
for i, t in enumerate(tests): run = max(run, min(1, (k - i) * t["p"])); t["p_holm"] = round(run, 4)
R = json.load(open(os.path.join(OUT, "results_arm_T.json"))); R["paired_vs_never_pays"] = tests
R["note_tests"] = "Paired by seed vs never_pays; sign-flip (seeded Monte Carlo, 200k, n>22); Holm over these 6 tests; exploratory."
json.dump(R, open(os.path.join(OUT, "results_arm_T.json"), "w"), indent=1); P.to_csv(os.path.join(OUT, "figT_per_game_detection.csv"), index=False)
print(pd.DataFrame(tests).to_string())
