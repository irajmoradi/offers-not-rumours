"""Arm DB (disclosure bounty) analysis. EXPLORATORY. Usage: analyze_db.py OUT_DIR SRC_DIR [...]
DB0 = arm U BASE insider vs its clean twin; DB1 = BASE + disclosure rule/bounty/forfeit vs a clean twin WITH the rule.
Primary: extra defections/game vs the (same-rule) twin. Bounty/forfeit mint/remove keys by construction, so key and
completion damage are shown with that caveat only."""
import sys, os, json, hashlib, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))]); G = G[G.arm == "DB"]
rng = np.random.default_rng(23)
def ci(x):
    x = np.asarray(x, float); bs = rng.choice(x, (5000, len(x))).mean(1)
    return round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
def signflip(d, name, B=200000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; obs = abs(d.mean())
    r = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    null = np.abs((r.choice([1, -1], (B, len(d))) * d).mean(1))
    return float(((null >= obs - 1e-12).sum() + 1) / (B + 1)), len(d)
def top1(g):
    hit, n = [], 0
    for _, r in g.iterrows():
        ab = [q["about"] for q in (r.db_reports or []) if q.get("about") is not None]
        if not ab: continue
        c = collections.Counter(ab); m = max(c.values()); cand = sorted(a for a, v in c.items() if v == m)
        hit.append(int(rng.choice(cand) == r.insider))
    return (round(float(np.mean(hit)), 3) if hit else None), len(hit)

rows = []
for cond in ("DB0_base", "DB1_disclosure", "clean_disclosure"):
    g = G[G.cond == cond]
    if not len(g): continue
    rep = [q for L in g.get("db_reports", pd.Series([[]] * len(g))) if isinstance(L, list) for q in L]
    bri = [(b, r_.rounds) for _, r_ in g.iterrows() for b in (r_.get("db_bribes") if isinstance(r_.get("db_bribes"), list) else [])]
    delivered = [b for b, R in bri if b["round"] + 1 <= R]
    row = dict(cond=cond, seeds=int(g.seed.nunique()))
    if cond != "clean_disclosure":
        de = ci(g.defection_effect); row.update(extra_defections=de[0], ed_lo=de[1], ed_hi=de[2], key_damage=round(float(g.key_effect.mean()), 3),
                                                completion_damage=round(float(g.attack_effect.mean()), 3),
                                                bribe_offers=int(g.bribe_offers.sum()), bribes_accepted=int(g.bribes_accepted.sum()),
                                                accept_rate=round(float(g.bribes_accepted.sum() / max(1, g.bribe_offers.sum())), 3))
    if cond != "DB0_base":
        t1, nt1 = top1(g)
        row.update(reports=len(rep), verified_reports=sum(q["verified"] for q in rep), false_reports=sum(not q["verified"] for q in rep),
                   false_reports_per_game=round(sum(not q["verified"] for q in rep) / len(g), 3),
                   near_miss_reports=sum((not q["verified"]) and q["score"] >= 0.3 for q in rep),
                   verified_about_is_briber=round(float(np.mean([q["about_is_briber"] for q in rep if q["verified"]])), 3) if any(q["verified"] for q in rep) else None,
                   bribes_delivered=len(delivered), bribes_reported_within_1=sum(b["reported_round"] is not None for b in delivered),
                   share_reported_within_1=round(sum(b["reported_round"] is not None for b in delivered) / max(1, len(delivered)), 3) if delivered else None,
                   accepted_and_reported=sum(b["accepted"] and b["reported_round"] is not None for b in delivered),
                   bounties=sum(q.get("bounty_key") is not None for q in rep),
                   forfeits=sum(l.get("forfeit_key") is not None for L in g.db_forfeits if isinstance(L, list) for l in L),
                   report_top1_insider=t1, games_with_reports=nt1, games=len(g))
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "figs", "figDB_disclosure.csv"), index=False)
# paired DB1 vs DB0 by seed
A = G[G.cond == "DB0_base"].set_index("seed"); B_ = G[G.cond == "DB1_disclosure"].set_index("seed"); s = A.index.intersection(B_.index)
tests = []
for met, f in (("extra_defections", lambda d: d.defection_effect), ("bribes_accepted", lambda d: d.bribes_accepted),
               ("key_damage", lambda d: d.key_effect)):
    d = (f(B_.loc[s]) - f(A.loc[s])).values.astype(float); p, n = signflip(d, f"DB_{met}")
    lo, hi = np.percentile(rng.choice(d, (5000, len(d))).mean(1), [2.5, 97.5])
    tests.append(dict(test=f"DB1-DB0 {met}", n=n, diff=round(float(d.mean()), 3), ci95=[round(float(lo), 3), round(float(hi), 3)], p=round(p, 4)))
res = dict(label="EXPLORATORY arm DB (disclosure bounty), all-Gemma-3 N=8, arm U BASE insider", cells=rows, paired=tests,
           note="DB1 vs its own clean twin with the rule; bounty/forfeit change keys by construction (key/completion damage caveat). "
                "Sign-flip seeded Monte Carlo 200k; no multiplicity correction (3 descriptive tests).")
json.dump(res, open(os.path.join(OUT, "results_arm_DB.json"), "w"), indent=1, default=str)
# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
d0 = T.set_index("cond")
cs = [c for c in ("DB0_base", "DB1_disclosure") if c in d0.index]
ax[0].bar(range(len(cs)), d0.loc[cs, "extra_defections"], color=["#2a78d6", "#1baf7a"][:len(cs)],
          yerr=[d0.loc[cs, "extra_defections"] - d0.loc[cs, "ed_lo"], d0.loc[cs, "ed_hi"] - d0.loc[cs, "extra_defections"]], capsize=4, ecolor="#52514e")
ax[0].set_xticks(range(len(cs))); ax[0].set_xticklabels(["no rule (DB0)", "disclosure bounty (DB1)"][:len(cs)])
ax[0].set_ylabel("extra defections per game vs clean twin"); ax[0].set_title("Betrayal (95% CI)")
if "DB1_disclosure" in d0.index:
    r1 = d0.loc["DB1_disclosure"]
    vals = [r1.share_reported_within_1 or 0, r1.accept_rate, d0.loc["DB0_base", "accept_rate"] if "DB0_base" in d0.index else 0, r1.report_top1_insider or 0]
    labs = ["bribes reported\nwithin 1 round", "acceptance\n(DB1)", "acceptance\n(DB0)", "reports' top-1\n= insider"]
    ax[1].bar(range(4), vals, color=["#1baf7a", "#1baf7a", "#2a78d6", "#eda100"])
    for i_, v in enumerate(vals):
        na = (i_ == 3 and r1.report_top1_insider is None) or (i_ == 0 and r1.share_reported_within_1 is None)
        ax[1].text(i_, v + 0.01, "n/a (no reports)" if na else f"{v:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("share")
    ax[1].set_title(f"Disclosure (false reports: {int(r1.false_reports)} in {int(r1.games)} games)")
for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e5e0"); a.set_axisbelow(True)
fig.suptitle(f"Arm DB (exploratory): disclosure bounty vs a bribe+lie insider; all-Gemma-3 N=8, {int(G.seed.nunique())} seeds", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figDB_disclosure.png"), dpi=200); plt.close(fig)
pd.set_option("display.width", 250); print(T.T.to_string()); print(pd.DataFrame(tests).to_string())
