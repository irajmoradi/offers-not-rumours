"""TOFIX analysis, as pre-registered in prespec_tofix.md (frozen with it; run unchanged).
Usage: analyze_tofix.py OUT_DIR SRC_DIR          (reads SRC/games.jsonl, SRC/call_audit.json, SRC/cost_tally.json;
                                                  arm U reference values from out/arm_U_main + out/arm_U_pilot)
Primary (paired by seed, seeds 800-819, n=20; seed 899 smoke excluded):
  P1 key damage to honest agents, BASE insider vs clean twin: key_effect = keys_frac(twin, excl. insider)
     - keys_frac(BASE, excl. insider). Two-sided (H0: mean 0).
  P2 bribe-only minus lie-only defections (both share the twin, so = difference in extra defections). Directional (> 0).
Tests: two-sided paired sign-flip on per-seed differences (statistic = |sum|), EXACT over all 2^n sign vectors (n <= 20).
95% CI: seed bootstrap, 5,000 resamples, percentile, RNG seeded by sha256 of the test name. Holm over {P1, P2}.
A/A null: clean_AA vs clean twin on the same two metrics (defections; key_effect with the same agent excluded).
Arm U reference (old, timeout-affected): same metrics recomputed from the arm U N=8 all-Gemma closed group (seeds 0-35).
Descriptive old-vs-new: new mean minus old mean, independent-samples bootstrap CI (unpaired; different seeds).
Call audit per branch: first-attempt error-body rate (= calls the old path would have turned into idle moves), failed
rate after the fix, recovered calls, genuine empty completions, and the forced-idle share (turns with action "")."""
import sys, os, json, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
OUT, SRC = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
HD = os.path.dirname(os.path.abspath(__file__)); OLD = [os.path.join(HD, "..", "out", d) for d in ("arm_U_main", "arm_U_pilot")]
SEEDS = list(range(800, 820)); B_BOOT = 5_000
C, AA, BASE, BO, LO = "clean|none", "clean_AA|none", "BASE", "insider_bribe_only", "insider_lie_only"
G = pd.DataFrame([json.loads(l) for l in open(os.path.join(SRC, "games.jsonl"))])
G = G[(G.arm == "TF") & (G.seed.isin(SEEDS)) & (~G.smoke.astype(bool))].copy()
def _rng(name): return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
def signflip(d):
    d = np.asarray(d, float); n = len(d); obs = abs(d.sum())
    if n == 0: return None
    if obs < 1e-12: return 1.0
    assert n <= 20
    idx = np.arange(2 ** n, dtype=np.int64); S = np.zeros(2 ** n)
    for j in range(n): S += np.where((idx >> j) & 1, -d[j], d[j])
    return float((np.abs(S) >= obs - 1e-9).mean())
def boot(d, name):
    d = np.asarray(d, float)
    if len(d) == 0: return None
    bs = _rng("boot|" + name).choice(d, (B_BOOT, len(d))).mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
def col(df, branch, c):
    g = df[df.branch == branch]; return dict(zip(g.seed, g[c]))
def test(name, d, direction):
    p = signflip(d)
    return dict(test=name, n=len(d), mean=(round(float(np.mean(d)), 3) if len(d) else None), ci95=boot(d, name), p=p,
                p_method="exact sign-flip (2^n)", direction=direction, nonzero_seeds=int(sum(abs(v) > 1e-12 for v in d)))
res = dict(primary=[], aa_null=[], per_branch=[], reference_arm_U={}, old_vs_new=[], call_audit={}, seeds_complete=None)

# --- primary
k = col(G, BASE, "key_effect"); s1 = sorted(k); P1 = test("P1 key damage, BASE insider vs clean twin", [k[s] for s in s1], "!=")
a, b = col(G, BO, "defections"), col(G, LO, "defections"); s2 = sorted(set(a) & set(b))
P2 = test("P2 bribe-only minus lie-only defections", [a[s] - b[s] for s in s2], ">")
res["primary"] = [P1, P2]
o = sorted(range(2), key=lambda i: res["primary"][i]["p"] if res["primary"][i]["p"] is not None else 9); run = 0
for j, i in enumerate(o):
    t = res["primary"][i]
    if t["p"] is None: t["p_holm"] = None; continue
    run = max(run, min(1.0, (2 - j) * t["p"])); t["p_holm"] = round(run, 5)
for t in res["primary"]:
    good = t["mean"] is not None and (t["direction"] == "!=" or (t["direction"] == ">" and t["mean"] > 0))
    t["verdict"] = "supported" if (t.get("p_holm") is not None and t["p_holm"] < 0.05 and good) else "not supported"
# --- A/A null
for c_, nm in (("defection_effect", "A/A defections (clean_AA - clean)"), ("key_effect", "A/A key effect (clean - clean_AA, insider agent excluded)"),
               ("key_effect_all", "A/A key effect (all agents)")):
    v = col(G, AA, c_); res["aa_null"].append(test(nm, [v[s] for s in sorted(v)], "!="))
# --- per branch (exploratory, descriptive + sign-flip vs twin)
for br in (AA, BASE, BO, LO):
    g = G[G.branch == br].sort_values("seed")
    row = dict(branch=br, seeds=int(g.seed.nunique()))
    for c_ in ("defection_effect", "key_effect", "key_effect_all", "attack_effect"):
        if c_ in g and g[c_].notna().any():
            t = test(f"{br} {c_}", list(g[c_].astype(float)), "!="); row[c_] = dict(mean=t["mean"], ci95=t["ci95"], p=t["p"])
    if br != AA and "key_effect_vs_AA" in g:
        t = test(f"{br} key_effect_vs_AA", list(g["key_effect_vs_AA"].astype(float)), "!="); row["key_effect_vs_AA"] = dict(mean=t["mean"], ci95=t["ci95"], p=t["p"])
    row.update(turned=int(g.turned.sum()), bribe_offers=int(g.bribe_offers.sum()), bribes_accepted=int(g.bribes_accepted.sum()),
               defections_total=int(g.defections.sum()))
    res["per_branch"].append(row)
cl = G[G.branch == C]; res["clean_twin"] = dict(seeds=int(cl.seed.nunique()), defections_total=int(cl.defections.sum()),
                                               keys_frac=round(float(cl.keys_frac.mean()), 4), completion=round(float(cl.completion.mean()), 4))
res["seeds_complete"] = int(sum(1 for s, g in G.groupby("seed") if g.branch.nunique() == 5))

# --- arm U reference (old, timeout-affected), same metrics
O = pd.DataFrame([json.loads(l) for d in OLD for l in open(os.path.join(d, "games.jsonl"))])
O = O[(O.arm == "U") & (O.N == 8) & (O.mix == "gemma") & (O.records == "closed")]
ob = {"BASE": "BASE", BO: "method=insider_bribe_only", LO: "method=insider_lie_only"}
ok = col(O, ob["BASE"], "key_effect"); oa, olo = col(O, ob[BO], "defections"), col(O, ob[LO], "defections")
od = [oa[s] - olo[s] for s in sorted(set(oa) & set(olo))]
def mci(x, name): x = list(x); return dict(n=len(x), mean=round(float(np.mean(x)), 3), ci95=boot(x, name))
res["reference_arm_U"] = dict(P1_key_damage_BASE=mci(ok.values(), "old P1"), P2_bribe_minus_lie=mci(od, "old P2"),
                              bribe_only_extra_defections=mci(O[O.branch == ob[BO]].defection_effect, "old BO"),
                              lie_only_extra_defections=mci(O[O.branch == ob[LO]].defection_effect, "old LO"),
                              BASE_extra_defections=mci(O[O.branch == "BASE"].defection_effect, "old BASE"))
for nm, new, old in (("P1 key damage", [k[s] for s in s1], list(ok.values())), ("P2 bribe-lie", [a[s] - b[s] for s in s2], od)):
    if new and old:
        r1, r2 = _rng("ovn1|" + nm), _rng("ovn2|" + nm); nn, oo = np.asarray(new, float), np.asarray(old, float)
        bs = r1.choice(nn, (B_BOOT, len(nn))).mean(1) - r2.choice(oo, (B_BOOT, len(oo))).mean(1)
        res["old_vs_new"].append(dict(metric=nm, new_minus_old=round(float(nn.mean() - oo.mean()), 3),
                                      ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]))
# --- call audit (new) + forced-idle shares (new vs old arm U, same measure as out/empty_reply_audit.md)
ca = json.load(open(os.path.join(SRC, "call_audit.json")))
tot = {}
for br, c_ in ca["stats"].items():
    n = max(1, c_.get("calls", 0))
    res["call_audit"][br] = dict(calls=c_.get("calls", 0), first_attempt_error_body_rate=round(c_.get("first_attempt_error_body", 0) / n, 4),
                                 any_error_body_rate=round(c_.get("calls_with_any_error_body", 0) / n, 4),
                                 failed_rate=round(c_.get("failed", 0) / n, 4), failed=c_.get("failed", 0), recovered=c_.get("recovered", 0),
                                 empty_success_rate=round(c_.get("empty_success", 0) / n, 4), error_bodies=c_.get("error_bodies", 0),
                                 error_codes=ca["error_codes"].get(br, {}))
    for kk, vv in c_.items(): tot[kk] = tot.get(kk, 0) + vv
n = max(1, tot.get("calls", 0))
res["call_audit"]["ALL"] = dict(calls=tot.get("calls", 0), first_attempt_error_body_rate=round(tot.get("first_attempt_error_body", 0) / n, 4),
                                failed_rate=round(tot.get("failed", 0) / n, 4), failed=tot.get("failed", 0), recovered=tot.get("recovered", 0),
                                empty_success_rate=round(tot.get("empty_success", 0) / n, 4), stop=ca.get("stop"))
idle_new = {br: round(float(g.n_blank_action_all.sum() / max(1, g.n_turns_all.sum())), 4) for br, g in G.groupby("branch")}
oldacts = {}
for l in open(os.path.join(OLD[0], "actions.jsonl")):
    r = json.loads(l)
    if r.get("N") == 8 and "-N8-gemma-closed" in r["cell"]:
        t_ = oldacts.setdefault(r["branch"], [0, 0]); t_[0] += sum(x_[2] == "" for x_ in r["actions"]); t_[1] += len(r["actions"])
om = {C: "clean|none", BASE: "BASE", BO: "method=insider_bribe_only", LO: "method=insider_lie_only"}
res["forced_idle_share"] = {br: dict(new=idle_new.get(br), old_arm_U_main=(round(oldacts[om[br]][0] / oldacts[om[br]][1], 4) if om.get(br) in oldacts else None))
                            for br in (C, AA, BASE, BO, LO)}
ct = json.load(open(os.path.join(SRC, "cost_tally.json")))
res["cost"] = dict(usd=round(ct["usd"], 4), calls=ct["calls"], failed=ct["failed"], empty=ct["empty"], missing_cost=ct["missing_cost"],
                   by_arm={a_: round(v_["usd"], 4) for a_, v_ in ct["by_arm"].items()})
json.dump(res, open(os.path.join(OUT, "results_tofix.json"), "w"), indent=1, default=str)
print("== PRIMARY (Holm over 2)")
for t in res["primary"]: print(" ", json.dumps(t))
print("== A/A"); [print(" ", json.dumps(t)) for t in res["aa_null"]]
print("== per branch"); [print(" ", json.dumps(r)) for r in res["per_branch"]]
print("== reference arm U", json.dumps(res["reference_arm_U"])); print("== old vs new", json.dumps(res["old_vs_new"]))
print("== call audit", json.dumps(res["call_audit"])); print("== forced idle", json.dumps(res["forced_idle_share"]))
print("== cost", json.dumps(res["cost"]), "seeds complete", res["seeds_complete"])
