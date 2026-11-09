"""Family-wide multiplicity correction for every reported inferential test in the betrayal study (planC_v2).
Reads raw p-values from each arm's results file (not from the paper or readouts), plus the few tests reported only in
READOUT_ADDENDUM_GHGR.md (entered by hand below, with their source). Families:
  PREREG_v21_primary  : H1 leak-free (final_v2_1/results.json)              -- its own family, as pre-registered
  PREREG_v21_secondary: H2a, H2b, H3, H4 (final_v2_1/results.json)          -- its own family, Holm as pre-registered
  PREREG_DC           : H-DC1..3 (arm_DC_v2_2/results_arm_DC.json)          -- its own family, Holm as pre-registered
  EXPLORATORY         : every other test in arms F, G, G_ada, U, UF, C, C16, dose(C2K), DB, SE, T, J -- ONE family
Outputs out/familywide_correction.csv and out/familywide_correction.md. Holm and BH are computed within each family;
for reference also across ALL tests pooled. p == 0 from a 20,000-draw sign-flip MC is set to its floor 1/20001.
Written by p1-critic-plan (independent of the per-arm analyze_*.py scripts). Arms I, H, K report CIs only (no p);
Keys & God (E1, swarm_hackathon) and single-agent / in-model tests are outside this script."""
import json, csv, os, re
O = f"{os.environ.get('PLANC_ROOT', '.')}/out"
FLOOR = 1 / 20001
T = []   # dict(family, arm, test, model, effect, p_raw, src, note)

def add(family, arm, test, p, model, effect=None, src="", note=""):
    if p is None: return
    p = float(p)
    if p == 0.0: p, note = FLOOR, (note + " p=0 from 20k MC -> floor").strip()
    T.append(dict(family=family, arm=arm, test=test, model=model, effect=effect, p_raw=p, src=src, note=note))

def J(path): return json.load(open(f"{O}/{path}"))
GEM, QW, FMIX = "Gemma-3-12B", "Qwen-2.5-7B", "arm-F mix (gemma-4-31b/26b, qwen3.8-27b, deepseek-v4-flash)"

# --- pre-registered v2.1
d = J("final_v2_1/results.json")
add("PREREG_v21_primary", "v2.1", "H1 leak-free: combined - best single detector", d["H1_leakfree"]["delta_combined_minus_best"]["p_one_sided"], QW,
    d["H1_leakfree"]["delta_combined_minus_best"].get("est"), "final_v2_1/results.json H1_leakfree", "one-sided")
for k, v in d["secondary_holm"].items():
    add("PREREG_v21_secondary", "v2.1", k, v["p_one_sided"], QW, (v.get("per_seed_mean") or {}).get("mean"), f"final_v2_1/results.json secondary_holm/{k}", "one-sided")
# --- pre-registered DC
for t in J("arm_DC_v2_2/results_arm_DC.json")["tests"]:
    name = next(t[k] for k in ("name", "test", "label", "comparison", "hypothesis") if k in t) if any(k in t for k in ("name", "test", "label", "comparison", "hypothesis")) else str(t)[:60]
    add("PREREG_DC", "DC", name, t["p"], GEM, t.get("diff", t.get("effect")), "arm_DC_v2_2/results_arm_DC.json tests")

E = "EXPLORATORY"
# --- arm F (v2.1 hypotheses re-run on stronger models; post-hoc arm)
d = J("arm_F_v2_2/results.json")
for k in ("H1_primary", "H1_leakfree"):
    add(E, "F", f"{k}: combined - best", d[k]["delta_combined_minus_best"]["p_one_sided"], FMIX, None, f"arm_F_v2_2/results.json {k}", "one-sided")
for k, v in d["secondary_holm"].items():
    if k.startswith("H4"): continue
    add(E, "F", k, v["p_one_sided"], FMIX, (v.get("per_seed_mean") or {}).get("mean"), f"arm_F_v2_2/results.json secondary_holm/{k}", "one-sided")
# --- arm G (Qwen) and G on Adaption (arm-F mix, 3 seeds)
for arm, path, model in (("G", "arm_G_v2_2/results_arm_G.json", QW), ("G_ada", "arm_G_ada_v2_2/results_arm_G.json", FMIX)):
    d = J(path)
    for c, v in d["conditions"].items():
        add(E, arm, f"outsider bribe {c} > 0", v["p_one_sided"], model, v["defection_effect"]["mean"], f"{path} conditions/{c}", "one-sided")
    for k in ("contrast_disguised_minus_flagged", "contrast_early_minus_late"):
        add(E, arm, k, d[k]["p_two_sided"], model, d[k].get("mean", d[k].get("diff")), f"{path} {k}")
    add(E, arm, "pooled outsider bribe > 0", d["pooled_outsider_bribe"]["p_one_sided"], model,
        (d["pooled_outsider_bribe"].get("defection_effect") or {}).get("mean"), f"{path} pooled_outsider_bribe", "one-sided")
# arm G (Qwen) pooled: the results file holds a 20k-draw MC one-sided p (0.00025); 14 of 14 non-zero seeds are positive,
# so the exact two-sided sign-flip p is 2^-13 = 1.2e-4 (as printed in P1).
# Sidedness: P1 reports post-hoc sign-flip tests TWO-sided. Arms G, G_ada and F's secondary tests are stored one-sided;
# convert. Sign-flip nulls are symmetric, so for an effect > 0, p2 = 2*p1 exactly (up to MC error); effect == 0 -> 1.
# Wrong-direction effects (< 0, or unknown with p1 > 0.5): approx p2 = 2*(1 - p1), floored at the exact minimum for n
# seeds (G_ada n = 3 -> 0.25); all have CIs including 0, so none can survive either way. F's H1 detector tests stay
# one-sided (a directional "combined beats best" hypothesis carried over from the pre-registration; P1 prints no p).
for t in T:
    if t["family"] != "EXPLORATORY" or "one-sided" not in t["note"] or t["test"].startswith("H1"): continue
    p1, e = t["p_raw"], t["effect"]
    if t["arm"] == "G" and t["test"] == "pooled outsider bribe > 0":
        p2, how = 2 ** -13, "exact two-sided 2^-13 (P1 text)"
    elif e is not None and e > 0:
        p2, how = min(1.0, 2 * p1), "two-sided = 2 x one-sided"
    elif e == 0:
        p2, how = 1.0, "effect 0 -> two-sided 1"
    else:
        p2, how = min(1.0, 2 * (1 - p1)), "APPROX two-sided (wrong direction)"
        if t["arm"] == "G_ada": p2 = max(p2, 0.25)
    t["p_raw"], t["note"] = p2, t["note"].replace("one-sided", "") + f" {how}; stored one-sided p {p1:.3g}"
# --- arm U: 34 OFAT cells vs BASE + 3 head-to-head
for c in J("arm_U_v2_2/results_arm_U.json")["cells"]:
    if c["p_vs_base"] is None: continue
    add(E, "U", f"{c['cell']} vs BASE", c["p_vs_base"], GEM + (" / mixed" if "mix=" in c["cell"] else ""), c["diff_vs_base_defections"], "arm_U_v2_2/results_arm_U.json cells")
for k, v in J("arm_U_v2_2/headtohead.json").items():
    add(E, "U", f"head-to-head {k}", v["p_two_sided"], GEM, v["diff"], "arm_U_v2_2/headtohead.json")
# --- arm UF (fractional factorial)
for e in J("arm_UF_v2_2/results_arm_UF.json")["effects"]:
    add(E, "UF", f"{e['kind']} {e['term']} on {e['response']}", e["p"], GEM + " (factor F: mixed)", e["effect"], "arm_UF_v2_2/results_arm_UF.json effects")
# --- arms C, C16 (contagion x bribe size), dose (C16 + C2K trend tests)
for arm, path in (("C", "arm_C_v2_2/results_arm_C.json"), ("C16", "arm_C16_v2_2/results_arm_C.json")):
    for k, v in J(path)["paired"].items():
        add(E, arm, k, v["p_two_sided"], GEM, v.get("diff", v.get("mean")), f"{path} paired/{k}")
for t in J("dose_N16_v2_2/results_dose_N16.json")["trend_tests"]:
    add(E, "dose_N16", t.get("test", t.get("name", "trend")), t["p"], GEM, t.get("slope", t.get("estimate")), "dose_N16_v2_2/results_dose_N16.json trend_tests")
# --- arms DB, SE, T
for arm, path, key in (("DB", "arm_DB_v2_2/results_arm_DB.json", "paired"), ("SE", "arm_SE_v2_2/results_arm_SE.json", "paired"),
                       ("T", "arm_T_v2_2/results_arm_T.json", "paired_vs_never_pays")):
    for t in J(path)[key]:
        name = next((t[k] for k in ("comparison", "name", "test", "label", "contrast") if k in t), "?")
        add(E, arm, name, t["p"], GEM, t.get("diff", t.get("effect")), f"{path} {key}")
# --- tests reported only in the addendum (no p in a results file)
add(E, "J", "ledger raw rule: never-pays vs always-pays (Fisher, ties split evenly: 0.74 vs 0.50)", 0.191, GEM, -0.24,
    "analysis/recompute_armJ_ledger.py (replaces the addendum's single-tie-break 0.90 vs 0.40, p = 0.001)", "hand-entered")

def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def bh(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i], reverse=True); adj = [0.0] * m; run = 1.0
    for r, i in enumerate(order):
        rank = m - r; run = min(run, ps[i] * m / rank); adj[i] = run
    return adj
for fam in sorted({t["family"] for t in T}):
    idx = [i for i, t in enumerate(T) if t["family"] == fam]; ps = [T[i]["p_raw"] for i in idx]
    for i, h, b in zip(idx, holm(ps), bh(ps)):
        T[i].update(m_family=len(idx), holm_family=h, bh_family=b)
allp = [t["p_raw"] for t in T]
for t, h, b in zip(T, holm(allp), bh(allp)):
    t.update(holm_all=h, bh_all=b)
for t in T:
    t["prereg"] = "y" if t["family"].startswith("PREREG") else "n"
    t["survives_holm"] = "y" if t["holm_family"] <= 0.05 else "n"
    t["survives_bh"] = "y" if t["bh_family"] <= 0.05 else "n"
cols = ["family", "prereg", "arm", "test", "model", "effect", "p_raw", "m_family", "holm_family", "bh_family", "survives_holm",
        "survives_bh", "holm_all", "bh_all", "src", "note"]
with open(f"{O}/familywide_correction.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for t in sorted(T, key=lambda t: (t["family"], t["p_raw"])):
        w.writerow({k: (f"{t[k]:.3g}" if isinstance(t.get(k), float) else t.get(k)) for k in cols})
# markdown summary
fams = {}
for t in T: fams.setdefault(t["family"], []).append(t)
L = ["# Family-wide multiplicity correction (betrayal study)", "",
     "Generated by `analysis/familywide_correction.py` from the per-arm results files. Holm (FWER 0.05) and BH (FDR 0.05) "
     "within each family; the three pre-registered families keep their own correction. p = 0 from a 20k-draw MC is set to 1/20001.", ""]
for fam, ts in sorted(fams.items()):
    L += [f"## {fam}: m = {len(ts)}; survive Holm {sum(t['survives_holm'] == 'y' for t in ts)}, BH {sum(t['survives_bh'] == 'y' for t in ts)}", "",
          "| arm | test | model | effect | raw p | Holm | BH |", "|---|---|---|---|---|---|---|"]
    for t in sorted(ts, key=lambda t: t["p_raw"]):
        if fam == E and t["bh_family"] > 0.05: continue
        eff = "" if t["effect"] is None else f"{t['effect']:+.3g}" if isinstance(t["effect"], (int, float)) else str(t["effect"])
        L.append(f"| {t['arm']} | {t['test']} | {t['model']} | {eff} | {t['p_raw']:.2g} | {t['holm_family']:.2g} | {t['bh_family']:.2g} |")
    if fam == E:
        L += ["", f"Not shown: the {sum(t['bh_family'] > 0.05 for t in ts)} exploratory tests failing BH (see the CSV)."]
    L.append("")
open(f"{O}/familywide_correction.md", "w").write("\n".join(L))
print("\n".join(L))
