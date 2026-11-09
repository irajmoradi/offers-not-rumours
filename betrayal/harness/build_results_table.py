"""out/adaption_upload/results_tables.csv: one tidy table (arm, cell, N, model, metric, estimate, ci_lo, ci_hi, n, note)
built from the analysis outputs (no new computation)."""
import os, json, csv
O = f"{os.environ.get('PLANC_ROOT', '.')}/out"; rows = []
def add(arm, cell, N, model, metric, est, lo=None, hi=None, n=None, note=""):
    rows.append(dict(arm=arm, cell=cell, N=N, model=model, metric=metric, estimate=est, ci_lo=lo, ci_hi=hi, n=n, note=note))
R = json.load(open(f"{O}/final_v2_1/results.json"))
for k, v in R["secondary_holm"].items():
    e = v.get("per_seed_mean") or v.get("per_seed_reduction") or v.get("C_minus_S") or {}
    add("v2.1", k, 8, "qwen-2.5-7b", "effect", e.get("mean"), *(e.get("ci95") or [None, None]), e.get("n"), f"Holm p={v.get('p_holm')}; reject={v.get('holm_reject')}")
for k, v in R["damage_descriptive"].items():
    e = v["attack_effect"]; add("v2.1", k, 8, "qwen-2.5-7b", "attack_damage_completion", e.get("mean"), *(e.get("ci95") or [None, None]), e.get("n"))
for key in ("H1_leakfree", "H1_primary", "H1_within_mixed_leakfree"):
    for inst, v in R[key]["eval_auc"].items():
        add("v2.1", key, 8, "qwen-2.5-7b" if "mixed" not in key else "mixed", f"AUROC:{inst}", v["auc"], *(v["ci95"] or [None, None]), R[key]["eval_positives"], R[key]["label"][:60])
def from_csv(path, arm, cellcols, metric_cols, Ncol="N", model=""):
    if not os.path.exists(path): return
    for r in csv.DictReader(open(path)):
        cell = "|".join(str(r.get(c, "")) for c in cellcols)
        for m, lo, hi in metric_cols:
            if r.get(m) not in (None, ""):
                add(arm, cell, r.get(Ncol, ""), r.get("model", model) or model, m, r[m], r.get(lo) if lo else None, r.get(hi) if hi else None, r.get("seeds") or r.get("n_seeds"))
from_csv(f"{O}/arm_G_v2_2/figs/figG1_defection_effect.csv", "G", ["timing", "recognition"], [("defection_effect", "de_lo", "de_hi"), ("acceptance_per_seen", "acc_lo", "acc_hi")], Ncol="none", model="qwen-2.5-7b")
from_csv(f"{O}/arm_H_v2_2/figs/figH1_damage_vs_N_hom_vs_het.csv", "H", ["swarm"], [("insider_damage", "dmg_lo", "dmg_hi"), ("defection_effect", "de_lo", "de_hi")])
from_csv(f"{O}/arm_I_v2_2/figs/figI1_fleets_damage_vs_N.csv", "I", ["fleet"], [("insider_damage", "dmg_lo", "dmg_hi"), ("defection_effect", "de_lo", "de_hi")])
from_csv(f"{O}/arm_I_v2_2/figs/figI2_key_damage_by_fleet.csv", "I", ["fleet"], [("key_damage_pp", "lo", "hi")])
from_csv(f"{O}/arm_J_v2_2/figs/figJ1_fulfillment.csv", "J", ["briber", "schedule"], [("extra_def_per_agent", "lo", "hi"), ("ledger_top1", "ledger_lo", "ledger_hi")], Ncol="none", model="gemma-3-12b")
from_csv(f"{O}/arm_K_v2_2/figs/figK1_contagion_by_stage_N.csv", "K", ["condition"], [("extra_defections", "lo", "hi"), ("sabotage_success_rate", None, None)], model="gemma-3-12b")
from_csv(f"{O}/arm_K_v2_2/figs/figK2_interview_detection.csv", "K-interview", ["condition"], [("said_outside_party", None, None), ("named_any_insider", None, None), ("named_true_insider", None, None), ("said_test", None, None)], model="gemma-3-12b")
from_csv(f"{O}/by_model/figs/fig_by_model_D.csv", "D-by-model", ["attacker", "model"], [("turned_per_exposed", None, None), ("extra_defections_per_agent", "extra_lo", "extra_hi")], Ncol="none")
from_csv(f"{O}/p1_review/p1_damage_decomposition_and_ledger.csv", "P1", ["group"], [("dmg_completion_honest_pp", None, None), ("total_loss_keys", None, None), ("defect_loss_keys", None, None), ("hoard_upper_keys", None, None), ("raw_ledger_top1", None, None), ("clean_any_zero_giver", None, None)])
from_csv(f"{O}/p1_review/p1_ledger_by_payment_schedule_armJ.csv", "P1-J", ["schedule"], [("raw_ledger_top1", None, None)], Ncol="none", model="gemma-3-12b")
os.makedirs(f"{O}/adaption_upload", exist_ok=True)
with open(f"{O}/adaption_upload/results_tables.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
