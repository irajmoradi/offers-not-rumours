"""Merge corrected test-awareness (ta2) results into god_dose summary.json / per_round.csv under NEW keys/columns,
and print per-arm stats + top-P example CoT snippets. Usage: python -I add_ta2.py <god_dose dir>"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cot_judge as cj   # noqa: E402

D = Path(sys.argv[1])
rows = {}
with open(D / "results_jev113-ta2.jsonl") as f:
    for ln in f:
        r = json.loads(ln)
        if r.get("ok"):
            rows[r["id"]] = r
rows = list(rows.values())
ARMS = ["No God", "God k=1", "God k=5"]


def agg(rs):
    n = len(rs)
    k = sum(r["noul_test_aware"] > 0.5 for r in rs)
    return {"n": n, "n_ta2_gt0.5": k, "frac_ta2_gt0.5": k / n, "mean_p_ta2": sum(r["noul_test_aware"] for r in rs) / n}


summ = json.loads((D / "summary.json").read_text())
ta2 = {"spec": "ta2 (corrected wording; noul10 test_aware dropped: it counted 'game')", "per_arm": {}, "per_arm_world": {}}
for arm in ARMS:
    rs = [r for r in rows if r["env"] == arm]
    a = agg(rs)
    worlds = sorted({r["world"] for r in rs})
    pw = {w: agg([r for r in rs if r["world"] == w]) for w in worlds}
    a["world_range_frac_ta2_gt0.5"] = [min(v["frac_ta2_gt0.5"] for v in pw.values()), max(v["frac_ta2_gt0.5"] for v in pw.values())]
    a["world_range_mean_p_ta2"] = [min(v["mean_p_ta2"] for v in pw.values()), max(v["mean_p_ta2"] for v in pw.values())]
    ta2["per_arm"][arm] = a
    ta2["per_arm_world"].update({f"{arm}|{w}": v for w, v in pw.items()})
summ["ta2"] = ta2
(D / "summary.json").write_text(json.dumps(summ, indent=1))

# per_round.csv: append new columns, keep old ones untouched
by = defaultdict(list)
for r in rows:
    by[(r["env"], r["world"], str(r["round"]))].append(r)
with open(D / "per_round.csv") as f:
    rd = list(csv.DictReader(f))
cols = list(rd[0].keys())
for c in ("frac_ta2_gt0.5", "mean_p_ta2"):
    if c not in cols:
        cols.append(c)
for row in rd:
    rs = by.get((row["env"], row["world"], row["round"]), [])
    if rs:
        a = agg(rs)
        row["frac_ta2_gt0.5"] = f"{a['frac_ta2_gt0.5']:.4f}"; row["mean_p_ta2"] = f"{a['mean_p_ta2']:.4f}"
    else:
        row["frac_ta2_gt0.5"] = row["mean_p_ta2"] = ""
with open(D / "per_round.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rd)

print(json.dumps(ta2["per_arm"], indent=1))
# top-3 examples per arm (CoT text: stored tail, else re-extract from the game's llm_calls.jsonl)
for arm in ARMS:
    top = sorted([r for r in rows if r["env"] == arm], key=lambda r: -r["noul_test_aware"])[:3]
    print(f"\n## {arm}")
    for r in top:
        txt = r.get("cot_tail")
        if not txt:
            for it in cj.iter_items(arm, Path(r["game_dir"])):
                if it["id"] == r["id"]:
                    txt = it["cot"]; break
        print(f"- P={r['noul_test_aware']:.3f} {r['world']} r{r['round']} {r['agent']}: {(txt or '')[-700:]!r}")
