"""Our side: plug Jennifer's returned arm F output into the Plan C v2 analysis.
Usage: analyze_arm_f.py <returned arm_f_out folder OR arm_f_out.zip> [DEST=$PLANC_ROOT/out/arm_F_returned]
Steps:
 1. Copy or unzip into DEST. The returned files are treated as data only and nothing in them is executed.
 2. Scan for credential-like strings (sk-/hf_/Bearer/long key-like tokens). Prints COUNTS only. If any are found, it
    stops before analysis.
 3. Check the required files are there.
 4. Run analyze_v2.py with REMAP=F:A, VERSION=armF -> DEST/results.json + DEST/figs.
 5. Write DEST/figs/armF_vs_v2_1.csv: arm F next to the v2.1 mixed swarm (D, OpenRouter) and the main grid (A).
    These are side by side only, never pooled."""
import sys, os, re, json, shutil, zipfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = sys.argv[1]; dest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "out", "arm_F_returned")
if os.path.exists(dest):
    sys.exit(f"{dest} exists; choose a new DEST (we never overwrite).")
os.makedirs(dest)
if src.endswith(".zip"):
    with zipfile.ZipFile(src) as z:
        for n in z.namelist():
            if n.startswith("/") or ".." in n.split("/"):
                sys.exit("unsafe path in zip; aborting")
        z.extractall(dest)
    inner = [os.path.join(dest, d) for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d)) and os.path.exists(os.path.join(dest, d, "games.jsonl"))]
    if inner:   # zip contained arm_f_out/ -> flatten
        for f in os.listdir(inner[0]): shutil.move(os.path.join(inner[0], f), dest)
        os.rmdir(inner[0])
else:
    for f in os.listdir(src):
        s = os.path.join(src, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copyfile)(s, os.path.join(dest, f))
pats = [re.compile(rb"sk-[A-Za-z0-9_\-]{20,}"), re.compile(rb"\bhf_[A-Za-z0-9]{20,}"), re.compile(rb"Bearer [A-Za-z0-9_\-\.]{20,}"),
        re.compile(rb"(?i)adaption[_-]?api[_-]?key\W{0,4}[A-Za-z0-9_\-]{16,}")]
hits = 0
for r, _, fs in os.walk(dest):
    for f in fs:
        b = open(os.path.join(r, f), "rb").read()
        hits += sum(1 for p in pats if p.search(b))
print("credential-pattern hits:", hits)
if hits:
    sys.exit("credential-like strings found in the returned folder; not analysing. Inspect privately and redact first.")
need = ["games.jsonl", "agents.jsonl", "cells.jsonl", "cost_tally.json", "messages.csv"]
missing = [f for f in need if not os.path.exists(os.path.join(dest, f))]
if missing:
    sys.exit(f"missing files: {missing}")
env = dict(os.environ, REMAP="F:A", VERSION="armF")
py = sys.executable
subprocess.run([py, "-I", os.path.join(HERE, "analyze_v2.py"), dest, os.path.join(dest, "figs")], env=env, check=True,
               stdout=subprocess.DEVNULL)
R = json.load(open(os.path.join(dest, "results.json")))
V = json.load(open(os.path.join(ROOT, "out", "final_v2_1", "results.json")))
def g(d, *ks):
    for k in ks:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d != {} else None
rows = []
def add(metric, f_val, d_val, a_val):
    rows.append(dict(metric=metric, arm_F_adaption=f_val, v2_1_D_mixed_openrouter=d_val, v2_1_A_main_qwen25=a_val))
tcF = R.get("turned_counts", {}).get("A", {}); tcD = V["turned_counts"].get("D", {}); tcA = V["turned_counts"].get("A", {})
add("turned agent-games / honest agent-games", f"{tcF.get('turned')}/{tcF.get('agent_games')}", f"{tcD.get('turned')}/{tcD.get('agent_games')}", f"{tcA.get('turned')}/{tcA.get('agent_games')}")
add("insider raises defection (per seed mean, CI)", json.dumps(g(R, "secondary_holm", "H2b_insider_raises_defection", "per_seed_mean")),
    None, json.dumps(g(V, "secondary_holm", "H2b_insider_raises_defection", "per_seed_mean")))
for k in ("insider|none", "insider|provenance", "insider|reputation+corroboration+quarantine", "d_evil|none"):
    add(f"attack damage {k}", json.dumps(g(R, "damage_descriptive", k, "attack_effect")), None, json.dumps(g(V, "damage_descriptive", k, "attack_effect")))
add("H1-style: combined AUROC (F own split / D own split / A held-out)", json.dumps(g(R, "H1_primary", "eval_auc", "combined")),
    json.dumps(g(V, "H1_within_mixed", "eval_auc", "combined")), json.dumps(g(V, "H1_primary", "eval_auc", "combined")))
add("insider identity top-1 (ledger)", json.dumps(g(R, "insider_identity_descriptive", "revealed_pref(-d_gives)")), None,
    json.dumps(g(V, "insider_identity_descriptive", "revealed_pref(-d_gives)")))
c = json.load(open(os.path.join(dest, "cost_tally.json")))
add("Adaption cost (usage.cost units) / calls / failed", f"{c.get('adaption_usd')} / {c.get('adaption_calls')} / {c.get('adaption_failed', 0)}", None, None)
import csv
os.makedirs(os.path.join(dest, "figs"), exist_ok=True)
with open(os.path.join(dest, "figs", "armF_vs_v2_1.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("arm F analysis:", os.path.join(dest, "results.json"), "| figures:", os.path.join(dest, "figs"),
      "| side-by-side (never pooled):", os.path.join(dest, "figs", "armF_vs_v2_1.csv"))
