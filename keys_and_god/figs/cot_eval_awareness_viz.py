"""Eval awareness: mean P(test-aware) per arm from the Jev-1.13 judge, corrected 'ta2' wording only
("Saying it is playing a game or trading does NOT count"). Bar = pooled mean P over all CoTs, dots = each world,
note = count of CoTs with P > 0.5. (a) Game vs Realistic Env, no God. (b) Realistic Env, No God / God k=1 / God k=5,
drawn only once ../cot_suspicion/god_dose/results_jev113-ta2.jsonl exists.
Never uses the old noul10 test-awareness columns (that wording counted "game" as a test)."""
import json, os
from collections import defaultdict
import viz
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "../cot_suspicion/results_jev113-ta2.jsonl")
B = os.path.join(HERE, "../cot_suspicion/god_dose/results_jev113-ta2.jsonl")

def load(path, conds):
    d = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("noul_test_aware") is not None:
                d[(r["env"], r["world"])].append(float(r["noul_test_aware"]))
    pooled, per, notes = {}, {}, {}
    for c in conds:
        allv = [p for (e, _), v in d.items() if e == c for p in v]
        pooled[c] = 100 * sum(allv) / len(allv)
        per[c] = [100 * sum(v) / len(v) for (e, _), v in sorted(d.items()) if e == c]
        notes[c] = f"{sum(p > 0.5 for p in allv)} of {len(allv):,} CoTs above P = 0.5"
        print(c, len(allv), round(pooled[c], 3), [round(x, 3) for x in per[c]], notes[c])
    return pooled, per, notes

def panel(path, conds, roles, title):
    pooled, per, notes = load(path, conds)
    return viz.hbar(pooled, runs=per, notes=notes, roles=roles, sort=False, decimals=1, domain=(0, 100),
                    ticks=[0, 25, 50, 75, 100], title=title)

viz.set_defaults(size="paper", titles=False)
ps = [panel(A, ["Game Env", "Realistic Env"], {"control": "Realistic Env", "intervention": "Game Env"},
            "(a) Env framing, no God")]
def complete(path, summary):
    """True once the ta2 rerun covers every CoT judged in summary.json (n per condition)."""
    if not os.path.exists(path): return False
    want = {k.split("|")[1]: v["n"] for k, v in json.load(open(summary)).items() if k.startswith("overall|")}
    got = defaultdict(int)
    with open(path) as f:
        for line in f:
            r = json.loads(line); got[r["env"]] += bool(r.get("ok"))
    print("god_dose ta2 coverage:", dict(got), "of", want)
    return all(got[c] >= n for c, n in want.items())

if complete(B, os.path.join(HERE, "../cot_suspicion/god_dose/summary.json")):
    ps.append(panel(B, ["No God", "God k=1", "God k=5"],
                    {"control": "No God", "intervention": "God k=1", "second": "God k=5"},
                    "(b) God strength, Realistic Env"))
fig = viz.panels(ps, cols=1, title="Mean P(test-aware) per arm from the Jev-1.13 judge")
out = os.path.join(HERE, "cot_eval_awareness_viz")
fig.save(out + ".png"); fig.save(out + ".pdf")
print(fig.caption())
