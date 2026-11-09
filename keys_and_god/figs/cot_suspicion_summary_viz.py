"""Overall % of CoTs with Jev-1.13 suspicion > 5: bar = pooled rate (summary.json overall), dots = each world.
(a) Game vs Realistic Env, no God, 3 worlds. (b) Realistic Env, No God / God k=1 / God k=5, 5 worlds."""
import json, os
import viz
HERE = os.path.dirname(os.path.abspath(__file__))
def load(p, conds):
    s = json.load(open(os.path.join(HERE, p)))
    pooled = {c: 100 * s[f"overall|{c}"]["frac_susp_gt5"] for c in conds}
    per = {c: [100 * v["frac_susp_gt5"] for k, v in sorted(s["per_env_world"].items()) if k.split("|")[0] == c] for c in conds}
    return pooled, per
viz.set_defaults(size="paper", titles=False)
pa, ra = load("../cot_suspicion/summary.json", ["Realistic Env", "Game Env"])
pb, rb = load("../cot_suspicion/god_dose/summary.json", ["No God", "God k=1", "God k=5"])
ra_roles = {"control": "Realistic Env", "intervention": "Game Env"}
rb_roles = {"control": "No God", "intervention": "God k=1", "second": "God k=5"}
a = viz.bars(pa, runs=ra, roles=ra_roles, sort=False, decimals=1, ylabel="CoTs with suspicion > 5 (%)",
             title="(a) Env framing, no God")
b = viz.bars(pb, runs=rb, roles=rb_roles, sort=False, decimals=2, ylabel="CoTs with suspicion > 5 (%)",
             title="(b) God strength, Realistic Env")
fig = viz.panels([a, b], cols=2, title="Suspicious CoTs rose with Game framing and with God strength")
out = os.path.join(HERE, "cot_suspicion_summary_viz")
fig.save(out + ".png"); fig.save(out + ".pdf")
print(fig.caption())
