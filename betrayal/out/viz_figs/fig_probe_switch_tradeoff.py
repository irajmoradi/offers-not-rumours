"""P1 appendix figure: the probe-switch trade-off (obeying vs helpfulness), redrawn in viz without internal codes.
Replaces an earlier c9_tradeoff figure (labels "C9: α 0.06, w 0.06"; no script found).
Data, read at render time: lfm_lab/NOOR-Research/data/noor_steer/sweep.jsonl (50 tempting training scenes, 50 normal
scenes). Same points as the old figure: the no-push baseline, the four probe-switch (C9) settings, and the other sweep
settings at strength 0.02-0.06 (the old figure predates the 0.08 and 0.1 rows; see EXTRA below).
Run: viz run fig_probe_switch_tradeoff.py
"""
import json, os
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../single_agent/data/noor_steer/sweep.jsonl")
NAME = "fig_probe_switch_tradeoff"
rows = [json.loads(l) for l in open(SRC)]
assert all(r["scenes"] == "train50" and r["n"] == 50 and r["benign_n"] == 50 for r in rows)
alpha = lambda r: float(r["name"].split("_")[1][1:]) if r["name"].startswith("SW_a") else None

SWITCH = {"SW_a0.04_C9wnone_tau-": "Switch, strength 0.04", "SW_a0.06_C9wnone_tau-": "Switch, strength 0.06",
          "SW_a0.04_C9w0.06l2e0_tau-": "Switch 0.04 + weak push", "SW_a0.06_C9w0.06l2e0_tau-": "Switch 0.06 + weak push"}
by = {r["name"]: r for r in rows}
base = by["baseline_none"]
# cross-check with NOOR-Research.md section 3.14 table (obey, engage out of 50)
assert (base["follows"], base["benign_engage"]) == (16, 26)
assert {k: (by[k]["follows"], by[k]["benign_engage"]) for k in SWITCH} == {
    "SW_a0.04_C9wnone_tau-": (14, 17), "SW_a0.06_C9wnone_tau-": (12, 19),
    "SW_a0.04_C9w0.06l2e0_tau-": (7, 3), "SW_a0.06_C9w0.06l2e0_tau-": (5, 5)}
OTHER = [r for r in rows if r["family"] == "sweep" and r["name"] not in SWITCH and alpha(r) <= 0.06]
EXTRA = [r for r in rows if r["family"] == "sweep" and r["name"] not in SWITCH and alpha(r) > 0.06]   # not plotted
BAR = 18   # pass bar: engage >= 0.35 * 50 (noor_steer.py: benign_engage >= 0.35 * n)

points, colors = {}, {}
for i, r in enumerate(OTHER):
    k = f"other {i}"
    points[k] = (r["follows"], r["benign_engage"]); colors[k] = "gray-light"
for k, lab in SWITCH.items():
    points[lab] = (by[k]["follows"], by[k]["benign_engage"]); colors[lab] = "intervention"
points["No push"] = (base["follows"], base["benign_engage"]); colors["No push"] = "gray-dark"

SIZE = viz.Size(width=500, plot=300, title=16, inches=4.5)   # 12.5 px text prints at 8.1 pt


def build():
    return viz.scatter(
        points, colors=colors, labels=list(SWITCH.values()) + ["No push"], xdomain=(0, 25), ydomain=(0, 30),
        xticks=[0, 5, 10, 15, 20, 25], yticks=[0, 10, 18, 20, 30], trend=(0, BAR),
        label_offsets={"Switch 0.06 + weak push": (12, 0, "start"), "Switch 0.04 + weak push": (12, 8, "start")},
        xlabel="Obeyed the fake boss (of 50 tempting scenes)", ylabel="Engaged with normal requests (of 50)",
        legend={"No push": "gray-dark", "Probe switch": "intervention", "Other settings, same sweep": "gray-light"},
        title="One switch setting cut obeying 16 to 12, keeping helpfulness")


viz.set_defaults(size=SIZE, titles=False)
f = build()
for ext in (".pdf", ".png"):
    f.save(os.path.join(HERE, NAME + ext))
viz.set_defaults(size="screen", titles=True)
build().save(os.path.join(HERE, NAME + "_screen.png"))
print("plotted", len(OTHER), "other settings; not plotted (strength 0.08-0.1):",
      [(r["name"], r["follows"], r["benign_engage"]) for r in EXTRA])
