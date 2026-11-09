"""P1 appendix figure "Contagion and bribe size" (8 agents) redrawn in viz without internal codes (the old
planC_v2/harness/analyze_c.py figure had an "Arm C" suptitle and cell codes). Same numbers, read at render time from
planC_v2/out/arm_C_v2_2/figs/figC_contagion.csv. Run: viz run figC_contagion_clean.py OUT.png"""
import ast, csv, os, sys
import viz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = {r["cell"]: r for r in csv.DictReader(open(os.path.join(OUT, "arm_C_v2_2/figs/figC_contagion.csv")))}
assert all(int(r["seeds"]) == 40 for r in C.values())
ONLY, CONT = "Outsider only", "Outsider with contagion"   # same names and colours as the 16-agent dose figure
cells = {"1-key bribe": ("outsider_only", "outsider+contagion"), "3-key bribe": ("outsider_only_3key", "outsider+contagion_3key")}
data = {g: [float(C[a]["extra_defections"]), float(C[b]["extra_defections"])] for g, (a, b) in cells.items()}
ci = {g: [(float(C[a]["ed_lo"]), float(C[a]["ed_hi"])), (float(C[b]["ed_lo"]), float(C[b]["ed_hi"]))]
      for g, (a, b) in cells.items()}
hops = {}
for c in ("outsider+contagion", "outsider+contagion_3key"):
    for k, v in ast.literal_eval(C[c]["hop_counts"]).items():
        hops[k] = hops.get(k, 0) + v
assert sum(hops.values()) == sum(int(C[c]["recruited_total"]) for c in ("outsider+contagion", "outsider+contagion_3key"))

P1W = viz.Size(width=610, plot=200, title=16, inches=5.5)
viz.set_defaults(size=P1W, titles=False)
left = viz.bars(data, series=[ONLY, CONT], ci=ci, sort=False, unit="", roles={"baseline": ONLY, "intervention": CONT},
                ylabel="Extra defections per game", title="(a) Extra defections, 8 agents")
right = viz.bars({f"Hop {k}": v for k, v in sorted(hops.items())}, sort=False, unit="", color="intervention",
                 ylabel="Agents recruited", title="(b) Recruits by chain depth")
fig = viz.panels([left, right], cols=2, gap=36, legend={ONLY: "baseline", CONT: "intervention"},
                 title="Contagion added defections, and chains rarely went past two hops",
                 subtitle="Error bars are 95% intervals")
fig.save(sys.argv[1])
print("wrote", sys.argv[1], hops)
