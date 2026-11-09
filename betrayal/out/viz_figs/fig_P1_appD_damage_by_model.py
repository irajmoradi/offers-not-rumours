"""P1 Appendix D: insider damage to honest agents' keys by swarm size, one panel per model (replaces
arm_I_v2_2/figs/figI3_key_damage_small_multiples.png, which plotted the old all-agent numbers).
Run: viz run fig_P1_appD_damage_by_model.py
Reads planC_v2/out/size_keydamage_honest.csv at render time (key_damage_honest_pp, ci_lo, ci_hi; honest agents only,
vs clean twin), the same source as P1 Figure 1c. No hand-typed values. Writes fig_P1_appD_damage_by_model.pdf/.png
(5.5 in, P1 text width), _screen.png, and its section in captions.md.
"""
import csv, os
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
NAME = "fig_P1_appD_damage_by_model"
P1W = viz.Size(width=610, plot=150, title=16, inches=5.5)   # P1 text width; text prints at 8 pt
FLEETS = [("all-Qwen", "Qwen", "sage"), ("all-llama-3.1-8b-instruct", "Llama", "gray-dark"),
          ("all-gpt-oss-20b", "gpt-oss", "gray"), ("all-gemma-3-12b-it", "Gemma", "blue"), ("mixed", "Mixed", "plum")]

H = list(csv.DictReader(open(os.path.join(OUT, "size_keydamage_honest.csv"))))
DATA = {}
for fleet, short, color in FLEETS:
    rs = sorted((r for r in H if r["fleet"] == fleet), key=lambda r: int(r["N"]))
    ns = [int(r["N"]) for r in rs]
    assert len(ns) == len(set(ns)) and ns[:3] == [4, 8, 16], (fleet, ns)
    mean = [float(r["key_damage_honest_pp"]) for r in rs]
    lo, hi = [float(r["ci_lo"]) for r in rs], [float(r["ci_hi"]) for r in rs]
    assert all(l <= m <= h for l, m, h in zip(lo, mean, hi)), fleet
    seeds = [int(r["seeds"]) for r in rs]
    DATA[short] = dict(N=ns, mean=mean, lo=lo, hi=hi, seeds=seeds, color=color)
# cross-check with P1 section 4.3 / Figure 1c (N=4: Qwen 21, Llama 18, gpt-oss 14, Gemma 17, mixed 18 pp)
assert {k: round(v["mean"][0]) for k, v in DATA.items()} == {"Qwen": 21, "Llama": 18, "gpt-oss": 14, "Gemma": 17, "Mixed": 18}
LO = min(min(v["lo"]) for v in DATA.values()); HI = max(max(v["hi"]) for v in DATA.values())
assert -10 <= LO and HI <= 30, (LO, HI)
XT = sorted({n for v in DATA.values() for n in v["N"]})


def panel(short):
    v = DATA[short]
    return viz.line(
        {short: v["mean"]}, x=v["N"], bands={short: (v["lo"], v["hi"])}, colors={short: v["color"]},
        xscale="log", xticks=XT, xtick_labels={n: str(n) for n in XT}, unit="", ref=0, end_labels=False,
        domain=(-10, 30), ticks=[-10, 0, 10, 20, 30], xlabel="Agents", ylabel="Key damage (pp)", title=short)


_n4 = [v["mean"][0] for v in DATA.values()]
_up = [m for v in DATA.values() for n, m in zip(v["N"], v["mean"]) if n >= 8]
TITLE = f"Insider damage fell from {min(_n4)}\u2013{max(_n4)} pp to at most {max(_up)} pp"


def build():
    return viz.panels([panel(s) for _, s, _ in FLEETS], cols=3, gap=28, row_gap=14, title=TITLE, legend=False,
                      subtitle="Bands are 95% intervals")


viz.set_defaults(size=P1W, titles=False)
f = build()
f.save(os.path.join(HERE, NAME + ".pdf"))
f.save(os.path.join(HERE, NAME + ".png"))
viz.set_defaults(size="screen", titles=True)
build().save(os.path.join(HERE, NAME + "_screen.png"))

later = [m for v in DATA.values() for n, m in zip(v["N"], v["mean"]) if n >= 8]
fmt = lambda x: f"{x}".replace("-", "−") if x < 0 else f"+{x}"
seeds = sorted({s for v in DATA.values() for s in v["seeds"]})
LONG = (f"**Insider damage by swarm size, per model.** {f.caption()} Insider damage to honest agents' keys (percentage "
        "points, honest agents only, a burned key counted as lost) vs the clean twin, for four single-model swarms and "
        "one mixed swarm (arm I†). Line: mean; band: 95% interval. At 4 agents the damage was "
        f"{min(v['mean'][0] for v in DATA.values())}–{max(v['mean'][0] for v in DATA.values())} pp; from 8 agents "
        f"up it was between {fmt(min(later))} and {fmt(max(later))} pp. {seeds[0]}–{seeds[-1]} seeds per point. "
        "Same data as Figure 1c. † Exploratory, not independently audited.")
CP = os.path.join(HERE, "captions.md")
txt = open(CP).read()
head = "## " + NAME + "\n"
if head in txt:
    i = txt.index(head); j = txt.find("\n## ", i + len(head))
    txt = txt[:i] + (txt[j + 1:] if j != -1 else "")
open(CP, "w").write(txt.rstrip("\n") + f"\n\n{head}\n{LONG}\n\nAlt text: {f.alt_text()}\n")
print(LONG)
