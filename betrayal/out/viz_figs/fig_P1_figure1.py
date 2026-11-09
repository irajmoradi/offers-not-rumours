"""P1 Figure 1 ("Bribes, not forgeries") rebuilt in the viz house style. Run: viz run fig_P1_figure1.py
Replaces the earlier keys_and_god/figs/fig1_headline.py (matplotlib); same numbers, read from the same result files.
  (a) Keys & God forger dose, 10 agents, gpt-oss-120b: swarm_hackathon/plan/figs/e1_stats.json
      (K=2 = g1_8 god-noise, n=8, and g1_16 god-nogod, n=16; K=18 = e1, n=16), as in plan/E1_RESULTS.md dose table.
  (b) Arm U head-to-head, all-Gemma-3, N=8, 36 seeds: planC_v2/out/arm_U_v2_2/figs/figU_cells.csv (= headtohead.json).
  (c) Arm I key damage vs N by fleet: planC_v2/out/arm_I_v2_2/figs/figI2_key_damage_by_fleet.csv (N = 4, 8, 16).
No new experiments, no API calls. Outputs next to this file: fig_P1_figure1.pdf/.png (5.5 in print, no titles in the
image) and fig_P1_figure1_screen.png (screen, with titles); appends the caption to captions.md.
"""
import csv, json, os
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
E1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../keys_and_god/results/e1_stats.json")
P1W = viz.Size(width=610, plot=170, title=16, inches=5.5)   # P1 text width; text prints at 8 pt
NAME = "fig_P1_figure1"


def rows(path):
    with open(os.path.join(OUT, path)) as f:
        return list(csv.DictReader(f))


# (a) ------------------------------------------------------------------------------------------------
S = json.load(open(E1))
w = lambda cell, c: S[cell][f"welfare:{c}"]
A_ROWS = [("2 moves", w("g1_8", "god-noise"), w("g1_16", "god-nogod")),
          ("18 moves", w("e1", "god-noise"), w("e1", "god-nogod"))]
# P1 quotes these to 3 decimals: K=2 -0.062 / +0.013, K=18 -0.140 / -0.094.
for (_, a, b), want in zip(A_ROWS, ((-0.062, 0.013), (-0.140, -0.094))):
    assert f"{a['mean']:.3f}" == f"{want[0]:.3f}" and f"{b['mean']:.3f}" == f"{want[1]:.3f}", (a["mean"], b["mean"])
    assert a["n"] in (8, 16) and b["n"] == 16


def panel_a():
    # One series in the narrow panel: the pre-registered contrast (vs the fork-matched noise twin). The vs-no-God
    # contrasts (+0.013 at 2 moves, -0.094 at 18) go in the caption.
    data = {k: a["mean"] for k, a, _ in A_ROWS}
    ci = {k: (a["t_lo"], a["t_hi"]) for k, a, _ in A_ROWS}
    return viz.bars(
        data, ci=ci, ref=0, unit="", decimals=3, sort=False, color="baseline",
        ylabel="Welfare vs noise twin", title="(a) Forger, 10 agents")


# (b) ------------------------------------------------------------------------------------------------
U = {r["cell"]: r for r in rows("arm_U_v2_2/figs/figU_cells.csv")}
H2H = json.load(open(os.path.join(OUT, "arm_U_v2_2/headtohead.json")))
B_PAIRS = [("Outsider", "posXmethod=outsider_forged", "pos=outsider", "outsider_bribe_vs_forged"),
           ("Insider", "method=insider_lie_only", "method=insider_bribe_only", "insider_bribe_only_vs_lie_only"),
           ("Both", "posXmethod=both_forged", "pos=both", "both_bribe_vs_forged")]


def panel_b():
    for _, a, b, k in B_PAIRS:
        assert abs(H2H[k]["b_mean"] - float(U[a]["extra_defections"])) < 6e-3
        assert abs(H2H[k]["a_mean"] - float(U[b]["extra_defections"])) < 6e-3
    data = {g: [float(U[a]["extra_defections"]), float(U[b]["extra_defections"])] for g, a, b, _ in B_PAIRS}
    ci = {g: [(float(U[a]["ed_lo"]), float(U[a]["ed_hi"])), (float(U[b]["ed_lo"]), float(U[b]["ed_hi"]))]
          for g, a, b, _ in B_PAIRS}
    return viz.dotplot(
        data, series=["Forgery or lies", "Bribes"], ci=ci, sort=False, unit="", decimals=2, ref=0,
        roles={"baseline": "Forgery or lies", "intervention": "Bribes"}, domain=(0, 4), ticks=[0, 1, 2, 3, 4],
        xlabel="Extra defections per game vs clean twin", title="(b) Bribes vs forgery")


# (c) ------------------------------------------------------------------------------------------------
FLEETS = {"all-Qwen": "Qwen", "all-llama-3.1-8b-instruct": "Llama", "all-gpt-oss-20b": "gpt-oss",
          "all-gemma-3-12b-it": "Gemma", "mixed": "Mixed"}


HONEST = "size_keydamage_honest.csv"   # honest agents only, vs clean twin (P1 section 4.3)
XS = [4, 8, 16]


def honest_series():
    H = rows(HONEST)
    series = {}
    for f, short in FLEETS.items():
        assert all(sum(1 for r in H if r["fleet"] == f and int(r["N"]) == x) == 1 for x in XS), f
        hit = {int(r["N"]): r for r in H if r["fleet"] == f}
        series[short] = [float(hit[x]["key_damage_honest_pp"]) for x in XS]
        assert [str(v) for v in series[short]] == [str(float(hit[x]["key_damage_honest_pp"])) for x in XS]
    # cross-check with P1 section 4.3 (N=4: Qwen 21, Llama 18, gpt-oss 14, Gemma 17, mixed 18 pp; -4 to +5 from 8 up)
    assert {k: round(v[0]) for k, v in series.items()} == {"Qwen": 21, "Llama": 18, "gpt-oss": 14, "Gemma": 17, "Mixed": 18}
    later = [v for vals in series.values() for v in vals[1:]]
    assert -4.5 <= min(later) and max(later) <= 5.0, (min(later), max(later))
    return series


def panel_c():
    series = honest_series()
    return viz.line(
        series, x=XS, xscale="log", xticks=XS, xtick_labels={4: "4", 8: "8", 16: "16"}, unit="", ref=0,
        colors={"Gemma": "blue", "Qwen": "sage", "Mixed": "plum", "Llama": "gray-dark", "gpt-oss": "gray"},
        end_labels=True, end_values=False, xlabel="Agents in the swarm",
        ylabel="Key damage to honest agents vs clean twin (pp)", height=280, domain=(-5, 25),
        ticks=[-5, 0, 5, 10, 15, 20, 25], title="(c) Insider damage vs swarm size")


TITLE = "Bribes added up to 3.11 defections per game, forgery at most 0.31"


def build():
    top = viz.panels([panel_a(), panel_b()], cols=2, gap=40)   # (a) and (b) side by side, (c) full width below
    return viz.panels([top, panel_c()], cols=1, row_gap=18, title=TITLE,
                      legend={"Forgery or lies": "baseline", "Bribes": "intervention"})


viz.set_defaults(size=P1W, titles=False)
f = build()
f.save(os.path.join(HERE, NAME + ".pdf"))
f.save(os.path.join(HERE, NAME + ".png"))
cap = f.caption()
viz.set_defaults(size="screen", titles=True)
build().save(os.path.join(HERE, NAME + "_screen.png"))
print(cap)

# caption -> captions.md (replaces this figure's section; make_figs.py keeps it when it rewrites the file)
a2, a18 = A_ROWS[0], A_ROWS[1]
fmt = lambda s: f"{s['mean']:+.3f} [{s['t_lo']:.3f}, {s['t_hi']:.3f}]".replace("-", "−")
HS = honest_series()
n4lo, n4hi = str(min(v[0] for v in HS.values())), str(max(v[0] for v in HS.values()))
_lat = [v for vals in HS.values() for v in vals[1:]]
later_lo, later_hi = str(min(_lat)).replace("-", "\u2212"), f"+{max(_lat)}"
def exact_signflip_outsider():
    """Exact two-sided sign-flip p for outsider bribe vs forgery, from per-seed effects (per_seed.json):
    with k non-zero paired differences all of one sign, p = 2 / 2**k (P1 section 4.2 quotes 0.031 for k = 6)."""
    S = json.load(open(os.path.join(HERE, "per_seed.json")))
    assert S["U_seeds"]["pos=outsider"] == S["U_seeds"]["posXmethod=outsider_forged"]
    diff = [x - y for x, y in zip(S["U"]["pos=outsider"], S["U"]["posXmethod=outsider_forged"])]
    nz = [x for x in diff if x != 0]
    assert len(nz) == 6 and all(x > 0 for x in nz)
    return 2 / 2 ** len(nz)


P_OUT = exact_signflip_outsider()

LONG = (f"{cap} (a) Keys & God (10 agents, gpt-oss-120b): change in group welfare when a scripted forger acts twice per "
        f"game or every round (18 moves), against the fork-matched noise twin: {fmt(a2[1])}, p = {a2[1]['p']:.2f}, "
        f"n = {a2[1]['n']} worlds, and {fmt(a18[1])}, p = {a18[1]['p']:.4f}, n = {a18[1]['n']}. Against no God: "
        f"{fmt(a2[2])} (n = {a2[2]['n']}) and {fmt(a18[2])} (n = {a18[2]['n']}). Error bars are t95 intervals over worlds. "
        "(b) Head-to-head (arm U†: 8 all-Gemma-3 agents, 36 seeds): extra defections per game vs the matched clean "
        "twin, with the same model, position, start round and number of attacker messages; the insider's control is "
        f"lies only. Paired differences: outsider +0.19 [0.06, 0.36], exact p = {P_OUT:.3f}; insider +0.89 [0.56, 1.25], p < 0.0001; "
        "both +2.81 [2.39, 3.25], p < 0.0001. Error bars are 95% seed-bootstrap intervals. "
        f"(c) Insider damage to honest agents' keys vs the clean twin (percentage points) fell "
        f"from {n4lo}\u2013{n4hi} at 4 agents to between {later_lo} and {later_hi} from 8 agents up, in four "
        "single-model swarms and one mixed swarm (arm I\u2020; 8\u201340 seeds per point; intervals in Appendix D). "
        "\u2020 Exploratory, not independently audited.")
CP = os.path.join(HERE, "captions.md")
txt = open(CP).read() if os.path.exists(CP) else "# Plan C (P1) figure captions\n"
head = "## " + NAME + "\n"
if head in txt:
    i = txt.index(head); j = txt.find("\n## ", i + len(head))
    txt = txt[:i] + (txt[j + 1:] if j != -1 else "")
txt = txt.rstrip("\n") + f"\n\n{head}\n{LONG}\n\nAlt text: {f.alt_text()}\n"
open(CP, "w").write(txt)
