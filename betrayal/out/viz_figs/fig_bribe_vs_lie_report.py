"""Report version (Apart single-column, 6.5 in) of the bribe-vs-lie result: arm U head-to-head, all-Gemma-3-12B, N=8,
36 seeds, extra defections per game vs the matched clean twin, 95% seed-bootstrap CIs. Self-contained: title,
setup subtitle, row descriptors, status notes and a source footer are all in the image.
Run: viz run fig_bribe_vs_lie_report.py   ->  fig_bribe_vs_lie_report.pdf (vector) and .png (2x)
Values are read at render time: arm_U_v2_2/figs/figU_cells.csv (checked against headtohead.json), per_seed.json
(exact sign-flip p), familywide_correction.csv (Holm family of 110 tests). No hand-typed numbers.
"""
import csv, json, os, re
import viz
from viz import svg as S, theme as th, text as T
from viz.figure import Figure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
NAME = "fig_bribe_vs_lie_report"
# 6.5 in print width; at 620 layout px the smallest text (12 px ticks) prints at 9.1 pt, labels (12.5 px) at 9.4 pt
REPORT = viz.Size(width=620, plot=230, title=20, inches=6.5, scale=2)

U = {r["cell"]: r for r in csv.DictReader(open(os.path.join(OUT, "arm_U_v2_2/figs/figU_cells.csv")))}
H2H = json.load(open(os.path.join(OUT, "arm_U_v2_2/headtohead.json")))
FW = {r["test"]: r for r in csv.DictReader(open(os.path.join(OUT, "familywide_correction.csv")))}
SEEDS = json.load(open(os.path.join(HERE, "per_seed.json")))

ROWS = [("Outsider", "posXmethod=outsider_forged", "pos=outsider", "outsider_bribe_vs_forged"),
        ("Insider", "method=insider_lie_only", "method=insider_bribe_only", "insider_bribe_only_vs_lie_only"),
        ("Both", "posXmethod=both_forged", "pos=both", "both_bribe_vs_forged")]
DESCRIPTORS = {"Outsider": "(not in the swarm)", "Insider": "(compromised member)"}
LIE, BRIBE = "Lies / forged rumours", "Bribes (fake promises of a key)"

data, ci, events = {}, {}, {}
for g, a, b, k in ROWS:
    va, vb = float(U[a]["extra_defections"]), float(U[b]["extra_defections"])
    assert abs(H2H[k]["b_mean"] - va) < 6e-3 and abs(H2H[k]["a_mean"] - vb) < 6e-3, k
    assert int(U[a]["seeds"]) == int(U[b]["seeds"]) == H2H[k]["n"] == 36
    data[g] = [va, vb]
    ci[g] = [(float(U[a]["ed_lo"]), float(U[a]["ed_hi"])), (float(U[b]["ed_lo"]), float(U[b]["ed_hi"]))]
    events[k] = (round(vb * 36), round(va * 36))   # extra defections summed over 36 seeds: bribe, lie/forgery

# outsider: exact two-sided sign-flip p from the per-seed paired differences (k non-zero, all positive -> 2 / 2**k)
assert SEEDS["U_seeds"]["pos=outsider"] == SEEDS["U_seeds"]["posXmethod=outsider_forged"]
diff = [x - y for x, y in zip(SEEDS["U"]["pos=outsider"], SEEDS["U"]["posXmethod=outsider_forged"])]
nz = [x for x in diff if x != 0]
assert all(x > 0 for x in nz)
P_OUT = 2 / 2 ** len(nz)
assert f"{P_OUT:.3f}" == "0.031" and events["outsider_bribe_vs_forged"] == (7, 0)
# correction status across the 110-test family
fo, fi = FW["head-to-head outsider_bribe_vs_forged"], FW["head-to-head insider_bribe_only_vs_lie_only"]
assert fo["m_family"] == fi["m_family"] == "110" and fo["survives_holm"] == "n" and fi["survives_holm"] == "y"

e = events["outsider_bribe_vs_forged"]
# Plain-language notes for report readers; the asserts above tie them to the stats (outsider fails Holm across the
# 110-test family, insider survives it). Precise p-values stay in the paper text and caption.
assert e[1] == 0
NOTES = {ROWS[0][0]: f"small, uncertain: {e[0]} extra defections over {H2H['outsider_bribe_vs_forged']['n']} games",
         ROWS[1][0]: "clear effect, holds up after correcting for the many comparisons we ran"}
TITLE = "Bribes, not lies, turned agents"
SUBTITLE = ("Gemma-3-12B swarms of 8 agents, 36 seeds;\nextra defections per game vs a matched game without the attacker; "
            "bars = 95% CI")
FOOTER = ("Exploratory study. Lies = the insider's betrayal accusations; forged rumours = the outsider LLM's "
          "hoarding gossip, sent under real agents' names.")


class WithFooter(Figure):
    """The viz figure plus one muted source line under it (viz has no footer of its own)."""
    def compose(self):
        inner, w, h = super().compose()
        # a short muted descriptor under two row labels (viz row labels are one line)
        # plain-language notes on a line under each bribe value label, wrapped to the figure's right edge
        for label, note in NOTES.items():
            val = f"{data[label][1]:.2f}"
            ms = list(re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>(?:<tspan[^>]*>)?' + re.escape(val)
                                  + r"(?:</tspan>)?</text>", inner))
            assert len(ms) == 1, (label, val, len(ms))
            x, y = float(ms[0].group(1)), float(ms[0].group(2))
            lines = T.wrap(note, w - 2 * th.PAD_X - x, 12, "sans")   # value labels sit in the body group (x0 = PAD_X)
            add = "".join(S.text(x, y + 17 + i * 15, ln, 12, th.MUTED) for i, ln in enumerate(lines))
            inner = inner[:ms[0].end()] + add + inner[ms[0].end():]
        for label, desc in DESCRIPTORS.items():
            m = re.search(r'<text x="([-\d.]+)" y="([-\d.]+)"([^>]*)>' + re.escape(label) + "</text>", inner)
            assert m, label
            x, y, rest = float(m.group(1)), float(m.group(2)), m.group(3)
            new = (f'<text x="{m.group(1)}" y="{y - 7:.2f}"{rest}>{label}</text>'
                   + S.text(x, y + 10, desc, 12, th.MUTED, "end"))
            inner = inner[:m.start()] + new + inner[m.end():]
        size = 12   # th.MIN_TEXT: prints at 9.1 pt at this size
        lines = T.wrap(FOOTER, w - 2 * th.PAD_X, size, "sans")
        lh = size * 1.35
        y0 = h - th.PAD_BOTTOM + 14
        foot = "".join(S.text(th.PAD_X, y0 + size + i * lh, ln, size, th.MUTED) for i, ln in enumerate(lines))
        return inner + foot, w, y0 + len(lines) * lh + th.PAD_BOTTOM


viz.set_defaults(size=REPORT, titles=True)
fig = viz.dotplot(
    data, series=[LIE, BRIBE], ci=ci, sort=False, unit="", decimals=2, ref=0,
    roles={"baseline": LIE, "intervention": BRIBE}, domain=(0, 4), ticks=[0, 1, 2, 3, 4], row_height=84, dot_size=7,
    xlabel="Extra defections per game vs matched attack-free game", title=TITLE, subtitle=SUBTITLE)
fig.__class__ = WithFooter
for ext in (".pdf", ".png"):
    fig.save(os.path.join(HERE, NAME + ext))
print("saved", NAME)
