"""Standalone version of P1 Figure 1b: bribes vs lies / forged rumours, by attacker position (arm U, all-Gemma-3-12B,
N=8, 36 seeds, extra defections per game vs the clean twin, 95% seed-bootstrap CIs).
Run: viz run fig_bribe_vs_lie_standalone.py
Reads planC_v2/out/arm_U_v2_2/figs/figU_cells.csv and checks it against headtohead.json; no hand-typed values.
Outputs: fig_bribe_vs_lie_standalone.pdf/.png (viz paper size, 6.75 in, no title in the image),
fig_bribe_vs_lie_standalone_screen.png (with title), and its section in captions.md.
"""
import csv, json, os
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
NAME = "fig_bribe_vs_lie_standalone"
PRINT = "paper"   # viz full paper width (6.75 in)

U = {r["cell"]: r for r in csv.DictReader(open(os.path.join(OUT, "arm_U_v2_2/figs/figU_cells.csv")))}
H2H = json.load(open(os.path.join(OUT, "arm_U_v2_2/headtohead.json")))
PAIRS = [("Outsider", "posXmethod=outsider_forged", "pos=outsider", "outsider_bribe_vs_forged"),
         ("Insider", "method=insider_lie_only", "method=insider_bribe_only", "insider_bribe_only_vs_lie_only"),
         ("Both", "posXmethod=both_forged", "pos=both", "both_bribe_vs_forged")]
LIE, BRIBE = "Lies / forged rumours", "Bribes (fake promises of a key)"

data, ci, events = {}, {}, {}
for g, a, b, k in PAIRS:
    va, vb = float(U[a]["extra_defections"]), float(U[b]["extra_defections"])
    assert abs(H2H[k]["b_mean"] - va) < 6e-3 and abs(H2H[k]["a_mean"] - vb) < 6e-3, k
    assert int(U[a]["seeds"]) == int(U[b]["seeds"]) == H2H[k]["n"] == 36
    data[g] = [va, vb]
    ci[g] = [(float(U[a]["ed_lo"]), float(U[a]["ed_hi"])), (float(U[b]["ed_lo"]), float(U[b]["ed_hi"]))]
    events[g] = (round(va * 36), round(vb * 36))   # extra defections summed over the 36 seeds
assert events["Outsider"] == (0, 7), events["Outsider"]

# Row notes carry the statistical status (viz keeps text off the plot except labels; notes sit after the values).
# "110 tests" is P1's correction family (the paper's abstract), not computed here.
NOTES = {"Insider": "survives correction across 110 tests",
         "Outsider": f"a lead only ({events['Outsider'][1]} vs {events['Outsider'][0]} events)"}


def build():
    return viz.dotplot(
        data, series=[LIE, BRIBE], ci=ci, sort=False, unit="", decimals=2, ref=0, notes=NOTES,
        roles={"baseline": LIE, "intervention": BRIBE}, domain=(0, 4), ticks=[0, 1, 2, 3, 4], row_height=72, dot_size=7,
        xlabel="Extra defections per game vs matched attack-free game",
        title="Bribes added more defections than lies or forged rumours in every position",
        subtitle="Error bars are 95% intervals")


viz.set_defaults(size=PRINT, titles=False)
f = build()
f.save(os.path.join(HERE, NAME + ".pdf"))
f.save(os.path.join(HERE, NAME + ".png"))
viz.set_defaults(size="screen", titles=True)
build().save(os.path.join(HERE, NAME + "_screen.png"))

d = lambda k: H2H[k]
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

LONG = (f"{f.caption()} Extra defections per game against a matched game without the attacker (Gemma-3-12B, N = 8) "
        "(36 seeds, paired by seed; same model, position, start round and number of attacker messages). "
        "The insider's comparison is its lies alone vs its bribes alone; the outsider's is forged rumours vs bribes; "
        "\"Both\" has an insider and an outsider attacking together. Paired differences: outsider "
        f"+{d('outsider_bribe_vs_forged')['diff']:.2f} [{d('outsider_bribe_vs_forged')['ci'][0]:.2f}, "
        f"{d('outsider_bribe_vs_forged')['ci'][1]:.2f}], exact p = {P_OUT:.3f} "
        f"({events['Outsider'][1]} vs {events['Outsider'][0]} extra defections; a lead only); insider "
        f"+{d('insider_bribe_only_vs_lie_only')['diff']:.2f} [{d('insider_bribe_only_vs_lie_only')['ci'][0]:.2f}, "
        f"{d('insider_bribe_only_vs_lie_only')['ci'][1]:.2f}], p < 0.0001 (survives correction across 110 tests); both "
        f"+{d('both_bribe_vs_forged')['diff']:.2f} [{d('both_bribe_vs_forged')['ci'][0]:.2f}, "
        f"{d('both_bribe_vs_forged')['ci'][1]:.2f}], p < 0.0001. Exploratory, not independently audited.")
CP = os.path.join(HERE, "captions.md")
txt = open(CP).read()
head = "## " + NAME + "\n"
if head in txt:
    i = txt.index(head); j = txt.find("\n## ", i + len(head))
    txt = txt[:i] + (txt[j + 1:] if j != -1 else "")
open(CP, "w").write(txt.rstrip("\n") + f"\n\n{head}\n{LONG}\n\nAlt text: {f.alt_text()}\n")
print(LONG)
