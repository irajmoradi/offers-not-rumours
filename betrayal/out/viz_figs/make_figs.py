"""Plan C (paper P1) result figures in the viz house style. Run: viz run make_figs.py
Reads only the existing analysis outputs in ../ (figs/*.csv, results_*.json) and per_seed.json
(made by `python3 -I extract_per_seed.py` from games.jsonl `defection_effect`; its means match the CSVs exactly).
No new experiments, no API calls. Each figure is saved three ways:
  <name>.pdf and <name>.png  paper size, no title in the image (the title goes in captions.md)
  <name>_screen.png          screen size with the title, for Discord
"""
import csv, json, os
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
SEEDS = json.load(open(os.path.join(HERE, "per_seed.json")))
EB = "Error bars are 95% intervals"
YL = "Extra defections per game"
SIZE = {"w": 1008}


def bw(frac, groups):
    """Bar width in layout px: a fraction of each group's width, so the per-seed dots have room."""
    return frac * (SIZE["w"] - 100) / groups


def rows(path):
    with open(os.path.join(OUT, path)) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x in ("", None) else float(x)


def cells(path, key):
    return {r[key]: r for r in rows(path)}


def ci_of(r):
    return (fnum(r["ed_lo"]), fnum(r["ed_hi"]))


# ---------------------------------------------------------------- 1. Arm U head-to-head
U = cells("arm_U_v2_2/figs/figU_cells.csv", "cell")
H2H = json.load(open(os.path.join(OUT, "arm_U_v2_2/headtohead.json")))


def fig_headtohead():
    pairs = [("Outsider", "posXmethod=outsider_forged", "pos=outsider"),
             ("Insider", "method=insider_lie_only", "method=insider_bribe_only"),
             ("Both positions", "posXmethod=both_forged", "pos=both")]
    # sanity: the head-to-head file's means are these cells' means
    for (_, a, b), k in zip(pairs, ("outsider_bribe_vs_forged", "insider_bribe_only_vs_lie_only", "both_bribe_vs_forged")):
        assert abs(H2H[k]["b_mean"] - float(U[a]["extra_defections"])) < 6e-3, k
        assert abs(H2H[k]["a_mean"] - float(U[b]["extra_defections"])) < 6e-3, k
    data = {g: [float(U[a]["extra_defections"]), float(U[b]["extra_defections"])] for g, a, b in pairs}
    ci = {g: [ci_of(U[a]), ci_of(U[b])] for g, a, b in pairs}
    runs = {g: [SEEDS["U"][a], SEEDS["U"][b]] for g, a, b in pairs}
    return viz.bars(
        data, series=["Forged messages or lies", "Bribes"], ci=ci, sort=False, unit="", decimals=2, bar_width=bw(0.3, 3),
        roles={"baseline": "Forged messages or lies", "intervention": "Bribes"}, ylabel=YL,
        title="Bribes raised defections more than forged messages or lies in every position",
        subtitle=EB)


# ---------------------------------------------------------------- 2. Dose-response at N=16
def fig_dose():
    D = rows("dose_N16_v2_2/figs/figDose_N16.csv")
    key = {(r["contagion"] == "True", int(r["size"])): r for r in D}
    names = {False: "Outsider only", True: "Outsider with contagion"}
    series = {names[c]: [float(key[(c, s)]["extra_defections"]) for s in (1, 2, 3)] for c in (False, True)}
    bands = {names[c]: ([float(key[(c, s)]["ed_lo"]) for s in (1, 2, 3)],
                        [float(key[(c, s)]["ed_hi"]) for s in (1, 2, 3)]) for c in (False, True)}
    src = {1: ("C16", ""), 2: ("C2K", "_2key"), 3: ("C16", "_3key")}
    runs = {}
    for c in (False, True):
        base = "outsider+contagion" if c else "outsider_only"
        per = [SEEDS[src[s][0]][base + src[s][1]] for s in (1, 2, 3)]
        seeds = [SEEDS[src[s][0] + "_seeds"][base + src[s][1]] for s in (1, 2, 3)]
        assert seeds[0] == seeds[1] == seeds[2]
        runs[names[c]] = [list(t) for t in zip(*per)]
    return viz.line(
        series, x=[1, 2, 3], bands=bands, xscale="category", unit="",
        xtick_labels={1: "1 key", 2: "2 keys", 3: "3 keys"},
        roles={"baseline": "Outsider only", "intervention": "Outsider with contagion"},
        xlabel="Size of the outsider's bribe (N=16)", ylabel=YL,
        title="Defections rose with the outsider's bribe size, with and without contagion",
        domain=(-0.5, 2), ticks=[-0.5, 0, 0.5, 1, 1.5, 2], subtitle="Bands are 95% intervals")


# ---------------------------------------------------------------- 3. Pay timing
T_NAMES = {"never_pays": "Never pays", "pays_now": "Pays at once", "pays_later3": "Pays 3 rounds later",
           "pays_at_end": "Pays at game end"}
T_ROLES = {"baseline": "Never pays", "intervention": "Pays at game end", "second": "Pays at once",
           "third": "Pays 3 rounds later"}


def fig_paytiming():
    T = cells("arm_T_v2_2/figs/figT_paytiming.csv", "cell")
    order = ["never_pays", "pays_now", "pays_later3", "pays_at_end"]
    data = {T_NAMES[c]: float(T[c]["extra_defections"]) for c in order}
    ci = {T_NAMES[c]: ci_of(T[c]) for c in order}
    runs = {T_NAMES[c]: SEEDS["T"][c] for c in order}
    return viz.bars(
        data, ci=ci, sort=False, unit="", roles=T_ROLES, ylabel=YL, bar_width=bw(0.5, 4),
        title="Extra defections stayed at 2.175 to 2.225 per game across pay timings",
        subtitle=EB)


def fig_paytiming_ledger():
    # Ties at the minimum split evenly (expected credit 1/k per tied agent), from the tie-split recompute
    # planC_v2/analysis/armT_ledger_by_round_tiesplit.py; replaces analyze_t.py's single random tie-break draw.
    R = rows("arm_T_v2_2/figs/figT_detection_over_rounds_tiesplit.csv")
    order = ["never_pays", "pays_now", "pays_later3", "pays_at_end"]
    final = {r["cell"]: float(r["ledger_top1"]) for r in R if r["round"] == "final"}
    assert {c: f"{final[c]:.2f}" for c in ("never_pays", "pays_now", "pays_at_end")} == \
        {"never_pays": "0.79", "pays_now": "0.45", "pays_at_end": "0.46"}, final   # P1 section 4 quotes these
    R = [r for r in R if r["round"] != "final"]
    rounds = sorted({int(r["round"]) for r in R})
    series = {}
    for c in order:
        m = {int(r["round"]): 100 * float(r["ledger_top1"]) for r in R if r["cell"] == c}
        assert len(m) == len(rounds), c
        series[T_NAMES[c]] = [m[x] for x in rounds]
    # first round at which the end-of-game payer falls clearly below the never-payer (> 10 points)
    gap = [x for x, a, b in zip(rounds, series["Never pays"], series["Pays at game end"]) if a - b > 10]
    assert gap and gap[0] == 12, gap
    return viz.line(
        series, x=rounds, roles=T_ROLES, unit="%", decimals=1, ref=12.5, ref_label="Chance (1 in 8)",
        domain=(0, 100), xlabel="Round", ylabel="Games where the ledger rule names the insider",
        title=f"Paying at game end did not lower ledger flags before round {gap[0]}")


# ---------------------------------------------------------------- 4. Dual control
def fig_dual():
    DC = cells("arm_DC_v2_2/figs/figDC_cells.csv", "cond")
    groups = [("Bribe-and-lie insider", "DC0_base", "DC1_base_dual"), ("Bribe-only insider", "DC0_bribe", "DC1_bribe_dual")]
    data = {g: [float(DC[a]["extra_defections"]), float(DC[b]["extra_defections"])] for g, a, b in groups}
    ci = {g: [ci_of(DC[a]), ci_of(DC[b])] for g, a, b in groups}
    runs = {g: [SEEDS["DC"][a], SEEDS["DC"][b]] for g, a, b in groups}
    return viz.bars(
        data, series=["No co-signer", "Co-signer required"], ci=ci, sort=False, unit="", decimals=2, bar_width=bw(0.38, 2),
        roles={"baseline": "No co-signer", "intervention": "Co-signer required"}, ylabel=YL,
        title="A co-signer on key burns cut defections for both insider types",
        subtitle=EB)


# ---------------------------------------------------------------- 5. Social-engineering levers
def fig_se():
    S = cells("arm_SE_v2_2/figs/figSE_levers.csv", "lever")
    names = {"SE-scarcity": "Urgency", "SE-recip": "Bribe", "SE-liking": "Flattery",
             "SE-social": "Social proof", "SE-rumour": "Forged rumour"}
    data = {names[k]: float(S[k]["extra_defections"]) for k in names}
    ci = {names[k]: ci_of(S[k]) for k in names}
    notes = {names[k]: f'{S[k]["accepted"]} of {S[k]["messages"]} messages acted on' for k in names}
    return viz.dotplot(
        data, ci=ci, notes=notes, sort=True, unit="", highlight=["Urgency", "Bribe"], ref=0, xlabel=YL,
        title="Only urgency and bribe messages raised defections, by 0.35 and 0.25",
        subtitle=EB)


# ---------------------------------------------------------------- 6. Defenses on Gemma (arm U)
def fig_defenses():
    names = {"defense=resilience": "Resilience prompt", "defense=ALL": "All seven defenses",
             "defense=reputation+corroboration+quarantine": "Rep + corroboration + quarantine",
             "defense=corroboration": "Corroboration", "defense=monitor": "Monitor", "defense=reputation": "Reputation",
             "defense=verify": "Verify", "defense=quarantine": "Quarantine",
             "defense=provenance+verify": "Provenance and verify", "defense=provenance": "Provenance"}
    names["BASE"] = "No defense"
    data = {names[k]: float(U[k]["extra_defections"]) for k in names}
    ci = {names[k]: ci_of(U[k]) for k in names}
    holm = ["Resilience prompt", "All seven defenses", "Rep + corroboration + quarantine", "Corroboration"]
    base = float(U["BASE"]["extra_defections"])
    return viz.dotplot(
        data, ci=ci, sort="asc", unit="", decimals=2, highlight=holm, ref=base,
        zero=True, xlabel=YL,
        title="A resilience prompt alone cut extra defections from 2.22 to 0",
        subtitle=EB)


# ---------------------------------------------------------------- 7. Contagion, N=8 vs N=16
def fig_n8_n16():
    C = rows("arm_C16_v2_2/figs/figC_N8_vs_N16.csv")
    k = {(int(r["N"]), r["cell"]): r for r in C}
    groups = [("1 key, outsider only", "outsider_only"), ("3 keys, outsider only", "outsider_only_3key"),
              ("1 key, with contagion", "outsider+contagion"), ("3 keys, with contagion", "outsider+contagion_3key")]
    data = {g: [float(k[(8, c)]["extra_defections"]), float(k[(16, c)]["extra_defections"])] for g, c in groups}
    ci = {g: [ci_of(k[(8, c)]), ci_of(k[(16, c)])] for g, c in groups}
    return viz.bars(
        data, series=["8 agents", "16 agents"], ci=ci, sort=False, unit="",
        roles={"ordered": ["8 agents", "16 agents"]}, ylabel=YL, domain=(-0.5, 2), ticks=[-0.5, 0, 0.5, 1, 1.5, 2],
        title="16 agents had more extra defections than 8 only with 3-key bribes",
        subtitle=EB)


FIGS = [("fig1_bribe_vs_forgery", fig_headtohead), ("fig2_dose_N16", fig_dose), ("fig3a_paytiming_defections", fig_paytiming),
        ("fig3b_paytiming_ledger", fig_paytiming_ledger), ("fig4_dual_control", fig_dual), ("fig5_se_levers", fig_se),
        ("fig6_defenses_gemma", fig_defenses), ("fig7_contagion_N8_N16", fig_n8_n16)]

caps = ["# Plan C (P1) figure captions\n",
        "Generated by make_figs.py from the existing CSV/JSON outputs. Escape each % as \\% in LaTeX.\n",
        "Measure in every defection figure: extra defections per game vs the matched clean twin (same seed and setup, no attack). "
        "Figures show means with 95% seed-bootstrap intervals (fig 3b: share of games); per-seed values are in per_seed.json. Gemma-3-12B, N=8 unless stated.\n"]
for name, build in FIGS:
    viz.set_defaults(size="paper", titles=False); SIZE["w"] = 748
    f = build()
    f.save(os.path.join(HERE, name + ".pdf"))
    f.save(os.path.join(HERE, name + ".png"))
    caps.append(f"## {name}\n\n{f.caption()}\n\nAlt text: {f.alt_text()}\n")
    viz.set_defaults(size="screen", titles=True); SIZE["w"] = 1008
    g = build()
    g.save(os.path.join(HERE, name + "_screen.png"))
CP = os.path.join(HERE, "captions.md")
keep = ""
if os.path.exists(CP):  # keep the P1 Figure 1 section written by fig_P1_figure1.py
    old = open(CP).read(); k = old.find("## fig_P1_figure1\n")
    keep = old[k:] if k != -1 else ""
open(CP, "w").write("\n".join(caps) + ("\n" + keep if keep else ""))
print("done")
