"""Figures for Noor's frozen results (NOOR-Research, final Nov 8 2026).

Reads only results files under ../data/ (single_agent/data/)  (json, csv and log text; nothing is executed
from there). Run from this folder:  viz run make_noor_figs.py
For each figure it writes <name>.png (screen size, with the title) and <name>.pdf (paper width, no title), and
captions.json with the caption text for the paper versions.
"""
import csv
import json
import math
import re
from pathlib import Path

import viz

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent


def wilson(k, n, z=1.959964):
    """95% Wilson interval, in percent."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def pct(k, n):
    return 100 * k / n


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def benign_log(path, arm):
    """engage count from a 'BENIGN <arm> {json}' line of a neuron-DPO log."""
    for line in open(path):
        if line.startswith(f"BENIGN {arm} "):
            d = json.loads(line.split(" ", 2)[2])
            return d["engage"], d["n"]
    raise KeyError(arm)


def eval_log(path):
    """tempt300 obey count from an 'EVAL l12_16: follows=k/n' line."""
    m = re.search(r"follows=(\d+)/(\d+)", open(path).read())
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------- (a) trade-off on the 300 + 100 finals
t300 = {r["name"]: r for r in jsonl(DATA / "noor_steer/tempt300.jsonl")}
dpo = {r["arm"]: r for r in json.load(open(DATA / "neuron_dpo/table.json"))}
abl = json.load(open(DATA / "gemma_ablation/summary.json"))["behaviour"]
abl_benign = re.search(r"engage (\d+)", open(DATA / "neuron_dpo/benign_abl.log").read())

base_obey = dpo["no training (unsteered)"]["follows"]          # 80 / 300
base_eng, _ = benign_log(DATA / "neuron_dpo/benign.log", "unsteered")  # 45 / 100
assert base_obey == abl["unsteered"]["follows"] == 80 and base_eng == t300["baseline_none"]["benign_engage"] == 45

dl12 = dpo["DPO style-matched (Gemma vs Gemma), LoRA L12-16 (neuron-guided)"]
dall = dpo["DPO style-matched (Gemma vs Gemma), LoRA all layers"]
e12, _ = benign_log(DATA / "neuron_dpo/benign.log", "style/lora_l12_16.pt")
eall, _ = benign_log(DATA / "neuron_dpo/benign.log", "style/lora_all.pt")
assert dl12["follows"] == dall["follows"] and e12 == eall  # both 0 / 300 and 0 / 100: drawn as one point
g_obey, _ = eval_log(DATA / "neuron_dpo/gentle.log")
g_eng, _ = benign_log(DATA / "neuron_dpo/benign.log", "gentle/lora_l12_16.pt")
a_obey, _ = eval_log(DATA / "neuron_dpo/active.log")
a_eng, _ = benign_log(DATA / "neuron_dpo/benign3.log", "active/lora_l12_16.pt")
F = t300["F_alpha0.1_pos"]
C6 = t300["C6_cosn2fade_on_F_alpha0.1_pos"]   # 3 / 300 and 2 / 100: sits on top of F, named in the caption
C9 = t300["SW_a0.06_C9wnone_tau-"]
C7 = t300["SW_a0.1_C7xC4_tau0.0"]

# label: (honest engagement %, obeyed %), group
A = {
    "No fix": ((base_eng, base_obey), "control"),
    "Neuron push": ((F["benign_engage"], F["follows"]), "push"),
    "Neuron push, probe switch": ((C9["benign_engage"], C9["follows"]), "push"),
    "Neuron push, two gates": ((C7["benign_engage"], C7["follows"]), "push"),
    "DPO on layers 12-16 or all": ((e12, dl12["follows"]), "dpo"),
    "Gentle DPO": ((g_eng, g_obey), "dpo"),
    "DPO on 52 active pairs": ((a_eng, a_obey), "dpo"),
    "Switch-off (n.s. vs random)": ((int(abl_benign.group(1)), abl["abl_obey10"]["follows"]), "off"),
}
A_POINTS = {k: (pct(e, 100), pct(o, 300)) for k, ((e, o), _) in A.items()}
A_COLORS = {k: {"control": "control", "push": "intervention", "dpo": "second", "off": "third"}[g]
            for k, (_, g) in A.items()}

# ---------------------------------------------------------------- (b) OBEY_ONLY, paired McNemar file
mc = json.load(open(DATA / "noor_steer/separate/mcnemar_obey_only_f0.1.json"))
o, h = mc["tempt300_obey"], mc["benign_heldout50_help"]
B = {"Obeyed the fake boss (300 test scenes)": [pct(o["base"], o["n"]), pct(o["steered"], o["n"])],
     "Completed the fair trade (50 held-out honest scenes)": [pct(h["base"], h["n"]), pct(h["steered"], h["n"])]}
B_CI = {"Obeyed the fake boss (300 test scenes)": [wilson(o["base"], o["n"]), wilson(o["steered"], o["n"])],
        "Completed the fair trade (50 held-out honest scenes)": [wilson(h["base"], h["n"]), wilson(h["steered"], h["n"])]}

# ---------------------------------------------------------------- (c) cosine by depth, letter-fixed, with null
rows = list(csv.DictReader(open(DATA / "noor_steer/separate/letterfix.csv")))
MODELS = {"google/gemma-2-2b-it": "Gemma-2-2B", "Qwen/Qwen2.5-1.5B-Instruct@989aa79": "Qwen2.5-1.5B",
          "HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2": "SmolLM2-1.7B"}
C = {}
for m, short in MODELS.items():
    rs = [r for r in rows if r["model"] == m]
    mid = [float(r["cos_obey_help_fixed"]) for r in rs if 0.4 <= float(r["depth"]) <= 0.7]
    C[short] = dict(depth=[float(r["depth"]) for r in rs], cos=[float(r["cos_obey_help_fixed"]) for r in rs],
                    lo=[float(r["null_lo"]) for r in rs], hi=[float(r["null_hi"]) for r in rs],
                    mid=sum(mid) / len(mid))
    print(f"{short}: mid-depth (0.4-0.7) fixed cos = {C[short]['mid']:.3f} over {len(mid)} layers")

# ---------------------------------------------------------------- (d) hosted models, real obey of 300
br = json.load(open(DATA / "tempt300/obey_breakdown.json"))
D = {m: v["give"] + v["defect"] for m, v in br.items()}   # outright key gifts + ordered attacks
NAME = {m: m.replace("@low", ", low effort").replace("@high", ", high effort") for m in br}
N_D = {m: v["n_conf"] + v["n_smite"] for m, v in br.items()}


# ---------------------------------------------------------------- figures
def figures():
    viz.set_roles(control=["No fix", "No push"], intervention=["Neuron push", "Push without the help part"],
                  second="DPO", third="Switch-off")
    figs = {}

    figs["fig_tradeoff"] = viz.scatter(
        A_POINTS, labels=True, colors=A_COLORS, xunit="%", yunit="%",
        xticks=[0, 10, 20, 30, 40, 50, 60, 70], yticks=[0, 5, 10, 15, 20, 25, 30],
        xlabel="Engaged with an honest trade offer", ylabel="Obeyed the fake boss",
        legend={"No fix": "control", "Neuron push": "intervention", "DPO": "second", "Switch-off": "third"},
        label_offsets={"DPO on layers 12-16 or all": (12, 4, "start"), "Switch-off (n.s. vs random)": (10, 0, "start")},
        title="Fixes that cut obeying below 10% cut honest engagement to 0-3%")

    figs["fig_obey_only"] = viz.bars(
        B, series=["No push", "Push without the help part"], ci=B_CI, decimals=1, sort=False,
        ticks=[0, 20, 40, 60, 80], ylabel="Share of scenes",
        title="Fair trades rose from 32% to 52%, but obeying did not fall",
        subtitle="Error bars are 95% intervals")

    panels = []
    for short, d in C.items():
        nul = [(lo + hi) / 2 for lo, hi in zip(d["lo"], d["hi"])]
        panels.append(viz.line(
            {"Obey vs help": d["cos"], "Null": nul}, x=d["depth"],
            bands={"Null": (d["lo"], d["hi"])}, roles={"intervention": "Obey vs help", "control": "Null"},
            unit="", domain=(-1, 1), ticks=[-1, -0.5, 0, 0.5, 1], xticks=[0, 0.25, 0.5, 0.75, 1],
            markers=False, end_labels=False, ref=0, marks={0.4: "Mid depth", 0.7: " "}, xlabel="Relative depth", ylabel="Cosine",
            title=short))
    figs["fig_cosine"] = viz.panels(
        panels, cols=3,
        legend={"Cosine of obey and help directions": "intervention", "Letter-balanced null, 95% range": "control"},
        title="Obey and help directions had cosine 0.32 to 0.46 at mid depth")

    viz.set_roles()
    figs["fig_hosted"] = viz.hbar(
        {NAME[m]: pct(k, N_D[m]) for m, k in D.items()}, sort=True, decimals=1, color="intervention",
        ci={NAME[m]: wilson(k, N_D[m]) for m, k in D.items()},
        notes={NAME[m]: f"{k} of {N_D[m]}" for m, k in D.items()}, domain=(0, 60), ticks=[0, 20, 40, 60],
        title="Hosted models obeyed a fake boss in 7.7% to 52.7% of scenes",
        subtitle="Error bars are 95% intervals")
    return figs


if __name__ == "__main__":
    for name, (pts) in A_POINTS.items():
        print(f"(a) {name}: engage {pts[0]:.1f}%  obey {pts[1]:.2f}%")
    print("(a) fade-gate variant (not drawn, on top of 'Neuron push'):", C6["follows"], C6["benign_engage"])
    print("(b)", B, "p obey", o["p_two_sided"], "p help", h["p_two_sided"])
    print("(d)", {m: (k, N_D[m]) for m, k in D.items()})

    captions = {}
    viz.set_defaults(size="screen", titles=True)
    for name, fig in figures().items():
        fig.save(OUT / f"{name}.png")
        captions[name] = {"title_caption": fig.caption(), "alt": fig.alt_text()}
    viz.set_defaults(size="paper", titles=False)
    for name, fig in figures().items():
        fig.save(OUT / f"{name}.pdf")
        fig.save(OUT / f"{name}_paper.png")
    json.dump(captions, open(OUT / "captions.json", "w"), indent=1)
