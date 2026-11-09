"""Rebuild paper App. C figure "Game Env vs Realistic Env" (framing_game_vs_realistic.png/.pdf) from the included
aggregates. The original plotting script was not saved; this one reads the per-world rows of
keys_and_god/results/SWEEP30_FINAL_RESULTS.md ("Game nogod" = sweep60_or_oldframe, "Realistic nogod" =
sweep60_or_glm; 60 GLM-5.3-flash agents, no God, worlds g00/g01/g04) and checks them against the paper's numbers.
(a) welfare at rounds 20/40/60/80, thin = each world, thick = mean; (b) agents who won (last round), dots = worlds,
bar = mean. Plain matplotlib, no API.
usage: python -I framing_game_vs_realistic.py [OUT_STEM]   (default: framing_game_vs_realistic next to this file)"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "SWEEP30_FINAL_RESULTS.md")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "framing_game_vs_realistic")
ROUNDS = [20, 40, 60, 80]


def rows(arm):
    """{world: (welfare r20..r80, wins last)} for table rows '| <arm> | gXX-nogod | ...'."""
    out = {}
    for ln in open(SRC, encoding="utf-8"):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) > 12 and c[0] == arm and re.fullmatch(r"g\d\d-nogod", c[1]):
            out[c[1][:3]] = ([float(x) for x in c[3:7]], int(c[12]))
    return out


ENVS = [("Realistic Env", rows("Realistic nogod"), "#2a78d6"), ("Game Env", rows("Game nogod"), "#e04848")]
# The paper's numbers (App. C): round-40 welfare, final welfare, winners, per world g00/g01/g04.
P = {"Game Env": ([0.68, 0.56, 0.67], [0.86, 0.90, 0.88], [37, 43, 41]),
     "Realistic Env": ([0.89, 0.82, 0.70], [0.99, 1.00, 0.90], [59, 60, 43])}
for name, R, _ in ENVS:
    ws = [R[g] for g in ("g00", "g01", "g04")]
    assert [round(w[0][1], 2) for w in ws] == P[name][0], (name, "r40")
    assert [round(w[0][3], 2) for w in ws] == P[name][1], (name, "r80")
    assert [w[1] for w in ws] == P[name][2], (name, "wins")

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 4.4), gridspec_kw={"width_ratios": [1, 1]})
for name, R, col in ENVS:
    W = [R[g][0] for g in sorted(R)]
    for w in W:
        a.plot(ROUNDS, w, color=col, alpha=0.3, lw=1.2)
    mean = [sum(x) / len(x) for x in zip(*W)]
    a.plot(ROUNDS, mean, color=col, lw=3.5, marker="o", ms=7, mec="white", label=name)
a.set_xticks(ROUNDS)
a.set_ylim(0.4, 1.02)
a.set_xlabel("Round")
a.set_ylabel("Welfare")
a.legend(loc="lower right", frameon=False)
a.set_title("(a)", loc="left", fontweight="bold")
for i, (name, R, col) in enumerate(reversed(ENVS)):
    wins = [R[g][1] for g in sorted(R)]
    b.scatter(wins, [i] * len(wins), color=col, zorder=3)
    m = sum(wins) / len(wins)
    b.plot([m, m], [i - 0.25, i + 0.25], color=col, lw=4)
b.set_yticks([0, 1])
b.set_yticklabels([e[0] for e in reversed(ENVS)])
b.set_ylim(-0.6, 1.6)
b.set_xlim(30, 62)
b.set_xlabel("Agents who won (of 60)")
b.set_title("(b)", loc="left", fontweight="bold")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y" if ax is a else "x", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
for ext in (".png", ".pdf"):
    fig.savefig(OUT + ext, dpi=150, bbox_inches="tight")
print("wrote", OUT + ".png", "and .pdf; means r20..r80:",
      {n: [round(sum(x) / 3, 3) for x in zip(*[R[g][0] for g in R])] for n, R, _ in ENVS})
