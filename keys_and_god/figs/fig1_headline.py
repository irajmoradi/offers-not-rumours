# Headline figure for P1 ("Bribes, not forgeries"). Every number is copied from a results file.
# Run: python3 -I fig1_headline.py <out.png>
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = sys.argv[1]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, SURF, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.0, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
fig.patch.set_facecolor(SURF)
for ax in (a, b, c):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def ci(ax, x, m, lo, hi, col, marker="o", hollow=False):
    ax.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle="round")
    ax.plot(x, m, marker, ms=7.5, color=SURF if hollow else col, mec=col, mew=2)


# (a) Keys & God, 10 agents, gpt-oss-120b, 20 rounds. Source: plan/E1_RESULTS.md dose table; plan/figs/e1_stats.json.
# K=2: God - B'' -0.062 [-0.207, 0.082] p=0.36 n=8 (g1_8); God - noGod +0.013 [-0.048, 0.073] p=0.72 n=16 (g1_16).
# K=18 (E1): God - B'' -0.140 [-0.204, -0.076] p=0.0002 n=16; God - noGod -0.094 [-0.162, -0.025] p=0.012 n=16.
ci(a, -0.12, -0.062, -0.207, 0.082, BLUE)
ci(a, 0.12, 0.013, -0.048, 0.073, GRAY, hollow=True)
ci(a, 0.88, -0.140, -0.204, -0.076, BLUE)
ci(a, 1.12, -0.094, -0.162, -0.025, GRAY, hollow=True)
a.axhline(0, color=INK2, lw=0.8)
a.set_xticks([0, 1])
a.set_xticklabels(["forger, 2 moves\nper game", "forger, every\nround (18 moves)"])
a.set_xlim(-0.6, 1.6)
a.set_ylim(-0.25, 0.14)
a.set_ylabel("change in group welfare (t95 over worlds)")
a.set_title("a  Forged authority (Keys & God, N=10)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
a.text(0.0, -0.24, "p = 0.36 / 0.72", ha="center", fontsize=7.5, color=INK2)
a.text(1.0, -0.24, "p = 0.0002 / 0.012", ha="center", fontsize=7.5, color=INK2)
a.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=2, ms=6, label="vs fork-matched noise B″"),
                  Line2D([], [], color=GRAY, marker="o", mfc=SURF, mew=2, lw=2, ms=6, label="vs no God")],
         loc="upper right", fontsize=7, frameon=False)

# (b) Arm U head-to-head (all-Gemma-3, N=8, 36 seeds, paired vs clean twins; exploratory, not independently audited).
# Source: jwin-planc analysis of planC_v2/out/arm_U_main (relayed by team lead): extra defections per game
# outsider forge 0.00 / bribe 0.19 (diff +0.19 [0.06, 0.36], p=0.033); insider lie-only 0.14 / bribe-only 1.03
# (diff +0.89 [0.56, 1.25], p<1e-4); both forge 0.31 / bribe 3.11 (diff +2.81 [2.39, 3.25], p<1e-4).
groups = ["outsider", "insider", "both"]
forge = [0.00, 0.14, 0.31]
bribe = [0.19, 1.03, 3.11]
diffs = ["+0.19 [0.06, 0.36]\np = 0.033", "+0.89 [0.56, 1.25]\np < 0.0001", "+2.81 [2.39, 3.25]\np < 0.0001"]
w = 0.36
for k in range(3):
    b.bar(k - w / 2, forge[k], width=w, color=GRAY)
    b.bar(k + w / 2, bribe[k], width=w, color=ORANGE)
    b.text(k - w / 2, forge[k] + 0.05, f"{forge[k]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.text(k + w / 2, bribe[k] + 0.05, f"{bribe[k]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.text(k, max(bribe[k], 0.6) + 0.42, diffs[k], ha="center", va="bottom", fontsize=7, color=INK)
b.set_xticks(range(3))
b.set_xticklabels(["outsider", "insider\n(lies vs bribes)", "both"], fontsize=8)
b.set_ylim(0, 4.6)
b.set_ylabel("extra defections per game vs clean twin")
b.set_title("b  Head-to-head: forge vs bribe (N=8)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
b.legend(handles=[Line2D([], [], color=GRAY, lw=6, label="forged messages / lies"),
                  Line2D([], [], color=ORANGE, lw=6, label="bribes")], loc="upper left", fontsize=7, frameon=False)

# (c) Arm I, key-based insider damage (pp) vs swarm size. EXPLORATORY, NOT AUDITED.
# Source: planC_v2/out/arm_I_v2_2/figs/figI2_key_damage_by_fleet.csv
fleets = [("all-Qwen-2.5-7B", BLUE, [(4, 21.1), (8, 3.6), (16, 0.3)]),
          ("all-Llama-3.1-8B", ORANGE, [(4, 16.4), (8, 2.8), (16, 0.7)]),
          ("all-gpt-oss-20b", AQUA, [(4, 16.7), (8, 1.5), (16, -0.3)]),
          ("all-Gemma-3-12B", YELLOW, [(4, 16.7), (8, -1.9), (16, -4.8)]),
          ("mixed (4 models)", MAGENTA, [(4, 20.8), (8, 2.1), (16, -0.3)])]
for name, col, pts in fleets:
    c.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.5, label=name)
c.axhline(0, color=INK2, lw=0.8)
c.set_xscale("log", base=2)
c.set_xticks([4, 8, 16])
c.set_xticklabels(["4", "8", "16"])
c.minorticks_off()
c.set_xlabel("agents in the swarm")
c.set_ylabel("insider damage to keys obtained (pp)")
c.set_ylim(-8, 25)
c.set_title("c  Larger swarms dilute one insider", loc="left", fontsize=9.5, fontweight="bold", color=INK)
c.legend(loc="upper right", fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=SURF)
print("wrote", OUT)
