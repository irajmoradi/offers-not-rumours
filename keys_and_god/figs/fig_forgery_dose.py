# Headline figure for P1. Every number is copied from a results file (see comments).
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BLUE, ORANGE, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})

fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
fig.patch.set_facecolor(SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# Panel a: tempt300 real ("unhedged") obey, 300 scenes per model.
# Source: lfm_lab/NOOR-Research/NOOR-Research.md section 3.6 (data/tempt300/TEMPT300.md); audit AUDIT_noor.md B14-B19.
models = [("glm-5.3-flash, high", 23, 0.05, 0.11), ("qwen3.8-27b, low", 39, 0.10, 0.17),
          ("glm-5.3-flash, low", 48, 0.12, 0.21), ("gemma-4-26b-a4b", 56, 0.15, 0.23),
          ("deepseek-v4-flash, high", 127, 0.37, 0.48), ("gemma-4-31b", 158, 0.47, 0.58)]
ys = list(range(len(models)))
for y, (name, k, lo, hi) in zip(ys, models):
    r = k / 300
    a.barh(y, r * 100, height=0.55, color=BLUE)
    a.plot([lo * 100, hi * 100], [y, y], color=INK, lw=1.2)
    a.text(hi * 100 + 1.5, y, f"{k}/300", va="center", color=INK2, fontsize=8)
a.set_yticks(ys)
a.set_yticklabels([m[0] for m in models])
a.set_xlim(0, 75)
a.set_xlabel("% of scenes the agent obeys the forged authority\n(outright key gift or ordered attack; 95% interval)")
a.set_title("a  Alone: one agent, one forged order", loc="left", fontsize=10.5, fontweight="bold", color=INK)
a.grid(axis="x", color="#e6e5e1", lw=0.6)
a.set_axisbelow(True)

# Panel b: swarm welfare change by forger dose. N=10, 20 rounds, gpt-oss-120b, G1 worlds.
# Source: plan/E1_RESULTS.md dose table; plan/figs/e1_stats.json (welfare:god-noise / god-nogod).
dose = [  # label, mean, lo, hi, n, kind
    ("1", -0.104, -0.213, 0.005, 8, "bpp"),
    ("2", -0.062, -0.207, 0.082, 8, "bpp"),
    ("4", -0.050, -0.178, 0.078, 8, "bpp"),
    ("6", 0.004, -0.097, 0.105, 16, "nogod"),
    ("18\n(every round)", -0.140, -0.204, -0.076, 16, "bpp"),
]
xs = [0, 1, 2, 3, 4]
for x, (lab, m, lo, hi, n, kind) in zip(xs, dose):
    col = BLUE if kind == "bpp" else GRAY
    b.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle="round")
    b.plot(x, m, "o", ms=8, color=col if kind == "bpp" else SURF, mec=col, mew=2)
    b.text(x, lo - 0.018, f"n={n}", ha="center", va="top", fontsize=7.5, color=INK2)
# Board removed, no God (reference). Source: AUDIT_planA.md results table; plan/figs/NUMBERS.txt "design: board off".
b.plot([5.4, 5.4], [-0.257, -0.089], color=ORANGE, lw=2)
b.plot(5.4, -0.173, "s", ms=8, color=ORANGE)
b.text(5.4, -0.257 - 0.018, "n=16", ha="center", va="top", fontsize=7.5, color=INK2)
b.axhline(0, color=INK2, lw=0.8)
b.axvline(4.7, color="#d6d5d0", lw=0.8, ls=":")
b.set_xticks(xs + [5.4])
b.set_xticklabels([d[0] for d in dose] + ["board\nremoved"])
b.set_xlabel("forged acts per game (10 agents, 20 rounds)")
b.set_ylabel("change in swarm welfare (95% interval)")
b.set_ylim(-0.30, 0.13)
b.set_title("b  Together: swarm welfare vs forger dose", loc="left", fontsize=10.5, fontweight="bold", color=INK)
b.grid(axis="y", color="#e6e5e1", lw=0.6)
b.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BLUE, marker="o", lw=2, ms=7, label="God vs re-sampled game"),
     Line2D([], [], color=GRAY, marker="o", mfc=SURF, mew=2, lw=2, ms=7, label="God vs no God"),
     Line2D([], [], color=ORANGE, marker="s", lw=2, ms=7, label="board removed vs board (no God)")]
b.legend(handles=h, loc="upper left", fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=SURF)
print("wrote", OUT)
