"""E1 dose-response: welfare delta vs number of God acts per game (K = 1, 2, 4, 6 from grid/sweep; 18 = E1).
usage: python e1_dose.py SWEEP_STATS_JSON OUT_PNG   (style follows infra/sweep/plot.py)"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GRAY, INK, MUTED, SURF, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
st = json.load(open(sys.argv[1]))
# (x = God acts per game, label for A-N, label for A-B, tick text)
CELLS = [(1, "k1", "k1", "1\nfraud"), (2, "g1_8", "g1_16", "2 (G1)\nfraud+cot"), (4, "k4", "k4", "4\nfraud+cot"),
         (6, "k6", "k6", "6 (G3)\nfraud+cot"), (18, "e1", "e1", "18 (E1)\nfraud r3-r20")]
fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
ax.set_xscale("log")
for x, la, lb, _ in CELLS:
    a = st[la].get("welfare:god-noise", {})
    b = st[lb].get("welfare:god-nogod", {})
    for s, filled, col, dx in ((a, True, BLUE, 0.94), (b, False, GRAY, 1.06)):
        if not s.get("n"):
            continue
        xx = x * dx
        ax.plot([xx, xx], [s["t_lo"], s["t_hi"]], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([xx], [s["mean"]], "o", ms=8, color=col if filled else SURF, mec=SURF if filled else col, mew=2,
                zorder=3)
    lo = min(s["t_lo"] for s in (a, b) if s.get("n"))
    lab = f"n={a['n']}/{b['n']}" if a.get("n") else f"n={b['n']}\nno noise arm"
    ax.annotate(lab, (x, lo), textcoords="offset points", xytext=(0, -14 if a.get("n") else -22), ha="center",
                fontsize=7, color=MUTED)
ax.set_xticks([c[0] for c in CELLS])
ax.set_xticklabels([c[3] for c in CELLS], fontsize=7)
ax.minorticks_off()
ax.set_xlim(0.75, 26)
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo - 0.04, yhi)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_xlabel("God acts per game (log scale)", color=INK, fontsize=9)
ax.set_ylabel("Δ welfare (mean, t95 over worlds)", color=INK, fontsize=8)
ax.plot([], [], "o", ms=7, color=BLUE, mec=SURF, label="God − fork-matched noise B″ (primary)")
ax.plot([], [], "o", ms=7, color=SURF, mec=GRAY, mew=2, label="God − noGod")
ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=INK)
ax.set_title("God dose-response, N=10, 20 rounds, G1 worlds (sim, gpt-oss-120b)", fontsize=8, color=MUTED, loc="left")
fig.tight_layout()
fig.savefig(sys.argv[2], facecolor=SURF)
