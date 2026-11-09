"""Sweep figures (plan/SWEEP_RESULTS.md), per factor, from the sweep_stats JSON (labels = conditions):
  sweep_<factor>_welfare.png     God effect vs the factor value: A - N (God - fork-matched noise; filled,
                                 primary) and A - B (God - noGod; hollow), mean with t95 bars, n below.
  sweep_<factor>_divergence.png  post-fork divergence from noGod: God (filled) vs noise (hollow).

    python infra/sweep/plot.py plan/figs/sweep_stats.json plan/figs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GRAY, INK, MUTED, SURF, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"

# factor -> (x label, [(label in the stats JSON, x value)])
FACTORS = {
    "start": ("God's first intervention round", [("start1", 1), ("base", 3), ("start8", 8), ("start14", 14)]),
    "board": ("Message board variant", [("base", 0), ("board_off", 1), ("board_delay2", 2), ("board_anon", 3),
                                        ("board_skip3", 4)]),
    "k": ("God budget K (interventions per game)", [("k1", 1), ("base", 2), ("k4", 4), ("k6", 6)]),
    "n": ("Number of agents N", [("n5", 5), ("base", 10), ("n20", 20)]),
    "keys": ("Keys per agent", [("keys3", 3), ("base", 5), ("keys8", 8)]),
    "rounds": ("Rounds per game", [("base", 20), ("r40", 40)]),
}
BOARD_TICKS = {0: "default", 1: "off", 2: "delay 2", 3: "anonymous", 4: "skip 3"}


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURF)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_title(title, fontsize=7, color=MUTED, loc="left")


def _point(ax, x: float, s: dict, filled: bool, color: str) -> None:
    ax.plot([x, x], [s["t_lo"], s["t_hi"]], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([x], [s["mean"]], "o", ms=8, color=color if filled else SURF, mec=color if not filled else SURF,
            mew=2, zorder=3)


def plot_pair(stats: dict, factor: str, keys: tuple[str, str], ylabel: str, title: str, out: Path) -> bool:
    xlabel, cells = FACTORS[factor]
    pts = [(lab, x) for lab, x in cells if lab in stats
           and (stats[lab].get(keys[0], {}).get("n") or stats[lab].get(keys[1], {}).get("n"))]
    if len(pts) < 2:
        return False
    xs = [x for _, x in pts]
    span = (max(xs) - min(xs)) or 1
    off = 0.025 * span
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=150, facecolor=SURF)
    _style(ax, xlabel, ylabel, title)
    lows = []
    for lab, x in pts:
        s0, s1 = stats[lab].get(keys[0], {}), stats[lab].get(keys[1], {})
        if s0.get("n"):
            _point(ax, x - off, s0, True, BLUE)
        if s1.get("n"):
            _point(ax, x + off, s1, False, GRAY)
        lo = min(s.get("t_lo", 9) for s in (s0, s1) if s.get("n"))
        n = (s0 if s0.get("n") else s1)["n"]
        lows.append(lo)
        ax.annotate(f"n={n}" + (" (low n)" if n < 8 else "") + ("" if s0.get("n") else "\nno noise arm"), (x, lo),
                    textcoords="offset points", xytext=(0, -12 if s0.get("n") else -20), ha="center", fontsize=7,
                    color=MUTED)
    ax.set_xticks(xs)
    if factor == "board":
        ax.set_xticklabels([BOARD_TICKS[x] for x in xs])
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(bottom=min(ylo, min(lows) - 0.08 * (yhi - ylo)))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return True


def main() -> None:
    stats = json.loads(Path(sys.argv[1]).read_text())
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    for f in FACTORS:
        p = outdir / f"sweep_{f}_welfare.png"
        if plot_pair(stats, f, ("welfare:god-noise", "welfare:god-nogod"), "Δ welfare",
                     "filled: God − fork-matched noise (primary); hollow: God − no-God; bars = t95", p):
            print(p)
        p = outdir / f"sweep_{f}_divergence.png"
        if plot_pair(stats, f, ("div_god", "div_noise"), "post-fork share of agents acting\ndifferently from no-God",
                     "filled: God branch; hollow: fork-matched noise branch; bars = t95", p):
            print(p)


if __name__ == "__main__":
    main()
