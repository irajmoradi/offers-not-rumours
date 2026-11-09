"""Paper figures (vector PDF + PNG, two-column width = 7.0 in) from finished grid runs, numbered as in
paper/draft.md section 5.

    python -m swarm.viz.paper_figs OUT_DIR --cell "G1 sim K=2=runs/grid_oss_sim_k2" \
        --cell "G3 sim K=6=runs/grid_oss_sim_k6" --cell "G2 seq K=2=runs/r2_grid_oss_seq_k2" \
        [--baseline "B″ fork-matched noise=noise2"] [--world median | --world 3]

    fig2_network.{pdf,png}     one G1 world (rule: median |delta welfare|, or --world N) at God's first move and
                               at the last round, with and without God
    fig3_divergence.{pdf,png}  share of agents whose actions have differed from the no-God branch by round r, per
                               cell; thin = one world, thick = mean; gray = noise baselines (B' and any --baseline)
    fig4_welfare.{pdf,png}     per-world delta welfare with mean and Student-t 95% CI (primary, as in the paper);
                               left: G1 vs its noise baselines (H1); right: G3, G2 and the paired interactions
                               H3 = G3 delta - G1 delta, H2 = G2 delta - G1 delta (same worlds, paired by index)
Prints every number drawn (mean, t95, bootstrap 95%, sign-flip p, n). Only finished branches are used; seeds
with an unhealthy branch (swarm.analysis.grid_stats.game_health) are dropped, as in grid_stats.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from swarm.viz.metrics import (  # noqa: E402,F401  (cell_deltas, summarize, ... re-exported for older imports)
    aligned_curves,
    cell_deltas,
    cum_curves,
    divergence,
    load_branch,
    mean_curve,
    paired_diff,
    parse_seeds,
    parse_spec,
    summarize,
    welfare,
    worlds,
)
from swarm.viz.story import median_world  # noqa: E402
from swarm.viz.style import GRID, INK, INK2, C  # noqa: E402

WIDE = 7.0   # in, two-column figure width
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
CELL_COLORS = [C["a"], "#eb6834", C["give"]]   # blue, orange, aqua: validated all-pairs (first three slots)
NOISE = ["#9a9993", "#52514e"]
PRIMARY = "#184f95"   # God − B'' rows (primary contrast): dark step of the blue ramp                 # B', B'' baselines: neutral grays (reference, not a series)


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


# ------------------------------------------------------------------ divergence
XMAX = 8    # rounds after the fork shown; every curve has saturated well before this

def fig3_divergence(cells: list[tuple[str, Path]], out_dir: Path,
                    baselines: list[tuple[str, str]] = (("B′ re-run", "noise"),)) -> list[Path]:
    """Share of agents diverged from no-God vs rounds since each branch forked (God: its first move; baselines:
    their first differing round), so God and noise curves start from the same point."""
    fig, axes = plt.subplots(1, len(cells), figsize=(WIDE, 2.3), sharey=True, squeeze=False)
    handles: dict[str, Any] = {}
    for (name, run), ax in zip(cells, axes[0], strict=True):
        series = []
        for j, (lab, br) in enumerate(baselines):
            c, f = aligned_curves(run, br)
            if c:
                series.append((f"{lab}", c, f, NOISE[j % len(NOISE)]))
        c, f = aligned_curves(run, "god")
        series.append(("A: God (forks at its first move F)", c, f, CELL_COLORS[0]))
        for lab, curves, forks, col in series:
            for ys in curves.values():
                ax.plot(range(len(ys)), ys, color=col, lw=0.6, alpha=0.3)
            mean = mean_curve(curves)
            fk = statistics.median(forks)
            ax.plot(range(len(mean)), mean, color=col, lw=2, label=lab)
            if lab.startswith("A"):
                ax.set_title(f"{name}\nF = r{fk:g}, n = {len(curves)}", loc="left", fontsize=8)
        ax.set_xlabel("rounds since fork F")
        ax.set_xlim(-0.3, XMAX + 0.3)
        ax.set_xticks(range(0, XMAX + 1, 2))
        ax.set_ylim(-0.02, 1.05)
        for ln in ax.get_legend_handles_labels()[0]:
            handles.setdefault(ln.get_label(), ln)
    fig.legend(handles=list(handles.values()), loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=7)
    axes[0][0].set_ylabel("share of agents acting\ndifferently from B (so far)")
    fig.tight_layout(w_pad=0.6)
    return _save(fig, out_dir, "paperfig3_divergence")


# ------------------------------------------------------------------ welfare
def _forest(ax, rows: list[tuple[str, dict[int, float], str]], h: int, xlim=(-0.5, 0.5)) -> list:
    """Rows top-aligned in a panel of h slots; dots = worlds, diamond + bar = mean and Student-t 95% CI."""
    rng = np.random.default_rng(0)
    out = []
    for y, (lab, d, col) in enumerate(rows):
        xs = list(d.values())
        s = summarize(xs)
        out.append((lab, s))
        yy = h - 1 - y
        ax.scatter(xs, yy + rng.uniform(-0.18, 0.18, len(xs)), s=11, color=col, alpha=0.6, lw=0, zorder=2)
        ax.plot([s["tlo"], s["thi"]], [yy, yy], color=INK, lw=1.8, zorder=3, solid_capstyle="round")
        ax.scatter([s["mean"]], [yy], s=36, marker="D", color=INK, zorder=4, edgecolors="white", lw=0.8)
        ax.text(xlim[1] + 0.08 * (xlim[1] - xlim[0]), yy, f"{s['mean']:+.3f}\n[{s['tlo']:+.3f}, {s['thi']:+.3f}]\np={s['p']:.2f}  n={s['n']}".replace("-", "\u2212"),
                va="center", fontsize=6.2, color=INK2, clip_on=False, linespacing=1.15)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks([h - 1 - y for y in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, h - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["−0.4", "", "0", "", "0.4"])
    ax.set_xlabel("Δ welfare per world")
    ax.grid(axis="y", visible=False)
    return out


def fig4_welfare(cells: list[tuple[str, Path]], out_dir: Path,
                 baselines: list[tuple[str, str]] = (("B′ re-run", "noise"),)) -> tuple[list[Path], list]:
    (n1, r1), rest = cells[0], cells[1:]
    d1 = cell_deltas(r1)
    short = {name: name.split()[0] for name, _ in cells}           # "G1 sim K=2" -> "G1"
    baselines = list(baselines)
    left = [("A − B (God − no God)", d1, CELL_COLORS[0])]
    for j, (lab, br) in enumerate(baselines):
        nz = cell_deltas(r1, br, "nogod")
        if nz:
            left.append((f"{lab} − B", nz, NOISE[j % len(NOISE)]))
    an = cell_deltas(r1, "god", "noisefork")
    if an:
        left.append(("A − B″ (God − matched noise)", an, PRIMARY))
    right = []
    for k, (name, run) in enumerate(rest, start=1):
        right.append((f"{name}: A − B", cell_deltas(run), CELL_COLORS[k % len(CELL_COLORS)]))
    for name, run in rest:
        pd = paired_diff(d1, cell_deltas(run))
        if pd:
            right.append((f"{short[name]} Δ − {short[n1]} Δ (paired)", pd, INK2))
    h = max(len(left), len(right))
    H = 0.36 * h + 0.95
    fig = plt.figure(figsize=(WIDE, H))
    bot, top = 0.45 / H, 1 - 0.42 / H
    hl = (top - bot) * len(left) / h
    axl = fig.add_axes([0.19, top - hl, 0.18, hl])
    axr = fig.add_axes([0.72, bot, 0.18, top - bot])
    lim = max(0.5, max(abs(x) for _, d, _ in left + right for x in d.values()) + 0.05)
    stats = _forest(axl, left, len(left), (-lim, lim)) + _forest(axr, right, h, (-lim, lim))
    axl.set_title(f"H1: {n1}, God vs. noise", loc="left", x=-1.05)
    axr.set_title("Dose (H1′, H3) and turn mode (H2)", loc="left", x=-1.1)
    hs = [Line2D([], [], marker="o", ls="", color=INK2, alpha=0.6, ms=4, label="one world"),
          Line2D([], [], marker="D", color=INK, ms=5, lw=1.8, label="mean, Student-t 95% CI; sign-flip p")]
    if len(left) < h:                                                  # in the space under the shorter left panel
        fig.legend(handles=hs, loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(0.04, top - hl - 0.62 / H - 0.06))
    else:
        fig.legend(handles=hs, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    return _save(fig, out_dir, "paperfig4a_grid_effects"), stats


# ------------------------------------------------------------------ sweep summary
def fig5_sweep(conds: list[tuple[str, str, Any]], design: tuple | None, out_dir: Path) -> tuple[list[Path], list]:
    """Left: God − B'' Δwelfare per sweep condition (dots = worlds, ◆ mean, bar = 95% t-CI, p), grouped by factor;
    hollow ○ = God − no God mean for reference. Right: design effect without God (e.g. board off vs on), welfare
    per world in the no-God branch of the two settings on the same worlds, with the paired mean Δ.
    conds: (group, label, spec); design: (label, ref_runs, variant_runs[, frozenset of the original worlds]) where
    runs may be 'a+b' (no-God-only extension runs join by world index)."""
    from swarm.viz.metrics import design_effect

    rows = [(g, lab, cell_deltas(sp, "god", "noisefork"), cell_deltas(sp, "god", "nogod")) for g, lab, sp in conds]
    rows = [r for r in rows if r[2]]
    groups = list(dict.fromkeys(g for g, *_ in rows))
    slots: list[tuple[str, Any]] = []                     # ("hdr", group) or ("row", row)
    for g in groups:
        slots.append(("hdr", g))
        slots += [("row", r) for r in rows if r[0] == g]
    H = 0.24 * len(slots) + 0.95
    fig = plt.figure(figsize=(WIDE, H))
    bot, top = 0.62 / H, 1 - 0.3 / H
    ax = fig.add_axes([0.2, bot, 0.32, top - bot])
    rng = np.random.default_rng(0)
    stats, yt, yl = [], [], []
    for k, (kind, v) in enumerate(slots):
        y = len(slots) - 1 - k
        if kind == "hdr":
            ax.text(-0.02, y, v, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                    fontweight="bold", color=INK)
            continue
        _g, lab, an, ab = v
        xs = list(an.values())
        st = summarize(xs)
        sb = summarize(list(ab.values()))
        stats.append((f"{_g}: {lab}: God − B″", st))
        ax.scatter(xs, y + rng.uniform(-0.2, 0.2, len(xs)), s=9, color=C["a"], alpha=0.5, lw=0, zorder=2)
        ax.plot([st["tlo"], st["thi"]], [y, y], color=PRIMARY, lw=2, zorder=3, solid_capstyle="round")
        ax.scatter([st["mean"]], [y], marker="D", s=30, color=PRIMARY, edgecolors="white", lw=0.8, zorder=4)
        ax.scatter([sb["mean"]], [y], marker="o", s=22, facecolors="none", edgecolors=INK2, lw=1, zorder=4)
        ax.text(1.03, y, f"{st['mean']:+.3f} [{st['tlo']:+.3f}, {st['thi']:+.3f}]  p={st['p']:.2f}  n={st['n']}"
                .replace("-", "\u2212"), transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color=INK2)
        yt.append(y)
        yl.append(lab)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7)
    ax.set_ylim(-0.6, len(slots) - 0.4)
    lim = max(0.35, max(abs(x) for r in rows for x in r[2].values()) + 0.03)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Δ welfare, God − B″ (fork-matched noise)")
    ax.grid(axis="y", visible=False)
    ax.set_title("God's effect beyond matched noise, per condition", loc="left", x=-0.3)
    hs = [Line2D([], [], marker="o", ls="", color=C["a"], alpha=0.6, ms=3.5, label="one world"),
          Line2D([], [], marker="D", color=PRIMARY, ms=4.5, lw=2, label="mean, 95% t-CI"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, ms=4.5, label="God − no God mean")]
    fig.legend(handles=hs, loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.02, -0.01), fontsize=6.5)
    if design:
        dlab, sref, svar, orig = (*design, None)[:4]
        de = design_effect(sref, svar)
        sd, kind = de["stats"], "paired"
        if sd:
            stats.append((f"design: {dlab} (no-God welfare, {kind})", sd))
        axd = fig.add_axes([0.87, bot + 0.08, 0.12, top - bot - 0.18])
        for i, (a_, b_) in de["pairs"].items():
            first = orig is None or i in orig
            axd.plot([0, 1], [a_, b_], color=INK2, lw=0.8, alpha=0.6 if first else 0.35, zorder=2)
            axd.scatter([0, 1], [a_, b_], s=10, zorder=3, color=INK2 if first else "none", edgecolors=INK2,
                        lw=0.8, alpha=0.8)
        if de["pairs"]:
            ma = statistics.fmean(v[0] for v in de["pairs"].values())
            mb = statistics.fmean(v[1] for v in de["pairs"].values())
            axd.plot([0, 1], [ma, mb], color=C["defect"], lw=2.2, marker="D", ms=5, zorder=4)
        axd.set_xticks([0, 1])
        axd.set_xticklabels(["board on", "board off"] if "board" in dlab else ["reference", dlab], fontsize=7)
        axd.set_xlim(-0.35, 1.35)
        axd.tick_params(labelsize=6.5)
        axd.grid(axis="x", visible=False)
        axd.set_title(f"Without God: {dlab}\n(welfare per world)", loc="left", fontsize=7.2, x=-0.45)
        ptxt = ("p<0.001" if sd["p"] < 0.001 else f"p={sd['p']:.3f}") if sd else ""
        txt = ((f"Δ {sd['mean']:+.3f} [{sd['tlo']:+.3f}, {sd['thi']:+.3f}]\n{ptxt}, "
                f"n={sd['n']} ({kind})").replace("-", "\u2212") if sd else "fewer than 2 shared worlds")
        if orig is not None and sd:
            n0 = sum(1 for i in de["pairs"] if i in orig)
            txt += f"\n● first {n0} worlds  ○ extension"
        axd.text(0.5, -0.3, txt, transform=axd.transAxes, ha="center", va="top", fontsize=6.3, color=INK2)
    return _save(fig, out_dir, "paperfig4b_sweep_effects"), stats


# ------------------------------------------------------------------ schematic
def fig1_schematic(out_dir: Path, name: str = "paperfig1_schematic") -> list[Path]:
    """Environment + God + paired-fork design, drawn with matplotlib primitives (vector, shared palette)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(WIDE, 3.25))
    # ---------------- left: the game
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.32, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-1.72, 1.36, "a  The game", fontsize=9, fontweight="bold", va="top")
    n, rx, ry = 10, 0.86, 0.86
    cx, cy = 0.2, -0.08
    pos = [(rx * np.cos(np.pi / 2 - 2 * np.pi * i / n) + cx, ry * np.sin(np.pi / 2 - 2 * np.pi * i / n) + cy)
           for i in range(n)]

    def arrow(p, q, col, ls="-", lw=1.4, rad=0.15, z=2, head=8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=col, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=9, shrinkB=9, zorder=z))

    for i in (0, 2, 5, 7):                                    # board posts
        arrow(pos[i], (cx, cy), C["dm"], lw=0.8, rad=0.0, head=6)
    arrow(pos[1], pos[3], C["dm"], rad=0.25)                  # DM
    arrow(pos[6], pos[4], C["give"], lw=2.0, rad=0.2)         # give
    arrow(pos[3], pos[4], C["defect"], ls=":", lw=1.8, rad=0.3)    # defect
    for i, (x, y) in enumerate(pos):
        won = i in (6,)
        ax.add_patch(Circle((x, y), 0.105, fc=C["win"] if won else "#d6d5cf", ec=INK if won else INK2, lw=0.9, zorder=3))
    ax.add_patch(FancyBboxPatch((cx - 0.24, cy - 0.12), 0.48, 0.24, boxstyle="round,pad=0.02", fc=GRID, ec=INK2,
                                lw=0.8, zorder=3))
    ax.text(cx, cy + 0.03, "BOARD", ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK2, zorder=4)
    ax.text(cx, cy - 0.06, "+ action log", ha="center", va="center", fontsize=5.3, color=INK2, zorder=4)
    # agent callout
    x0, y0 = pos[1]
    ax.annotate("agent: holds 5 keys,\nneeds keys for 3 infos;\nscore = share of\nneeds held",
                (x0 + 0.1, y0 + 0.05), (1.12, 0.98), fontsize=5.8, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    # legend of actions
    ly = -1.2
    for k, (col, ls, lab) in enumerate(((C["dm"], "-", "DM / board post"), (C["give"], "-", "give key"),
                                        (C["defect"], ":", "defect (burn a need)"))):
        xx = -1.7 + k * 1.12
        ax.plot([xx, xx + 0.22], [ly, ly], color=col, ls=ls, lw=2)
        ax.text(xx + 0.27, ly, lab, fontsize=6, va="center", color=INK2)
    # God
    gx, gy = -1.42, 0.95
    ax.add_patch(FancyBboxPatch((gx - 0.2, gy - 0.12), 0.4, 0.24, boxstyle="round,pad=0.02", fc=INK, ec="none", zorder=4))
    ax.text(gx, gy, "GOD", ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=5)
    ax.text(gx - 0.3, gy - 0.18, "hidden; sees all\nactions and DMs;\nbudget of K moves", ha="left", va="top",
            fontsize=5.6, color=INK2)
    t1, t2 = pos[9], pos[7]
    arrow((gx + 0.2, gy), t1, C["god"], ls="--", lw=1.6, rad=-0.15, z=4)
    arrow((gx, gy - 0.32), t2, C["god"], ls="--", lw=1.6, rad=0.3, z=4)
    for t in (t1, t2):
        ax.add_patch(Circle(t, 0.15, fc="none", ec=C["god"], lw=1.6, zorder=4))
    ax.text(-0.95, 1.2, "\u2460 forged DM that appears\n    to come from a real agent", fontsize=5.8,
            color=C["god_ink"], ha="left", va="center")
    ax.text(-1.74, -0.62, "\u2461 overwrite an agent's\n    private notes\n    (CoT control)", fontsize=5.8,
            color=C["god_ink"], ha="left", va="center")
    ax.text(1.74, -0.98, "agents are never\ntold God exists", fontsize=5.6, color=INK2, ha="right", va="center",
            style="italic")
    # ---------------- right: paired fork
    bx = fig.add_axes([0.575, 0.0, 0.425, 1.0])
    bx.set_xlim(0, 10)
    bx.set_ylim(0, 10)
    bx.axis("off")
    bx.text(0.0, 9.75, "b  Paired fork (one world)", fontsize=9, fontweight="bold", va="top")
    F, end, ym = 3.6, 7.6, 5.6
    bx.plot([0.4, F], [ym, ym], color=INK2, lw=3, solid_capstyle="round")
    bx.text(0.4, ym + 0.35, "shared prefix\n(same seeds,\ncached replies)", ha="left", va="bottom",
            fontsize=5.8, color=INK2)
    bx.text(0.4, ym - 0.45, "r1", fontsize=6, color=INK2, ha="center", va="top")
    bx.text(F, ym - 0.45, "F", fontsize=6.5, color=INK, ha="center", va="top", fontweight="bold")
    branches = [(ym + 2.1, C["a"], "-", "A: with God", "God acts at F"), (ym, C["b"], "-", "B: no God", ""),
                (ym - 2.1, "#7a7974", "--", "B″: no God, every agent\nresampled from F", "")]
    for y, col, ls, lab, _note in branches:
        xs = np.linspace(F, end, 30)
        ys = ym + (y - ym) * (1 - np.exp(-(xs - F) * 2.2))
        bx.plot(xs, ys, color=col, lw=2.2, ls=ls)
        bx.text(end + 0.15, y, lab, fontsize=6.3, va="center", color=INK)
    bx.plot([F], [ym], marker="o", color=C["god"], ms=6, zorder=5)
    bx.text(F + 0.25, ym - 0.45, "= God's first move", fontsize=5.6, color=C["god_ink"], va="top", ha="left")
    bx.plot([0.4, end], [2.6, 2.6], color=NOISE[0], lw=1.2, ls=":")
    bx.text(end + 0.15, 2.6, "B′: re-run\nfrom r1 (G1)", fontsize=5.8, va="center", color=INK2)
    bx.text(0.0, 0.15, "Outcomes compared per world, then over worlds:\n"
            "God effect = A − B″ (primary), also A − B; noise = B″ − B.\n"
            "Divergence of A and B″ from B is measured from F.", fontsize=6, color=INK2, va="bottom")
    return _save(fig, out_dir, name)


# ------------------------------------------------------------------ board mechanism
BOARD_METRICS = [("welfare", "welfare (mean score)", "{:.2f}"), ("win_rate", "win rate", "{:.0%}"),
                 ("gives", "keys given / agent", "{:.2f}"), ("dms", "DMs / agent", "{:.1f}"),
                 ("defects_wrong", "wrong defects / agent", "{:.2f}"),
                 ("defects_correct", "correct defects / agent", "{:.2f}")]


BOARD_SHORT = {"welfare": "welfare", "win_rate": "win rate", "gives": "keys given", "dms": "DMs",
               "defects_wrong": "wrong defects", "defects_correct": "correct defects"}


def fig6_board_mechanism(ref_runs: str, var_runs: str, out_dir: Path, orig: frozenset[int] | None = None,
                         labels: tuple[str, str] = ("on", "off"), width: float = 4.0) -> tuple[list[Path], list]:
    """No-God games on identical worlds with the public board on and off, laid out for a narrow figure (default 4 in
    printed width, 2 columns x 3 rows, axis text >= 9 pt, annotations >= 8 pt): per metric, per-world paired lines
    (filled = original worlds, hollow = extension), condition means with Student-t 95% CIs, and the paired
    Δ (off − on) with its t95 and exact sign-flip p. Counts are per agent."""
    from swarm.analysis.grid_stats import t_ci
    from swarm.viz.metrics import design_effect

    fig, axes = plt.subplots(3, 2, figsize=(width, width * 1.5))
    stats = []
    for ax, (m, _title, fmt) in zip(axes.flat, BOARD_METRICS, strict=True):
        means: list[str] = []
        de = design_effect(ref_runs, var_runs, m)
        pairs, sd = de["pairs"], de["stats"]
        stats.append((f"board mechanism: {m} (board {labels[1]} − board {labels[0]}, paired)", sd))
        for i, (a_, b_) in pairs.items():
            first = orig is None or i in orig
            ax.plot([0, 1], [a_, b_], color=INK2, lw=0.6, alpha=0.4 if first else 0.28, zorder=1)
            ax.scatter([0, 1], [a_, b_], s=8, zorder=2, color=INK2 if first else "none", edgecolors=INK2, lw=0.6)
        for x, k in ((0, 0), (1, 1)):
            vals = [v[k] for v in pairs.values()]
            mu = statistics.fmean(vals)
            lo, hi = t_ci(vals)
            xm = x + (0.16 if x else -0.16)
            col = C["a"] if x == 0 else C["defect"]
            ax.plot([xm] * 2, [lo, hi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
            ax.scatter([xm], [mu], marker="D", s=34, zorder=4, color=col, edgecolors="white", lw=0.7)
            means.append(fmt.format(mu))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(labels), fontsize=9)
        ax.set_xlim(-0.45, 1.45)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        if m == "win_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.tick_params(labelsize=8.5, length=2)
        ax.grid(axis="x", visible=False)
        ax.set_title(BOARD_SHORT[m], loc="left", fontsize=9.5, pad=30)
        if sd:
            pt = "p<0.001" if sd["p"] < 0.001 else f"p={sd['p']:.2f}"
            if m == "win_rate":           # percentage points
                dtxt = f"Δ {100 * sd['mean']:+.0f} pp [{100 * sd['tlo']:+.0f}, {100 * sd['thi']:+.0f}], {pt}"
            else:
                d = 2 if abs(sd["mean"]) < 10 else 1
                dtxt = f"Δ {sd['mean']:+.{d}f} [{sd['tlo']:+.{d}f}, {sd['thi']:+.{d}f}], {pt}"
            ax.text(0.0, 1.02, f"mean {means[0]} \u2192 {means[1]}\n" + dtxt.replace("-", "\u2212"),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK2, linespacing=1.2)
    n = len(design_effect(ref_runs, var_runs, "welfare")["pairs"])
    hs = [Line2D([], [], color=INK2, marker="o", ms=3, lw=0.6, label="one world" + (
              f" (● first {len(orig)}, ○ extension)" if orig else "")),
          Line2D([], [], color=C["a"], marker="D", ms=4.5, lw=2.2, label="mean, 95% t-CI (on)"),
          Line2D([], [], color=C["defect"], marker="D", ms=4.5, lw=2.2, label="mean, 95% t-CI (off)")]
    fig.legend(handles=hs, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=8)
    fig.suptitle(f"Board on vs off, no God, {n} paired worlds\n(counts are per agent)",
                 x=0.02, ha="left", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.965), h_pad=1.0, w_pad=1.6)
    return _save(fig, out_dir, "paperfig5_board_mechanism"), stats


# ------------------------------------------------------------------ network overview
def fig_network_overview(runs: list[tuple[str, str]], out_dir: Path, note: str = "", name: str = "network_overview",
                         branch: str = "nogod") -> tuple[list[Path], list]:
    """Per topology (one run each): the contact graph of world 0 (node size = degree, hubs outlined), the degree
    distribution pooled over the run's worlds, and the knowledge-diffusion S-curve (mean share of the other
    players' names known vs round, known_ids runs; thin = one world). All read from the logs (game_start contacts,
    snapshots' known sets) via swarm.viz.metrics."""
    import networkx as nx
    from matplotlib.gridspec import GridSpec

    from swarm.viz.animate import hub_agents, network_layout
    from swarm.viz.metrics import branch_by_world, load_branch

    k = len(runs)
    fig = plt.figure(figsize=(WIDE, 3.9))
    gs = GridSpec(2, k + 1, figure=fig, height_ratios=[1.55, 1], width_ratios=[1] * k + [1.25],
                  hspace=0.6, wspace=0.5)
    axc = fig.add_subplot(gs[:, k])
    out = []
    cols = [C["a"], "#eb6834", C["give"]]
    for j, (lab, run) in enumerate(runs):
        dirs = branch_by_world(run, branch)
        bs = {i: load_branch(d) for i, d in sorted(dirs.items())}
        b0 = bs[min(bs)]
        net = b0.network or {}
        axg = fig.add_subplot(gs[0, j])
        axg.axis("off")
        pos = network_layout(b0.agents, net.get("edges", []), gap=(0.5, 0.5))
        G = nx.Graph()
        G.add_nodes_from(b0.agents)
        G.add_edges_from(tuple(e) for e in net.get("edges", []))
        deg = net.get("degree", {})
        hubs = hub_agents(deg)
        nx.draw_networkx_edges(G, pos, ax=axg, edge_color="#9a9993", width=0.9)
        nx.draw_networkx_nodes(G, pos, ax=axg, node_size=[25 + 22 * deg.get(a, 0) for a in b0.agents],
                               node_color=[cols[j % 3] for _ in b0.agents],
                               edgecolors=[INK if a in hubs else "white" for a in b0.agents],
                               linewidths=[1.6 if a in hubs else 0.6 for a in b0.agents])
        st = nx.average_clustering(G)
        axg.set_title(lab, fontsize=7.5, loc="left")
        axg.text(0.0, -0.04, f"world 0: {G.number_of_edges()} edges\nmean degree "
                 f"{2 * G.number_of_edges() / max(1, len(G)):.1f}, clustering {st:.2f}", transform=axg.transAxes,
                 fontsize=5.8, color=INK2, va="top", fontweight="normal")
        axd = fig.add_subplot(gs[1, j])
        degs = [d for b in bs.values() for d in (b.network or {}).get("degree", {}).values()]
        mx = max(degs) if degs else 1
        cnt = np.bincount(degs, minlength=mx + 1) if degs else np.zeros(1)
        axd.bar(range(len(cnt)), cnt / max(1, len(degs)), color=cols[j % 3], width=0.75)
        axd.set_xlabel("degree", fontsize=7)
        axd.set_xticks(range(0, len(cnt), max(1, len(cnt) // 5 + (len(cnt) % 5 > 0))))
        axd.tick_params(labelsize=6.5)
        if j == 0:
            axd.set_ylabel("share of agents", fontsize=7)
        axd.grid(axis="x", visible=False)
        axd.set_title(f"degree, {len(bs)} worlds", fontsize=6.8, loc="left", fontweight="normal")
        curves = {i: [b.known_share(r) for r in range(b.R + 1)] for i, b in bs.items() if b.known}
        curves = {i: [y for y in c if y is not None] for i, c in curves.items()}
        if curves:
            for c in curves.values():
                axc.plot(range(len(c)), c, color=cols[j % 3], lw=0.6, alpha=0.35)
            mean = mean_curve(curves)
            axc.plot(range(len(mean)), mean, color=cols[j % 3], lw=2, label=lab)
            out.append((f"diffusion {lab}: share of names known r0 → r{len(mean) - 1}",
                        {"n": len(curves), "mean": mean[-1], "tlo": mean[0], "thi": mean[-1], "lo": float("nan"),
                         "hi": float("nan"), "p": float("nan")}))
    axc.set_ylim(0, 1.02)
    axc.set_xlabel("round", fontsize=7.5)
    axc.set_ylabel("names known (share of others)", fontsize=7)
    axc.set_title("Name diffusion (known_ids)", fontsize=7.5, loc="left")
    axc.tick_params(labelsize=6.5)
    axc.legend(fontsize=6.5, loc="lower right")
    if note:
        fig.text(0.01, 0.995, note, fontsize=8, color=C["defect"], fontweight="bold", va="top", ha="left")
    return _save(fig, out_dir, name), out


# ------------------------------------------------------------------ timeline (optional paper fig)
def fig_timeline(game_dir: Path, out_dir: Path, title: str | None = None) -> list[Path]:
    """Per-agent timeline of one God branch (swarm.viz.figures.timeline_fig), saved as vector PDF + PNG."""
    from swarm.runlog.trace import load_game
    from swarm.viz.figures import timeline_fig

    fig = timeline_fig(load_game(game_dir, with_llm_calls=False, with_snapshots=False), title)
    fig.set_size_inches(WIDE, fig.get_size_inches()[1] * WIDE / fig.get_size_inches()[0] * 1.15)
    return _save(fig, out_dir, "paperfig6_timeline")


# ------------------------------------------------------------------ network pair
def fig2_network(run: Path, world: int, out_dir: Path) -> list[Path]:
    from swarm.viz.animate import draw_panel, layout

    gd, nd = run / "games" / f"g{world:02d}-god", run / "games" / f"g{world:02d}-nogod"
    god, nogod = load_branch(gd), load_branch(nd)
    _, ever = divergence(gd, nd)
    pos = layout(god.agents, [god, nogod])
    for a in god.agents:   # pull the ellipse in so side labels stay inside their panel
        pos[a] = pos[a] * np.array([0.74, 1.0])
    r0 = god.powers[0].round if god.powers else 3
    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 4.6))
    for row, r in enumerate((r0, god.R)):
        wg, wn = god.welfare(r), nogod.welfare(r)
        for ax, lab, b, kw in ((axes[row][0], f"A: with God · round {r} · welfare {wg:.2f}", god, {"ever": ever}),
                               (axes[row][1], f"B: without God · round {r} · welfare {wn:.2f}", nogod,
                                {"show_god": False})):
            draw_panel(ax, "", b, pos, r, scale=0.42, **kw)
            ax.set_title(lab, loc="left", fontsize=7, color=C["a"] if b is god else INK2, pad=1)
    for ax in axes.flat:
        for t in ax.texts:
            t.set_fontsize(min(t.get_fontsize(), 6.0) if t.get_text() not in ("GOD", "BOARD") else 4.5)
    hs = [Line2D([], [], color=C["dm"], lw=1, label="DM (this round)"),
          Line2D([], [], color=C["give"], lw=1.8, label="key given"),
          Line2D([], [], color=C["defect"], lw=1.6, ls=":", label="defect"),
          Line2D([], [], color=C["god"], lw=1.8, ls="--", label="God's move"),
          Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=C["god"], mew=1.8, label="God's target"),
          Line2D([], [], marker="o", ls="", ms=8, color=C["a"], alpha=0.35, label="acts differently than in B"),
          Line2D([], [], marker="o", ls="", ms=5, color=C["win"], mec=INK, label="won (size = score)")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.09, wspace=0.0, hspace=0.07)
    fig.add_artist(Line2D([0.5, 0.5], [0.1, 0.98], color=GRID, lw=0.8))
    return _save(fig, out_dir, "paperfig2_network")


def _cells(xs: list[str]) -> list[tuple[str, str]]:
    return [(c.rsplit("=", 1)[0], c.rsplit("=", 1)[1]) for c in xs]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cell", action="append", default=[],
                    help='"label=RUN[,NOISE_RUN[@BRANCH]][#seeds]" (fig4; the first is the main cell, fig2 world)')
    ap.add_argument("--div-cell", action="append", default=[], help="cells for fig3 (default: --cell)")
    ap.add_argument("--sweep", action="append", default=[], help='fig5 rows: "group|label=SPEC"')
    ap.add_argument("--design", help='fig5 design panel: "label=REF_SPEC|VARIANT_SPEC"')
    ap.add_argument("--board", help='fig6: "REF_RUNS|VARIANT_RUNS[|original seeds]" (no-God board on vs off)')
    ap.add_argument("--network", action="append", default=[], help='network_overview: "label=run_dir" per topology')
    ap.add_argument("--network-note", default="", help="banner on network_overview (e.g. dry-run placeholder)")
    ap.add_argument("--only-network", action="store_true", help="only (re)draw network_overview")
    ap.add_argument("--world", default="median", help='"median" (rule) or a game index of the main cell')
    a = ap.parse_args(argv)
    if a.network:
        pn, sn = fig_network_overview(_cells(a.network), a.out_dir, a.network_note)
        print(*pn)
        for lab, st in sn:
            print(f"  {lab}: {st['tlo']:.3f} -> {st['thi']:.3f} (n={st['n']} worlds)")
        if a.only_network:
            return
    cells = _cells(a.cell)
    base = [("B′: re-run from r1", "noise"), ("B″: fork-matched noise (resampled at F)", "noisefork")]
    print(*fig3_divergence(_cells(a.div_cell) or cells, a.out_dir, base))
    paths, stats = fig4_welfare(cells, a.out_dir, [("B′ re-run", "noise"), ("B″ matched", "noisefork")])
    print(*paths)
    if a.sweep:
        conds = [(c.split("|", 1)[0], c.split("|", 1)[1].rsplit("=", 1)[0], c.rsplit("=", 1)[1]) for c in a.sweep]
        design = None
        if a.design:          # "label=REF_RUNS|VARIANT_RUNS[|orig seeds]"
            dl, ds = a.design.split("=", 1)
            parts = ds.split("|")
            design = (dl, parts[0], parts[1], parse_seeds(parts[2]) if len(parts) > 2 else None)
        p5, s5 = fig5_sweep(conds, design, a.out_dir)
        print(*p5)
        stats += s5
    if a.board:
        ps = a.board.split("|")
        p6, s6 = fig6_board_mechanism(ps[0], ps[1], a.out_dir, parse_seeds(ps[2]) if len(ps) > 2 else None)
        print(*p6)
        stats += s6
    for lab, s in stats:
        print(f"  {lab.replace(chr(10), ' ')}: {s['mean']:+.3f} t95 [{s['tlo']:+.3f}, {s['thi']:+.3f}] "
              f"boot [{s['lo']:+.3f}, {s['hi']:+.3f}] p={s['p']:.4f} n={s['n']}")
    main_run = parse_spec(cells[0][1]).run
    if a.world == "median":
        w, why = median_world({i: abs(d) for i, d in cell_deltas(main_run).items()})
        print(f"  fig2/fig6 world: g{w:02d} ({why})")
    else:
        w = int(a.world)
    print(*fig2_network(main_run, w, a.out_dir))
    print(*fig1_schematic(a.out_dir))
    print(*fig_timeline(main_run / "games" / f"g{w:02d}-god", a.out_dir,
                        f"World g{w:02d} with God ({cells[0][0]}): who did what, per round"))


if __name__ == "__main__":
    main()
