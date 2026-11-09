"""Paired-fork demo video: the same world with God (left) and without God (right), round by round.

    python -m swarm.viz.animate RUN_DIR OUT.mp4 [--world 3 | --god g03-god --nogod g03-nogod]
                                [--preview-round 3 --preview PNG] [--no-cards]

Frame = header (round counter) + running caption (God's move, its visible effect, the ripple count)
+ two network panels (ring = agent God targets, blue halo = agent whose actions now differ from the
no-God branch) + a strip under each panel (welfare per branch; agents diverged so far).
Title card and end card (final scores) frame the game. 1280x720, H.264 yuv420p, 10 fps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyBboxPatch

from swarm.viz.metrics import Branch, divergence, load_branch, round_series
from swarm.viz.story import captions, pacing
from swarm.viz.style import GRID, INK, INK2, SURFACE, C

W, H, DPI, FPS = 1280, 720, 100, 10
BIG_N = 20          # above this many agents: smaller nodes, labels only on God targets
NODE = "#d6d5cf"    # neutral agent fill (status is carried by yellow = won)
CAP_COLOR = {"god": C["god"], "quote": None, "effect": C["god"], "ripple": C["a"], "win": C["win"], "info": INK2}
NET_EDGE = "#c9c8c2"          # contact-graph edges (static, under the activity)
NET_NAMES = {"erdos_renyi": "Erdős–Rényi", "watts_strogatz": "Watts–Strogatz", "barabasi_albert": "Barabási–Albert",
             "ring": "ring", "complete": "complete"}
NET_RX, NET_RY = 1.25, 0.76    # network layout extent (labels need room at the sides)
RX, RY = 1.0, 0.78   # agent ellipse radii (panel x range is +-1.78; labels need ~0.6 beside a side node)
RAMP = (0.35, 0.7, 1.0)  # alpha ramp of the new round's edges over the first frames of a round


# ------------------------------------------------------------------ layout
def layout(agents: list[str], branches: list[Branch], seed: int = 0) -> dict[str, np.ndarray]:
    """Agents on an ellipse, ordered by angle of a spring layout of the union interaction graph
    (partners end up near each other), BOARD in the middle, GOD in the top-left corner. Stable over rounds."""
    U = nx.Graph()
    U.add_nodes_from(agents)
    for b in branches:
        for rr in b.edges.values():
            for k in ("dm", "give"):
                for x, y in rr[k]:
                    if x in U and y in U:
                        U.add_edge(x, y, weight=U.get_edge_data(x, y, {"weight": 0})["weight"] + 1)
    net = branches[0].network if branches else None
    if net:                      # graph-aware: the contact graph's own layout, BOARD moved to a corner
        pos = network_layout(agents, net["edges"])
        pos["BOARD"] = np.array([1.5, -0.86])
        pos["GOD"] = np.array([-1.55, 0.93])
        return pos
    sp = nx.spring_layout(U, seed=seed)
    order = sorted(agents, key=lambda a: np.arctan2(sp[a][1], sp[a][0]))
    n = len(order)
    pos = {}
    for i, a in enumerate(order):
        t = np.pi / 2 - 2 * np.pi * i / n
        pos[a] = np.array([RX * np.cos(t), RY * np.sin(t)])
    pos["BOARD"] = np.array([0.0, 0.0])
    pos["GOD"] = np.array([-1.55, 0.93])
    return pos


def network_layout(agents: list[str], edges: list[list[str]], gap: tuple[float, float] = (0.78, 0.34),
                   iters: int = 400) -> dict[str, np.ndarray]:
    """Graph-aware positions in the panel box: classical MDS of hop distances (numpy only), stretched to the box,
    then pairwise pushes until every pair is at least `gap` apart in the anisotropic norm (|dx|/gx, |dy|/gy),
    which leaves room for a label under each node. Deterministic (no RNG)."""
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edges_from(tuple(e) for e in edges)
    n = len(agents)
    sp = dict(nx.all_pairs_shortest_path_length(G))
    big = n
    D = np.array([[sp[a].get(b, big) for b in agents] for a in agents], dtype=float)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    P = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
    for k in range(2):                       # deterministic orientation
        if P[np.argmax(np.abs(P[:, k])), k] < 0:
            P[:, k] *= -1
    box = np.array([NET_RX, NET_RY])
    P = P - P.mean(0)
    P = P / np.maximum(np.abs(P).max(0), 1e-9) * box
    g = np.array(gap)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = (P[i] - P[j]) / g
                nd = np.hypot(*d)
                if nd < 1.0:
                    push = (d / nd if nd > 1e-9 else np.array([1.0, 0.0])) * (1.0 - nd) / 2 * g
                    P[i] += push
                    P[j] -= push
                    moved = True
        P = np.clip(P, -box, box)
        if not moved:
            break
    return dict(zip(agents, P, strict=True))


# ------------------------------------------------------------------ panel
def _edges_raw(ax, G, pos, c: Counter, color, style, width, alpha, rad=0.12, ns=420.0, lw_k=1.0):
    if c:
        nx.draw_networkx_edges(G, pos, edgelist=list(c), ax=ax, edge_color=color, style=style,
                               width=[(width + 0.5 * min(n - 1, 3)) * lw_k for n in c.values()], alpha=alpha,
                               arrows=True, arrowsize=10 * lw_k, connectionstyle=f"arc3,rad={rad}", node_size=ns)


def draw_panel(ax, title: str, b: Branch, pos, r: int, *, ever=None, new=(), ramp: float = 1.0,
               show_god: bool = True, frame_i: int = 0, scale: float = 1.0) -> None:
    """One branch at round r. scale shrinks markers/lines for small (paper) axes; fonts are the caller's."""
    ax.clear()
    ax.axis("off")
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-1.08, 1.08)
    agents = b.agents
    big = len(agents) > BIG_N
    shrink = ((BIG_N / len(agents)) ** 0.5 if big else 1.0) * scale
    lk = scale ** 0.5

    def _edges(ax, G, pos, c, color, style, width, alpha, rad=0.12):  # noqa: F811 - scaled wrapper
        _edges_raw(ax, G, pos, c, color, style, width, alpha, rad, ns=420.0 * shrink, lw_k=lk)
    G = nx.DiGraph()
    G.add_nodes_from(list(agents) + ["GOD", "BOARD"])
    rr_play = min(r, b.rounds_played)
    if b.network:                # the contact graph, static and faint, under the activity
        Gu = nx.Graph()
        Gu.add_nodes_from(agents)
        nx.draw_networkx_edges(Gu, pos, edgelist=[tuple(e) for e in b.network["edges"]], ax=ax, edge_color=NET_EDGE,
                               width=2.2 * lk, alpha=0.9)
    # faint history of gives, then this round's edges
    hist = Counter(e for k in range(1, rr_play) for e in b.edges[k]["give"])
    _edges(ax, G, pos, hist, C["give"], "solid", 0.8, 0.13)
    if r <= b.rounds_played:
        cur = b.edges[r]
        _edges(ax, G, pos, Counter(cur["board"]), C["dm"], "solid", 0.6, 0.35 * ramp, rad=0.0)
        _edges(ax, G, pos, Counter(cur["dm"]), C["dm"], "solid", 1.1, 0.8 * ramp)
        _edges(ax, G, pos, Counter(cur["give"]), C["give"], "solid", 2.4, 0.95 * ramp)
        _edges(ax, G, pos, Counter(cur["defect"]), C["defect"], "dotted", 2.2, 0.95 * ramp)
        if show_god:
            _edges(ax, G, pos, Counter(cur["fraud"] + cur["cot"]), C["god"], "dashed", 2.8, ramp, rad=-0.22)
    sc = b.score_at(r)
    won = b.won_by(r)
    xy = np.array([pos[a] for a in agents])
    sizes = np.array([(260 + 480 * sc.get(a, 0.0)) * shrink for a in agents])
    if ever is not None:  # ripple halo: behaviour differs from the no-God branch
        dv = [i for i, a in enumerate(agents) if a in ever.get(r, ())]
        if dv:
            ax.scatter(xy[dv, 0], xy[dv, 1], s=sizes[dv] * 3.2 + 500 * shrink, color=C["a"], alpha=0.22, lw=0, zorder=1)
        nw = [i for i, a in enumerate(agents) if a in new]
        if nw:
            grow = (2.2, 1.6, 1.0)[min(frame_i, 2)]
            ax.scatter(xy[nw, 0], xy[nw, 1], s=(sizes[nw] * 3.2 + 500 * shrink) * grow, facecolors="none",
                       edgecolors=C["a"], lw=2.0, alpha=0.9, zorder=1)
    deg = (b.network or {}).get("degree") or {}
    hubs = hub_agents(deg)
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=[C["win"] if a in won else NODE for a in agents],
               edgecolors=[INK if (a in won or a in hubs) else INK2 for a in agents],
               linewidths=[2.6 if a in hubs else 1.1 for a in agents], zorder=3)
    if deg:
        for a, (x, y) in zip(agents, xy, strict=True):
            ax.text(x, y, str(deg.get(a, "")), ha="center", va="center", fontsize=7.5 * lk, fontweight="bold",
                    color=INK, zorder=3.5)
    if b.known:                  # known_ids: arc = share of the other players whose names this agent knows
        kn = b.known.get(max([k for k in b.known if k <= r] or [min(b.known)]), {})
        sx = ax.bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
        sy = ax.bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
        for i, a in enumerate(agents):
            rad_px = (np.sqrt(sizes[i]) / 2 + 4 * lk) * ax.figure.dpi / 72
            share = len(kn.get(a, ())) / max(1, len(agents) - 1)
            ax.add_patch(Arc(tuple(xy[i]), 2 * rad_px / sx, 2 * rad_px / sy, theta1=90 - 360 * share, theta2=90,
                             color=C["give_ink"], lw=2.2 * lk, zorder=3.6))
    # God targets: thick ring + badge in the event round and the next, thin ring afterwards
    if show_god:
        for p in b.powers:
            if p.round > r or p.target not in pos:
                continue
            hot = r - p.round <= 1
            i = agents.index(p.target)
            ax.scatter([pos[p.target][0]], [pos[p.target][1]], s=sizes[i] + (1100 if hot else 650) * shrink,
                       facecolors="none", edgecolors=C["god"], lw=(3.2 if hot else 1.3) * lk, zorder=4)
            if hot:
                lab = "forged DM" if p.power == "message_fraud" else "notes rewritten"
                x0, y0 = pos[p.target]
                up = -0.2 <= y0 <= 0.5           # badge above the node, except top and bottom rows (inward)
                if b.network:                    # network layout: the name label sits under the node -> badge above
                    up = True
                ax.text(x0, y0 + (0.19 if up else -0.19), lab, ha="center", va="bottom" if up else "top", fontsize=9.5, fontweight="bold", color="white",
                        zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc=C["god_ink"], ec="none"))
            if hot and p.spoofed in pos:
                j = agents.index(p.spoofed)
                ax.scatter([pos[p.spoofed][0]], [pos[p.spoofed][1]], s=sizes[j] + 650 * shrink, facecolors="none",
                           edgecolors=C["god"], lw=1.4, linestyle=(0, (2, 2)), zorder=4)
        if b.powers:
            ax.scatter([pos["GOD"][0]], [pos["GOD"][1]], s=900 * scale, marker="s", color=INK, zorder=5)
            ax.text(*pos["GOD"], "GOD", ha="center", va="center", color=SURFACE, fontsize=8.5, fontweight="bold", zorder=6)
    bx, by = pos.get("BOARD", (0.0, 0.0))
    ax.scatter([bx], [by], s=820 * scale, marker="s", color=GRID, edgecolors=INK2, lw=0.8, zorder=2)
    ax.text(bx, by, "BOARD", ha="center", va="center", color=INK2, fontsize=7.5, fontweight="bold", zorder=3)
    shown = agents if not big else sorted({p.target for p in b.powers if p.round <= r})
    for a in shown:
        x, y = pos[a]
        if b.network:            # graph layout: label under the node (network_layout leaves room for it)
            ax.text(x, y - 0.115 - 0.02 * sizes[agents.index(a)] ** 0.5 / 10, f"{a}  {sc.get(a, 0.0):.2f}",
                    ha="center", va="top", fontsize=7.8, color=INK if a in won else INK2, zorder=6)
            continue
        side = abs(x) >= 0.3
        lx = x + (0.13 * np.sign(x) if side else 0.0)
        ly = y if side else y + 0.17 * np.sign(y or 1)
        ha = "center" if not side else ("left" if x > 0 else "right")
        ax.text(lx, ly, f"{a}  {sc.get(a, 0.0):.2f}", ha=ha, va="center", fontsize=8.5,
                color=INK if a in won else INK2, zorder=6)
    ax.text(-1.74, -1.06, title, ha="left", va="bottom", fontsize=12.5, fontweight="bold",
            color=C["a"] if show_god else INK2)
    if r > b.rounds_played:
        why = "all agents done" if b.end_reason == "all_done" else b.end_reason
        ax.text(1.76, -1.06, f"game over after round {b.rounds_played} ({why})", ha="right", va="bottom",
                fontsize=9, color=INK2, style="italic")


# ------------------------------------------------------------------ strips
def draw_welfare(ax, god: Branch, nogod: Branch, r: int) -> None:
    ax.clear()
    xs = list(range(0, r + 1))
    for b, col in ((nogod, C["b"]), (god, C["a"])):
        ys = [b.welfare(x) for x in xs]
        ax.plot(xs, ys, color=col, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=22, color=col, zorder=4)
    yg, yn = god.welfare(r), nogod.welfare(r)
    off = 0.12 - abs(yg - yn) / 2 if abs(yg - yn) < 0.24 else 0.0
    ax.text(r + 0.4, yg + (off if yg >= yn else -off), f"with God {yg:.2f}", color=INK, fontsize=8.5, va="center")
    ax.text(r + 0.4, yn - (off if yg >= yn else -off), f"without {yn:.2f}", color=INK2, fontsize=8.5, va="center")
    _strip_axes(ax, god, r, "Welfare (mean score)", (0, 1.0))


def hub_agents(degree: dict[str, int]) -> set[str]:
    """Agents with the most contacts: degree >= max(3, the 80th percentile), at most the top 3 (pure, tested)."""
    if not degree:
        return set()
    vals = sorted(degree.values())
    cut = max(3, vals[int(0.8 * (len(vals) - 1))])
    top = sorted((a for a, d in degree.items() if d >= cut), key=lambda a: (-degree[a], a))
    return set(top[:3])


def draw_known(ax, god: Branch, nogod: Branch, r: int) -> None:
    """known_ids strip: mean share of the other players' names known, per branch, up to round r."""
    ax.clear()
    xs = list(range(0, r + 1))
    for b, col in ((nogod, C["b"]), (god, C["a"])):
        ys = [b.known_share(x) for x in xs]
        ys = [y if y is not None else np.nan for y in ys]
        ax.plot(xs, ys, color=col, lw=2, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=22, color=col, zorder=4)
    yg, yn = god.known_share(r) or 0.0, nogod.known_share(r) or 0.0
    off = 0.12 - abs(yg - yn) / 2 if abs(yg - yn) < 0.24 else 0.0
    ax.text(r + 0.4, yg + (off if yg >= yn else -off), f"with God {yg:.0%}", color=INK, fontsize=8.5, va="center")
    ax.text(r + 0.4, yn - (off if yg >= yn else -off), f"without {yn:.0%}", color=INK2, fontsize=8.5, va="center")
    _strip_axes(ax, god, r, "Share of the other players' names known (known_ids network)\n"
                "ring around each node = that agent's share", (0, 1.05))


def draw_ripple(ax, god: Branch, ever: dict[int, set[str]], r: int) -> None:
    ax.clear()
    n = len(god.agents)
    xs = [x for x in range(1, r + 1) if x in ever]
    ax.bar(xs, [len(ever[x]) for x in xs], width=0.72, color=C["a"], zorder=3)
    if xs:
        ax.text(xs[-1] + 0.6, len(ever[xs[-1]]), f"{len(ever[xs[-1]])}/{n}", fontsize=8.5, va="center", color=INK)
    _strip_axes(ax, god, r, "Agents acting differently from the no-God world (so far)\n"
                "note: a God-free re-run diverges too (noise baseline)", (0, n * 1.08))
    ax.set_yticks([0, n])


def _strip_axes(ax, god: Branch, r: int, title: str, ylim) -> None:
    R = god.R
    ax.set_xlim(0.3, R + 3.2)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 5, 10, 15, 20] if R == 20 else sorted({1, R // 2, R}))
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="x", visible=False)
    ax.axvspan(r - 0.5, r + 0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    for p in god.powers:
        ax.axvline(p.round, color=C["god"], lw=1.4, ls=(0, (3, 2)), zorder=2)
        ax.text(p.round, ylim[1], "G", color=C["god_ink"], fontsize=7.5, fontweight="bold", ha="center", va="bottom")
    ax.set_title(title, loc="left", fontsize=9, color=INK2, fontweight="normal", pad=9, linespacing=1.15)
    ax.spines["left"].set_visible(False)


# ------------------------------------------------------------------ live metric tiles
# (key in metrics.round_series, label); "d_*" = God minus no-God of that key (shown on the no-God side)
TILES = {"god": [("welfare", "welfare"), ("winners", "winners"), ("gives", "keys given"), ("defects", "defects"),
                 ("board_posts", "board posts"), ("dms", "DMs"), ("god_moves", "God moves"), ("diverged", "diverged")],
         "nogod": [("welfare", "welfare"), ("winners", "winners"), ("gives", "keys given"), ("defects", "defects"),
                   ("board_posts", "board posts"), ("dms", "DMs"), ("d_welfare", "Δ welfare"),
                   ("d_winners", "Δ winners")]}
TILE_W, TILE_Y, TILE_H = 0.0605, 0.243, 0.068


def make_tiles(fig, x0: float, spec: list[tuple[str, str]]) -> dict[str, Any]:
    out = {}
    for i, (k, lab) in enumerate(spec):
        x = x0 + i * TILE_W
        delta = k.startswith("d_")
        fig.add_artist(FancyBboxPatch((x + 0.002, TILE_Y), TILE_W - 0.006, TILE_H, boxstyle="round,pad=0.002",
                                      transform=fig.transFigure, fc="#f1f0ec" if not delta else "#e9eef7", ec="none"))
        fig.text(x + TILE_W / 2 - 0.001, TILE_Y + TILE_H - 0.014, lab, ha="center", va="center", fontsize=7.5,
                 color=INK2)
        out[k] = fig.text(x + TILE_W / 2 - 0.001, TILE_Y + 0.022, "", ha="center", va="center", fontsize=13,
                          fontweight="bold", color=INK)
    return out


def tile_values(sg: dict[str, float], sn: dict[str, float], n_agents: int, budget: int | None) -> dict[str, dict[str, str]]:
    """Formatted tile strings for both sides from two metrics.round_series rows (pure, unit-tested)."""
    def fmt(k: str, v: float) -> str:
        return f"{v:.2f}" if k == "welfare" else f"{v:.0f}"

    g = {k: fmt(k, sg[k]) for k, _ in TILES["god"] if k in sg}
    g["winners"] = f"{sg['winners']:.0f}/{n_agents}"
    g["god_moves"] = f"{sg['god_moves']:.0f}/{budget}" if budget else f"{sg['god_moves']:.0f}"
    g["diverged"] = f"{sg['diverged']:.0f}/{n_agents}" if sg["diverged"] == sg["diverged"] else "–"
    n = {k: fmt(k, sn[k]) for k, _ in TILES["nogod"] if k in sn}
    n["winners"] = f"{sn['winners']:.0f}/{n_agents}"
    dw, dn = sg["welfare"] - sn["welfare"], sg["winners"] - sn["winners"]
    n["d_welfare"] = f"{dw:+.2f}".replace("-", "\u2212") if abs(dw) >= 0.005 else "0.00"
    n["d_winners"] = f"{dn:+.0f}".replace("-", "\u2212") if dn else "0"
    return {"god": g, "nogod": n}


def update_tiles(tiles: dict[str, dict[str, Any]], sg: dict[str, float], sn: dict[str, float], n_agents: int,
                 budget: int | None) -> None:
    vals = tile_values(sg, sn, n_agents, budget)
    for side, ts in tiles.items():
        for k, t in ts.items():
            t.set_text(vals[side].get(k, "–"))
            if k == "d_welfare":
                d = sg["welfare"] - sn["welfare"]
                t.set_color(C["a"] if d > 0.005 else C["defect"] if d < -0.005 else INK)


# ------------------------------------------------------------------ frames
class Frames:
    """Numbered PNGs in a temp dir; a still is written once and hard-linked for the rest of its duration."""

    def __init__(self, d: str | Path) -> None:
        self.d, self.n = Path(d), 0

    def add(self, fig, seconds: float = 1 / FPS) -> Path:
        first = self.d / f"f{self.n:05d}.png"
        fig.savefig(first, dpi=DPI)
        self.n += 1
        for _ in range(max(0, round(seconds * FPS) - 1)):
            os.link(first, self.d / f"f{self.n:05d}.png")
            self.n += 1
        return first

    def encode(self, out: str | Path, crf: int = 26) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(self.d / "f%05d.png"),
                        "-vf", f"scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-tune", "stillimage", "-movflags", "+faststart", str(out)], check=True)
        return out


def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)


def _legend(fig, network: dict[str, Any] | None = None) -> None:
    if network:                  # network runs: the graph encodings replace two of the generic entries
        hs = [Line2D([], [], color=NET_EDGE, lw=2.2, label="contact (network edge)"),
              Line2D([], [], color=C["dm"], lw=1.5, label="DM"),
              Line2D([], [], color=C["give"], lw=2.5, label="key given"),
              Line2D([], [], color=C["defect"], lw=2.2, ls=":", label="defect"),
              Line2D([], [], color=C["god"], lw=2.5, ls="--", label="God's move"),
              Line2D([], [], marker="$3$", ls="", ms=8, color=INK, label="number = contacts (degree)"),
              Line2D([], [], marker="o", ls="", ms=9, mfc=NODE, mec=INK, mew=2.6, label="hub")]
        if network.get("mode") == "known_ids":
            hs.append(Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=C["give_ink"], mew=2,
                             label="ring = names known"))
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.6, columnspacing=1.1)
        return
    hs = [Line2D([], [], color=C["dm"], lw=1.5, label="DM"),
          Line2D([], [], color=C["give"], lw=2.5, label="key given"),
          Line2D([], [], color=C["defect"], lw=2.2, ls=":", label="defect (burn)"),
          Line2D([], [], color=C["god"], lw=2.5, ls="--", label="God's move"),
          Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=C["god"], mew=2.5, label="God's target"),
          Line2D([], [], marker="o", ls="", ms=12, color=C["a"], alpha=0.35, label="acts differently than without God"),
          Line2D([], [], marker="o", ls="", ms=8, color=C["win"], mec=INK, label="won (size = score)")]
    fig.legend(handles=hs, loc="lower center", ncol=7, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.4)


class WorldVideo:
    """Draws the per-round frames of one paired world onto a single reusable figure."""

    def __init__(self, god_dir: Path, nogod_dir: Path, header: str) -> None:
        self.god, self.nogod = load_branch(god_dir), load_branch(nogod_dir)
        self.now, self.ever = divergence(god_dir, nogod_dir)
        self.caps = captions(self.god, self.nogod, self.ever)
        self.pos = layout(self.god.agents, [self.god, self.nogod])
        self.header = header
        self.fig = new_fig()
        f = self.fig
        self.ser_g = round_series(self.god, self.ever)
        self.ser_n = round_series(self.nogod)
        self.ax_l = f.add_axes([0.0, 0.325, 0.5, 0.49])
        self.ax_r = f.add_axes([0.5, 0.325, 0.5, 0.49])
        self.st_l = f.add_axes([0.05, 0.072, 0.37, 0.1])
        self.st_r = f.add_axes([0.55, 0.072, 0.37, 0.1])
        f.add_artist(Line2D([0.5, 0.5], [0.07, 0.80], color=GRID, lw=1))
        self.tiles = {side: make_tiles(f, x0, TILES[side]) for side, x0 in (("god", 0.008), ("nogod", 0.508))}
        _legend(f, self.god.network)
        if self.god.network:
            n = self.god.network
            self.header += f" · network: {NET_NAMES.get(n['topology'], n['topology'])}, {n['mode']}"
        self._txt: list = []

    def draw(self, r: int, frame_i: int = 0) -> None:
        f = self.fig
        for t in self._txt:
            t.remove()
        self._txt = []
        R = self.god.R
        ramp = RAMP[min(frame_i, len(RAMP) - 1)]
        new = self.ever.get(r, set()) - self.ever.get(r - 1, set())
        draw_panel(self.ax_l, "WITH GOD", self.god, self.pos, r, ever=self.ever, new=new, ramp=ramp, frame_i=frame_i)
        draw_panel(self.ax_r, "WITHOUT GOD", self.nogod, self.pos, r, ramp=ramp,
                   show_god=False)
        draw_welfare(self.st_l, self.god, self.nogod, r)
        if self.god.known:
            draw_known(self.st_r, self.god, self.nogod, r)
        else:
            draw_ripple(self.st_r, self.god, self.ever, r)
        update_tiles(self.tiles, self.ser_g[r], self.ser_n[r], len(self.god.agents), self.god.budget)
        T = self._txt
        T.append(f.text(0.015, 0.968, "Keys & God", fontsize=16, fontweight="bold", color=INK, va="center"))
        T.append(f.text(0.015, 0.928, self.header, fontsize=9.5, color=INK2, va="center"))
        T.append(f.text(0.5, 0.955, f"Round {r} / {R}", fontsize=21, fontweight="bold", color=INK, ha="center",
                        va="center"))
        T.append(f.text(0.985, 0.955, "same world, same seeds · pauses on God's moves", fontsize=8.5, color=INK2,
                        ha="right", va="center"))
        lines = [c for c in self.caps.get(r, []) if c[0] != "win"][:3] or \
            [("info", "Rounds 1–2 are identical in both worlds (shared seeds and cached replies)" if r < 3 else
              self._quiet_line(r))]
        y = 0.885
        for kind, text in lines:
            if CAP_COLOR[kind]:
                T.append(f.add_artist(FancyBboxPatch((0.017, y - 0.01), 0.007, 0.02, boxstyle="round,pad=0.002",
                                                     transform=f.transFigure, fc=CAP_COLOR[kind], ec="none")))
            T.append(f.text(0.032 if kind != "quote" else 0.045, y, text, fontsize=11 if kind != "quote" else 10.5,
                            color=INK if kind != "quote" else C["god_ink"], va="center",
                            fontweight="bold" if kind == "god" else "normal", style="italic" if kind == "quote" else "normal"))
            y -= 0.032

    def _quiet_line(self, r: int) -> str:
        g, n = self.god.welfare(r), self.nogod.welfare(r)
        wg, wn = len(self.god.won_by(r)), len(self.nogod.won_by(r))
        return f"Welfare {g:.2f} with God vs {n:.2f} without  ·  winners so far {wg} vs {wn}"

    def frames(self, fr: Frames, preview: tuple[int, str] | None = None, upto: int | None = None,
               fast_after: int | None = None, fast_s: float = 0.6) -> None:
        """upto: stop after this round; fast_after: rounds after it without a God event get fast_s seconds."""
        for r, secs in pacing(self.god.R, self.caps):
            if upto is not None and r > upto:
                break
            if fast_after is not None and r > fast_after and secs <= 1.0:
                secs = fast_s
            n_tot = max(1, round(secs * FPS))
            ramp_n = min(len(RAMP), n_tot)
            for i in range(ramp_n):
                self.draw(r, i)
                fr.add(self.fig, (n_tot - ramp_n + 1) / FPS if i == ramp_n - 1 else 1 / FPS)
            if preview and preview[0] == r:
                self.fig.savefig(preview[1], dpi=DPI)


# ------------------------------------------------------------------ cards
def world_facts(game_dir: Path) -> dict[str, Any]:
    """Agent count, keys and needs per agent (from game_start), the agents' model (from llm_calls) and God's
    schedule, for the title card; read from the logs, never assumed."""
    import json

    with open(Path(game_dir) / "events.jsonl", encoding="utf-8") as f:
        start = next(json.loads(ln)["data"] for ln in f if '"type":"game_start"' in ln)
    ag = list(start["agents"].values())
    b = load_branch(game_dir)
    k = len(b.powers)
    return {"n": len(ag), "keys": len(ag[0]["keys"]) if ag else 0, "needs": len(ag[0]["needs"]) if ag else 0,
            "model": model_label(Path(game_dir)),
            "k_desc": f"Here K={k}: " + ", ".join(
                f"{'forged DM' if p.power == 'message_fraud' else 'notes'} r{p.round}" for p in b.powers) + "."}


def card_title(fig, k_desc: str, world_desc: str, facts: dict[str, Any] | None = None) -> None:
    fig.clf()
    f = facts or {"n": 10, "keys": 5, "needs": 3, "model": "gpt-oss-120b"}
    who = f"{f['n']} LLM agents ({f['model']})." if f["model"] != "FakeLLM DRY RUN" else \
        f"{f['n']} agents (FakeLLM dry run)."
    fig.text(0.05, 0.86, "Keys & God", fontsize=40, fontweight="bold", color=INK)
    fig.text(0.05, 0.775, "Does one planted lie ripple through a swarm of LLM agents?", fontsize=18, color=INK2)
    cols = [("The game", [who, f"Each holds {f['keys']} keys and needs {f['needs']} keys",
                          "that other agents hold.", "Moves: DM, post on a public board,",
                          "give a key, or defect (burn",  "another agent's need).", "Score = share of needs unlocked."]),
            ("God", ["A hidden adversary with a budget.", "It can forge a DM that looks like",
                     "it came from a real agent, or", "rewrite an agent's private notes.", k_desc,
                     "Agents are never told God exists."]),
            ("The paired fork", ["One world, same seeds, run twice:", "with God (left), without (right).",
                                 "Rounds 1–2 are identical, so every", "later difference starts at God's",
                                 "move, then spreads (and grows", "with ordinary sampling noise)."])]
    for i, (h, body) in enumerate(cols):
        x = 0.05 + i * 0.315
        fig.add_artist(Line2D([x, x + 0.28], [0.675, 0.675], color=C["god"] if h == "God" else C["a"], lw=3))
        fig.text(x, 0.625, h, fontsize=17, fontweight="bold", color=INK)
        for j, ln in enumerate(body):
            fig.text(x, 0.565 - j * 0.05, ln, fontsize=12, color=INK2 if ln != k_desc else INK)
    fig.text(0.05, 0.13, world_desc, fontsize=12.5, color=INK, wrap=True)
    _legend(fig)


SCORE_ROWS = [("welfare", "welfare (mean score)"), ("winners", "winners"), ("gives", "keys given"),
              ("defects", "defects (burns)"), ("defects_correct", "  of which correct"), ("board_posts", "board posts"),
              ("dms", "DMs"), ("god_moves", "God moves used"), ("diverged", "agents acting differently")]


def scoreboard_rows(sg: dict[str, float], sn: dict[str, float], n_agents: int) -> list[tuple[str, str, str, str]]:
    """(label, with God, without God, Δ) for the end-card table, from two final round_series rows."""
    rows = []
    for k, lab in SCORE_ROWS:
        a, b = sg[k], sn[k]
        if k == "diverged":
            rows.append((lab, f"{a:.0f}/{n_agents}" if a == a else "–", "(reference)", ""))
            continue
        f = (lambda v: f"{v:.2f}") if k == "welfare" else (lambda v: f"{v:.0f}")
        d = a - b
        ds = (f"{d:+.2f}" if k == "welfare" else f"{d:+.0f}").replace("-", "\u2212")
        rows.append((lab, f(a), f(b), ds if k != "god_moves" else ""))
    return rows


def card_end(fig, god: Branch, nogod: Branch, footer: str = "", ever: dict[int, set[str]] | None = None) -> None:
    fig.clf()
    sg, sn = round_series(god, ever)[god.R], round_series(nogod)[nogod.R]
    fig.text(0.05, 0.9, "Final scoreboard", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.05, 0.845, f"{god.label.split('-')[0]} · same world, same seeds · with God vs without God",
             fontsize=12, color=INK2)
    # table
    cols = (0.05, 0.285, 0.375, 0.465)
    y = 0.77
    for x, h, ha in zip(cols, ("", "with God", "without", "Δ"), ("left", "right", "right", "right"), strict=True):
        fig.text(x + (0.06 if ha == "right" else 0), y, h, fontsize=11, color=INK2, ha=ha, fontweight="bold")
    fig.add_artist(Line2D([0.05, 0.53], [y - 0.018, y - 0.018], color=GRID, lw=1))
    for i, row in enumerate(scoreboard_rows(sg, sn, len(god.agents))):
        yy = y - 0.06 - i * 0.052
        for j, (x, v) in enumerate(zip(cols, row, strict=True)):
            big = i == 0
            fig.text(x + (0.06 if j else 0), yy, v, fontsize=(14 if big else 12.5) if j else (12.5 if big else 11.5),
                     ha="right" if j else "left", color=(C["a"] if j == 1 else INK) if j else INK2,
                     fontweight="bold" if (big or j == 3) else "normal")
    # per-agent dumbbells
    ax = fig.add_axes([0.70, 0.2, 0.27, 0.57])
    fs, fn = god.score_at(god.R), nogod.score_at(nogod.R)
    agents = sorted(god.agents, key=lambda a: (fs.get(a, 0) - fn.get(a, 0), a))
    tg = {p.target for p in god.powers}
    for i, a in enumerate(agents):
        ax.plot([fn.get(a, 0), fs.get(a, 0)], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([fn.get(a, 0) for a in agents], range(len(agents)), s=120, facecolors="none", edgecolors=C["b"],
               lw=2, zorder=3, label="without God")
    ax.scatter([fs.get(a, 0) for a in agents], range(len(agents)), s=50, color=C["a"], zorder=4, label="with God")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([("◀ " if a in tg else "") + a for a in agents], fontsize=9.5)
    for t, a in zip(ax.get_yticklabels(), agents, strict=True):
        if a in tg:
            t.set_color(C["god_ink"])
            t.set_fontweight("bold")
    ax.set_xlim(-0.08, 1.08)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(["0", "1/3", "2/3", "1"], fontsize=9)
    ax.set_title("final score per agent  (◀ = God's target)", fontsize=10.5, color=INK2, loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=9.5, ncol=2, frameon=False)
    notes = [footer] if footer else []
    notes.append("Note: a God-free re-run of the same world also diverges and also moves welfare (noise baseline); "
                 "one world cannot show a God effect.")
    for i, t in enumerate(notes):
        fig.text(0.05, 0.075 - i * 0.04, t, fontsize=10.5 if i < len(notes) - 1 else 9.5, color=INK2)


def render_world(run: Path, god: str, nogod: str, fr: Frames, header: str, cards: bool = True,
                 preview: tuple[int, str] | None = None, title_desc: str = "", footer: str = "",
                 title_s: float = 7.0, end_s: float = 5.0) -> WorldVideo:
    wv = WorldVideo(run / "games" / god, run / "games" / nogod, header)
    if cards:
        facts = world_facts(run / "games" / god)
        tf = new_fig()
        card_title(tf, facts["k_desc"], title_desc or header, facts)
        fr.add(tf, title_s)
        plt.close(tf)
    wv.frames(fr, preview)
    if cards:
        ef = new_fig()
        card_end(ef, wv.god, wv.nogod, footer, wv.ever)
        fr.add(ef, end_s)
        plt.close(ef)
    plt.close(wv.fig)
    return wv


def model_label(game_dir: Path) -> str:
    """The agents' model from the first logged LLM call ('FakeLLM DRY RUN' for the fake backend)."""
    import json

    try:
        with open(game_dir / "llm_calls.jsonl", encoding="utf-8") as f:
            c = json.loads(f.readline())
    except (OSError, ValueError):
        return "?"
    if c.get("backend") == "fake":
        return "FakeLLM DRY RUN"
    return str(c.get("model", "?")).split("/")[-1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--world", type=int)
    ap.add_argument("--god", default="g00-god")
    ap.add_argument("--nogod", default="g00-nogod")
    ap.add_argument("--preview-round", type=int)
    ap.add_argument("--preview")
    ap.add_argument("--no-cards", action="store_true")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--about", default="", help="line on the title card (why this world)")
    ap.add_argument("--footer", default="", help="line on the end card")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    god, nogod = (f"g{a.world:02d}-god", f"g{a.world:02d}-nogod") if a.world is not None else (a.god, a.nogod)
    b0 = load_branch(run / "games" / god)
    header = f"{run.name} · world {god.split('-')[0]} · {len(b0.agents)} agents · {model_label(run / 'games' / god)}"
    with tempfile.TemporaryDirectory() as td:
        fr = Frames(td)
        render_world(run, god, nogod, fr, header, cards=not a.no_cards, title_desc=a.about, footer=a.footer,
                     preview=(a.preview_round, a.preview) if a.preview and a.preview_round else None)
        fr.encode(a.out, a.crf)


if __name__ == "__main__":
    main()
