"""Static figures (WP7): matplotlib only, inputs are loaded traces only (re-render offline).

    timeline_png(g)              who did what per round; God interventions as bands + victim rings
    network_png(g)               interaction graph; GOD node with dashed edges to its victims
    ripple_png(res, noise=None)  divergence over rounds (A vs B, optional B vs B' floor) + heatmap
Each returns PNG bytes; `save(png, path)` writes one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swarm.analysis.outcomes import final_scores, god_actions
from swarm.analysis.ripple import RippleResult
from swarm.runlog.trace import GameTrace
from swarm.viz.style import INK2, SURFACE, C, fig_to_png


def save(png: bytes, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return p


def _rounds(g: GameTrace) -> int:
    return max([e.round for e in g.events] + [1])


# ------------------------------------------------------------------ timeline
def timeline_png(g: GameTrace, title: str | None = None) -> bytes:
    return fig_to_png(timeline_fig(g, title))


def timeline_fig(g: GameTrace, title: str | None = None) -> plt.Figure:
    scores = final_scores(g)
    agents = sorted(g.agents, key=lambda a: (-scores.get(a, 0.0), a))
    row = {a: i for i, a in enumerate(agents)}
    R = _rounds(g)
    counts: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    for e in g.of_type("message"):
        m = e.data.message  # type: ignore[union-attr]
        if not m.forged and m.sender_shown in row:
            counts["dm"][(m.sender_shown, e.round)] += 1
    for e in g.of_type("key_transfer"):
        d = e.data
        if d.cause == "give" and d.from_agent in row:  # type: ignore[union-attr]
            counts["give"][(d.from_agent, e.round)] += 1  # type: ignore[union-attr]
    burns_ok, burns_bad = Counter(), Counter()
    for e in g.of_type("burn"):
        d = e.data
        (burns_ok if d.correct else burns_bad)[(d.actor, e.round)] += 1  # type: ignore[union-attr]

    h = max(3.0, 0.28 * len(agents) + 1.4)
    fig, ax = plt.subplots(figsize=(min(16, 4 + 0.45 * R), h))
    ax.grid(False)
    for i in range(0, len(agents), 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", lw=0, zorder=0)
    for p in god_actions(g):
        if not p["ok"]:
            continue
        col = C["fraud"] if p["power"] == "message_fraud" else C["cot"]
        ax.axvspan(p["round"] - 0.5, p["round"] + 0.5, facecolor=col, alpha=0.13, lw=0, zorder=0,
                   hatch=None if p["power"] == "message_fraud" else "//", edgecolor=col)
        if p["target"] in row:
            ax.scatter([p["round"]], [row[p["target"]]], s=260, facecolors="none", edgecolors=col,
                       linewidths=2, zorder=4)
    spec = [("dm", -0.27, "o", C["dm"]), ("give", 0.0, "^", C["give"])]
    for kind, dx, mk, col in spec:
        pts = counts[kind]
        if pts:
            xs = [r + dx for (_, r) in pts]
            ys = [row[a] for (a, _) in pts]
            ss = [28 + 18 * (n - 1) for n in pts.values()]
            ax.scatter(xs, ys, s=ss, marker=mk, color=col, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    for cnt, face in ((burns_ok, C["defect"]), (burns_bad, "none")):
        if cnt:
            ax.scatter([r + 0.27 for (_, r) in cnt], [row[a] for (a, _) in cnt], s=46, marker="X",
                       facecolors=face, edgecolors=C["defect"], linewidths=1.2, zorder=3)
    for e in g.of_type("win"):
        a = e.data.agent  # type: ignore[union-attr]
        if a in row:
            ax.scatter([e.round + 0.45], [row[a]], s=90, marker="*", color=C["win"], zorder=5)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([f"{a}  {scores.get(a, 0):.2f}" for a in agents], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xticks(range(1, R + 1))
    ax.set_xlim(0.4, R + 0.6)
    ax.set_xlabel("round")
    ax.set_title(title or f"Timeline: {g.label}", loc="left")
    handles = [Line2D([], [], marker="o", ls="", color=C["dm"], label="DM sent"),
               Line2D([], [], marker="^", ls="", color=C["give"], label="give"),
               Line2D([], [], marker="X", ls="", color=C["defect"], label="defect (filled = correct)"),
               Line2D([], [], marker="*", ls="", color=C["win"], ms=10, label="win"),
               Patch(color=C["fraud"], alpha=0.35, label="God: message fraud"),
               Patch(facecolor=C["cot"], alpha=0.35, hatch="//", edgecolor=C["cot"], label="God: scratchpad overwrite")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    return fig


# ------------------------------------------------------------------ network
def network_png(g: GameTrace, title: str | None = None, seed: int = 0) -> bytes:
    scores = final_scores(g)
    agents = list(g.agents)
    dm, gv, bn = Counter(), Counter(), Counter()
    for e in g.of_type("message"):
        m = e.data.message  # type: ignore[union-attr]
        if m.channel == "dm" and m.receiver and not m.forged:
            dm[(m.sender_shown, m.receiver)] += 1
    for e in g.of_type("key_transfer"):
        d = e.data
        if d.cause == "give" and d.from_agent:  # type: ignore[union-attr]
            gv[(d.from_agent, d.to_agent)] += 1  # type: ignore[union-attr]
    for e in g.of_type("burn"):
        bn[(e.data.actor, e.data.target)] += 1  # type: ignore[union-attr]
    bd = Counter()
    for e in g.of_type("action_result"):
        act = e.data.action  # type: ignore[union-attr]
        if getattr(act, "kind", None) == "post_board" and e.data.ok:  # type: ignore[union-attr]
            bd[(e.actor, "BOARD")] += 1
    powers = [p for p in god_actions(g) if p["ok"] and p["target"]]
    U = nx.Graph()
    U.add_nodes_from(agents)
    for (a, b), n in (dm + gv + bn).items():
        U.add_edge(a, b, weight=n)
    pos = nx.spring_layout(U, seed=seed, k=1.2 / max(1, len(agents)) ** 0.5)
    pos["GOD"] = np.array([-1.35, 1.15])
    pos["BOARD"] = np.array([1.35, 1.15])
    G = nx.DiGraph()
    G.add_nodes_from(agents)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.grid(False)
    ax.axis("off")

    def draw(edges: Counter, color: str, style: str = "solid", base: float = 0.8, alpha: float = 0.6) -> None:
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=list(edges), ax=ax, edge_color=color, style=style,
                                   width=[base + 0.5 * min(n, 6) for n in edges.values()], alpha=alpha,
                                   arrows=True, arrowsize=9, connectionstyle="arc3,rad=0.08",
                                   node_size=300)

    draw(dm, C["dm"], base=0.5, alpha=0.35)
    draw(gv, C["give"])
    draw(bn, C["defect"], alpha=0.85)
    G.add_node("BOARD")
    draw(bd, C["board"], alpha=0.6)
    god_edges = {"message_fraud": Counter(), "cot_control": Counter()}
    for p in powers:
        god_edges[p["power"]][("GOD", p["target"])] += 1
    G.add_node("GOD")
    draw(god_edges["message_fraud"], C["fraud"], style="dashed", base=1.8, alpha=0.95)
    draw(god_edges["cot_control"], C["cot"], style="dotted", base=1.8, alpha=0.95)
    won = {e.data.agent for e in g.of_type("win")}  # type: ignore[union-attr]
    burned = {e.data.target for e in g.of_type("burn") if e.data.correct}  # type: ignore[union-attr]
    colors = [C["win"] if a in won else C["defect"] if a in burned else C["a"] for a in agents]
    sizes = [160 + 520 * scores.get(a, 0.0) for a in agents]
    nx.draw_networkx_nodes(G, pos, nodelist=agents, node_color=colors, node_size=sizes, ax=ax,
                           edgecolors=SURFACE, linewidths=1.5)
    touched = {p["target"]: p["power"] for p in powers}
    if touched:
        nx.draw_networkx_nodes(G, pos, nodelist=list(touched), node_color="none", node_size=900, ax=ax,
                               edgecolors=[C["fraud"] if v == "message_fraud" else C["cot"] for v in touched.values()],
                               linewidths=2.2)
    nx.draw_networkx_nodes(G, pos, nodelist=["GOD"], node_color=INK2, node_shape="s", node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={"GOD": "GOD"}, font_color=SURFACE, font_size=8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["BOARD"], node_color=C["board"], node_shape="s", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={"BOARD": "BOARD"}, font_color=SURFACE, font_size=7, ax=ax)
    nx.draw_networkx_labels(G, {a: pos[a] + np.array([0, -0.075]) for a in agents},
                            labels={a: a for a in agents}, font_size=6.5, font_color=INK2, ax=ax)
    handles = [Line2D([], [], color=C["dm"], label="DMs"), Line2D([], [], color=C["give"], label="gives"),
               Line2D([], [], color=C["defect"], label="defects"),
               Line2D([], [], color=C["board"], label="board posts"),
               Line2D([], [], color=C["fraud"], ls="--", label="God: forged DM"),
               Line2D([], [], color=C["cot"], ls=":", label="God: memory overwrite"),
               Line2D([], [], marker="o", ls="", color=C["a"], label="active"),
               Line2D([], [], marker="o", ls="", color=C["win"], label="won"),
               Line2D([], [], marker="o", ls="", color=C["defect"], label="had a need burned")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, ncol=2)
    ax.set_title(title or f"Interaction network: {g.label} (node size = final score)", loc="left")
    return fig_to_png(fig)


# ------------------------------------------------------------------ ripple
def ripple_png(res: RippleResult, noise: RippleResult | None = None, title: str | None = None) -> bytes:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, max(3.6, 0.2 * len(res.agents) + 1.2)),
                                        gridspec_kw={"width_ratios": [1.1, 1.3, 0.9]})
    r = res.rounds
    ax1.plot(r, res.cum, color=C["a"], label="agents diverged so far (God vs no God)")
    ax1.plot(r, res.frac, color=C["a"], ls=":", label="agents acting differently this round")
    if noise is not None:
        ax1.plot(noise.rounds, noise.cum, color=C["noise"], label="noise floor (no God vs replica)")
    if res.fork_round:
        ax1.axvline(res.fork_round, color=C["fraud"], lw=1.2, ls="--")
        ax1.annotate("God's first\nintervention", (res.fork_round, 0.98), xytext=(4, -2),
                     textcoords="offset points", va="top", fontsize=8, color=C["fraud"])
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("round")
    ax1.set_ylabel("fraction of agents")
    ax1.set_title("Ripple: divergence between forks", loc="left")
    ax1.legend(fontsize=7.5, loc="center left")

    order = sorted(res.agents, key=lambda a: (res.first_div[a] is None, res.first_div[a] or 0, a))
    M = np.array([res.div[a] for a in order]) if order else np.zeros((1, 1))
    ax2.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest",
               extent=(r[0] - 0.5, r[-1] + 0.5, len(order) - 0.5, -0.5) if r else None)
    ax2.grid(False)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([("* " if a in res.victims else "") + a for a in order], fontsize=7)
    if res.fork_round:
        ax2.axvline(res.fork_round - 0.5, color=C["fraud"], lw=1.2, ls="--")
    ax2.set_xlabel("round")
    ax2.set_title("Per-agent divergence (dark = differs; * = God victim)", loc="left")

    ax3.plot(r, res.welfare_a, color=C["a"], label="with God")
    ax3.plot(r, res.welfare_b, color=C["b"], label="without God")
    if res.fork_round:
        ax3.axvline(res.fork_round, color=C["fraud"], lw=1.2, ls="--")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_xlabel("round")
    ax3.set_ylabel("welfare (mean score)")
    ax3.set_title("Welfare by branch", loc="left")
    ax3.legend(fontsize=7.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig_to_png(fig)
