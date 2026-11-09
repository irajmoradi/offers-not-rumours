"""
Animations replayed from a finished game's event log. Run a game with
keep_log=True, then call these to render what happened round by round.

    from llm_game import new_game, play
    from animate import animate_game, animate_metric_race, show_gif
    st = new_game(6, "complete", 0, 12, 0.7, visibility="private", keep_log=True)
    play(st, backend, seed=0)
    animate_game(st, "game.gif"); show_gif("game.gif")

Needs matplotlib + pillow (both standard on Colab/Kaggle).
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def _events_by_round(st):
    rounds = {}
    for ev in getattr(st, "log", []):
        rounds.setdefault(ev.get("round", 0), []).append(ev)
    return rounds


def animate_game(st, path="game.gif", fps=1.4, seed=1, title="Swarm replay"):
    """Round-by-round network: green = gifts (cumulative), red = defections,
    purple dashed = hidden-adversary messages. Node color = keys still needed
    (dark = done); gold ring = finished; red ring = lost a key."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    if not getattr(st, "log", None):
        raise ValueError("No event log. Build the game with keep_log=True.")
    N = st.cfg.N
    G = st.G
    pos = nx.spring_layout(G, seed=seed, k=1.2)
    by_round = _events_by_round(st)
    rounds = sorted(by_round)
    need0 = [len(s) for s in st.needs0]

    gifts = {}             # (i,j) -> count, cumulative
    needs_left = [set(s) for s in st.needs0]
    burned = [set() for _ in range(N)]
    done = [False] * N
    coop = defect = 0

    fig, ax = plt.subplots(figsize=(6.2, 6))

    def draw(frame):
        nonlocal coop, defect
        ax.clear(); ax.axis("off")
        r = rounds[frame]
        this_def, this_d_evil = [], []
        for ev in by_round[r]:
            t = ev.get("type")
            if t == "give":
                e = (ev["agent"], ev["to"]); gifts[e] = gifts.get(e, 0) + 1; coop += 1
                if ev["key"] in needs_left[ev["to"]]:
                    needs_left[ev["to"]].discard(ev["key"])
            elif t == "defect":
                defect += 1; this_def.append((ev["actor"], ev["target"]))
                if ev.get("success"):
                    needs_left[ev["target"]].discard(ev.get("key"))
                    burned[ev["target"]].add(ev.get("key"))
                    if ev.get("gained") is not None:
                        needs_left[ev["actor"]].discard(ev["gained"])
                else:
                    if ev.get("lost") is not None:
                        needs_left[ev["actor"]].discard(ev["lost"]); burned[ev["actor"]].add(ev["lost"])
            elif t == "d_evil":
                this_d_evil.append((ev["pretend_from"], ev["to"]))
            elif t == "win":
                done[ev["agent"]] = True
        for i in range(N):
            if not needs_left[i]:
                done[i] = True
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ddd", width=0.6)
        if gifts:
            nx.draw_networkx_edges(G if False else nx.DiGraph(list(gifts)), pos, ax=ax,
                                   edge_color="#009E73", width=[0.6 + gifts[e] for e in gifts],
                                   alpha=0.6, arrowsize=9, connectionstyle="arc3,rad=0.08")
        if this_d_evil:
            nx.draw_networkx_edges(nx.DiGraph(this_d_evil), pos, ax=ax, edge_color="#CC79A7",
                                   style="dashed", width=2.0, arrowsize=14, connectionstyle="arc3,rad=0.2")
        if this_def:
            nx.draw_networkx_edges(nx.DiGraph(this_def), pos, ax=ax, edge_color="#D55E00",
                                   width=3.0, arrowsize=16, connectionstyle="arc3,rad=-0.2")
        frac = [1 - len(needs_left[i]) / max(need0[i], 1) for i in range(N)]
        ring = ["#F0C000" if done[i] else ("#D55E00" if burned[i] else "#888") for i in range(N)]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=frac, cmap="viridis", vmin=0, vmax=1,
                               node_size=560, edgecolors=ring, linewidths=2.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=9)
        ax.set_title(f"{title}  —  round {r}\ngifts {coop}   defections {defect}", fontsize=11)
        return ax,

    anim = FuncAnimation(fig, draw, frames=len(rounds), interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def animate_metric_race(states, labels, path="race.gif", fps=2.0, title="Cooperation vs. betrayal over rounds"):
    """Animated line race: cumulative gifts and defections per round, averaged
    over a list of games (one color per label group)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    COL = {"gifts": "#009E73", "defections": "#D55E00"}

    def series(group):
        maxr = max((max(_events_by_round(s) or [0]) for s in group), default=1)
        g = np.zeros(maxr + 1); d = np.zeros(maxr + 1)
        for s in group:
            br = _events_by_round(s)
            for r in range(1, maxr + 1):
                evs = br.get(r, [])
                g[r] += sum(e.get("type") == "give" for e in evs)
                d[r] += sum(e.get("type") == "defect" for e in evs)
        n = max(len(group), 1)
        return np.cumsum(g) / n, np.cumsum(d) / n, maxr

    groups = {}
    for s, lab in zip(states, labels):
        groups.setdefault(lab, []).append(s)
    data = {lab: series(grp) for lab, grp in groups.items()}
    maxr = max(v[2] for v in data.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = ["-", "--", ":", "-."]

    def draw(frame):
        ax.clear()
        for k, (lab, (g, d, mr)) in enumerate(data.items()):
            xs = range(min(frame + 1, len(g)))
            ax.plot(list(xs), g[:frame + 1], styles[k % 4], color=COL["gifts"],
                    label=f"{lab}: gifts")
            ax.plot(list(xs), d[:frame + 1], styles[k % 4], color=COL["defections"],
                    label=f"{lab}: defections")
        ax.set_xlim(0, maxr); ax.set_xlabel("round"); ax.set_ylabel("cumulative (per game)")
        ax.set_title(f"{title}  —  round {frame}")
        ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)
        return ax,

    anim = FuncAnimation(fig, draw, frames=maxr + 1, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def show_gif(path):
    from IPython.display import Image, display
    display(Image(filename=path))
