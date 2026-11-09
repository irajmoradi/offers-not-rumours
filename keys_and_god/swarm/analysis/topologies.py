"""Topology comparison for game.network (plan/TOPOLOGIES.md). No LLM, deterministic.

    PYTHONNOUSERSITE=1 $PY -m swarm.analysis.topologies \
        [--worlds 30] [--procs 16] [--out plan/figs]

For ER3 (configs/net40_er3_*), sbm (net40_sbm_contacts_sim), community_hubs (net40_hubclusters_known_sim)
and powerlaw_cluster (net40_powerlaw_known_sim), x key_placement random / local_bias with local_strength 0.5 (local_bias50) / local_bias, on the REAL world
generator and graph builder (world seeds of game_idx 0..worlds-1, so 0-3 are the worlds a live run
plays):
  geometry  graph stats (swarm.game.network.graph_summary) and the hop distance from each need to its
            nearest holder (swarm.analysis.game_theory_audit.need_distances, the audit's method);
  sim       the audit's scripted-agent simulation (game_theory_audit.Sim: real rules.apply transitions,
            simultaneous, reshuffled priority, winner_policy continue, 30 rounds), all-C (cooperators
            that relay toward far needers in contacts mode) and all-Cn (never relay), under the
            information modes board (neighbour board only, the LLM default), gossip2 and gossipAll,
            in both network modes (contacts / known_ids). Welfare = mean score = the best possible
            score under that information model.
Outputs: <out>/topologies.json, <out>/topologies.png (one example graph per topology, coloured by
community, hubs and bridges highlighted, plus the need-distance distributions).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from swarm.analysis.game_theory_audit import Sim, hash_job, load_cfg, need_distances, world
from swarm.config import RunConfig
from swarm.game import network

TOPOS = {  # label: (config, display name)
    "er3": ("net40_er3_contacts_sim", "Erdős–Rényi, mean degree 3"),
    "sbm": ("net40_sbm_contacts_sim", "SBM, 4 communities"),
    "hubs": ("net40_hubclusters_known_sim", "community_hubs (4 clusters, hub clique)"),
    "plc": ("net40_powerlaw_known_sim", "powerlaw_cluster (Holme–Kim)"),
}
PLACEMENTS = {"random": ["game.network.key_placement=random"],
              "local_bias50": ["game.network.key_placement=local_bias", "game.network.local_strength=0.5"],
              "local_bias": ["game.network.key_placement=local_bias"]}
MODES = ("contacts", "known_ids")
SIM_CELLS = [("C", "board"), ("C", "gossip2"), ("C", "gossipAll"), ("Cn", "gossipAll")]


def cfg_for(topo: str, mode: str, placement: str) -> RunConfig:
    return load_cfg(TOPOS[topo][0], [f"game.network.mode={mode}", *PLACEMENTS[placement],
                                     "game.winner_policy=continue", "game.max_rounds=30"])


def geometry(topo: str, placement: str, n_worlds: int) -> dict[str, Any]:
    cfg = cfg_for(topo, "contacts", placement)
    near, matched, stats, no_nb = [], [], [], []
    for gi in range(n_worlds):
        st = world(cfg, gi)
        d = need_distances(st)
        near += d["nearest"]
        matched += d["matched"]
        adj = {a: list(s.contacts or []) for a, s in st.agents.items()}
        comm = network.make_graph_full(cfg.game.network, list(st.agents), _seed(cfg, gi))[1]
        stats.append(network.graph_summary(adj, comm))
        hold: dict[int, set[str]] = {}
        for a, s in st.agents.items():
            for kc in s.keys:
                hold.setdefault(kc.info_id, set()).add(a)
        no_nb += [all(not (hold[x] & set(adj[a])) for x in s.needs) for a, s in st.agents.items()]
    num = ("mean_degree", "max_degree", "clustering", "avg_path", "diameter", "modularity", "between_community_edges")
    hist = Counter(near)
    return {"graph": {k: round(statistics.mean(s[k] for s in stats), 3) for k in num},
            "nearest_hist": {str(k): hist[k] for k in sorted(hist)},
            "mean_nearest": round(statistics.mean(near), 3), "mean_matched": round(statistics.mean(matched), 3),
            "share_le1": round(sum(x <= 1 for x in near) / len(near), 3),
            "share_le2": round(sum(x <= 2 for x in near) / len(near), 3),
            "share_ge3": round(sum(x >= 3 for x in near) / len(near), 3),
            "agents_no_need_at_neighbour": round(statistics.mean(1.0 if f else 0.0 for f in no_nb), 3)}


def _seed(cfg: RunConfig, gi: int) -> int:
    from swarm.runner.spec import world_seed

    return world_seed(cfg, gi)


def run_job(job: tuple) -> dict[str, Any]:
    topo, mode, placement, pop, info, gi = job
    sim = Sim(cfg_for(topo, mode, placement), gi, pop, info, order_seed=hash_job(job))
    res = sim.play()
    return {"job": list(job), "welfare": res["welfare"], "rounds": res["rounds"], "finished": res["finished"],
            "relays": res["stats"].get("relay_gives", 0)}


def jobs(n_worlds: int) -> list[tuple]:
    out = []
    for topo in TOPOS:
        for placement in PLACEMENTS:
            for mode in MODES:
                for pop, info in SIM_CELLS:
                    if pop == "Cn" and mode == "known_ids":      # relays only exist in contacts mode
                        continue
                    out += [(topo, mode, placement, pop, info, gi) for gi in range(n_worlds)]
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        topo, mode, placement, pop, info, _ = r["job"]
        cells.setdefault(f"{topo}|{mode}|{placement}|{pop}|{info}", []).append(r)
    return {k: {"n": len(v), "welfare": round(statistics.mean(x["welfare"] for x in v), 3),
                "finished": round(statistics.mean(1.0 if x["finished"] else 0.0 for x in v), 3),
                "rounds": round(statistics.mean(x["rounds"] for x in v), 1),
                "relays": round(statistics.mean(x["relays"] for x in v), 1)} for k, v in sorted(cells.items())}


# ------------------------------------------------------------------ figure
COMM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # dataviz slots 1-4 (CVD all-pairs >= 9.1)
COMM_MARKERS = ("o", "s", "^", "D")                          # secondary encoding (orange/yellow dE 13.7)
INK, INK2, MUTED, SURFACE, EDGE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#c9c8c2"


def fig(geo: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig_, axes = plt.subplots(2, 4, figsize=(15, 7.6), gridspec_kw={"height_ratios": [2.3, 1]},
                              facecolor=SURFACE)
    for col, (topo, (_cfgname, title)) in enumerate(TOPOS.items()):
        cfg = cfg_for(topo, "contacts", "random")
        st = world(cfg, 0)
        adj, comm = network.make_graph_full(cfg.game.network, list(st.agents), _seed(cfg, 0))
        g = nx.Graph()
        g.add_nodes_from(sorted(adj))
        g.add_edges_from((a, b) for a in adj for b in adj[a] if a < b)
        planted = comm is not None
        if comm is None:                                  # detected (4 groups, greedy modularity)
            parts = nx.algorithms.community.greedy_modularity_communities(g, cutoff=4, best_n=4)
            comm = {a: c for c, p in enumerate(sorted(parts, key=lambda p: (-len(p), min(p)))) for a in p}
        btw = nx.betweenness_centrality(g)
        deg = dict(g.degree())
        hubs = sorted(g, key=lambda a: (-deg[a], a))[:3]
        bridges = [a for a in sorted(g, key=lambda a: (-btw[a], a)) if a not in hubs][:3]
        # layout: communities pulled apart (weighted spring), seeded
        w = {(a, b): (1.0 if comm[a] == comm[b] else 0.15) for a, b in g.edges()}
        nx.set_edge_attributes(g, w, "w")
        pos = nx.spring_layout(g, weight="w", seed=3, iterations=200)
        ax = axes[0, col]
        ax.set_facecolor(SURFACE)
        cross = [(a, b) for a, b in g.edges() if comm[a] != comm[b]]
        nx.draw_networkx_edges(g, pos, edgelist=[e for e in g.edges() if e not in cross], ax=ax,
                               edge_color=EDGE, width=0.9)
        nx.draw_networkx_edges(g, pos, edgelist=cross, ax=ax, edge_color=INK2, width=1.1, style="dashed")
        for c in sorted(set(comm.values())):
            nodes = [a for a in g if comm[a] == c]
            nx.draw_networkx_nodes(g, pos, nodelist=nodes, ax=ax, node_color=COMM_COLORS[c % 4],
                                   node_shape=COMM_MARKERS[c % 4],
                                   node_size=[40 + 22 * deg[a] for a in nodes], edgecolors=SURFACE, linewidths=1.0)
        for a in hubs + bridges:
            nx.draw_networkx_nodes(g, pos, nodelist=[a], ax=ax, node_color="none", node_shape="o",
                                   node_size=40 + 22 * deg[a] + 160, edgecolors=INK, linewidths=1.8 if a in hubs else 1.0)
            lab = ("hub " if a in hubs else "bridge ") + a.split("_")[1]
            x, y = pos[a]
            ax.annotate(lab, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=7.5, color=INK)
        s = network.graph_summary(adj, None if not planted else comm)
        ax.set_title(f"{title}\n<k>={s['mean_degree']}  C={s['clustering']:.2f}  L={s['avg_path']:.2f}  "
                     f"D={s['diameter']}  Q={s['modularity']:.2f}"
                     f"{'' if planted else ' (detected)'}", fontsize=9, color=INK, loc="left")
        ax.axis("off")
        # need-distance distribution: random vs local_bias
        ax2 = axes[1, col]
        ax2.set_facecolor(SURFACE)
        hops = range(1, 8)
        pls = (("random", INK2), ("local_bias50", "#86b6ef"), ("local_bias", "#2a78d6"))
        for k, (pl, color) in enumerate(pls):
            h = geo[f"{topo}|{pl}"]["nearest_hist"]
            tot = sum(h.values())
            ys = [h.get(str(x), 0) / tot for x in hops]
            ax2.bar([x + (k - 1) * 0.27 for x in hops], ys, width=0.25, color=color,
                    label=f"{pl} (≤2 hops: {geo[f'{topo}|{pl}']['share_le2']:.0%})")
        ax2.set_xticks(list(hops))
        ax2.set_xlabel("hops from a need to its nearest holder")
        if col == 0:
            ax2.set_ylabel("share of need slots")
        ax2.set_ylim(0, 0.8)
        ax2.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax2.set_axisbelow(True)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        ax2.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig_.suptitle("Network topologies, N=40 (world 0 of each config). Colour + shape = community "
                  "(planted, or 4 greedy-modularity groups); dashed = between-community edge;\n"
                  "node size = degree; thick ring = top-3 hub (degree), thin ring = top-3 other bridge (betweenness).",
                  fontsize=8.5, color=INK2, x=0.01, ha="left")
    fig_.tight_layout(rect=(0, 0, 1, 0.95))
    fig_.savefig(out / "topologies.png", dpi=150, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="plan/figs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = {f"{t}|{p}": geometry(t, p, a.worlds) for t in TOPOS for p in PLACEMENTS}
    js = jobs(a.worlds)
    with Pool(a.procs) as pool:
        res = pool.map(run_job, js, chunksize=2)
    summ = summarize(res)
    (out / "topologies.json").write_text(json.dumps({"worlds": a.worlds, "geometry": geo, "sim": summ}, indent=1))
    fig(geo, out)
    for k, v in geo.items():
        print(f"GEO {k:18s} {v['graph']} le1={v['share_le1']} le2={v['share_le2']} ge3={v['share_ge3']} "
              f"mean={v['mean_nearest']} no_nb={v['agents_no_need_at_neighbour']}")
    for k, v in summ.items():
        print(f"SIM {k:42s} W={v['welfare']:.3f} fin={v['finished']:.2f} rounds={v['rounds']} relays={v['relays']}")


if __name__ == "__main__":
    main()
