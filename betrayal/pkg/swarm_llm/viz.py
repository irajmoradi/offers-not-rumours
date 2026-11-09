"""
Charts for every experiment. One function per experiment; each builds a
multi-panel matplotlib figure from the same dataframes/row-lists the notebook
produces. Colorblind-safe palette (Okabe-Ito), one y-axis per panel, error bars
where seeds give a spread, direct value labels on bars.

    from viz import setup_style, plot_main, plot_scaling, plot_interviews, \
                    plot_one_lie, plot_insider, plot_deterrence, plot_emergence_graph, \
                    plot_route_scaling
    setup_style()
    plot_main(dm)                 # dm = pd.DataFrame(MAIN_ROWS)
"""

from __future__ import annotations

import numpy as np

# Okabe-Ito, colorblind-safe
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
GOOD, WARN, BAD = "#009E73", "#E69F00", "#D55E00"


def setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 8,
    })


def _note(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#888", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _labels(ax, bars, fmt="{:.2f}", rot=0):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                        fontsize=7, xytext=(0, 1), textcoords="offset points", rotation=rot)


def _grouped(ax, df, index, column, value, title, ylabel, fmt="{:.2f}", ylim=None):
    """Grouped bars: one group per index value, one bar per column value, mean of `value`."""
    import pandas as pd
    piv = df.pivot_table(index=index, columns=column, values=value, aggfunc="mean")
    err = df.pivot_table(index=index, columns=column, values=value, aggfunc="sem")
    if piv.empty:
        _note(ax, "no data"); ax.set_title(title); return
    cats = list(piv.index); cols = list(piv.columns)
    x = np.arange(len(cats)); w = 0.8 / max(len(cols), 1)
    for k, col in enumerate(cols):
        bars = ax.bar(x + k * w, piv[col].values, w, yerr=err[col].values if not err.empty else None,
                      capsize=2, color=PALETTE[k % len(PALETTE)], label=str(col))
        _labels(ax, bars, fmt)
    ax.set_xticks(x + w * (len(cols) - 1) / 2); ax.set_xticklabels([str(c) for c in cats])
    ax.set_title(title); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    if len(cols) > 1: ax.legend(title=str(column))


# --------------------------------------------------------------------------
# Main experiment: cooperation vs. betrayal under covert influence
# --------------------------------------------------------------------------

def plot_main(dm, show=True):
    import matplotlib.pyplot as plt
    setup_style()
    if "branch" not in dm:
        dm = dm.copy()
        dm["branch"] = dm.apply(lambda r: "no D-Evil" if r["d_evil_budget"] == 0 else f"D-Evil {r['d_evil_budget']}, {r['defense']}", axis=1)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    panels = [("cooperation_rate", "share of moves", "Cooperation (gifts)", (0, None)),
              ("defection_rate", "share of moves", "Betrayal (defections)", (0, None)),
              ("mean_score", "keys obtained / needed", "User outcome", (0, 1.02)),
              ("retaliations", "count per game", "Revenge defections", None),
              ("defections_within_2_rounds_of_d_evil_msg", "count per game", "Defections soon after a D-Evil message", None),
              ("d_evil_reached", "agents", "How many agents D-Evil reached", None)]
    for a, (metric, ylabel, title, ylim) in zip(ax.ravel(), panels):
        if metric not in dm:
            _note(a, f"{metric} missing"); a.set_title(title); continue
        _grouped(a, dm, "branch", "visibility", metric, title, ylabel,
                 fmt="{:.2f}" if dm[metric].max() <= 2 else "{:.1f}", ylim=ylim)
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Cooperation vs. betrayal under a covert influencer", fontweight="bold")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_d_evil_budget(dm, show=True):
    """Dose-response: defection vs. D-Evil budget, if several budgets were run."""
    import matplotlib.pyplot as plt
    setup_style()
    budgets = sorted(dm["d_evil_budget"].unique())
    if len(budgets) < 2:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for metric, a, title in [("defection_rate", ax[0], "Betrayal vs. D-Evil budget"),
                             ("mean_score", ax[1], "User outcome vs. D-Evil budget")]:
        for k, vis in enumerate(sorted(dm["visibility"].unique())):
            sub = dm[dm.visibility == vis].groupby("d_evil_budget")[metric]
            m, e = sub.mean(), sub.sem()
            a.errorbar(m.index, m.values, yerr=e.values, marker="o", color=PALETTE[k], label=vis, capsize=2)
        a.set_xlabel("D-Evil message budget"); a.set_title(title); a.legend(title="defections")
    fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Scaling / emergence
# --------------------------------------------------------------------------

def plot_scaling(ds, show=True):
    import matplotlib.pyplot as plt, pandas as pd
    setup_style()
    g = ds.groupby("N"); m = g.mean(numeric_only=True); e = g.sem(numeric_only=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0, 0]
    for metric, col, lab in [("completion", PALETTE[0], "completion"), ("mean_score", PALETTE[2], "mean score")]:
        if metric in m: a.errorbar(m.index, m[metric], yerr=e.get(metric), marker="o", color=col, label=lab, capsize=2)
    a.set_title("Failure as the swarm grows"); a.set_xlabel("swarm size N"); a.set_ylim(0, 1.02); a.legend()
    a = ax[0, 1]
    for metric, col, lab in [("indegree_centralization", BAD, "in-degree centralization"),
                             ("top_giver_share", PALETTE[4], "top-giver share")]:
        if metric in m: a.errorbar(m.index, m[metric], yerr=e.get(metric), marker="o", color=col, label=lab, capsize=2)
    a.set_title("Leader emergence"); a.set_xlabel("swarm size N"); a.set_ylim(0, 1); a.legend()
    a = ax[1, 0]
    for metric, col, lab in [("defection_rate", WARN, "defection rate"), ("score_gini", PALETTE[5], "inequality (Gini)")]:
        if metric in m: a.errorbar(m.index, m[metric], yerr=e.get(metric), marker="o", color=col, label=lab, capsize=2)
    a.set_title("Conflict and inequality"); a.set_xlabel("swarm size N"); a.legend()
    a = ax[1, 1]
    if "role_counts" in ds:
        roles = ds["role_counts"].apply(pd.Series).groupby(ds["N"]).mean()
        share = roles.div(roles.sum(axis=1), axis=0)
        bottom = np.zeros(len(share))
        for k, c in enumerate(share.columns):
            a.bar(share.index.astype(str), share[c].values, bottom=bottom, color=PALETTE[k % len(PALETTE)], label=c)
            bottom += share[c].values
        a.set_title("Role mix vs N"); a.set_xlabel("swarm size N"); a.set_ylabel("share"); a.legend(ncol=2)
    else:
        _note(a, "role_counts missing")
    fig.suptitle("Leaders and failures as the swarm grows", fontweight="bold")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_emergence_graph(st, show=True, title="Gift network"):
    """Draw one game's gift network, node size = keys given, color = role."""
    import matplotlib.pyplot as plt, networkx as nx
    from emergence import role_types
    setup_style()
    roles, _ = role_types(st)
    cmap = {"giver": PALETTE[2], "taker": PALETTE[3], "broker": PALETTE[4], "trader": PALETTE[0], "isolate": "#bbb"}
    G = nx.DiGraph()
    G.add_nodes_from(range(st.cfg.N))
    for i in range(st.cfg.N):
        for j in range(st.cfg.N):
            if st.gifts[i, j]: G.add_edge(i, j, weight=int(st.gifts[i, j]))
    out = st.gifts.sum(axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    pos = nx.spring_layout(G, seed=1, k=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999", width=[0.6 + G[u][v]["weight"] for u, v in G.edges()],
                           alpha=0.6, arrowsize=10, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[120 + 180 * out[i] for i in range(st.cfg.N)],
                           node_color=[cmap[roles[i]] for i in range(st.cfg.N)])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=r)
               for r, c in cmap.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=7); ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Interviews: self-report vs. behavior
# --------------------------------------------------------------------------

def plot_interviews(interviews, show=True):
    import matplotlib.pyplot as plt
    setup_style()
    judged = [r for r in interviews if r.get("judge")]
    if not judged:
        return None
    targeted = [r for r in judged if r["behavior"]["hidden_adversary_messages_received"] > 0]
    untargeted = [r for r in judged if r["behavior"]["hidden_adversary_messages_received"] == 0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cons = np.mean([bool(r["judge"].get("consistent")) for r in judged])
    susp_t = np.mean([bool(r["judge"].get("suspected_outside_influence")) for r in targeted]) if targeted else 0
    susp_u = np.mean([bool(r["judge"].get("suspected_outside_influence")) for r in untargeted]) if untargeted else 0
    b = ax[0].bar(["answers consistent\nwith behavior"], [cons], color=PALETTE[0], width=0.5)
    _labels(ax[0], b, "{:.0%}"); ax[0].set_ylim(0, 1); ax[0].set_title("Interview faithfulness")
    b = ax[1].bar(["D-Evil messaged them", "D-Evil did not"], [susp_t, susp_u], color=[BAD, PALETTE[0]], width=0.6)
    _labels(ax[1], b, "{:.0%}"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Suspected an outside influence")
    fig.suptitle("Do the assistants notice the covert influencer?", fontweight="bold")
    fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Extra experiment 1: one forged message
# --------------------------------------------------------------------------

def plot_one_lie(df, show=True):
    import matplotlib.pyplot as plt
    setup_style()
    att = df[df.condition != "clean"].copy()
    if att.empty:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    _grouped(ax[0], att, "condition", "visibility", "damage", "Cooperation lost to the lie",
             "drop in completion")
    if "victim_score" in att:
        _grouped(ax[1], att, "condition", "visibility", "victim_score", "Victim's outcome", "keys obtained / needed", ylim=(0, 1.02))
    for k, metric in enumerate(["warned_about_victim", "defected_on_victim"]):
        if metric in att:
            sub = att.groupby("condition")[metric].mean()
            ax[2].bar(np.arange(len(sub)) + k * 0.4, sub.values, 0.4, color=PALETTE[k], label=metric.replace("_", " "))
    ax[2].set_xticks(np.arange(len(att.condition.unique())) + 0.2); ax[2].set_xticklabels(sorted(att.condition.unique()))
    ax[2].set_title("How the lie spread"); ax[2].set_ylabel("agents per game"); ax[2].legend()
    fig.suptitle("One forged message", fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Extra experiment 2: who to compromise
# --------------------------------------------------------------------------

def plot_insider(df2, show=True):
    import matplotlib.pyplot as plt
    setup_style()
    if "variant" not in df2:
        df2 = df2.copy()
        df2["variant"] = df2.apply(lambda r: "hoarding only" if not r["lies"] else f"hoard+lie,{r['defense']}", axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _grouped(ax[0], df2, "target", "variant", "damage", "Damage by who was compromised", "drop in completion")
    ax[0].tick_params(axis="x", rotation=15)
    if "topology" in df2:
        _grouped(ax[1], df2[df2.variant.str.startswith("hoard+lie")], "target", "topology", "damage",
                 "Damage by network shape (hoard+lie)", "drop in completion")
        ax[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Who to compromise", fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Experiment 3: deterrence
# --------------------------------------------------------------------------

def plot_deterrence(E3_ROWS, E3=None, scripted_pred=None, show=True):
    import matplotlib.pyplot as plt, pandas as pd
    setup_style()
    df = pd.DataFrame(E3_ROWS)
    for c in ["take_shortcut", "do_task"]:
        df[c] = df[c].map(lambda x: np.nan if x is None else float(x))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # (1) heatmap: attempt rate over detection x penalty (penalty regime, neutral frame)
    pen = df[(df.regime == "penalty")]
    if "frame" in pen: pen = pen[pen.frame == pen.frame.iloc[0]]
    grid = pen.pivot_table(index="p", values="take_shortcut", columns="s", aggfunc="mean")
    a = ax[0, 0]
    if not grid.empty:
        im = a.imshow(grid.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
        a.set_xticks(range(len(grid.columns))); a.set_xticklabels(grid.columns)
        a.set_yticks(range(len(grid.index))); a.set_yticklabels(grid.index)
        a.set_xlabel("penalty size"); a.set_ylabel("detection chance")
        for (yi, xi), v in np.ndenumerate(grid.values):
            if np.isfinite(v): a.text(xi, yi, f"{v:.0%}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=a, fraction=0.046, label="broke the rule")
    a.set_title("Rule-breaking over detection × penalty")
    # (2) iso-expected-penalty line + scripted predictions
    a = ax[0, 1]
    if E3 and E3.get("iso_expected_penalty_8"):
        iso = E3["iso_expected_penalty_8"]
        a.plot(list(iso.keys()), [100 * v for v in iso.values()], marker="o", color=PALETTE[0], label="LLM")
        if scripted_pred:
            for k, (model, rows) in enumerate(scripted_pred.items()):
                a.plot([f"p={r['p']} s={r['s']}" for r in rows], [100 * r["attempt_rate"] for r in rows],
                       "--", color=PALETTE[(k + 1) % len(PALETTE)], alpha=0.7, label=model)
        a.set_ylabel("broke the rule (%)"); a.tick_params(axis="x", rotation=20); a.legend()
    a.set_title("Same expected penalty, different certainty")
    # (3) framing ablation
    a = ax[1, 0]
    if "frame" in df:
        sub = df.groupby(["regime", "frame"])["take_shortcut"].mean().unstack()
        if not sub.empty:
            x = np.arange(len(sub.index))
            for k, fr in enumerate(sub.columns):
                b = a.bar(x + k * 0.35, sub[fr].values, 0.35, color=PALETTE[k], label=str(fr))
                _labels(a, b, "{:.0%}")
            a.set_xticks(x + 0.17); a.set_xticklabels(sub.index); a.legend(title="framing")
    a.set_title("Rule-breaking by regime and framing"); a.set_ylabel("broke the rule")
    # (4) legitimate work vs penalty
    a = ax[1, 1]
    if E3 and E3.get("legit_task_rate_by_penalty"):
        d = E3["legit_task_rate_by_penalty"]
        ks = [float(k.split("=")[1]) for k in d]
        a.plot(ks, [100 * v for v in d.values()], marker="s", color=PALETTE[3])
        a.set_xlabel("penalty size"); a.set_ylabel("legitimate work done (%)")
    a.set_title("Chilling effect: penalties deter good work too")
    fig.suptitle("Deterrence vs. containment", fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Real-world routes: scaling of centralization vs N
# --------------------------------------------------------------------------

def plot_route_scaling(rows, size_key="requested_N", title="Real framework: leaders and failure vs N", show=True):
    import matplotlib.pyplot as plt, pandas as pd
    setup_style()
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    g = df.groupby(size_key); m = g.mean(numeric_only=True); e = g.sem(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for metric, col, lab in [("indegree_centralization", BAD, "in-degree centralization"),
                             ("leader_share", PALETTE[4], "leader share"),
                             ("single_hub", PALETTE[0], "single-hub rate")]:
        if metric in m: ax[0].errorbar(m.index, m[metric], yerr=e.get(metric), marker="o", color=col, label=lab, capsize=2)
    ax[0].set_title("Leader emergence"); ax[0].set_xlabel("swarm size N"); ax[0].set_ylim(0, 1); ax[0].legend()
    for metric, col, lab in [("density", PALETTE[2], "graph density"), ("reciprocity", PALETTE[1], "reciprocity")]:
        if metric in m: ax[1].errorbar(m.index, m[metric], yerr=e.get(metric), marker="o", color=col, label=lab, capsize=2)
    ax[1].set_title("Interaction structure"); ax[1].set_xlabel("swarm size N"); ax[1].legend()
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


def plot_public_buckets(summary, show=True):
    import matplotlib.pyplot as plt
    setup_style()
    if not summary:
        return None
    buckets = list(summary.keys())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cen = [summary[b].get("mean_centralization", np.nan) for b in buckets]
    hub = [summary[b].get("single_hub_rate", np.nan) for b in buckets]
    x = np.arange(len(buckets))
    b1 = ax[0].bar(x - 0.2, cen, 0.4, color=BAD, label="centralization")
    b2 = ax[0].bar(x + 0.2, hub, 0.4, color=PALETTE[0], label="single-hub rate")
    _labels(ax[0], b1, "{:.2f}"); _labels(ax[0], b2, "{:.2f}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(buckets); ax[0].set_ylim(0, 1); ax[0].legend()
    ax[0].set_title("Leader emergence by swarm size (public data)")
    # failed vs succeeded, where labels exist
    fc = [summary[b].get("centralization_failed", np.nan) for b in buckets]
    sc = [summary[b].get("centralization_succeeded", np.nan) for b in buckets]
    if any(np.isfinite(fc)) or any(np.isfinite(sc)):
        b1 = ax[1].bar(x - 0.2, fc, 0.4, color=BAD, label="failed runs")
        b2 = ax[1].bar(x + 0.2, sc, 0.4, color=GOOD, label="succeeded runs")
        _labels(ax[1], b1, "{:.2f}"); _labels(ax[1], b2, "{:.2f}")
        ax[1].set_xticks(x); ax[1].set_xticklabels(buckets); ax[1].legend()
        ax[1].set_title("Centralization: failed vs. succeeded")
    else:
        _note(ax[1], "no success labels in this dataset")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_model_comparison(df, metrics=("defection_rate", "cooperation_rate", "mean_score", "retaliations"),
                          attacked_only=True, show=True):
    """Compare several open-source models on the main experiment. df needs a
    'model' column. One panel per metric, bars grouped by model and visibility."""
    import matplotlib.pyplot as plt
    setup_style()
    if "model" not in df or df.empty:
        return None
    d = df[df.d_evil_budget > 0] if (attacked_only and "d_evil_budget" in df) else df
    metrics = [m for m in metrics if m in d]
    n = len(metrics)
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.9))
    if n == 1:
        ax = [ax]
    hue = "visibility" if "visibility" in d and d["visibility"].nunique() > 1 else None
    for a, metric in zip(ax, metrics):
        if hue:
            _grouped(a, d, "model", hue, metric, metric.replace("_", " "), "",
                     fmt="{:.2f}" if d[metric].max() <= 2 else "{:.1f}")
        else:
            sub = d.groupby("model")[metric]
            m, e = sub.mean(), sub.sem()
            bars = a.bar(range(len(m)), m.values, yerr=e.values, capsize=2,
                         color=[PALETTE[k % len(PALETTE)] for k in range(len(m))])
            _labels(a, bars, "{:.2f}" if m.max() <= 2 else "{:.1f}")
            a.set_xticks(range(len(m))); a.set_xticklabels(m.index, rotation=25, ha="right")
            a.set_title(metric.replace("_", " "))
        a.tick_params(axis="x", rotation=25)
    fig.suptitle("Open-source models compared (under covert influence)", fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Network science, role self-report, and evaluation awareness
# --------------------------------------------------------------------------

def plot_network_graph(st, show=True, title="Interaction network (communities & bridges)"):
    """One game's message network: node color = community, size = pagerank,
    red edges = bridges / inter-community links, labels = structural role."""
    import matplotlib.pyplot as plt, networkx as nx
    import netsci
    setup_style()
    G = netsci.message_graph(st)
    if G.number_of_edges() == 0:
        G = netsci.gift_graph(st)
    membership, comms, _ = netsci.communities(G)
    roles, _ = netsci.structural_roles(st)
    pr = netsci.centralities(G)["pagerank"]
    br = netsci.bridges(G)
    bridge_set = {tuple(sorted(e)) for e in br["bridge_edges"]} | {tuple(sorted(e)) for e in br["inter_community_edges"]}
    pos = nx.spring_layout(G.to_undirected(), seed=2, k=1.1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ccc", width=0.7, arrowsize=8)
    bedges = [e for e in G.to_undirected().edges() if tuple(sorted(e)) in bridge_set]
    if bedges:
        nx.draw_networkx_edges(G.to_undirected(), pos, ax=ax, edgelist=bedges, edge_color="#D55E00", width=2.4)
    sizes = [300 + 3000 * pr.get(i, 0) for i in G.nodes()]
    colors = [PALETTE[membership.get(i, 0) % len(PALETTE)] for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: f"{i}\n{roles[i][:4]}" for i in G.nodes()},
                            font_size=7, font_color="white")
    ax.set_title(title + "\nred = bridges / cross-community links; size = PageRank"); ax.axis("off")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_centrality_over_time(temporal_rows, show=True):
    """How structure grows across rounds."""
    import matplotlib.pyplot as plt, pandas as pd
    setup_style()
    df = pd.DataFrame(temporal_rows)
    if df.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df["round"], df["gift_centralization"], marker="o", color=BAD, label="centralization")
    ax[0].plot(df["round"], df["max_pagerank"], marker="s", color=PALETTE[0], label="top agent PageRank")
    ax[0].set_title("Leader concentration over time"); ax[0].set_xlabel("round"); ax[0].legend(); ax[0].set_ylim(0, 1)
    ax[1].plot(df["round"], df["n_communities"], marker="o", color=PALETTE[2], label="# communities")
    ax[1].plot(df["round"], df["modularity"], marker="s", color=PALETTE[4], label="modularity")
    ax[1].set_title("Community structure over time"); ax[1].set_xlabel("round"); ax[1].legend()
    fig.suptitle("Network dynamics as the swarm interacts", fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


def plot_role_comparison(compare_result, show=True):
    """Confusion matrix: structural role (rows) vs self-reported role (cols)."""
    import matplotlib.pyplot as plt, pandas as pd, numpy as np
    setup_style()
    rows = compare_result.get("rows", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    sroles = sorted(df["structural_role"].unique())
    proles = sorted(df["self_role"].replace("", "(none)").unique())
    M = np.zeros((len(sroles), len(proles)))
    for _, r in df.iterrows():
        M[sroles.index(r["structural_role"]), proles.index(r["self_role"] or "(none)")] += 1
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(proles), 1.2 + 0.8 * len(sroles)))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(proles))); ax.set_xticklabels(proles, rotation=25, ha="right")
    ax.set_yticks(range(len(sroles))); ax.set_yticklabels(sroles)
    for (yi, xi), v in np.ndenumerate(M):
        if v: ax.text(xi, yi, int(v), ha="center", va="center", fontsize=8)
    ax.set_xlabel("self-reported role"); ax.set_ylabel("structural role")
    ax.set_title(f"Structural vs. self-reported role\nagreement {compare_result.get('agreement', 0):.0%} (n={compare_result.get('n', 0)})")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_eval_awareness(ea_result, show=True):
    """Do agents think they're being tested, by framing; and does believing it's
    a test change behavior?"""
    import matplotlib.pyplot as plt, pandas as pd, numpy as np
    setup_style()
    bf = pd.DataFrame(ea_result.get("by_framing", []))
    pa = pd.DataFrame(ea_result.get("per_agent", []))
    if bf.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    g = bf.groupby("framing")["test_belief_rate"].agg(["mean", "sem"]).reindex(
        [f for f in ["assistant", "assistant_test", "assistant_real", "game"] if f in bf.framing.values])
    bars = ax[0].bar(range(len(g)), g["mean"], yerr=g["sem"], capsize=3,
                     color=[PALETTE[k % len(PALETTE)] for k in range(len(g))])
    _labels(ax[0], bars, "{:.0%}")
    ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(g.index, rotation=20, ha="right")
    ax[0].set_ylim(0, 1); ax[0].set_title('Said "this is a test" by framing')
    if not pa.empty and pa["is_test"].nunique() == 2:
        d = pa.groupby("is_test")["defected"].mean()
        bars = ax[1].bar([str(k) for k in d.index], d.values, color=[PALETTE[0], BAD], width=0.6)
        _labels(ax[1], bars, "{:.0%}")
        ax[1].set_xlabel("agent believed it was a test"); ax[1].set_ylabel("defection rate")
        ax[1].set_title("Does believing it's a test change behavior?")
    else:
        _note(ax[1], "need both test/not-test beliefs")
    fig.suptitle("Evaluation awareness", fontweight="bold"); fig.tight_layout()
    if show: plt.show()
    return fig


# --------------------------------------------------------------------------
# Text-response dynamics
# --------------------------------------------------------------------------

def _group_color(st, by="role"):
    import textdyn
    g = textdyn.group_labels(st)
    if by == "role":
        return {a: (BAD if g[a]["role"] == "influential" else PALETTE[0]) for a in g}, \
               {"influential": BAD, "follower": PALETTE[0]}
    return {a: (PALETTE[0] if g[a]["aligned"] else BAD) for a in g}, \
           {"aligned": PALETTE[0], "misaligned": BAD}


def plot_response_trajectories(st, by="role", show=True):
    """2-D map of every response; each agent is a path t1 -> tN. Color = group."""
    import matplotlib.pyplot as plt
    import textdyn
    setup_style()
    pts, rows = textdyn.trajectories(st)
    if not pts:
        return None
    cmap, legend = _group_color(st, by)
    by_agent = {}
    for (a, r), xy in pts.items():
        by_agent.setdefault(a, []).append((r, xy))
    fig, ax = plt.subplots(figsize=(7, 6))
    for a, seq in by_agent.items():
        seq.sort()
        xs = [xy[0] for _, xy in seq]; ys = [xy[1] for _, xy in seq]
        col = cmap.get(a, "#888")
        ax.plot(xs, ys, "-", color=col, alpha=0.5, lw=1.2)
        ax.scatter(xs, ys, color=col, s=18, alpha=0.7)
        ax.scatter([xs[0]], [ys[0]], color=col, s=70, marker="o", edgecolors="white", zorder=3)  # start
        ax.scatter([xs[-1]], [ys[-1]], color=col, s=120, marker="*", edgecolors="white", zorder=3)  # end
        ax.annotate(f"a{a}", (xs[-1], ys[-1]), fontsize=8, color=col, xytext=(4, 4), textcoords="offset points")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k)
               for k, c in legend.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="#555", markerfacecolor="#555", markersize=7, label="start"),
                plt.Line2D([0], [0], marker="*", color="#555", markerfacecolor="#555", markersize=11, label="end")]
    ax.legend(handles=handles, fontsize=8)
    ax.set_title(f"Response trajectories over time (color = {by})\neach path is one agent, t1 → tN")
    ax.set_xlabel("semantic axis 1"); ax.set_ylabel("semantic axis 2")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_self_drift(st, by="role", ref="first", show=True):
    """Per-agent language drift over rounds. ref='first' plots similarity to the
    agent's first response (falls = evolving away); 'prev' plots round-to-round."""
    import matplotlib.pyplot as plt
    import textdyn
    setup_style()
    sd = textdyn.self_drift(st)
    if not sd:
        return None
    cmap, legend = _group_color(st, by)
    key = "cos_to_first" if ref == "first" else "cos_to_prev"
    fig, ax = plt.subplots(figsize=(8, 4))
    for a, d in sd.items():
        xs = [s["round"] for s in d["series"]]; ys = [s[key] for s in d["series"]]
        ax.plot(xs, ys, "-o", color=cmap.get(a, "#888"), ms=3, alpha=0.75)
    handles = [plt.Line2D([0], [0], color=c, label=k) for k, c in legend.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_xlabel("round")
    ax.set_ylabel("cosine similarity to " + ("first response" if ref == "first" else "previous response"))
    ax.set_title("How each agent's language drifts over time\n(lower = more change)")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_group_similarity(st, by="role", show=True):
    """Within-group vs between-group response similarity over rounds."""
    import matplotlib.pyplot as plt, pandas as pd
    import textdyn
    setup_style()
    rows = textdyn.group_similarity_over_time(st, by=by)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [c for c in df.columns if c != "round"]
    for k, c in enumerate(cols):
        ax.plot(df["round"], df[c], "-o", color=PALETTE[k % len(PALETTE)], ms=3, label=c.replace("_", " "))
    ax.set_xlabel("round"); ax.set_ylabel("mean cosine similarity"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    ax.set_title(f"Do {by} groups talk alike? within vs. between, over time")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_similarity_heatmap(st, rnd, show=True):
    """Agent-by-agent response similarity at one round."""
    import matplotlib.pyplot as plt, numpy as np
    import textdyn
    setup_style()
    ids, M = textdyn.pairwise_at_round(st, rnd)
    if len(ids) < 2:
        return None
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(ids), 1 + 0.6 * len(ids)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([f"a{i}" for i in ids], rotation=0)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"a{i}" for i in ids])
    for (yi, xi), v in np.ndenumerate(M):
        ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    ax.set_title(f"Response similarity between agents, round {rnd}")
    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_model_within_swarm(rows, show=True):
    """Heterogeneous swarm: cooperation / betrayal / outcome by model, when
    different models played different agents in one game. `rows` from
    llm_game.by_model_behavior."""
    import matplotlib.pyplot as plt
    setup_style()
    if not rows:
        return None
    models = [r["model"] for r in rows]
    import numpy as np
    x = np.arange(len(models))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, key, title in zip(ax, ["gifts_per_agent", "defects_per_agent", "mean_score"],
                             ["Cooperation (gifts/agent)", "Betrayal (defects/agent)", "User outcome (mean score)"]):
        bars = a.bar(x, [r[key] for r in rows], color=[PALETTE[k % len(PALETTE)] for k in range(len(models))])
        _labels(a, bars, "{:.2f}")
        a.set_xticks(x); a.set_xticklabels(models, rotation=20, ha="right"); a.set_title(title)
    fig.suptitle("Within one swarm: how each model behaved", fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_composition(df, show=True):
    """Homogeneous (one model) vs. mixed swarm: cooperation, betrayal, outcome,
    leader concentration. `df` needs a 'composition' column (from the swarm-
    composition experiment)."""
    import matplotlib.pyplot as plt, numpy as np
    setup_style()
    if "composition" not in df or df.empty:
        return None
    d = df[df.get("d_evil_budget", 1) > 0] if "d_evil_budget" in df else df
    order = sorted(d["composition"].unique(), key=lambda s: (s.startswith("mixed"), s))
    metrics = [("cooperation_rate", "Cooperation (gifts share)"),
               ("defection_rate", "Betrayal (defections share)"),
               ("mean_score", "User outcome"),
               ("indegree_centralization", "Leader concentration")]
    metrics = [(k, t) for k, t in metrics if k in d]
    fig, ax = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.9))
    if len(metrics) == 1:
        ax = [ax]
    for a, (key, title) in zip(ax, metrics):
        g = d.groupby("composition")[key].agg(["mean", "sem"]).reindex(order)
        colors = [BAD if c.startswith("mixed") else PALETTE[k % len(PALETTE)] for k, c in enumerate(order)]
        bars = a.bar(range(len(order)), g["mean"], yerr=g["sem"], capsize=3, color=colors)
        _labels(a, bars, "{:.2f}")
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right"); a.set_title(title)
    fig.suptitle("Swarm composition: homogeneous models vs. mixed (red)", fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_defense_comparison(df, show=True):
    """Damage to the swarm by defense against a malicious insider (lower is
    better), plus the bribe outcome. `df` has columns defense, damage,
    keys_burned, bribes_accepted, broken_promises."""
    import matplotlib.pyplot as plt, numpy as np
    setup_style()
    if "defense" not in df or df.empty:
        return None
    g = df.groupby("defense").mean(numeric_only=True)
    g = g.sort_values("damage", ascending=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.5 + 0.5 * len(g)))
    colors = []
    for d in g.index:
        if d == "provenance":
            colors.append(BAD)            # highlight: provenance fails
        elif "+" in d:
            colors.append(GOOD)           # highlight: the combined defense
        else:
            colors.append(PALETTE[0])
    y = np.arange(len(g))
    bars = ax[0].barh(y, 100 * g["damage"].values, color=colors)
    for b, v in zip(bars, g["damage"].values):
        ax[0].text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{100*v:.0f}%", va="center", fontsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(g.index, fontsize=8)
    ax[0].set_xlabel("swarm damage (drop in completion, %)")
    ax[0].set_title("Defenses vs. a malicious insider\nred = provenance (fails); green = combined defense")
    ax[0].invert_yaxis()
    if "bribes_accepted" in g and "broken_promises" in g:
        w = 0.4
        ax[1].barh(y - w/2, g["bribes_accepted"].values, w, color=BAD, label="bribes accepted (induced betrayals)")
        ax[1].barh(y + w/2, g["broken_promises"].values, w, color=WARN, label="promises broken (never paid)")
        ax[1].set_yticks(y); ax[1].set_yticklabels(g.index, fontsize=8); ax[1].invert_yaxis()
        ax[1].set_xlabel("per game"); ax[1].set_title("Bribe-and-renege outcomes"); ax[1].legend(fontsize=7)
    fig.suptitle("Malicious insider: lie + bribe-to-betray + renege", fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
