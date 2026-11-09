"""Network-aware visuals: layout, hubs, known-name series (must match swarm.analysis.network), dashboard section,
overview figure; non-network runs keep their old encodings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swarm.viz.animate import NET_RX, NET_RY, hub_agents, network_layout
from swarm.viz.metrics import load_branch, network_cell, round_series


def test_network_layout_is_deterministic_spaced_and_in_box() -> None:
    agents = [f"a{i}" for i in range(10)]
    edges = [[f"a{i}", f"a{(i + 1) % 10}"] for i in range(10)] + [["a0", "a5"]]
    p1, p2 = network_layout(agents, edges), network_layout(agents, edges)
    assert all(np.allclose(p1[a], p2[a]) for a in agents)
    xy = np.array([p1[a] for a in agents])
    assert (np.abs(xy[:, 0]) <= NET_RX + 1e-9).all() and (np.abs(xy[:, 1]) <= NET_RY + 1e-9).all()
    g = np.array((0.78, 0.34))
    d = [np.hypot(*((xy[i] - xy[j]) / g)) for i in range(10) for j in range(i + 1, 10)]
    assert min(d) > 0.6                                   # pushed apart (box clipping can leave a little overlap)


def test_hub_agents() -> None:
    assert hub_agents({}) == set()
    assert hub_agents({"a": 6, "b": 2, "c": 2, "d": 1, "e": 3}) == {"a", "e"}   # 80th pct = 3
    assert hub_agents({"a": 2, "b": 2}) == set()                     # nobody reaches degree 3
    assert len(hub_agents({c: 5 for c in "abcdef"})) == 3            # at most 3


def test_known_share_matches_analysis(net_run: Path) -> None:
    from swarm.analysis.network import diffusion, load_game

    d = net_run / "games" / "g00-god"
    b = load_branch(d)
    assert b.network and b.network["mode"] == "known_ids" and b.known
    curve = diffusion(load_game(d))["coverage_by_round"]
    for r, v in curve.items():
        assert abs(b.known_share(r) - v) < 1e-3
    rs = round_series(b)
    assert all(rs[i]["known_share"] <= rs[i + 1]["known_share"] + 1e-12 for i in range(len(rs) - 1))
    assert sum(b.network["degree"].values()) == 2 * len(b.network["edges"])


def test_non_network_run_has_no_network(fake_run: Path) -> None:
    b = load_branch(fake_run / "games" / "g00-god")
    assert b.network is None and b.known == {} and b.known_share(3) is None
    assert np.isnan(round_series(b)[1]["known_share"])
    assert network_cell(fake_run) is None


def test_network_dashboard_and_overview(net_run: Path, fake_run: Path, tmp_path: Path) -> None:
    from swarm.viz.dashboard import build
    from swarm.viz.paper_figs import fig_network_overview

    nc = network_cell(net_run)
    assert nc and nc["graphs"] and nc["centrality_rows"] and nc["mode"] == "known_ids"
    build([("net", str(net_run)), ("plain", str(fake_run))], tmp_path / "d.html")
    page = (tmp_path / "d.html").read_text()
    assert page.count("<h3>Network (") == 1 and "Name diffusion" in page
    paths, stats = fig_network_overview([("ER", str(net_run))], tmp_path, note="dry run")
    assert all(p.stat().st_size > 2000 for p in paths) and stats and stats[0][1]["n"] >= 1


def test_network_video_frame(net_run: Path, tmp_path: Path) -> None:
    from swarm.viz.animate import WorldVideo

    wv = WorldVideo(net_run / "games" / "g00-god", net_run / "games" / "g00-nogod", "h")
    assert "network: Erdős–Rényi, known_ids" in wv.header
    wv.draw(2, 2)
    wv.fig.savefig(tmp_path / "f.png", dpi=50)
    assert (tmp_path / "f.png").stat().st_size > 5000


def test_title_card_facts_come_from_the_logs(net_run: Path) -> None:
    import json

    from swarm.viz.animate import world_facts

    d = net_run / "games" / "g00-god"
    f = world_facts(d)
    start = next(json.loads(ln)["data"] for ln in open(d / "events.jsonl") if '"type":"game_start"' in ln)
    assert f["n"] == len(start["agents"]) and f["model"] == "FakeLLM DRY RUN"
    assert f["keys"] == len(next(iter(start["agents"].values()))["keys"]) and f["k_desc"].startswith("Here K=")
