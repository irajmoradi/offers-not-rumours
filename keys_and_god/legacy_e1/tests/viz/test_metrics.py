"""Shared metrics layer: per-round series, outcome metrics (must match swarm.analysis.grid_results), health,
cell loading, statistics, world pairing; plus a dashboard smoke test."""

from __future__ import annotations

import json
from pathlib import Path

from swarm.viz.metrics import (
    ROUND_FIELDS,
    branch_health,
    deltas,
    game_metrics,
    load_branch,
    load_cell,
    mde,
    paired_or_unpaired,
    round_series,
    summarize,
    world_hash,
)


def test_round_series_cumulative_and_consistent(fake_run: Path) -> None:
    b = load_branch(fake_run / "games" / "g00-god")
    rs = round_series(b, ever={r: set() for r in range(1, b.R + 1)})
    assert len(rs) == b.R + 1 and set(ROUND_FIELDS) <= set(rs[0])
    for k in ("gives", "defects", "board_posts", "dms", "god_moves", "winners"):
        assert all(rs[i][k] <= rs[i + 1][k] for i in range(b.R)), k          # cumulative
    assert rs[-1]["gives"] == len(b.gives) and rs[-1]["defects"] == len(b.defects)
    assert rs[-1]["god_moves"] == len(b.powers)
    assert abs(rs[-1]["welfare"] - b.welfare(b.R)) < 1e-12
    assert rs[0]["gives"] == 0 and rs[0]["welfare"] == 0


def test_game_metrics_match_grid_results(fake_run: Path) -> None:
    from swarm.analysis.grid_results import action_counts
    from swarm.analysis.grid_stats import game_metrics as gs_metrics

    d = fake_run / "games" / "g00-god"
    m = game_metrics(d)
    ref = {**gs_metrics(d), **action_counts(d)}
    for k, v in ref.items():
        assert abs(m[k] - v) < 1e-12, k


def test_branch_health_counts(fake_run: Path) -> None:
    d = fake_run / "games" / "g00-god"
    h = branch_health(d)
    n_lines = sum(1 for _ in open(d / "llm_calls.jsonl", encoding="utf-8"))
    assert h["calls"] == n_lines and h["paid"] + h["cache_hits"] + h["errors"] == h["calls"]
    assert h["turns"] > 0 and 0 <= h["parse_failures"] <= h["turns"] and h["unhealthy"] == []


def test_load_cell_and_deltas(synth_run: Path) -> None:
    c = load_cell(synth_run)
    assert c["worlds"] and c["status"].get("done", 0) >= 2
    w = next(iter(c["worlds"].values()))
    assert {"god", "nogod"} <= set(w["branches"]) and "god" in w["pairs"] and w["world_hash"]
    d = deltas(c, "welfare", "god")
    i = next(iter(d))
    assert abs(d[i] - (c["worlds"][i]["branches"]["god"]["welfare"] - c["worlds"][i]["branches"]["nogod"]["welfare"])) < 1e-12
    assert all(0 <= v <= 1 for v in deltas(c, "div_final_cum", "god").values())


def test_world_hash_pairs_identical_worlds_only(synth_run: Path, fake_run: Path) -> None:
    a = load_cell(synth_run, with_health=False)
    kind, s = paired_or_unpaired(a, a, "welfare")
    if len(a["worlds"]) >= 2:
        assert kind == "paired" and s["mean"] == 0
    else:                                   # one world: nothing to pair, no CI
        assert kind == "n/a"
    g = synth_run / "games"
    assert world_hash(g / "g00-god") == world_hash(g / "g00-nogod")      # branches of one world share it
    b = load_cell(fake_run, with_health=False)                            # different config -> different worlds
    kind, _ = paired_or_unpaired(a, b, "welfare")
    assert kind in ("unpaired", "n/a")


def test_summarize_and_mde() -> None:
    s = summarize([0.1, -0.1, 0.2, 0.0, 0.05], n_boot=300)
    assert s["n"] == 5 and s["tlo"] < s["mean"] < s["thi"] and 0 <= s["p"] <= 1
    assert abs(mde(0.19, 16) - (2.131 + 0.8416) * 0.19 / 4) < 1e-9


def test_dashboard_self_contained(synth_run: Path, tmp_path: Path) -> None:
    from swarm.viz.dashboard import build

    empty = tmp_path / "empty_run"
    (empty / "games").mkdir(parents=True)
    info = build([("synth", synth_run), ("empty", empty)], tmp_path / "d.html")
    page = (tmp_path / "d.html").read_text()
    assert info["cells"] == ["synth"] and info["skipped"] == ["empty"]
    assert "<svg" in page and "95% t-CI" in page and "re-run" in page
    assert not any(x in page for x in ('src="http', "src='http", 'href="http', "url(http", "<script src", "<link"))
    assert "<metadata" not in page
    c = load_cell(synth_run, with_health=False)
    mean = json.loads(json.dumps(summarize(list(deltas(c, "welfare", "god").values()))["mean"]))
    assert f"{mean:+.3f}".replace("-", "−") in page                    # numbers come from the logs


def test_cell_round_means(synth_run: Path) -> None:
    from swarm.viz.metrics import cell_round_means, worlds

    m = cell_round_means(synth_run)
    assert {"god", "nogod"} <= set(m)
    ws = worlds(synth_run)
    finals = [load_branch(b["god"]).welfare(load_branch(b["god"]).R) for b in ws.values()]
    assert abs(m["god"]["welfare"][-1] - sum(finals) / len(finals)) < 1e-9
    assert m["god"]["welfare"][0] == 0 and m["god"]["n"] == [float(len(ws))]


def test_cell_spec_joins_external_branch(synth_run: Path) -> None:
    from swarm.viz.metrics import cell_deltas, parse_spec, worlds

    sp = parse_spec(f"{synth_run},{synth_run}@noise#0-0")
    assert sp.run == synth_run and sp.extra == (("noisefork", synth_run, "noise"),) and sp.seeds == frozenset({0})
    ws = worlds(sp)
    assert set(ws) <= {0} and all(b["noisefork"] == b["noise"] for b in ws.values())
    a = cell_deltas(sp, "god", "noisefork")
    b = cell_deltas(synth_run, "god", "noise")
    assert a == {i: b[i] for i in a}
    c = load_cell(sp, with_health=False)
    for i, w in c["worlds"].items():
        assert "noisefork" in w["pairs"]
        d = deltas(c, "div_post_fork", "god", "noisefork")[i]
        assert abs(d - (w["pairs"]["god"]["div_post_fork"] - w["pairs"]["noisefork"]["div_post_fork"])) < 1e-12


def test_fig5_sweep_renders(synth_run: Path, tmp_path: Path) -> None:
    from swarm.viz.paper_figs import fig5_sweep

    sp = f"{synth_run},{synth_run}@noise"
    paths, stats = fig5_sweep([("group", "cond", sp)], ("variant", str(synth_run), str(synth_run)), tmp_path)
    assert all(p.stat().st_size > 2000 for p in paths)
    design = [st for lab, st in stats if lab.startswith("design")]
    assert all(st["mean"] == 0 for st in design)          # a cell against itself: no design effect


def test_design_effect_joins_runs_and_pairs_same_worlds(synth_run: Path, fake_run: Path) -> None:
    from swarm.viz.metrics import branch_by_world, design_effect

    one = branch_by_world(str(synth_run))
    assert one and branch_by_world(f"{synth_run}+{fake_run}").keys() >= one.keys()
    de = design_effect(str(synth_run), str(synth_run))
    assert set(de["pairs"]) == set(one) and all(a == b for a, b in de["pairs"].values()) and not de["dropped"]
    other = design_effect(str(synth_run), str(fake_run))       # different worlds at the same index -> dropped
    assert not other["pairs"] or other["dropped"] == []


def test_fig6_board_mechanism_renders(synth_run: Path, tmp_path: Path) -> None:
    from swarm.viz.metrics import game_metrics
    from swarm.viz.paper_figs import fig6_board_mechanism

    m = game_metrics(synth_run / "games" / "g00-god")
    assert {"defects_correct", "defects_wrong"} <= set(m) and m["defects_correct"] >= 0
    paths, _ = fig6_board_mechanism(str(synth_run), str(synth_run), tmp_path, frozenset({0}))
    assert all(p.stat().st_size > 2000 for p in paths)
