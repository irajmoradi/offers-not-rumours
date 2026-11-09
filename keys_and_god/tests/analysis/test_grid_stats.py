"""grid_stats: paired deltas, sign-flip p, bootstrap CI, interaction pairing by seed."""

from __future__ import annotations

import json
import random

from swarm.analysis import grid_stats as gs


def _run(tmp_path, name, welfare_by_seed, run_seed=0):
    rd = tmp_path / name
    games = []
    for i, branches in welfare_by_seed.items():
        for br, w in branches.items():
            d = rd / "games" / f"g{i:02d}-{br}"
            d.mkdir(parents=True)
            k = round(w * 3)   # 1 agent, 3 needs
            (d / "llm_calls.jsonl").write_text('{"error": null}\n')
            (d / "events.jsonl").write_text('{"type":"turn","data":{"parse_ok":true,"errors":[],"actions":[{"kind":"noop"}]}}\n')
            (d / "outcome.json").write_text(json.dumps({"rounds_played": 5, "per_agent": {
                "a": {"score": k / 3, "won_round": 2 if k == 3 else None, "burned": [], "lost": []}}}))
            games.append({"status": "done", "run_seed": run_seed, "game_idx": i, "branch": br,
                          "dir": f"games/g{i:02d}-{br}"})
    rd.mkdir(exist_ok=True)
    (rd / "manifest.json").write_text(json.dumps({"games": games}))
    return rd


def test_deltas_signflip_and_interaction(tmp_path, capsys):
    a = gs.load_run(_run(tmp_path, "a", {i: {"god": 1 / 3, "nogod": 1.0, "noise": 1.0} for i in range(6)}))
    d = list(gs.deltas(a, "welfare").values())
    assert len(d) == 6 and all(abs(x + 2 / 3) < 1e-9 for x in d)
    assert gs.signflip_p(d, random.Random(0)) == 2 / 64          # all-same-sign: 2 of 2^6 flips
    lo, hi = gs.boot_ci(d, 200, random.Random(0))
    assert abs(lo + 2 / 3) < 1e-9 and abs(hi + 2 / 3) < 1e-9
    _run(tmp_path, "b", {i: {"god": 1.0, "nogod": 1.0} for i in range(6)})
    gs.main([str(tmp_path / "a"), str(tmp_path / "b"), "--boot", "200"])
    out = capsys.readouterr().out
    assert "noise-nogod +0.000" in out and "paired by seed, n=6" in out and "welfare   +0.667" in out


def test_unhealthy_branch_excludes_its_seed(tmp_path, capsys):
    rd = _run(tmp_path, "c", {i: {"god": 1 / 3, "nogod": 1.0} for i in range(3)})
    bad = rd / "games" / "g01-god"
    (bad / "llm_calls.jsonl").write_text('{"error": "LLMError: HTTP 400"}\n' * 3 + '{"error": null}\n')
    (bad / "events.jsonl").write_text(
        '{"type":"turn","data":{"parse_ok":true,"errors":["llm_error:LLMError"],"actions":[{"kind":"noop"}]}}\n' * 4)
    r = gs.load_run(rd)
    assert sorted(r["seeds"]) == [0, 2] and len(r["unhealthy"]) == 1 and "llm errors 3/4" in r["unhealthy"][0]
    assert sorted(gs.load_run(rd, keep_unhealthy=True)["seeds"]) == [0, 1, 2]
    gs.main([str(rd), "--boot", "50"])
    assert "UNHEALTHY games/g01-god" in capsys.readouterr().out


def test_t_ci_is_wider_than_percentile_bootstrap_for_one_outlier_seed():
    """Review: 7 zeros and one 2.0 -> the bootstrap's lower bound is exactly 0 (looks 'positive'),
    the t interval covers 0 on both sides."""
    xs = [0.0] * 7 + [2.0]
    lo, hi = gs.t_ci(xs)
    assert lo < 0 < hi and abs((lo + hi) / 2 - 0.25) < 1e-9
    blo, _ = gs.boot_ci(xs, 2000, random.Random(0))
    assert blo == 0.0
    assert gs.t_ci([1.0]) != gs.t_ci([1.0])        # nan for n < 2
