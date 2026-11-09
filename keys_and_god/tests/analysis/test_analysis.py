"""Outcomes and ripple over real FakeLLM / synth runs (fixtures from tests/runner/conftest.py)."""

from __future__ import annotations

from pathlib import Path

from swarm.analysis.outcomes import agent_table, game_summary, god_actions, god_touched
from swarm.analysis.ripple import action_sig, ripple
from swarm.runlog.trace import load_run
from swarm.schema import SendMessage


def test_summary_and_agent_table(fake_run: Path) -> None:
    run = load_run(fake_run)
    for g in run.games:
        s = game_summary(g)
        assert s["status"] == "done" and s["rounds"] == 6
        assert 0.0 <= s["welfare"] <= 1.0
        rows = agent_table(g)
        assert len(rows) == 8
    a = run.pairs()[0]["god"]
    assert game_summary(a)["forged"] == 1 and game_summary(a)["overwrites"] == 1
    assert {p["power"] for p in god_actions(a)} == {"message_fraud", "cot_control"}
    touched = god_touched(a)
    assert sum(len(v) for v in touched.values()) == 2


def test_ripple_self_is_zero(fake_run: Path) -> None:
    a = load_run(fake_run).pairs()[0]["god"]
    res = ripple(a, a)
    assert all(f == 0 for f in res.frac) and all(v is None for v in res.first_div.values())


def test_ripple_series_shapes(fake_run: Path) -> None:
    br = load_run(fake_run).pairs()[0]
    res = ripple(br["god"], br["nogod"])
    n = len(res.rounds)
    assert n == 6 and len(res.frac) == len(res.cum) == len(res.welfare_a) == len(res.welfare_b) == n
    assert all(x <= y for x, y in zip(res.cum, res.cum[1:], strict=False))          # cumulative is monotone
    assert set(res.victims) <= set(res.agents)
    s = res.summary()
    assert s["fork_round"] == 3 and s["pre_fork_frac"] == 0.0


def test_noise_floor_available(synth_run: Path) -> None:
    br = load_run(synth_run).pairs()[0]
    noise = ripple(br["nogod"], br["noise"], fork_from=br["god"])
    assert len(noise.rounds) == 8


def test_action_sig_coarse_ignores_text() -> None:
    a, b = SendMessage(to="x", text="hello"), SendMessage(to="x", text="bye")
    assert action_sig(a) == action_sig(b)
    assert action_sig(a, "strict") != action_sig(b, "strict")
