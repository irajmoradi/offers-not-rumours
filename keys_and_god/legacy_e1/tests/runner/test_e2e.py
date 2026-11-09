"""End-to-end (no GPU): python -m swarm.run on configs/demo.yaml with the FakeLLM backend."""

from __future__ import annotations

import json
from pathlib import Path

from swarm.analysis.ripple import ripple
from swarm.runlog.trace import load_run
from swarm.runner.run import main


def test_fake_run_completes_with_report(fake_run: Path) -> None:
    man = json.loads((fake_run / "manifest.json").read_text())
    assert man["status"] == "done"
    assert {g["label"] for g in man["games"]} == {"g00-god", "g00-nogod"}
    assert all(g["status"] == "done" for g in man["games"])
    html = (fake_run / "report.html").read_text()
    assert html.startswith("<!doctype html>") and "data:image/png;base64," in html
    for section in ('id="summary"', 'id="ripple"', 'id="timeline"', 'id="network"', 'id="god"'):
        assert section in html
    assert "<script" not in html.lower()          # static, self-contained


def test_fake_run_god_fires_and_fork_is_paired(fake_run: Path) -> None:
    run = load_run(fake_run)
    pair = run.pairs()[0]
    a, b = pair["god"], pair["nogod"]
    powers = [e.data for e in a.of_type("god_power")]
    assert sorted(p.power for p in powers) == ["cot_control", "message_fraud"]
    assert all(p.ok for p in powers)
    assert not b.of_type("god_power")
    assert a.game_id == b.game_id and a.meta["seed"] == b.meta["seed"]
    res = ripple(a, b)
    assert res.fork_round == 3
    assert res.pre_fork_frac == 0.0                 # common random numbers: identical before the fork
    assert any(v is not None for v in res.first_div.values())


def test_fake_run_leak_oracle_scanned_every_request(fake_run: Path) -> None:
    run = load_run(fake_run)
    for g in run.games:
        assert not g.parse_errors
        n_calls = len(g.llm_calls)
        assert n_calls == len(g.of_type("turn")) > 0
        assert g.meta["leak_checked"] == n_calls and g.meta["leak_hits"] == 0


def test_fake_run_logs_have_no_secrets_and_manifest_has_provenance(fake_run: Path) -> None:
    man = json.loads((fake_run / "manifest.json").read_text())
    assert man["models"]["agent"]["backend"] == "fake"
    assert man["prompts"]["agent"]["hash"]
    assert man["seeds"]["per_game"] and "git" in man and "usage" in man
    assert man["usage"]["total_calls"] > 0
    for p in fake_run.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".html", ".yaml"}:
            txt = p.read_text(errors="ignore")
            assert "sk-or-" not in txt and "Bearer " not in txt


def test_cli_synth_returns_zero(tmp_path: Path) -> None:
    rc = main(["--config", "configs/demo.yaml", "--engine", "synth", "--out", str(tmp_path),
               "--run-id", "cli", "game.n_agents=6", "game.max_rounds=5"])
    assert rc == 0
    assert (tmp_path / "cli" / "report.html").exists()


def test_synth_run_has_noise_branch(synth_run: Path) -> None:
    run = load_run(synth_run)
    assert set(run.pairs()[0]) == {"god", "nogod", "noise"}
    assert (synth_run / "report.html").exists()
