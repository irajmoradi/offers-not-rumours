"""Shared fixtures for runner/analysis/viz tests: one FakeLLM end-to-end run of the REAL engine
(configs/demo.yaml scaled down, paired fork on) and one synth-engine run, each built once per session."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from swarm.config import REPO_ROOT
from swarm.runner.run import load_experiment, run_experiment

DEMO = REPO_ROOT / "configs" / "demo.yaml"
FAKE_OVERRIDES = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8",
                  "game.max_rounds=6"]


def _run(tmp: Path, run_id: str, overrides: list[str]) -> Path:
    cfg, opts = load_experiment(DEMO, overrides)
    return asyncio.run(run_experiment(cfg, opts, overrides=overrides, out_root=tmp, run_id=run_id))


@pytest.fixture(scope="session")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real engine + LLMAgent + ScriptedGod + LeakOracle, FakeLLM backend, paired fork, report."""
    return _run(tmp_path_factory.mktemp("runs"), "fake_e2e", FAKE_OVERRIDES)


@pytest.fixture(scope="session")
def synth_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synth engine (no LLM at all), paired fork + noise replica."""
    return _run(tmp_path_factory.mktemp("runs"), "synth_e2e",
                ["runner.engine=synth", "runner.fork.noise_replica=true", "game.max_rounds=8"])
