"""Shared fixtures (WP0): tiny_cfg, fake_llm, tmp_run_dir, fixed_state. Auto-skips gpu/openrouter tests."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest
from hypothesis import settings

from swarm.config import RunConfig, from_dict
from swarm.llm.base import FakeLLM
from swarm.schema import GameState
from tests.fixtures.fixed_state import make_fixed_state

# Internal rule errors must surface in tests instead of becoming silent failed actions (rules.apply
# reads this at call time).
os.environ.setdefault("SWARM_STRICT_RULES", "1")

settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("deep", max_examples=500, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def _vllm_up() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu = any("gpu" in i.keywords for i in items) and _vllm_up()
    orr = bool(os.environ.get("OPENROUTER_API_KEY"))
    for item in items:
        if "gpu" in item.keywords and not gpu:
            item.add_marker(pytest.mark.skip(reason="vLLM not reachable on localhost:8000"))
        if "openrouter" in item.keywords and not orr:
            item.add_marker(pytest.mark.skip(reason="OPENROUTER_API_KEY not set"))


TINY = {
    "run": {"name": "tiny", "seed": 7},
    "game": {"n_agents": 6, "max_rounds": 5},
    "roles": {r: {"backend": "fake"} for r in ("agent", "god", "judge", "summarizer")},
    "context": {"tokenizer": "words"},
    "llm": {"cache": {"mode": "off"}},
}


@pytest.fixture
def tiny_cfg() -> RunConfig:
    return from_dict(TINY)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / "tiny_run"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def fixed_state() -> GameState:
    return make_fixed_state()
