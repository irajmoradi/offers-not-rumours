"""Re-use the session-scoped run fixtures of tests/runner, plus a FakeLLM run with a social network."""

from pathlib import Path

import pytest

from tests.runner.conftest import FAKE_OVERRIDES, _run, fake_run, synth_run  # noqa: F401


@pytest.fixture(scope="session")
def net_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real engine + FakeLLM, Erdos-Renyi contact graph in known_ids mode (names learned over the game)."""
    return _run(tmp_path_factory.mktemp("runs"), "net_e2e",
                FAKE_OVERRIDES + ["game.network.topology=erdos_renyi", "game.network.mode=known_ids"])
