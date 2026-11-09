"""Doctor: offline checks pass on the demo config; a bad config and a dead vLLM FAIL."""

from __future__ import annotations

from swarm.doctor import check_vllm, main, run_checks
from swarm.runner.run import load_experiment


def test_offline_demo_ok() -> None:
    checks = run_checks("configs/demo.yaml", network=False)
    assert [c.name for c in checks] == ["config", "disk", "ledger", "key"]
    assert checks[0].status == "OK"


def test_bad_config_fails() -> None:
    checks = run_checks("configs/demo.yaml", ["game.n_agents=-3"], network=False)
    assert checks[0].status == "FAIL" and len(checks) == 1
    assert main(["--offline", "game.n_agents=-3"]) == 1


def test_dead_vllm_fails() -> None:
    cfg, _ = load_experiment("configs/demo.yaml", ["llm.vllm.base_url=http://127.0.0.1:9/v1"])
    assert check_vllm(cfg, timeout=0.5).status == "FAIL"
