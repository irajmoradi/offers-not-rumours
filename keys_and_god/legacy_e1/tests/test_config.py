from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swarm.config import REPO_ROOT, RunConfig, expand_sweep, from_dict, load, parse_override


def test_defaults_validate_and_match_lead_decisions():
    c = RunConfig()
    assert c.game.turn_mode == "sequential" and c.game.defect_gain == "mint"
    assert c.game.wrong_defect_penalty == "lose_need" and c.game.winner_policy == "continue"
    v = c.game.view_log
    assert (v.content, v.actor_identity, v.reveal_provenance) == ("actions_only", "shown", False)
    assert v.defect_visibility == "victim_then_delayed" and v.defect_delay_rounds == 3 and v.grep_enabled
    assert c.game.window_turns == c.game.n_agents
    assert c.reasoning.cot_replay_window == 0 and c.reasoning.cot_mode in ("none", "native", "explicit")
    assert c.god.powers.cot_control.targets == ["scratchpad"] and c.god.powers.cot_control.mode == "replace"
    assert c.god.tool_call_safety_ceiling == 50
    assert c.context.policy == "full_truncate" and c.context.budget_tokens == 14000
    assert c.roles.agent.max_tokens == 700
    assert c.llm.openrouter.enabled is False and c.llm.openrouter.global_cap_usd == 20.0
    assert c.llm.openrouter.model == "qwen/qwen3.6-35b-a3b"
    assert c.model_for("agent") == ("vllm", "Qwen/Qwen3-8B")


def test_base_yaml_equals_code_defaults():
    assert load(REPO_ROOT / "configs" / "base.yaml") == RunConfig()


def test_extra_forbid_rejects_typo():
    with pytest.raises(ValidationError):
        from_dict({"game": {"n_agent": 10}})


def test_overrides_parse_types():
    c = from_dict({}, ["game.n_agents=10", "game.view_log.reveal_provenance=true",
                       "run.name=foo"])
    assert c.game.n_agents == 10 and c.game.view_log.reveal_provenance is True
    assert c.run.name == "foo"
    assert parse_override("a.b=3") == (["a", "b"], 3)
    assert parse_override("x=[1,2]") == (["x"], [1, 2])
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_extends_deep_merge_and_list_replace(tmp_path: Path):
    base = {"game": {"n_agents": 8, "max_rounds": 9},
            "god": {"powers": {"cot_control": {"targets": ["scratchpad"]}}}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"extends": "base.yaml", "game": {"n_agents": 12}}))
    c = load(tmp_path / "exp.yaml", ["game.max_rounds=11"])
    assert (c.game.n_agents, c.game.max_rounds) == (12, 11)
    (tmp_path / "exp2.yaml").write_text(yaml.safe_dump({"extends": "exp.yaml"}))
    with pytest.raises(ValueError, match="one level"):
        load(tmp_path / "exp2.yaml")


@pytest.mark.parametrize("over, msg", [
    (["god.powers.cot_control.targets=[cot]"], "cot_replay_window"),
    (["god.powers.cot_control.targets=[cot]", "reasoning.cot_replay_window=2", "reasoning.cot_mode=none"],
     "cot_mode"),
    (["god.powers.cot_control.targets=[summary]"], "summarizing"),
    (["reasoning.scratchpad.enabled=false"], "scratchpad"),
    (["context.budget_tokens=16000"], "max_model_len"),
    (["roles.agent.backend=openrouter"], "openrouter.enabled"),
    (["context.policy=rolling_summary"], "stub"),
    (["context.policy=hybrid"], "stub"),
    (["game.immediate_reply=true"], "not implemented"),
    (["game.observe_mode=tools"], ""),
    (["game.actions_per_turn=3"], ""),
    (["game.turn_mode=parallel"], ""),
])
def test_cross_field_errors(over, msg):
    with pytest.raises(ValidationError, match=msg or None):
        from_dict({}, over)


def test_valid_alternatives():
    c = from_dict({}, ["god.powers.cot_control.targets=[cot, scratchpad]", "reasoning.cot_replay_window=2",
                       "context.policy=window", "game.turn_mode=simultaneous", "game.defect_gain=steal",
                       "game.wrong_defect_penalty=delay", "game.winner_policy=sit_out",
                       "llm.openrouter.enabled=true", "roles.god.backend=openrouter",
                       "game.view_log.window_turns=5"])
    assert c.model_for("god") == ("openrouter", "qwen/qwen3.6-35b-a3b")
    assert c.game.window_turns == 5


def test_sweep_expansion_and_hash():
    c = from_dict({"sweep": {"god.budget.k": [0, 1, 3], "run.seed": [0, 1]}})
    cells = expand_sweep(c)
    assert len(cells) == 6
    assert {(p["god.budget.k"], p["run.seed"]) for p, _ in cells} == {(k, s) for k in (0, 1, 3) for s in (0, 1)}
    assert all(cc.sweep == {} for _, cc in cells)
    assert len({cc.config_hash() for _, cc in cells}) == 6
    assert RunConfig().config_hash() == RunConfig().config_hash()
    assert from_dict({}, ["run.seed=1"]).config_hash() != RunConfig().config_hash()


def test_config_hash_ignores_global_cap():
    """The $2 -> $6 cap raise must not re-key games: base hash pinned from commit 1993b5c (cap $2)."""
    from swarm.runner.run import load_experiment

    pinned = {"base": "b442ad7af4a87da8bb2ff289c1e794cd7b98f44742c8a7066966921cfad35ee1"}
    assert RunConfig().config_hash() == pinned["base"]
    for name in ("base", "demo", "demo_openrouter", "demo_or_10x50_sim"):
        hs = {load_experiment(REPO_ROOT / "configs" / f"{name}.yaml",
                              [] if cap is None else [f"llm.openrouter.global_cap_usd={cap}"])[0].config_hash()
              for cap in (None, 2.0, 6.0, 100.0)}
        assert len(hs) == 1, name
    assert load_experiment(REPO_ROOT / "configs" / "base.yaml", [])[0].config_hash() == pinned["base"]


OPERATIONAL_OVERRIDES = [
    "run.name=renamed", "run.n_games=7", "run.out_dir=elsewhere", "run.max_concurrency_games=4",
    "llm.budget_usd=0.75", "llm.vllm.base_url=http://x:1/v1", "llm.vllm.max_concurrency=3",
    "llm.openrouter.max_concurrency=16", "llm.openrouter.global_cap_usd=9", "llm.openrouter.ledger_path=l.jsonl",
    "llm.openrouter.price_in_per_mtok=0.02", "llm.openrouter.price_out_per_mtok=0.05",
    "llm.openrouter.base_url=https://x/v1", "llm.openrouter.api_key_env=OTHER_KEY",
    "llm.cache.mode=read_only", "llm.cache.path=c/x.sqlite", "llm.retry.max_retries=2", "llm.retry.timeout_s=9",
]
SEMANTIC_OVERRIDES = [
    "run.seed=3", "run.replica=r1", "run.branch=b", "game.n_agents=10", "game.max_rounds=12",
    "game.turn_mode=simultaneous", "game.board.anonymous=true", "reasoning.cot_mode=none",
    "roles.agent.model=openai/gpt-oss-120b", "roles.agent.max_tokens=900", "roles.agent.temperature=0.2",
    "llm.openrouter.model=openai/gpt-oss-120b", "god.kind=none", "god.budget.k=1",
    "context.budget_tokens=3000", "llm.openrouter.reasoning_effort=low", "llm.openrouter.provider_sort=price",
]


@pytest.mark.parametrize("ov", OPERATIONAL_OVERRIDES)
def test_config_hash_ignores_operational_knobs(ov):
    """CODE_REVIEW #6: caps, budgets, concurrency, paths, retry, price estimates and run labels are
    not game semantics; changing them keeps the hash (game ids, seeds, cache keys)."""
    base = from_dict({})
    assert from_dict({}, [ov]).config_hash() == base.config_hash()
    sim = from_dict({}, ["game.turn_mode=simultaneous"])
    assert from_dict({}, ["game.turn_mode=simultaneous", ov]).config_hash() == sim.config_hash()


@pytest.mark.parametrize("ov", SEMANTIC_OVERRIDES)
def test_config_hash_keeps_game_semantics(ov):
    assert from_dict({}, [ov]).config_hash() != RunConfig().config_hash()


def test_config_hash_all_operational_at_once_and_new_fields_default_off():
    allops = from_dict({}, OPERATIONAL_OVERRIDES)
    assert allops.config_hash() == RunConfig().config_hash()
    d = RunConfig().model_dump(mode="json")
    assert d["llm"]["openrouter"]["reasoning_effort"] is None and d["llm"]["openrouter"]["provider_sort"] is None


def test_salvage_truncated_default_off_and_hashed_only_when_on():
    assert RunConfig().reasoning.salvage_truncated is False
    on = from_dict({}, ["reasoning.salvage_truncated=true"])
    assert on.config_hash() != RunConfig().config_hash()
