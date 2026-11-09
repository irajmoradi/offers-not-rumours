"""Run specs and experiment loading: paired forks share seeds (CRN); the noise replica does not."""

from __future__ import annotations

import json

import pytest

from swarm.config import REPO_ROOT
from swarm.runner.run import load_experiment
from swarm.runner.spec import RunnerOpts, build_specs

DEMO = REPO_ROOT / "configs" / "demo.yaml"


def test_demo_config_shape() -> None:
    cfg, opts = load_experiment(DEMO)
    assert cfg.game.n_agents == 20 and cfg.game.max_rounds == 15
    assert cfg.game.turn_mode == "sequential"
    assert cfg.god.kind == "scripted" and cfg.god.budget.k == 2
    assert sorted(s.power for s in cfg.god.schedule) == ["cot_control", "message_fraud"]
    assert cfg.model_for("agent")[0] == "vllm"
    assert opts.fork.enabled and opts.engine == "real"


def test_runner_overrides_split() -> None:
    cfg, opts = load_experiment(DEMO, ["runner.engine=synth", "game.max_rounds=4"])
    assert opts.engine == "synth" and cfg.game.max_rounds == 4


def test_unknown_runner_key_rejected() -> None:
    with pytest.raises(ValueError):
        RunnerOpts.from_dict({"bogus": 1})


def test_fork_pairs_share_seeds_noise_differs() -> None:
    cfg, opts = load_experiment(DEMO, ["runner.fork.noise_replica=true", "run.n_games=2"])
    groups = build_specs(cfg, opts)
    assert len(groups) == 2
    for g in groups:
        a, b, n = g
        assert (a.branch, b.branch, n.branch) == ("god", "nogod", "noise")
        assert a.god_active and not b.god_active and not n.god_active
        assert a.game_id == b.game_id != n.game_id
        assert a.world_seed == b.world_seed == n.world_seed
        assert a.cfg.run.replica == b.cfg.run.replica != n.cfg.run.replica
    assert groups[0][0].game_id != groups[1][0].game_id


def test_fork_without_god_rejected() -> None:
    cfg, opts = load_experiment(DEMO, ["god.kind=none"])
    with pytest.raises(ValueError):
        build_specs(cfg, opts)


GRID_K2 = REPO_ROOT / "configs" / "grid_oss_sim_k2.yaml"


def test_control_god_reuses_base_cell_control() -> None:
    """A sweep cell that changes only `god` and passes the base cell's god section as
    fork.control_god gets exactly the base cell's control games (game_id, config, seeds)."""
    base_cfg, base_opts = load_experiment(GRID_K2, ["run.n_games=2"])
    base = build_specs(base_cfg, base_opts)
    g1_god = base_cfg.god.model_dump(mode="json")
    over = ["run.n_games=2", "god.start_round=14", "god.budget.k=1",
            "god.schedule=[{round: 14, power: message_fraud}]"]
    plain = build_specs(*load_experiment(GRID_K2, over))
    cfg, opts = load_experiment(GRID_K2, over + [f"runner.fork.control_god={json.dumps(g1_god)}"])
    swept = build_specs(cfg, opts)
    for b, p, s in zip(base, plain, swept, strict=True):
        bn, pn, sn = (next(x for x in g if x.branch == "nogod") for g in (b, p, s))
        assert pn.game_id != bn.game_id                      # default: own control
        assert sn.game_id == bn.game_id and sn.cfg.config_hash() == bn.cfg.config_hash()
        sg = next(x for x in s if x.branch == "god")
        assert sg.game_id == bn.game_id and sg.cfg.god.start_round == 14 and sg.cfg.god.kind == "scripted"
        assert sg.world_seed == bn.world_seed


def test_control_god_default_unchanged() -> None:
    cfg, opts = load_experiment(GRID_K2, ["run.n_games=1"])
    assert opts.fork.control_god is None
    a, b, _ = build_specs(cfg, opts)[0]
    assert b.cfg.god.kind == "none" and b.cfg.god.schedule == cfg.god.schedule
    assert a.game_id == b.game_id == build_specs(cfg, RunnerOpts.from_dict(
        {"fork": {"enabled": True, "noise_replica": True, "control_god": None}}))[0][1].game_id


def test_noise_fork_branch_shares_prefix_then_resamples() -> None:
    """B'' (fork.noise_fork): same game_id/world as the pair; agent seeds equal to the control's
    before God's first scheduled round and different from it on."""
    from swarm.rng import derive_seed
    from swarm.runner.spec import fork_round

    over = ["run.n_games=1", "runner.fork.noise_fork=true", "runner.fork.noise_replica=false",
            "god.start_round=8", "god.schedule=[{round: 8, power: message_fraud}, {round: 10, power: cot_control}]"]
    cfg, opts = load_experiment(GRID_K2, over)
    assert fork_round(cfg) == 8
    a, b, nf = build_specs(cfg, opts)[0]
    assert (a.branch, b.branch, nf.branch) == ("god", "nogod", "noisefork")
    assert nf.game_id == b.game_id == a.game_id and nf.world_seed == b.world_seed
    assert nf.cfg.god.kind == "none" and nf.cfg.run.resample_from_round == 8
    assert nf.cfg.run.replica == b.cfg.run.replica
    # the seed rule used by LLMAgent._request (checked on real requests in the next test)
    def seed(c, rnd):
        r = c.run
        rep = r.replica if (r.resample_from_round is None or rnd < r.resample_from_round) \
            else f"{r.replica}@r{r.resample_from_round}"
        return derive_seed(r.seed, b.game_id, "turn", "x", rnd, 0, branch=r.branch, replica=rep)
    assert all(seed(nf.cfg, r) == seed(b.cfg, r) for r in range(1, 8))
    assert all(seed(nf.cfg, r) != seed(b.cfg, r) for r in range(8, 21))
    # default: no B''; hash-neutral at None
    cfg0, opts0 = load_experiment(GRID_K2, ["run.n_games=1"])
    assert [s.branch for s in build_specs(cfg0, opts0)[0]] == ["god", "nogod", "noise"]
    explicit = load_experiment(GRID_K2, ["run.n_games=1", "run.resample_from_round=null"])[0]
    assert cfg0.config_hash() == explicit.config_hash()


def test_noise_fork_resamples_in_real_agent_requests() -> None:
    """LLMAgent seeds: B'' equals the control before F, differs from F on (checked on real requests)."""
    from swarm.agents.llm_agent import LLMAgent
    from swarm.config import RunConfig
    from swarm.prompts import load_prompt_set
    from swarm.schema import ChatMessage

    def mk(rfr):
        cfg = RunConfig.model_validate({"run": {"resample_from_round": rfr}})
        ag = LLMAgent.__new__(LLMAgent)
        ag.cfg, ag.agent_id = cfg, "ann"
        ag.role = cfg.roles.agent
        ag.cot_mode = cfg.reasoning.cot_mode
        ag.prompts = load_prompt_set("agent/v1")
        return ag

    msgs = [ChatMessage(role="user", content="x")]
    base, nf = mk(None), mk(5)
    for rnd in range(1, 9):
        s0 = base._request(msgs, game_id="g1", round=rnd, attempt=0, purpose="turn").seed
        s1 = nf._request(msgs, game_id="g1", round=rnd, attempt=0, purpose="turn").seed
        assert (s0 == s1) == (rnd < 5)
