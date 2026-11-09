"""Code-review regressions for runner/wiring.py (plan/CODE_REVIEW.md)."""

from __future__ import annotations

import pytest

from swarm.config import from_dict
from swarm.runner.wiring import check_schedule


def _cfg(turn_mode: str, after_turns: int, n_agents: int = 4):
    return from_dict({"game": {"n_agents": n_agents, "turn_mode": turn_mode},
                      "god": {"kind": "scripted", "start_round": 1,
                              "schedule": [{"round": 1, "after_turns": after_turns,
                                            "power": "message_fraud"}]}})


def test_schedule_after_turns_rejected_in_simultaneous_mode() -> None:
    """Simultaneous mode consults God only at slot 0, so after_turns > 0 used to be a silent no-op."""
    check_schedule(_cfg("simultaneous", 0))
    with pytest.raises(ValueError, match="simultaneous"):
        check_schedule(_cfg("simultaneous", 2))


def test_schedule_after_turns_beyond_round_rejected_in_sequential_mode() -> None:
    check_schedule(_cfg("sequential", 3))
    with pytest.raises(ValueError, match="never fires"):
        check_schedule(_cfg("sequential", 4))


def test_schedule_check_ignores_non_scripted_god() -> None:
    cfg = from_dict({"game": {"turn_mode": "simultaneous"}, "god": {"kind": "none",
                     "schedule": [{"round": 1, "after_turns": 5, "power": "message_fraud"}]}})
    check_schedule(cfg)


def test_ledger_sink_numbers_own_actions_per_turn() -> None:
    """Review bug 1: LedgerSink gave every own action item_id '<agent>:<round>:own_action:0', so the
    context policy (items keyed by id) showed a 2-action turn's second action twice and lost the first."""
    from swarm.agents.llm_agent import LLMAgent
    from swarm.context.base import WordCounter
    from swarm.runner.wiring import LedgerSink
    from swarm.schema import ActionResultData, EventBody, GiveKey, MemorySink, PinnedFacts, PostBoard

    sink = LedgerSink(MemorySink(), WordCounter())
    acts = [PostBoard(text="looking for info 6"), GiveKey(to="bob", info_id=1)]
    for rnd in (1, 2):
        for a in acts:
            sink.emit(EventBody(round=rnd, turn_index=0, phase="resolve", actor="ann",
                                data=ActionResultData(agent="ann", action=a, ok=True)))
    ids = [it.item_id for it in sink.ledgers["ann"]]
    assert len(ids) == len(set(ids)) == 4
    cfg = from_dict({"roles": {r: {"backend": "fake"} for r in ("agent", "god", "judge", "summarizer")},
                     "context": {"tokenizer": "words"}, "god": {"kind": "none"}})
    agent = LLMAgent("ann", None, cfg)   # build_messages only; no LLM call
    pinned = PinnedFacts(agent="ann", round=3, max_rounds=5, needs_open=[], needs_burned=[], holdings=[],
                         can_give=[], score=0.0, peers=["bob"])
    bundle = agent.policy.build(agent="ann", round=3, ledger=sink.ledgers["ann"], current=[], memory=[],
                                pinned=pinned, budget=cfg.context.budget_tokens, counter=WordCounter())
    user = agent.build_messages(bundle)[1].content
    assert user.count("You posted on the board: looking for info 6") == 2      # once per round
    assert user.count("You gave bob your key for info 1") == 2


def test_runner_openrouter_client_gets_reasoning_effort_and_provider_sort(tmp_path, monkeypatch) -> None:
    """The runner builds OpenRouter clients itself (not via registry.from_cfg): it must pass
    reasoning_effort/provider_sort, or gpt-oss gets reasoning.enabled=false -> HTTP 400 on every call."""
    from swarm.runlog.writer import RunWriter
    from swarm.runner.wiring import _build_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test0000000000000000000000000000")
    cfg = from_dict({"roles": {"agent": {"backend": "openrouter", "model": "openai/gpt-oss-120b"}},
                     "llm": {"cache": {"mode": "read_only"},
                             "openrouter": {"enabled": True, "ledger_path": str(tmp_path / "l.jsonl"),
                                            "reasoning_effort": "low", "provider_sort": "price"}}})
    c = _build_client(cfg, "agent", RunWriter(tmp_path / "run", "r"), None)
    assert (c.reasoning_effort, c.provider_sort) == ("low", "price")


def test_budget_stopped_game_gets_status_budget_exceeded(tmp_path, monkeypatch) -> None:
    """Review #11: a game the engine ended on BudgetExceeded is status budget_exceeded, not done."""
    import asyncio

    from swarm.runner import run as runmod
    from swarm.runner.run import load_experiment, run_experiment
    from swarm.schema import Outcome

    async def fake_play(spec, gw, clients, engine):
        return Outcome(game_id=spec.game_id, rounds_played=3, end_reason="budget_exceeded", per_agent={},
                       welfare=0.0, win_rate=0.0)
    monkeypatch.setattr(runmod, "play_game", fake_play)
    from swarm.config import REPO_ROOT
    cfg, opts = load_experiment(REPO_ROOT / "configs" / "demo.yaml", ["roles.agent.backend=fake",
                                "llm.cache.mode='off'", "runner.report=false"])
    rd = asyncio.run(run_experiment(cfg, opts, out_root=tmp_path, run_id="bx"))
    import json
    man = json.loads((rd / "manifest.json").read_text())
    assert man["status"] == "budget_exceeded"
    assert {g["status"] for g in man["games"]} == {"budget_exceeded"}


def test_log_prompt_trace_switch(tmp_path) -> None:
    """Review #5: with run.log_prompt_trace the real runner logs prompt hashes and context_built;
    off (default) it logs neither (events.jsonl unchanged). The config hash ignores the switch."""
    import json

    from tests.runner.conftest import FAKE_OVERRIDES, _run

    base = [*FAKE_OVERRIDES, "game.max_rounds=3", "runner.report=false"]
    off = _run(tmp_path / "off", "off", base)
    on = _run(tmp_path / "on", "on", [*base, "run.log_prompt_trace=true"])

    def turns(rd):
        evs = [json.loads(x) for x in (rd / "games" / "g00-god" / "events.jsonl").read_text().splitlines()]
        return evs, [e["data"] for e in evs if e["type"] == "turn"]
    e_off, t_off = turns(off)
    e_on, t_on = turns(on)
    assert all(t["prompt_hash"] is None for t in t_off) and not any(e["type"] == "context_built" for e in e_off)
    assert all(t["prompt_hash"] and t["prompt_set_hash"] for t in t_on)
    assert sum(e["type"] == "context_built" for e in e_on) == len(t_on)
    assert e_off[0]["game_id"] == e_on[0]["game_id"]          # same config hash -> same game id
