"""LLMAgent round-trips with FakeLLM: prompt building, parse, retry, grep follow-up, hashes."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from swarm.agents.llm_agent import LLMAgent, system_context
from swarm.config import from_dict
from swarm.llm.base import MALFORMED_REPLY, AuthError, BudgetExceeded, FakeLLM, TransientError
from swarm.prompts import load_prompt_set, prompt_hash
from swarm.schema import (
    AgentMessageView,
    GiveKey,
    GrepLog,
    GrepResult,
    KeyView,
    LogLine,
    MemoryView,
    Noop,
    Observation,
    PinnedFacts,
    PostBoard,
    SendMessage,
)
from tests.conftest import TINY

BANNED = re.compile(r"\b(god|adversar\w*|deity|overseer|tamper\w*|forg\w*|mind control)\b", re.I)


def obs(agent="amber_fox", round=2, text="hi, I have 3", log=()):
    pin = PinnedFacts(agent=agent, round=round, max_rounds=5, needs_open=[3, 4], needs_burned=[],
                      holdings=[KeyView(info_id=1, key=1111)], can_give=[1], score=0.0,
                      peers=["blue_owl", "coral_elk"])
    return Observation(agent=agent, round=round, turn_index=6, pinned=pin,
                       inbox=[AgentMessageView(msg_id="m1", channel="dm", sender="blue_owl", text=text)],
                       log=list(log))


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def cfg():
    return from_dict(TINY)


def test_round_trip(cfg):
    reply = json.dumps({"thought": "trade", "note": "owl has 3",
                        "actions": [{"kind": "give_key", "to": "blue_owl", "info_id": 1},
                                    {"kind": "send_message", "to": "blue_owl", "text": "sent 1"}]})
    f = FakeLLM(script=[reply])
    ag = LLMAgent("amber_fox", f, cfg)
    out = run(ag.act(obs(), memory=[MemoryView(kind="scratchpad", round=1, text="old note")], game_id="g1"))
    assert out.actions == [GiveKey(to="blue_owl", info_id=1), SendMessage(to="blue_owl", text="sent 1")]
    assert out.cot == "trade" and out.note == "owl has 3" and out.parse_ok and out.retries == 0
    assert len(out.llm_call_ids) == 1
    req = f.requests[0]
    assert req.call_id == out.llm_call_ids[0]
    assert req.max_tokens == 700 and req.thinking is False
    assert req.meta.role == "agent" and req.meta.agent == "amber_fox" and req.meta.round == 2
    assert req.meta.prompt_hash == ag.last.prompt_hash == prompt_hash(req.messages)
    user = req.messages[1].content
    assert "DM from blue_owl: hi, I have 3" in user
    assert "Your note (round 1): old note" in user
    assert ag.last.bundle is not None and ag.last.prompt_set_hash == ag.prompts.hash


def test_retry_then_success(cfg):
    good = '{"actions": [{"kind": "post_board", "text": "x"}]}'
    f = FakeLLM(script=[MALFORMED_REPLY, good])
    out = run(LLMAgent("amber_fox", f, cfg).act(obs(), game_id="g1"))
    assert out.actions == [PostBoard(text="x")] and out.parse_ok and out.retries == 1
    assert "bad_json" in out.errors
    retry_msgs = f.requests[1].messages
    assert [m.role for m in retry_msgs] == ["system", "user", "assistant", "user"]
    assert f.requests[0].seed != f.requests[1].seed and len(set(out.llm_call_ids)) == 2


def test_malformed_twice_is_noop(cfg):
    f = FakeLLM(script=["garbage", "still garbage"])
    out = run(LLMAgent("amber_fox", f, cfg).act(obs(), game_id="g1"))
    assert out.actions == [Noop()] and not out.parse_ok and out.retries == 1
    assert out.errors.count("no_json") == 2


def test_transient_llm_error_is_noop(cfg):
    f = FakeLLM(raise_on={0: TransientError("boom")})
    out = run(LLMAgent("amber_fox", f, cfg).act(obs(), game_id="g1"))
    assert out.actions == [Noop()] and not out.parse_ok and "llm_error:TransientError" in out.errors


@pytest.mark.parametrize("exc", [BudgetExceeded("global", 2.0, 0.1, 2.0), AuthError("401")])
def test_fatal_errors_propagate(cfg, exc):
    f = FakeLLM(raise_on={0: exc})
    with pytest.raises(type(exc)):
        run(LLMAgent("amber_fox", f, cfg).act(obs(), game_id="g1"))


def test_grep_followup(cfg):
    first = '{"actions": [{"kind": "grep_log", "pattern": "blue_owl"}]}'
    second = '{"note": "owl gave", "actions": [{"kind": "give_key", "to": "blue_owl", "info_id": 1}]}'
    f = FakeLLM(script=[first, second])
    hit = LogLine(round=1, kind="give", actor="blue_owl", target="coral_elk", info_id=5)
    seen = []

    def grep(a: GrepLog) -> GrepResult:
        seen.append(a.pattern)
        return GrepResult(pattern=a.pattern, hits=[hit])

    ag = LLMAgent("amber_fox", f, cfg)
    out = run(ag.act(obs(), game_id="g1", grep=grep))
    assert seen == ["blue_owl"]
    assert out.actions == [GrepLog(pattern="blue_owl"), GiveKey(to="blue_owl", info_id=1)]
    assert out.note == "owl gave" and len(out.llm_call_ids) == 2
    assert 'Log search "blue_owl": 1 hits' in f.requests[1].messages[1].content
    assert ag.last.grep is not None


def test_native_mode_uses_reasoning(cfg):
    c = from_dict({**TINY, "reasoning": {"cot_mode": "native"}})
    f = FakeLLM(script=['{"actions": [{"kind": "noop"}]}'], reasoning="deep thoughts")
    out = run(LLMAgent("amber_fox", f, c).act(obs(), game_id="g1"))
    assert f.requests[0].thinking is True and out.cot == "deep thoughts"


def test_cot_none_mode(cfg):
    c = from_dict({**TINY, "reasoning": {"cot_mode": "none"}})
    f = FakeLLM(script=['{"thought": "ignored", "actions": [{"kind": "noop"}]}'])
    out = run(LLMAgent("amber_fox", f, c).act(obs(), game_id="g1"))
    assert out.cot is None and '"thought"' not in f.requests[0].messages[0].content


def test_prompt_hash_stable(cfg):
    def one():
        f = FakeLLM()
        ag = LLMAgent("amber_fox", f, cfg)
        run(ag.act(obs(), game_id="g1"))
        return ag.last.prompt_hash, f.requests[0].seed, f.requests[0].call_id, ag.prompts.hash
    a, b = one(), one()
    assert a == b
    f2 = FakeLLM()
    ag2 = LLMAgent("amber_fox", f2, cfg)
    run(ag2.act(obs(text="different"), game_id="g1"))
    assert ag2.last.prompt_hash != a[0]


def test_system_prompt_identical_across_agents_and_rounds(cfg):
    f = FakeLLM()
    for agent, rnd in [("amber_fox", 1), ("blue_owl", 3), ("coral_elk", 5)]:
        run(LLMAgent(agent, f, cfg).act(obs(agent=agent, round=rnd), game_id="g1"))
    systems = {r.messages[0].content for r in f.requests}
    assert len(systems) == 1
    s = systems.pop()
    assert "amber_fox" not in s and "Round" not in s


def test_no_banned_words_in_agent_prompts(cfg):
    for mode in ("none", "native", "explicit"):
        c = from_dict({**TINY, "reasoning": {"cot_mode": mode}})
        f = FakeLLM()
        run(LLMAgent("amber_fox", f, c).act(obs(), game_id="g1"))
        for m in f.requests[0].messages:
            assert not BANNED.search(m.content), BANNED.search(m.content)


def test_framing_slot_empty_by_default_and_rendered_when_set(cfg, tmp_path):
    ps = load_prompt_set("agent/v1", framing="framing/none")
    assert ps.framing_text == ""
    assert ps.render("system.j2", **system_context(cfg)) == load_prompt_set("agent/v1").render(
        "system.j2", **system_context(cfg))


@pytest.mark.parametrize("finish,salvage,expect_retry", [("length", True, False), ("length", False, True),
                                                         ("stop", True, True)])
def test_salvage_only_for_length_cutoffs_when_enabled(finish, salvage, expect_retry):
    """reasoning.salvage_truncated keeps the complete actions of a max_tokens cut-off instead of
    retrying; never for a reply that stopped normally; default off."""
    cut = ('{"thought": "t", "actions": [{"kind": "give_key", "to": "blue_owl", "info_id": 1}, '
           '{"kind": "send_message", "to": "blue_owl", "text": "abc')
    full = json.dumps({"thought": "t2", "actions": [{"kind": "noop"}]})
    c = from_dict({**TINY, "reasoning": {"salvage_truncated": salvage}})
    f = FakeLLM(script=[cut, full], finish_reason=finish)
    out = run(LLMAgent("amber_fox", f, c).act(obs(), game_id="g1"))
    if expect_retry:
        assert out.retries == 1 and len(f.requests) == 2
    else:
        assert out.retries == 0 and len(f.requests) == 1 and out.actions == [GiveKey(to="blue_owl", info_id=1)]
