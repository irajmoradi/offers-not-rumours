from __future__ import annotations

import asyncio
import json

import pytest

from swarm.llm.base import MALFORMED_REPLY, NOOP_REPLY, FakeLLM, LLMClient, TransientError, hash_policy
from swarm.rng import derive_seed, make_call_id
from swarm.schema import CallMeta, ChatMessage, ChatRequest


def req(i: int, content: str = "hello world", thinking: bool = False, role: str = "agent") -> ChatRequest:
    return ChatRequest(call_id=make_call_id("g", role, "a", i, 0), seed=derive_seed(0, i),
                       messages=[ChatMessage(role="user", content=content)], max_tokens=50,
                       temperature=0.0, thinking=thinking, meta=CallMeta(role=role, agent="a", round=i))


def run(coro):
    return asyncio.run(coro)


def test_is_llm_client():
    assert isinstance(FakeLLM(), LLMClient)


def test_default_noop_and_usage():
    r = run(FakeLLM().chat(req(1)))
    assert r.text == NOOP_REPLY and json.loads(r.text)["actions"] == [{"kind": "noop"}]
    assert r.call_id == req(1).call_id and r.usage.prompt_tokens == 2 and r.backend == "fake"


def test_script_then_policy_and_determinism():
    async def go():
        f = FakeLLM(script=["s0", "s1"], policy=lambda q: f"p:{q.meta.round}")
        return [(await f.chat(req(i))).text for i in range(4)], f

    out1, f1 = run(go())
    out2, _ = run(go())
    assert out1 == out2 == ["s0", "s1", "p:2", "p:3"]
    assert [q.meta.round for q in f1.requests] == [0, 1, 2, 3] and f1.n_calls == 4


def test_fail_every_and_raise_on():
    async def go():
        f = FakeLLM(fail_every=2, raise_on={3: TransientError("boom")})
        out = [(await f.chat(req(i))).text for i in range(3)]
        with pytest.raises(TransientError):
            await f.chat(req(3))
        return out

    assert run(go()) == [NOOP_REPLY, MALFORMED_REPLY, NOOP_REPLY]


def test_reasoning_only_when_thinking():
    f = FakeLLM(reasoning=lambda q: "because " + q.messages[0].content)
    assert run(f.chat(req(1))).reasoning is None
    r = run(f.chat(req(1, thinking=True)))
    assert r.reasoning == "because hello world" and r.usage.reasoning_tokens == 3


def test_policy_order_independent_under_concurrency():
    pol = hash_policy(["A", "B", "C", "D"])

    async def go(order):
        f = FakeLLM(policy=pol, latency_s=0.001)
        rs = await asyncio.gather(*(f.chat(req(i, content=f"c{i}")) for i in order))
        return {r.call_id: r.text for r in rs}

    assert run(go(range(8))) == run(go(reversed(range(8))))


def test_by_call_id_and_sent_text():
    f = FakeLLM(by_call_id={req(5).call_id: "special"})
    assert run(f.chat(req(5))).text == "special"
    run(f.chat(req(6, content="god stuff", role="god")))
    assert "god stuff" not in f.sent_text() and "hello world" in f.sent_text()
