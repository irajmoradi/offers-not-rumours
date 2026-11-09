"""vLLM / OpenAI-compat client: body, retries, errors, cache, concurrency, accounting (mock transport)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from swarm.config import RetryCfg, from_dict
from swarm.llm.base import AuthError, LLMError, RateLimited, TransientError
from swarm.llm.cache import CacheMiss, ResponseCache, cache_key
from swarm.llm.registry import build_client
from swarm.llm.vllm import VLLMClient
from swarm.schema import CallMeta, ChatMessage, ChatRequest
from tests.conftest import TINY

FAST = RetryCfg(max_retries=3, base_delay_s=0.0, max_delay_s=0.0)


def req(content="hi", seed=1, thinking=False, max_tokens=700, call_id="c1"):
    return ChatRequest(call_id=call_id, messages=[ChatMessage(role="user", content=content)],
                       max_tokens=max_tokens, temperature=0.6, seed=seed, thinking=thinking,
                       meta=CallMeta(role="agent"))


def ok_body(text='{"actions": []}', reasoning=None):
    return {"model": "Qwen/Qwen3-8B",
            "choices": [{"message": {"content": text, "reasoning": reasoning}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class Script:
    """Mock transport: a list of (status, json|None, headers) served in order; records bodies."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []
        self.headers = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        status, body, hdrs = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if status == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(status, json=body, headers=hdrs or {})


def client(script, **kw):
    return VLLMClient(transport=httpx.MockTransport(script), retry=kw.pop("retry", FAST), **kw)


def run(c):
    return asyncio.run(c)


def test_body_has_thinking_flag_and_defaults():
    s = Script((200, ok_body(reasoning="r"), None))
    c = client(s)
    r = run(c.chat(req(thinking=True)))
    b = s.bodies[0]
    assert b["chat_template_kwargs"] == {"enable_thinking": True}
    assert b["model"] == "Qwen/Qwen3-8B" and b["seed"] == 1 and b["max_tokens"] == 700
    assert r.reasoning == "r" and r.backend == "vllm" and r.call_id == "c1"
    run(c.chat(req(thinking=False, seed=2)))
    assert s.bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_retry_on_5xx_timeout_and_empty_then_success():
    s = Script((500, {}, None), ("timeout", None, None), (200, ok_body(text=""), None), (200, ok_body(), None))
    c = client(s)
    r = run(c.chat(req()))
    assert r.retries == 3 and c.totals.retries == 3 and len(s.bodies) == 4


def test_retries_exhausted_raises():
    s = Script((503, {}, None))
    with pytest.raises(TransientError):
        run(client(s, retry=RetryCfg(max_retries=2, base_delay_s=0, max_delay_s=0)).chat(req()))
    assert len(s.bodies) == 3


def test_auth_error_not_retried():
    s = Script((401, {"error": "bad"}, None))
    with pytest.raises(AuthError):
        run(client(s).chat(req()))
    assert len(s.bodies) == 1


def test_400_not_retried():
    s = Script((400, {"error": "context too long"}, None))
    with pytest.raises(LLMError) as ei:
        run(client(s).chat(req()))
    assert not isinstance(ei.value, (TransientError, RateLimited)) and len(s.bodies) == 1


def test_429_honours_retry_after():
    slept = []

    async def sleep(d):
        slept.append(d)

    s = Script((429, {}, {"retry-after": "0.25"}), (200, ok_body(), None))
    c = VLLMClient(transport=httpx.MockTransport(s), retry=RetryCfg(max_retries=2, max_delay_s=5), sleep=sleep)
    run(c.chat(req()))
    assert slept == [0.25]


def test_length_finish_with_empty_text_returned_not_retried():
    body = ok_body(text="")
    body["choices"][0]["finish_reason"] = "length"
    s = Script((200, body, None))
    r = run(client(s).chat(req()))
    assert r.text == "" and r.finish_reason == "length" and len(s.bodies) == 1


def test_cache_hit_and_key_sensitivity(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    s = Script((200, ok_body(), None))
    c = client(s, cache=cache)
    r1 = run(c.chat(req(call_id="a")))
    r2 = run(c.chat(req(call_id="b")))
    assert not r1.cache_hit and r2.cache_hit and r2.call_id == "b" and len(s.bodies) == 1
    run(c.chat(req(seed=99)))
    run(c.chat(req(max_tokens=100)))
    assert len(s.bodies) == 3 and len(cache) == 3
    assert c.totals.cache_hits == 1 and c.totals.calls == 4
    assert cache_key("vllm", "m", req(call_id="x")) == cache_key("vllm", "m", req(call_id="y"))
    assert cache_key("vllm", "m", req()) != cache_key("vllm", "m2", req())


def test_cache_persists_and_read_only_miss_raises(tmp_path):
    p = tmp_path / "c.sqlite"
    run(client(Script((200, ok_body(), None)), cache=ResponseCache(p)).chat(req()))
    ro = client(Script((500, {}, None)), cache=ResponseCache(p, "read_only"))
    assert run(ro.chat(req())).cache_hit
    with pytest.raises(CacheMiss):
        run(ro.chat(req(seed=5)))


def test_concurrency_semaphore():
    state = {"cur": 0, "max": 0}

    async def handler(request):
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return httpx.Response(200, json=ok_body())

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await handler(request)

    c = VLLMClient(transport=T(), max_concurrency=3, retry=FAST)

    async def many():
        await asyncio.gather(*(c.chat(req(seed=i, call_id=str(i))) for i in range(12)))

    run(many())
    assert state["max"] == 3 and c.totals.calls == 12
    assert c.totals.prompt_tokens == 120 and c.totals.completion_tokens == 60


def test_recorder_gets_llmcall():
    recs = []
    s = Script((200, ok_body(), None))
    run(client(s, recorder=recs.append, run_id="r1").chat(req()))
    assert len(recs) == 1 and recs[0].run_id == "r1" and recs[0].response is not None


def test_registry_backends(tmp_path):
    cfg = from_dict(TINY)
    assert build_client(cfg, "agent").backend == "fake"
    v = from_dict({**TINY, "roles": {**TINY["roles"], "agent": {"backend": "vllm"}}})
    c = build_client(v, "agent")
    assert isinstance(c, VLLMClient) and c.model == "Qwen/Qwen3-8B" and c.cache is None
    o = from_dict({**TINY, "roles": {**TINY["roles"], "agent": {"backend": "vllm"}},
                   "llm": {"cache": {"mode": "read_write", "path": str(tmp_path / "x.sqlite")}}})
    assert build_client(o, "agent").cache is not None


def test_single_flight_shares_one_request_between_concurrent_identical_calls(tmp_path):
    """Paired fork branches run concurrently: identical pre-fork requests must hit the API once."""
    n = {"posts": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        n["posts"] += 1
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=ok_body("same"))

    c = VLLMClient(transport=httpx.MockTransport(handler), retry=FAST, cache=ResponseCache(tmp_path / "c.sqlite"))

    async def both():
        return await asyncio.gather(c.chat(req(call_id="a")), c.chat(req(call_id="b")),
                                    c.chat(req(content="other", call_id="x")))

    a, b, x = run(both())
    assert n["posts"] == 2                      # "hi" once (shared) + "other" once
    assert a.text == b.text == "same" and {a.call_id, b.call_id} == {"a", "b"}
    assert [a.cache_hit, b.cache_hit].count(True) == 1 and x.cache_hit is False
    assert c.totals.cache_hits == 1 and not c._inflight


def test_single_flight_follower_retries_after_leader_failure(tmp_path):
    n = {"posts": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        n["posts"] += 1
        await asyncio.sleep(0.02)
        if n["posts"] == 1:
            return httpx.Response(400, json={"error": "bad"})
        return httpx.Response(200, json=ok_body("ok"))

    c = VLLMClient(transport=httpx.MockTransport(handler), retry=FAST, cache=ResponseCache(tmp_path / "c.sqlite"))

    async def both():
        return await asyncio.gather(c.chat(req(call_id="a")), c.chat(req(call_id="b")), return_exceptions=True)

    a, b = run(both())
    assert isinstance(a, LLMError) and b.text == "ok" and n["posts"] == 2 and not c._inflight
