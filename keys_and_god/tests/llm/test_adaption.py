"""Adaption Responses backend: body mapping, response/usage/cost parsing, retries, incomplete re-ask,
own ledger, key never logged, config + hash rules. HTTP is mocked (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from swarm.config import RetryCfg, from_dict
from swarm.llm.adaption import AdaptionClient, _GLOBAL_SEM
from swarm.llm.base import AuthError, BudgetExceeded, LLMError, TransientError
from swarm.llm.budget import GlobalLedger, RunBudget
from swarm.llm.cache import ResponseCache
from swarm.llm.registry import build_client
from swarm.llm.secrets import install_log_redaction, load_adaption_key
from swarm.schema import CallMeta, ChatMessage, ChatRequest
from tests.conftest import TINY

KEY = "ad-zqcanary-0123456789abcdef0123456789abcdef"
FAST = RetryCfg(max_retries=2, base_delay_s=0.0, max_delay_s=0.0)


def req(seed=1, max_tokens=1000, system="You are agent A.", user="Your turn."):
    msgs = ([ChatMessage(role="system", content=system)] if system else []) + [
        ChatMessage(role="user", content=user)]
    return ChatRequest(call_id=f"c{seed}", messages=msgs, max_tokens=max_tokens, temperature=0.6, seed=seed,
                       response_format={"type": "json_object"}, meta=CallMeta(role="agent"))


def resp(text='{"actions": [{"kind": "noop"}]}', status="completed", cost=0.0012, inp=900, out=300, rsn=200):
    return {"id": "resp_1", "object": "response", "status": status, "model": "qwen3.8-27b",
            "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking..."}]},
                       {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
            "output_text": text,
            "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                      "output_tokens_details": {"reasoning_tokens": rsn}, "cost": cost}}


def mk(tmp_path, handler, **kw):
    return AdaptionClient(enabled=True, api_key=SecretStr(KEY), transport=httpx.MockTransport(handler),
                          ledger=kw.pop("ledger", GlobalLedger(tmp_path / "adaption_ledger.jsonl", 3.0)),
                          retry=kw.pop("retry", FAST), sleep=kw.pop("sleep", _nosleep), **kw)


async def _nosleep(_s):
    return None


def run(c):
    return asyncio.run(c)


def test_disabled_by_default():
    with pytest.raises(LLMError):
        AdaptionClient(api_key=SecretStr(KEY))
    assert from_dict(TINY).llm.adaption.enabled is False


def test_config_requires_enabled_and_registry_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTION_API_KEY", KEY)   # release: hermetic (no local key file needed)
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "adaption", "model": "qwen3.8-27b"}}}
    with pytest.raises(ValueError, match="llm.adaption.enabled"):
        from_dict(raw)
    raw["llm"] = {"cache": {"mode": "off"}, "adaption": {"enabled": True, "max_concurrency": 32,
                                                         "ledger_path": str(tmp_path / "l.jsonl")}}
    cfg = from_dict(raw)
    assert cfg.model_for("agent") == ("adaption", "qwen3.8-27b")
    c = build_client(cfg, "agent")
    assert isinstance(c, AdaptionClient) and c.max_concurrency == 32 and c.model == "qwen3.8-27b"
    assert KEY not in repr(c)


def test_hash_unchanged_by_default_and_operational_knobs():
    base = from_dict(TINY).config_hash()
    # the new llm.adaption block at its defaults does not re-key old configs
    assert from_dict({**TINY, "llm": {"cache": {"mode": "off"}, "adaption": {}}}).config_hash() == base
    on = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "adaption"}},
          "llm": {"cache": {"mode": "off"}, "adaption": {"enabled": True}}}
    h1 = from_dict(on).config_hash()
    on2 = json.loads(json.dumps(on))
    on2["llm"]["adaption"].update(max_concurrency=64, global_cap_usd=9.0, ledger_path="x.jsonl",
                                  price_in_per_mtok=0.1)
    assert from_dict(on2).config_hash() == h1 != base


def test_body_mapping_and_response_parsing(tmp_path):
    seen = {}

    def handler(r: httpx.Request):
        seen["url"] = str(r.url)
        seen["auth"] = r.headers["authorization"]
        seen["body"] = json.loads(r.content)
        return httpx.Response(200, json=resp())

    c = mk(tmp_path, handler)
    rq = ChatRequest(call_id="c9", temperature=0.6, seed=3, max_tokens=1000, meta=CallMeta(role="agent"),
                     messages=[ChatMessage(role="system", content="S1"), ChatMessage(role="user", content="u1"),
                               ChatMessage(role="assistant", content="a1"),
                               ChatMessage(role="user", content="u2")],
                     response_format={"type": "json_object"}, stop=["\n\n"])
    out = run(c.chat(rq))
    b = seen["body"]
    assert seen["url"].endswith("/api/v1/responses")
    assert seen["auth"] == f"Bearer {KEY}"
    assert b == {"model": "qwen3.8-27b", "instructions": "S1", "max_output_tokens": 1000, "temperature": 0.6,
                 "store": False, "stream": True, "reasoning": {"effort": "low"}, "input": [{"role": "user", "content": "u1"},
                                           {"role": "assistant", "content": "a1"},
                                           {"role": "user", "content": "u2"}]}
    assert out.text == '{"actions": [{"kind": "noop"}]}' and out.finish_reason == "stop"
    assert out.reasoning == "thinking..."
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.reasoning_tokens) == (900, 300, 200)
    assert out.cost_usd == pytest.approx(0.0012) and out.provider == "adaption" and out.backend == "adaption"
    assert c.ledger.total() == pytest.approx(0.0012)


def test_output_text_fallback_to_message_items(tmp_path):
    d = resp()
    d.pop("output_text")
    c = mk(tmp_path, lambda r: httpx.Response(200, json=d))
    assert run(c.chat(req())).text == '{"actions": [{"kind": "noop"}]}'


def test_retries_429_and_5xx_then_ok(tmp_path):
    codes = iter([429, 503])
    sleeps = []

    async def sl(s):
        sleeps.append(s)

    def handler(r):
        code = next(codes, 200)
        if code == 429:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        if code != 200:
            return httpx.Response(code, text="upstream")
        return httpx.Response(200, json=resp())

    c = mk(tmp_path, handler, sleep=sl)
    out = run(c.chat(req()))
    assert out.retries == 2 and c.totals.retries == 2 and len(sleeps) == 2


def test_gives_up_after_max_retries(tmp_path):
    c = mk(tmp_path, lambda r: httpx.Response(502, text="bad gateway"))
    with pytest.raises(TransientError):
        run(c.chat(req()))
    assert c.totals.errors == 1 and c.ledger.total() == 0.0 and c.ledger.reserved == 0.0


def test_auth_and_400_not_retried(tmp_path):
    n = {"k": 0}

    def h401(r):
        n["k"] += 1
        return httpx.Response(401, text=f"bad key {KEY}")

    with pytest.raises(AuthError) as ei:
        run(mk(tmp_path, h401).chat(req()))
    assert KEY not in str(ei.value) and n["k"] == 1

    def h400(r):
        n["k"] += 1
        return httpx.Response(400, json={"error": {"message": "bad param"}})

    with pytest.raises(LLMError):
        run(mk(tmp_path, h400).chat(req()))
    assert n["k"] == 2


def test_failed_status_is_retried(tmp_path):
    seq = iter([{"status": "failed", "error": {"code": "server_error", "message": "provider"}}, resp()])
    c = mk(tmp_path, lambda r: httpx.Response(200, json=next(seq)))
    assert run(c.chat(req())).retries == 1


def test_incomplete_empty_reasks_once_with_more_tokens(tmp_path):
    bodies = []
    seq = iter([resp(text="", status="incomplete", cost=0.001, out=1000, rsn=1000), resp(cost=0.002)])

    def handler(r):
        bodies.append(json.loads(r.content))
        return httpx.Response(200, json=next(seq))

    c = mk(tmp_path, handler)
    out = run(c.chat(req(max_tokens=1000)))
    assert [b["max_output_tokens"] for b in bodies] == [1000, 2000]
    assert out.text.startswith("{") and out.finish_reason == "stop" and c.incomplete_retries == 1
    assert out.cost_usd == pytest.approx(0.003)           # both attempts billed
    assert out.usage.completion_tokens == 1300 and out.usage.prompt_tokens == 1800


def test_incomplete_empty_twice_returns_length_empty(tmp_path):
    n = {"k": 0}

    def handler(r):
        n["k"] += 1
        return httpx.Response(200, json=resp(text="", status="incomplete"))

    out = run(mk(tmp_path, handler).chat(req()))
    assert n["k"] == 2 and out.text == "" and out.finish_reason == "length"


def test_incomplete_with_text_is_truncation(tmp_path):
    c = mk(tmp_path, lambda r: httpx.Response(200, json=resp(text='{"thought": "x", "actions": [', status="incomplete")))
    out = run(c.chat(req()))
    assert out.finish_reason == "length" and c.incomplete_retries == 0


def test_own_ledger_cap_refuses_before_call(tmp_path):
    led = tmp_path / "adaption_ledger.jsonl"
    led.write_text(json.dumps({"cost_usd": 2.999}) + "\n")
    sent = {"k": 0}

    def handler(r):
        sent["k"] += 1
        return httpx.Response(200, json=resp())

    c = mk(tmp_path, handler, ledger=GlobalLedger(led, 3.0))
    with pytest.raises(BudgetExceeded):
        run(c.chat(req()))
    assert sent["k"] == 0


def test_run_budget_shared(tmp_path):
    rb = RunBudget(0.0001)
    c = mk(tmp_path, lambda r: httpx.Response(200, json=resp()), run_budget=rb)
    with pytest.raises(BudgetExceeded):
        run(c.chat(req()))
    assert rb.reserved == 0.0


def test_cache_and_single_flight(tmp_path):
    n = {"k": 0}

    async def handler_async(r):
        n["k"] += 1
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=resp())

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await handler_async(request)

    cache = ResponseCache(tmp_path / "c.sqlite", "read_write")
    c = AdaptionClient(enabled=True, api_key=SecretStr(KEY), transport=T(), retry=FAST, cache=cache,
                       ledger=GlobalLedger(tmp_path / "l.jsonl", 3.0))

    async def go():
        return await asyncio.gather(*(c.chat(req(seed=5)) for _ in range(4)))

    outs = run(go())
    assert n["k"] == 1 and sum(o.cache_hit for o in outs) == 3
    again = run(c.chat(req(seed=5)))
    assert again.cache_hit and n["k"] == 1


def test_semaphore_caps_in_flight(tmp_path):
    state = {"cur": 0, "max": 0}

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1
            return httpx.Response(200, json=resp())

    c = AdaptionClient(enabled=True, api_key=SecretStr(KEY), transport=T(), retry=FAST, max_concurrency=3,
                       ledger=GlobalLedger(tmp_path / "l.jsonl", 3.0))

    async def go():
        await asyncio.gather(*(c.chat(req(seed=i)) for i in range(12)))

    run(go())
    assert state["max"] == 3


def test_key_file_first_then_env(tmp_path):
    f = tmp_path / ".adaptionkey"
    f.write_text(KEY + "\n")
    assert load_adaption_key(f, env={"ADAPTION_API_KEY": "ad-other-0123456789"}).get_secret_value() == KEY
    assert load_adaption_key(tmp_path / "missing", env={"ADAPTION_API_KEY": "ad-env-0123456789"}) \
        .get_secret_value() == "ad-env-0123456789"
    assert load_adaption_key(tmp_path / "missing", env={"OPENROUTER_API_KEY": "sk-or-x0123456789"}) is None


def test_key_never_logged(tmp_path, caplog, capsys):
    install_log_redaction()
    load_adaption_key(tmp_path / "missing", env={"ADAPTION_API_KEY": KEY})   # registers it for redact()
    calls = []

    def handler(r):
        return httpx.Response(500, text=f"echo {KEY}")

    c = mk(tmp_path, handler, recorder=calls.append)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TransientError):
            run(c.chat(req()))
    blob = caplog.text + capsys.readouterr().out + "".join(x.model_dump_json() for x in calls)
    assert KEY not in blob and KEY not in repr(c)


# ------------------------------------------------------------------ streaming (SSE)
def sse(events, ctype="text/event-stream; charset=utf-8"):
    body = "".join(f"event: {e.get('type')}\ndata: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": ctype}, content=body.encode())


def stream_events(text='{"actions": []}', status="completed", final_text=True, cost=0.002):
    final = {"id": "r", "status": status, "model": "qwen3.8-27b",
             "usage": {"input_tokens": 1200, "output_tokens": 400, "output_tokens_details": {"reasoning_tokens": 150},
                       "cost": cost}}
    if final_text:
        final["output_text"] = text
    half = len(text) // 2
    return [{"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
            {"type": "response.reasoning_text.delta", "delta": "let me "},
            {"type": "response.reasoning_text.delta", "delta": "think"},
            {"type": "response.output_text.delta", "delta": text[:half]},
            {"type": "response.output_text.delta", "delta": text[half:]},
            {"type": f"response.{status}", "response": final}]


def test_stream_final_event_usage_and_cost(tmp_path):
    c = mk(tmp_path, lambda r: sse(stream_events()))
    out = run(c.chat(req()))
    assert out.text == '{"actions": []}' and out.finish_reason == "stop" and out.reasoning == "let me think"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.reasoning_tokens) == (1200, 400, 150)
    assert out.cost_usd == pytest.approx(0.002) and c.ledger.total() == pytest.approx(0.002)


def test_stream_text_from_deltas_when_final_lacks_output_text(tmp_path):
    c = mk(tmp_path, lambda r: sse(stream_events(text='{"a": 1}', final_text=False)))
    assert run(c.chat(req())).text == '{"a": 1}'


def test_stream_cut_and_html_504_are_retried_and_counted(tmp_path):
    seq = iter([sse(stream_events()[:4]),                                       # no final event
                httpx.Response(504, headers={"content-type": "text/html"}, text="<html>504 Gateway Time-out</html>"),
                httpx.Response(200, headers={"content-type": "text/html"}, text="<html>oops</html>"),
                sse(stream_events())])
    c = mk(tmp_path, lambda r: next(seq), retry=RetryCfg(max_retries=3, base_delay_s=0.0, max_delay_s=0.0))
    out = run(c.chat(req()))
    assert out.retries == 3 and c.fail_counts == {"stream_cut": 1, "504": 1, "nonjson": 1}


def test_stream_failed_and_error_events_retry(tmp_path):
    seq = iter([sse([{"type": "error", "error": {"message": "provider down"}}]),
                sse(stream_events(status="failed")), sse(stream_events())])
    c = mk(tmp_path, lambda r: next(seq), retry=RetryCfg(max_retries=3, base_delay_s=0.0, max_delay_s=0.0))
    assert run(c.chat(req())).retries == 2
    assert c.fail_counts["stream_error"] == 1 and c.fail_counts["failed_status"] == 1


def test_read_timeout_is_transient_and_counted(tmp_path):
    n = {"k": 0}

    def handler(r):
        n["k"] += 1
        if n["k"] == 1:
            raise httpx.ReadTimeout("slow", request=r)
        return sse(stream_events())

    c = mk(tmp_path, handler)
    assert run(c.chat(req())).retries == 1 and c.fail_counts["timeout"] == 1


def test_no_cap_omits_max_output_tokens(tmp_path):
    bodies = []

    def handler(r):
        bodies.append(json.loads(r.content))
        return sse(stream_events())

    c = mk(tmp_path, handler, uncapped_out_tokens=8000, price_out_per_mtok=2.0, price_in_per_mtok=0.0)
    rq = req(max_tokens=1000).model_copy(update={"max_tokens": None})
    run(c.chat(rq))
    assert "max_output_tokens" not in bodies[0]
    assert c._estimate(rq) == pytest.approx(8000 * 2.0 / 1e6)


def test_no_cap_incomplete_empty_is_transient(tmp_path):
    seq = iter([sse(stream_events(text="", status="incomplete")), sse(stream_events())])
    c = mk(tmp_path, lambda r: next(seq))
    rq = req().model_copy(update={"max_tokens": None})
    out = run(c.chat(rq))
    assert out.retries == 1 and c.fail_counts["incomplete_empty"] == 1 and c.incomplete_retries == 0


def test_effort_minimal_maps_to_low_and_none_omits(tmp_path, caplog):
    bodies = []

    def handler(r):
        bodies.append(json.loads(r.content))
        return sse(stream_events())

    with caplog.at_level(logging.WARNING):
        c = mk(tmp_path, handler, reasoning_effort="minimal")
    assert "minimal" in caplog.text and c.reasoning_effort == "low"
    run(c.chat(req()))
    run(mk(tmp_path, handler, reasoning_effort=None, stream=False).chat(req(seed=2)))
    assert bodies[0]["reasoning"] == {"effort": "low"}
    assert "reasoning" not in bodies[1] and "stream" not in bodies[1]


def test_429_backoff_jitter_with_retry_after_floor(tmp_path):
    sleeps = []

    async def sl(s):
        sleeps.append(s)

    codes = iter([429] * 4)

    def handler(r):
        if next(codes, None) == 429:
            return httpx.Response(429, headers={"retry-after": "5"})
        return sse(stream_events())

    c = mk(tmp_path, handler, sleep=sl, max_retries_429=6, rate_limit_max_delay_s=60.0,
           retry=RetryCfg(max_retries=1, base_delay_s=2.0, max_delay_s=30.0))
    out = run(c.chat(req()))
    assert out.retries == 4 and c.fail_counts["429"] == 4      # 429s get more than retry.max_retries=1
    assert all(5.0 <= s <= 60.0 for s in sleeps) and len(set(sleeps)) == 4   # floored + jittered
    assert all(s <= 5.0 + 2.0 * 2 ** i for i, s in enumerate(sleeps))


def test_global_concurrency_cap_across_clients(tmp_path):
    state = {"cur": 0, "max": 0}

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1
            return sse(stream_events())

    _GLOBAL_SEM.clear()
    a, b = (AdaptionClient(enabled=True, api_key=SecretStr(KEY), transport=T(), retry=FAST, max_concurrency=10,
                           global_max_concurrency=4, ledger=GlobalLedger(tmp_path / f"l{i}.jsonl", 3.0))
            for i in range(2))

    async def go():
        await asyncio.gather(*(x.chat(req(seed=i)) for i in range(10) for x in (a, b)))

    run(go())
    _GLOBAL_SEM.clear()
    assert state["max"] == 4


def test_max_tokens_null_not_on_vllm():
    # realistic-framing merge: no cap is allowed on adaption, openrouter and fake (the field is omitted); vllm needs one
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "vllm", "max_tokens": None}}}
    with pytest.raises(ValueError, match="max_tokens=null"):
        from_dict(raw)
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "fake", "max_tokens": None}}}
    assert from_dict(raw).roles.agent.max_tokens is None
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "adaption", "max_tokens": None}},
           "llm": {"cache": {"mode": "off"}, "adaption": {"enabled": True}}}
    assert from_dict(raw).roles.agent.max_tokens is None


def test_effort_is_hashed_when_enabled():
    on = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "adaption"}},
          "llm": {"cache": {"mode": "off"}, "adaption": {"enabled": True}}}
    hi = json.loads(json.dumps(on))
    hi["llm"]["adaption"]["reasoning_effort"] = "high"
    assert from_dict(on).config_hash() != from_dict(hi).config_hash()


# ------------------------------------------------------------------ RPM token bucket
def test_rpm_bucket_refill_and_burst():
    from swarm.llm.adaption import RpmBucket

    now = {"t": 0.0}
    b = RpmBucket(rpm=60, burst=3, clock=lambda: now["t"])     # 1 token / s, 3 burst

    async def sleep_adv(s):
        now["t"] += s

    async def go():
        import swarm.llm.adaption as mod
        orig = mod.asyncio.sleep
        mod.asyncio.sleep = sleep_adv
        try:
            for _ in range(5):
                await b.acquire()
        finally:
            mod.asyncio.sleep = orig

    run(go())
    assert now["t"] == pytest.approx(2.0) and b.waited_s == pytest.approx(2.0)   # 3 free, then 1/s


def test_rpm_bucket_limits_request_starts(tmp_path):
    import time as _t

    from swarm.llm.adaption import _RPM
    starts = []

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            starts.append(_t.monotonic())
            return sse(stream_events())

    _RPM.clear()
    c = AdaptionClient(enabled=True, api_key=SecretStr(KEY), transport=T(), retry=FAST, max_concurrency=50,
                       rpm=600, rpm_burst=2, ledger=GlobalLedger(tmp_path / "l.jsonl", 3.0))   # 10/s after 2

    async def go():
        await asyncio.gather(*(c.chat(req(seed=i)) for i in range(6)))

    run(go())
    _RPM.clear()
    assert len(starts) == 6 and starts[-1] - starts[0] >= 0.35     # 4 tokens at 10/s ~ 0.4 s


def test_rpm_off_when_none(tmp_path):
    c = mk(tmp_path, lambda r: sse(stream_events()), rpm=None)
    assert run(c.chat(req())).text
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "adaption"}},
           "llm": {"cache": {"mode": "off"}, "adaption": {"enabled": True}}}
    a = from_dict(raw).llm.adaption
    assert (a.rpm, a.rpm_burst, a.global_max_concurrency) == (380.0, 60, 300)


def test_extra_keys_lanes_route_by_bucket_and_429_hits_only_that_key(tmp_path):
    from swarm.llm.ratelimit import FileRateLimiter

    ka, kb = "pt_live_FAKEkeyAAAA", "pt_live_FAKEkeyBBBB"
    seen = []

    def handler(r):
        k = r.headers["authorization"].split()[-1]
        seen.append(k)
        if k == ka and seen.count(ka) == 1:                     # key A's first POST: 429
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate"})
        return httpx.Response(200, json=resp())

    c = mk(tmp_path, handler)
    la = FileRateLimiter(tmp_path / "a.json", rpm=600, burst=5)
    lb = FileRateLimiter(tmp_path / "b.json", rpm=600, burst=5)
    c.key_lanes = [(SecretStr(ka), la), (SecretStr(kb), lb)]
    outs = [run(c.chat(req(seed=i))) for i in range(4)]
    assert set(seen) == {ka, kb}                                 # both keys carry traffic
    assert {o.provider for o in outs} <= {"adaption:k0", "adaption:k1"}
    assert json.loads((tmp_path / "a.json").read_text())["n429"] == 1      # the 429 counted on A's bucket only
    assert json.loads((tmp_path / "b.json").read_text())["n429"] == 0
    assert all(o.text for o in outs)


def test_extra_keys_from_cfg_loads_each_key_by_path(tmp_path):
    for n, k in (("main", "pt_live_FAKEmain"), ("second", "label: pt_live_FAKEsecond")):
        (tmp_path / n).write_text(k + "\n")
    cfg = from_dict({"llm": {"adaption": {
        "enabled": True, "key_file": str(tmp_path / "main"), "rpm": 100, "rpm_shared": True,
        "rpm_state_path": str(tmp_path / "m.json"), "rpm_log_path": str(tmp_path / "m.log"),
        "ledger_path": str(tmp_path / "l.jsonl"),
        "extra_keys": [{"key_file": str(tmp_path / "second"), "rpm_state_path": str(tmp_path / "s.json"),
                        "rpm_log_path": str(tmp_path / "s.log")}]}}})
    c = AdaptionClient.from_cfg(cfg.llm.adaption)
    assert [k.get_secret_value() for k, _ in c.key_lanes] == ["pt_live_FAKEmain", "pt_live_FAKEsecond"]
    assert c.key_lanes[1][1].path == tmp_path / "s.json"
    base = from_dict({"llm": {"adaption": {"enabled": True}}})
    assert cfg.config_hash() == from_dict({"llm": {"adaption": {"enabled": True, "rpm": 100, "rpm_shared": True}}}).config_hash() == base.config_hash()
