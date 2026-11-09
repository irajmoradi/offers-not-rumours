"""OpenRouter preset: off by default, key handling (never logged), global ledger cap, run budget."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from swarm.config import RetryCfg, from_dict
from swarm.llm.base import AuthError, BudgetExceeded, LLMError
from swarm.llm.cache import ResponseCache
from swarm.llm.openrouter import GlobalLedger, OpenRouterClient, RunBudget, estimate_cost, load_key
from swarm.llm.registry import build_client
from swarm.llm.secrets import install_log_redaction
from swarm.schema import CallMeta, ChatMessage, ChatRequest
from tests.conftest import TINY

KEY = "sk-or-v1-zqcanary0123456789abcdef0123456789abcdef"
FAST = RetryCfg(max_retries=1, base_delay_s=0.0, max_delay_s=0.0)


def req(seed=1, max_tokens=700, content="hello"):
    return ChatRequest(call_id=f"c{seed}", messages=[ChatMessage(role="user", content=content)],
                       max_tokens=max_tokens, temperature=0.6, seed=seed, meta=CallMeta(role="agent"))


def body(cost=0.001):
    return {"model": "qwen/qwen3.6-35b-a3b",
            "choices": [{"message": {"content": '{"actions": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost}}


def mk(tmp_path, handler, **kw):
    return OpenRouterClient(enabled=True, api_key=SecretStr(KEY), transport=httpx.MockTransport(handler),
                            ledger=kw.pop("ledger", GlobalLedger(tmp_path / "ledger.jsonl", 2.0)),
                            retry=FAST, **kw)


def run(c):
    return asyncio.run(c)


def test_disabled_by_default():
    with pytest.raises(LLMError):
        OpenRouterClient(api_key=SecretStr(KEY))
    cfg = from_dict(TINY)
    assert cfg.llm.openrouter.enabled is False


def test_registry_refuses_when_disabled():
    raw = {**TINY, "roles": {**TINY["roles"], "agent": {"backend": "openrouter"}},
           "llm": {"cache": {"mode": "off"}, "openrouter": {"enabled": True}}}
    cfg = from_dict(raw)
    object.__setattr__(cfg.llm.openrouter, "enabled", False)   # simulate a bypassed validator
    with pytest.raises(LLMError):
        build_client(cfg, "agent")


def test_body_and_cost(tmp_path):
    seen = []

    def h(r):
        seen.append((json.loads(r.content), r.headers.get("authorization")))
        return httpx.Response(200, json=body(0.0042))

    c = mk(tmp_path, h)
    r = run(c.chat(req()))
    b, auth = seen[0]
    assert b["model"] == "qwen/qwen3.6-35b-a3b" and b["reasoning"] == {"enabled": False}
    assert b["usage"] == {"include": True} and "chat_template_kwargs" not in b
    assert auth == f"Bearer {KEY}"
    assert r.cost_usd == pytest.approx(0.0042)
    led = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert led[0]["cost_usd"] == pytest.approx(0.0042) and led[0]["call_id"] == "c1"


def test_reasoning_effort_and_provider_sort(tmp_path):
    """Mandatory-reasoning models (gpt-oss) reject reasoning.enabled=false: send effort instead."""
    seen = []

    def h(r):
        seen.append(json.loads(r.content))
        out = body()
        out["choices"][0]["message"]["reasoning"] = "think think"
        return httpx.Response(200, json=out)

    c = mk(tmp_path, h, reasoning_effort="low", provider_sort="price")
    r = run(c.chat(req()))
    assert seen[0]["reasoning"] == {"effort": "low"} and seen[0]["provider"] == {"sort": "price"}
    assert r.text == '{"actions": []}' and r.reasoning == "think think"   # JSON is the content only
    cfg = from_dict({"llm": {"openrouter": {"enabled": True, "reasoning_effort": "low",
                                            "provider_sort": "price"}}})
    c2 = OpenRouterClient.from_cfg(cfg.llm.openrouter, api_key=SecretStr(KEY))
    assert (c2.reasoning_effort, c2.provider_sort) == ("low", "price")
    assert "provider" not in mk(tmp_path, h).body(req())


def test_global_cap_refuses_before_call(tmp_path):
    lp = tmp_path / "ledger.jsonl"
    lp.write_text(json.dumps({"cost_usd": 1.999}) + "\n")
    calls = []

    def h(r):
        calls.append(1)
        return httpx.Response(200, json=body())

    c = mk(tmp_path, h, ledger=GlobalLedger(lp, 2.0))
    with pytest.raises(BudgetExceeded) as ei:
        run(c.chat(req()))
    assert ei.value.scope == "global" and calls == []


def test_ledger_accumulates_across_clients(tmp_path):
    lp = tmp_path / "ledger.jsonl"
    for _ in range(3):
        run(mk(tmp_path, lambda r: httpx.Response(200, json=body(0.5)), ledger=GlobalLedger(lp, 2.0))
            .chat(req(max_tokens=10)))
    assert GlobalLedger(lp).total() == pytest.approx(1.5)
    with pytest.raises(BudgetExceeded):
        run(mk(tmp_path, lambda r: httpx.Response(200, json=body(0.5)), ledger=GlobalLedger(lp, 1.5))
            .chat(req(max_tokens=10)))


def test_run_budget(tmp_path):
    rb = RunBudget(0.001)
    c = mk(tmp_path, lambda r: httpx.Response(200, json=body()), run_budget=rb)
    with pytest.raises(BudgetExceeded) as ei:
        run(c.chat(req(max_tokens=700)))   # est = 700 * 4/1e6 = 0.0028 > 0.001
    assert ei.value.scope == "run"
    assert estimate_cost(req(max_tokens=700, content="x" * 35), 1.0, 4.0) == pytest.approx(0.00281)


def test_load_key_env_then_file(tmp_path):
    f = tmp_path / "keyfile"
    f.write_text(KEY + "\n")
    k = load_key(f, env={})
    assert k is not None and k.get_secret_value() == KEY and KEY not in repr(k) and KEY not in str(k)
    k2 = load_key(f, env={"OPENROUTER_API_KEY": "sk-or-envkey-123456789"})
    assert k2.get_secret_value() == "sk-or-envkey-123456789"
    assert load_key(tmp_path / "missing", env={}) is None


def test_key_never_logged(tmp_path, caplog, capsys):
    install_log_redaction()
    load_key(_write(tmp_path, KEY), env={})   # registers KEY with redact()
    recs = []

    def h401(r):
        return httpx.Response(401, json={"error": {"message": f"invalid key {KEY}"}})

    cache = ResponseCache(tmp_path / "cache.sqlite")
    c = mk(tmp_path, h401, recorder=recs.append, cache=cache)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(AuthError) as ei:
        run(c.chat(req()))
    # transient path too (logs a warning with the error)
    c2 = mk(tmp_path, lambda r: httpx.Response(500, text=f"oops {KEY}"), recorder=recs.append, cache=cache)
    with pytest.raises(LLMError) as ei2:
        run(c2.chat(req(seed=2)))
    c3 = mk(tmp_path, lambda r: httpx.Response(400, text=f"bad request {KEY}"), recorder=recs.append)
    with pytest.raises(LLMError) as ei3:
        run(c3.chat(req(seed=3)))
    c4 = mk(tmp_path, lambda r: httpx.Response(200, json=body()), recorder=recs.append, cache=cache)
    run(c4.chat(req(seed=4)))
    out = capsys.readouterr()
    blobs = [str(ei.value), repr(ei.value), str(ei2.value), str(ei3.value), repr(c), str(c.__dict__.get("_api_key")),
             caplog.text, out.out, out.err, *(r.model_dump_json() for r in recs),
             (tmp_path / "ledger.jsonl").read_text()]
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.name != "keyfile":
            blobs.append(p.read_bytes().decode("utf-8", "replace"))
    for b in blobs:
        assert KEY not in b


def _write(tmp_path, key):
    p = tmp_path / "keyfile"
    p.write_text(key)
    return p


@pytest.mark.openrouter
def test_live_openrouter_smoke(tmp_path):
    c = OpenRouterClient(enabled=True, ledger=GlobalLedger())
    r = run(c.chat(req(max_tokens=20, content="Reply with the word ok.")))
    assert r.text


def test_provider_pin_and_provider_logged(tmp_path):
    """provider_order/allow_fallbacks pin providers; the serving provider is kept on the response."""
    seen = []

    def h(r):
        seen.append(json.loads(r.content))
        out = body()
        out["provider"] = "SomeProvider"
        return httpx.Response(200, json=out)

    c = mk(tmp_path, h, provider_sort="price", provider_order=["SomeProvider"], allow_fallbacks=False)
    r = run(c.chat(req()))
    assert seen[0]["provider"] == {"sort": "price", "order": ["SomeProvider"], "allow_fallbacks": False}
    assert r.provider == "SomeProvider"
    cfg = from_dict({"llm": {"openrouter": {"enabled": True, "provider_order": ["A", "B"],
                                            "allow_fallbacks": False}}})
    c2 = OpenRouterClient.from_cfg(cfg.llm.openrouter, api_key=SecretStr(KEY))
    assert (c2.provider_order, c2.allow_fallbacks) == (["A", "B"], False)
    assert from_dict({}).config_hash() == from_dict(
        {"llm": {"openrouter": {"provider_order": None, "allow_fallbacks": None}}}).config_hash()
    assert from_dict({}).config_hash() != cfg.config_hash()
    plain = run(mk(tmp_path, lambda r: httpx.Response(200, json=body())).chat(req(seed=2)))
    assert plain.provider is None


def test_queued_calls_reserve_nothing(tmp_path):
    """Sweep incident: every PENDING call used to reserve its worst-case estimate before the
    concurrency semaphore, so 8 queued calls blew a cap that 2 in-flight calls fit easily
    (false budget_exceeded far below the cap). Now only calls in flight hold a reservation."""
    peak = {"reserved": 0.0}
    budget = RunBudget(0.0)

    async def slow(request):
        peak["reserved"] = max(peak["reserved"], budget.reserved)
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=body(cost=0.00001))

    est = estimate_cost(req(), 1.0, 4.0)
    budget.cap = 2.5 * est                       # room for 2 in-flight reservations, not 8
    c = OpenRouterClient(enabled=True, api_key=SecretStr(KEY), transport=httpx.MockTransport(slow),
                         ledger=GlobalLedger(tmp_path / "l.jsonl", 2.0), retry=FAST, max_concurrency=2,
                         run_budget=budget)

    async def many():
        return await asyncio.gather(*(c.chat(req(seed=i)) for i in range(8)))
    out = run(many())
    assert len(out) == 8 and peak["reserved"] <= 2 * est + 1e-12
    assert budget.reserved == 0.0 and abs(budget.spent - 8 * 0.00001) < 1e-12


def test_one_budget_per_run_across_paid_clients(tmp_path):
    """Review #9: two OpenRouter models in one run share ONE RunBudget and ONE GlobalLedger."""
    from swarm.runner.wiring import _paid_budget

    cfg = from_dict(TINY)
    shared: dict = {}
    b1, l1 = _paid_budget(cfg, shared)
    b2, l2 = _paid_budget(cfg, shared)
    assert b1 is b2 and l1 is l2 and b1.cap == cfg.llm.budget_usd
