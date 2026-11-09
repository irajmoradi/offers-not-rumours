"""The LLM cache key depends only on what is sent (backend, model, messages, seed, sampling params),
never on CallMeta: a prompt-template edit that leaves the rendered text identical (e.g. a new
{% if %} block that is off) changes PromptSet.hash / meta.prompt_set_hash but keeps every cache key,
seed and game id, so cached control games still replay for free."""

from __future__ import annotations

from swarm.agents.llm_agent import LLMAgent
from swarm.llm.base import FakeLLM
from swarm.llm.cache import cache_key
from swarm.schema import CallMeta, ChatMessage, ChatRequest
from tests.game.helpers import run_cfg


def _req(**meta):
    return ChatRequest(call_id="c1", messages=[ChatMessage(role="user", content="hi")], max_tokens=10,
                       temperature=0.0, seed=7, meta=CallMeta(role="agent", **meta))


def test_cache_key_ignores_call_meta():
    a = _req(prompt_set="agent/v1", prompt_set_hash="0" * 64, prompt_hash="x")
    b = _req(prompt_set="agent/v2", prompt_set_hash="f" * 64, prompt_hash="y", round=9, agent="z")
    assert cache_key("openrouter", "m", a) == cache_key("openrouter", "m", b)
    c = a.model_copy(update={"messages": [ChatMessage(role="user", content="hi!")]})
    assert cache_key("openrouter", "m", a) != cache_key("openrouter", "m", c)
    assert cache_key("openrouter", "m", a) != cache_key("openrouter", "m", a.model_copy(update={"seed": 8}))


def test_request_seed_and_call_id_ignore_prompt_set_hash():
    """Same rendered prompt -> same request apart from meta, whatever PromptSet.hash is."""
    cfg = run_cfg()
    ag = LLMAgent("ann", FakeLLM(), cfg)
    msgs = [ChatMessage(role="system", content=ag.system_text), ChatMessage(role="user", content="t")]
    r1 = ag._request(msgs, game_id="g1", round=1, attempt=0, purpose="turn")
    ag.prompts.hash = "deadbeef" * 8           # simulate an edited template set (same rendered text)
    r2 = ag._request(msgs, game_id="g1", round=1, attempt=0, purpose="turn")
    assert r1.meta.prompt_set_hash != r2.meta.prompt_set_hash
    assert r1.sampling() == r2.sampling() and r1.call_id == r2.call_id
    assert cache_key("fake", "m", r1) == cache_key("fake", "m", r2)
