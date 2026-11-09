"""Live vLLM smoke (auto-skipped when localhost:8000 is down)."""

from __future__ import annotations

import asyncio

import pytest

from swarm.llm.vllm import VLLMClient
from swarm.schema import CallMeta, ChatMessage, ChatRequest


@pytest.mark.gpu
def test_live_vllm_smoke():
    c = VLLMClient()
    r = asyncio.run(c.chat(ChatRequest(
        call_id="smoke", messages=[ChatMessage(role="user", content='Reply with {"actions": []} only.')],
        max_tokens=50, temperature=0.0, seed=0, meta=CallMeta(role="agent"))))
    assert r.text and r.usage.completion_tokens > 0 and r.reasoning is None
