"""LLM I/O records (llm/base.py re-exports them). Requests never contain headers or keys."""

from __future__ import annotations

from typing import Any, Literal

from swarm import SCHEMA_VERSION
from swarm.schema.base import AgentId, GameId, Record, RunId

Role = Literal["agent", "god", "judge", "summarizer"]


class ChatMessage(Record):
    """One chat message sent to an LLM."""
    role: Literal["system", "user", "assistant"]
    content: str


class CallMeta(Record):
    """Logged, never sent to the backend."""

    role: Role
    agent: AgentId | None = None
    game_id: GameId | None = None
    round: int | None = None
    attempt: int = 0
    purpose: str = "turn"
    prompt_set: str | None = None
    prompt_set_hash: str | None = None
    prompt_hash: str | None = None


class ChatRequest(Record):
    """Backend-independent LLM request. call_id and seed are deterministic."""
    call_id: str                       # rng.make_call_id(...) by the caller; deterministic
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float
    top_p: float | None = None
    seed: int
    thinking: bool = False
    response_format: dict[str, Any] | None = None
    stop: list[str] | None = None
    meta: CallMeta

    def sampling(self) -> dict[str, Any]:
        """Everything that affects the completion (cache-key input besides model/backend)."""
        return self.model_dump(mode="json", exclude={"call_id", "meta"})


class Usage(Record):
    """Token usage of one call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


class ChatResponse(Record):
    """Backend-independent LLM response (echoes call_id)."""
    call_id: str
    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hit: bool = False
    retries: int = 0
    backend: str = ""
    model: str = ""
    provider: str | None = None        # upstream provider that served the call (OpenRouter), if reported


class LLMCall(Record):
    """One line of llm_calls.jsonl (cache hits included)."""

    v: int = SCHEMA_VERSION
    run_id: RunId
    backend: str
    model: str
    cache_key: str | None = None
    request: ChatRequest
    response: ChatResponse | None = None
    error: str | None = None
