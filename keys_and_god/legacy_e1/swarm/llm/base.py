"""LLM contract (WP0, FROZEN) + FakeLLM.

    class LLMClient(Protocol):
        async def chat(self, req: ChatRequest) -> ChatResponse

Backends (WP2): vllm and openrouter (one AsyncOpenAI-based class, two presets) and fake (below).
Callers build ChatRequest with a deterministic call_id (rng.make_call_id) and seed (rng.derive_seed).
The response echoes call_id. Errors: raise the classes below; never return an error string as text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from swarm.schema import ChatMessage, ChatRequest, ChatResponse, LLMCall, Usage

__all__ = [
    "ChatMessage", "ChatRequest", "ChatResponse", "Usage", "LLMCall", "LLMClient",
    "LLMError", "BudgetExceeded", "AuthError", "RateLimited", "TransientError",
    "FakeLLM", "NOOP_REPLY", "MALFORMED_REPLY",
]


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can serve a ChatRequest (backends, middleware, FakeLLM)."""
    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Serve one request."""
        ...


class LLMError(Exception):
    """Base for all LLM-layer errors."""


class BudgetExceeded(LLMError):
    """A paid call would exceed the per-run or global USD cap. Raised before the call."""
    def __init__(self, scope: str, spent: float, estimate: float, cap: float) -> None:
        super().__init__(f"budget exceeded ({scope}): spent={spent:.4f} + est={estimate:.4f} > cap={cap:.4f}")
        self.scope, self.spent, self.estimate, self.cap = scope, spent, estimate, cap


class AuthError(LLMError):
    """401/403. Never retried. Message must not contain the key."""


class RateLimited(LLMError):
    """429 from the provider; retry after retry_after_s if given."""
    def __init__(self, msg: str = "rate limited", retry_after_s: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class TransientError(LLMError):
    """5xx, timeouts, connection errors, empty completions: retried by WP2's retry layer."""


# ======================================================================= FakeLLM
NOOP_REPLY = json.dumps({"note": "", "actions": [{"kind": "noop"}]})
MALFORMED_REPLY = "Sure! Here is my move: {\"actions\": [ {\"kind\": "


def _words(s: str) -> int:
    return len(s.split())


class FakeLLM:
    """Deterministic, scriptable LLMClient for tests. Modes (combinable; precedence top to bottom):

      raise_on={call_idx: Exception}   raise on the k-th call (0-based)
      fail_every=k                     every k-th call (1-based: calls k, 2k, ...) returns MALFORMED_REPLY
      script=[text, ...]               canned replies in call order (then falls through to policy/default)
      policy=fn(req) -> str            reply computed from the request (order-independent; use this
                                       under concurrency)
      default                          NOOP_REPLY

    Also: reasoning (str or fn(req)->str, returned only when req.thinking), latency_s (awaited and
    reported), finish_reason. Usage is faked as whitespace word counts. Every request is recorded in
    `.requests` (in call order) for leak oracles.
    NOTE: script / fail_every / raise_on are keyed by call ORDER, so they are deterministic only when
    calls are sequential. Under concurrency use `policy` or `by_call_id`.
    """

    backend = "fake"

    def __init__(
        self,
        script: Sequence[str] | None = None,
        policy: Callable[[ChatRequest], str] | None = None,
        *,
        by_call_id: Mapping[str, str] | None = None,
        fail_every: int | None = None,
        raise_on: Mapping[int, Exception] | None = None,
        reasoning: str | Callable[[ChatRequest], str] | None = None,
        latency_s: float = 0.0,
        finish_reason: str = "stop",
        model: str = "fake",
    ) -> None:
        self.script = list(script or [])
        self.policy = policy
        self.by_call_id = dict(by_call_id or {})
        self.fail_every = fail_every
        self.raise_on = dict(raise_on or {})
        self.reasoning = reasoning
        self.latency_s = latency_s
        self.finish_reason = finish_reason
        self.model = model
        self.requests: list[ChatRequest] = []
        self._n = 0

    @property
    def n_calls(self) -> int:
        """Number of chat() calls so far."""
        return self._n

    def _text(self, idx: int, req: ChatRequest) -> str:
        if req.call_id in self.by_call_id:
            return self.by_call_id[req.call_id]
        if self.fail_every and (idx + 1) % self.fail_every == 0:
            return MALFORMED_REPLY
        if idx < len(self.script):
            return self.script[idx]
        if self.policy is not None:
            return self.policy(req)
        return NOOP_REPLY

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Serve one request."""
        idx = self._n
        self._n += 1
        self.requests.append(req)
        if idx in self.raise_on:
            raise self.raise_on[idx]
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        text = self._text(idx, req)
        reasoning = None
        if req.thinking and self.reasoning is not None:
            reasoning = self.reasoning(req) if callable(self.reasoning) else self.reasoning
        prompt_tokens = sum(_words(m.content) for m in req.messages)
        return ChatResponse(
            call_id=req.call_id, text=text, reasoning=reasoning, finish_reason=self.finish_reason,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=_words(text),
                        reasoning_tokens=_words(reasoning or "")),
            cost_usd=0.0, latency_s=self.latency_s, cache_hit=False, retries=0,
            backend=self.backend, model=self.model,
        )

    def sent_text(self, roles: tuple[str, ...] = ("agent", "summarizer")) -> str:
        """All message bytes sent for the given CallMeta roles (for truth-leak oracles)."""
        return "\n".join(m.content for r in self.requests if r.meta.role in roles for m in r.messages)


def hash_policy(choices: Sequence[str]) -> Callable[[ChatRequest], str]:
    """A deterministic, order-independent policy: picks a reply by sha256 of the request messages."""

    def _pick(req: ChatRequest) -> str:
        h = hashlib.sha256("".join(m.content for m in req.messages).encode()).digest()
        return choices[int.from_bytes(h[:4], "big") % len(choices)]

    return _pick
