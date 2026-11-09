"""One OpenAI-compatible async backend (httpx). vllm.py and openrouter.py are presets of it.

Per call, `OpenAICompatClient.chat(req)` does, in order:
  cache lookup -> concurrency semaphore -> pre-call hook (budget; first attempt only) -> retry loop -> POST
  -> post-call hook (cost) -> cache write -> accounting (totals + optional LLMCall recorder)

- Retries: 429 (honours Retry-After), 5xx, timeouts, connection errors and empty completions,
  with full-jitter exponential backoff. 400 is never retried; 401/403 raise AuthError at once.
- Requests carry messages and params only. Headers (and any API key) never reach a log, an
  exception message, the cache or an LLMCall.
- Single-flight: concurrent calls with the same cache key share ONE request (the followers get the
  leader's response marked cache_hit, cost 0). This lets paired fork branches run concurrently and
  still see an identical pre-fork prefix (needs the cache to be on; key = cache_key()).
- Semaphores and the httpx client are created per running event loop, so one client object can
  be reused across `asyncio.run` calls (tests, sweeps).

Imports: config, llm.base, llm.cache, llm.secrets, httpx. The only network I/O in the package.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import SecretStr

from swarm.config import RetryCfg
from swarm.llm.base import (
    AuthError,
    BudgetExceeded,
    ChatRequest,
    ChatResponse,
    LLMCall,
    LLMError,
    RateLimited,
    TransientError,
    Usage,
)
from swarm.llm.cache import ResponseCache, cache_key
from swarm.llm.secrets import redact

log = logging.getLogger(__name__)

Recorder = Callable[[LLMCall], None]


@dataclass
class Totals:
    """Running accounting for one client (goes into the manifest)."""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Manifest-ready summary (p50 latency instead of the raw list)."""
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        return {"calls": self.calls, "cache_hits": self.cache_hits, "errors": self.errors,
                "retries": self.retries, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "reasoning_tokens": self.reasoning_tokens,
                "cost_usd": round(self.cost_usd, 6), "latency_p50_s": round(p50, 3)}


class OpenAICompatClient:
    """Generic /chat/completions client. Subclasses override `_extra_body`, `_headers`,
    `_cost`, `_before_call` and `_after_call`."""

    backend = "openai_compat"

    def __init__(self, *, base_url: str, model: str, api_key: SecretStr | None = None,
                 max_concurrency: int = 8, retry: RetryCfg | None = None,
                 cache: ResponseCache | None = None, recorder: Recorder | None = None,
                 run_id: str = "", transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Callable[[float], Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.max_concurrency = max_concurrency
        self.retry = retry or RetryCfg()
        self.cache = cache
        self.recorder = recorder
        self.run_id = run_id
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._jitter = random.Random(0)   # backoff jitter only; never affects logged content
        self._per_loop: dict[int, tuple[asyncio.Semaphore, httpx.AsyncClient]] = {}
        self._inflight: dict[str, asyncio.Future[ChatResponse]] = {}
        self.totals = Totals()

    def __repr__(self) -> str:   # never show the key
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r})"

    # ------------------------------------------------------------------ hooks
    def _headers(self) -> dict[str, str]:
        key = self._api_key.get_secret_value() if self._api_key else "EMPTY"
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _extra_body(self, req: ChatRequest) -> dict[str, Any]:
        return {}

    def _cost(self, data: dict[str, Any], usage: Usage) -> float:
        return 0.0

    async def _before_call(self, req: ChatRequest) -> Any:
        """Budget check. Returns a token handed to _after_call. May raise BudgetExceeded."""
        return None

    async def _after_call(self, req: ChatRequest, token: Any, resp: ChatResponse | None) -> None:
        return None

    # ------------------------------------------------------------------ plumbing
    def _loop_state(self) -> tuple[asyncio.Semaphore, httpx.AsyncClient]:
        loop = asyncio.get_running_loop()
        st = self._per_loop.get(id(loop))
        if st is None:
            st = (asyncio.Semaphore(self.max_concurrency),
                  httpx.AsyncClient(transport=self._transport, timeout=None))
            self._per_loop = {id(loop): st}   # drop clients of dead loops
        return st

    def body(self, req: ChatRequest) -> dict[str, Any]:
        """The JSON body sent to /chat/completions (no headers, no key)."""
        b: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "seed": req.seed,
        }
        if req.top_p is not None:
            b["top_p"] = req.top_p
        if req.stop:
            b["stop"] = list(req.stop)
        if req.response_format:
            b["response_format"] = req.response_format
        b.update(self._extra_body(req))
        return b

    def _backoff(self, attempt: int) -> float:
        return min(self.retry.max_delay_s, self.retry.base_delay_s * 2 ** attempt) * self._jitter.random()

    async def _post_once(self, client: httpx.AsyncClient, req: ChatRequest) -> dict[str, Any]:
        timeout = self.retry.timeout_thinking_s if req.thinking else self.retry.timeout_s
        try:
            r = await client.post(f"{self.base_url}/chat/completions", json=self.body(req),
                                  headers=self._headers(), timeout=timeout)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout after {timeout}s ({type(e).__name__})") from None
        except httpx.TransportError as e:
            raise TransientError(f"connection error ({type(e).__name__})") from None
        if r.status_code in (401, 403):
            raise AuthError(f"HTTP {r.status_code} from {self.backend}: check the API key "
                            f"(OPENROUTER_API_KEY for openrouter)")
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            try:
                ra_s = float(ra) if ra is not None else None
            except ValueError:
                ra_s = None
            raise RateLimited(f"HTTP 429 from {self.backend}", retry_after_s=ra_s)
        if r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code} from {self.backend}")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code} from {self.backend}: {redact(r.text[:300])}")
        try:
            return r.json()
        except ValueError:
            raise TransientError("non-JSON response body") from None

    def _to_response(self, req: ChatRequest, data: dict[str, Any], latency: float,
                     retries: int) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            if data.get("error"):
                raise TransientError(f"provider error: {redact(str(data['error'])[:200])}")
            raise TransientError("no choices in response")
        ch = choices[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or None
        finish = ch.get("finish_reason")
        u = data.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        usage = Usage(prompt_tokens=int(u.get("prompt_tokens") or 0),
                      completion_tokens=int(u.get("completion_tokens") or 0),
                      reasoning_tokens=int(details.get("reasoning_tokens") or 0))
        if not text.strip() and finish != "length":
            raise TransientError("empty completion")
        return ChatResponse(call_id=req.call_id, text=text, reasoning=reasoning, finish_reason=finish,
                            usage=usage, cost_usd=self._cost(data, usage), latency_s=latency,
                            cache_hit=False, retries=retries, backend=self.backend,
                            model=str(data.get("model") or self.model),
                            provider=(str(data["provider"]) if data.get("provider") else None))

    async def _call_with_retry(self, req: ChatRequest, held: list[Any]) -> ChatResponse:
        """POST with retries. The budget hook runs INSIDE the concurrency semaphore on the first
        attempt (its token is appended to `held`), so only calls really in flight hold a reservation;
        calls queued behind the semaphore reserve nothing (the sweep's false budget_exceeded)."""
        sem, client = self._loop_state()
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                async with sem:
                    if not held:
                        held.append(await self._before_call(req))
                    data = await self._post_once(client, req)
                return self._to_response(req, data, time.perf_counter() - t0, attempt)
            except (TransientError, RateLimited) as e:
                if attempt >= self.retry.max_retries:
                    raise
                delay = self._backoff(attempt)
                if isinstance(e, RateLimited) and e.retry_after_s is not None:
                    delay = min(self.retry.max_delay_s, e.retry_after_s)
                log.warning("%s call %s attempt %d failed (%s); retrying in %.2fs",
                            self.backend, req.call_id, attempt, e, delay)
                attempt += 1
                self.totals.retries += 1
                await self._sleep(delay)

    def _record(self, req: ChatRequest, key: str | None, resp: ChatResponse | None,
                error: str | None) -> None:
        if self.recorder is None:
            return
        self.recorder(LLMCall(run_id=self.run_id, backend=self.backend, model=self.model,
                              cache_key=key, request=req, response=resp,
                              error=redact(error) if error else None))

    # ------------------------------------------------------------------ public
    async def chat(self, req: ChatRequest) -> ChatResponse:
        """LLMClient.chat. Raises AuthError, BudgetExceeded, CacheMiss, LLMError, TransientError."""
        key = cache_key(self.backend, self.model, req) if self.cache is not None else None
        if self.cache is not None and key is not None:
            hit = self.cache.get(key)
            if hit is None:
                fut = self._inflight.get(key)
                if fut is not None and fut.get_loop() is asyncio.get_running_loop():
                    try:
                        hit = await asyncio.shield(fut)
                    except Exception:
                        hit = None          # the leader failed: make our own call below
            if hit is not None:
                resp = hit.model_copy(update={"call_id": req.call_id, "cache_hit": True, "latency_s": 0.0,
                                              "cost_usd": 0.0, "retries": 0})
                self.totals.calls += 1
                self.totals.cache_hits += 1
                self._record(req, key, resp, None)
                return resp
            if key not in self._inflight or self._inflight[key].get_loop() is not asyncio.get_running_loop():
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                try:
                    resp = await self._chat_miss(req, key)
                except BaseException as e:
                    if not fut.done():
                        fut.set_exception(e if isinstance(e, Exception) else LLMError("cancelled"))
                        fut.exception()     # mark retrieved: no 'never retrieved' warning
                    raise
                else:
                    if not fut.done():
                        fut.set_result(resp)
                    return resp
                finally:
                    if self._inflight.get(key) is fut:
                        del self._inflight[key]
        return await self._chat_miss(req, key)

    async def _chat_miss(self, req: ChatRequest, key: str | None) -> ChatResponse:
        """Retry loop (budget hook inside the semaphore) -> cost hook -> cache write -> accounting."""
        held: list[Any] = []
        try:
            resp = await self._call_with_retry(req, held)
        except BudgetExceeded:
            raise                                 # refused before sending: nothing reserved or recorded
        except LLMError as e:
            if held:
                await self._after_call(req, held[0], None)
            self.totals.errors += 1
            self._record(req, key, None, f"{type(e).__name__}: {e}")
            raise
        except BaseException:                     # cancelled mid-call: release the reservation
            if held:
                await self._after_call(req, held[0], None)
            raise
        await self._after_call(req, held[0], resp)
        if self.cache is not None and key is not None and resp.text.strip():
            self.cache.put(key, resp)
        t = self.totals
        t.calls += 1
        t.prompt_tokens += resp.usage.prompt_tokens
        t.completion_tokens += resp.usage.completion_tokens
        t.reasoning_tokens += resp.usage.reasoning_tokens
        t.cost_usd += resp.cost_usd
        t.latency_s += resp.latency_s
        t.latencies.append(resp.latency_s)
        self._record(req, key, resp, None)
        return resp

    async def aclose(self) -> None:
        """Close the per-loop httpx clients."""
        for _, client in self._per_loop.values():
            await client.aclose()
        self._per_loop = {}
