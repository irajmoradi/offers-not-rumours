"""Adaption Labs backend (POST {base_url}/responses, an OpenAI-Responses-style API). OFF by default.

Request mapping (engine ChatRequest -> Responses body):
  system messages      -> `instructions` (joined with blank lines)
  user/assistant msgs  -> `input`: [{role, content}, ...]
  max_tokens           -> `max_output_tokens`; OMITTED when None (no cap)
  temperature          -> `temperature`;   store=false always;   stream=true (llm.adaption.stream)
  reasoning_effort     -> `reasoning: {effort: X}` (undocumented but honoured). "minimal" is broken on
                          Adaption (it reasons MORE, runs away) and is sent as "low" with a warning.
  response_format      -> not sent (no text.format documented): JSON is left to the prompt
  seed / top_p / stop / thinking -> not sent (undocumented)

Streaming (SSE): a non-streamed call gets an HTML "504 Gateway Time-out" at ~60 s; streamed calls run
to completion. Text = the final response's output_text (else the accumulated response.output_text.delta
events), reasoning = final reasoning items (else reasoning_text deltas), usage + cost = the final
event's (response.completed / response.incomplete / response.failed) response.usage. httpx read
timeout llm.adaption.read_timeout_s (~300 s, between chunks).

Response mapping: status completed -> finish "stop"; incomplete -> "length"; usage.{input_tokens,
output_tokens, output_tokens_details.reasoning_tokens, cost} -> Usage + cost_usd; provider "adaption".

Errors / retries (OpenAICompatClient's retry loop; each failure kind is counted in `fail_counts`):
  429                       RateLimited: up to max_retries_429 retries, exponential backoff with full
                            jitter, floored at Retry-After, capped at rate_limit_max_delay_s
  5xx (incl. HTML 504), non-JSON / non-SSE body, stream cut before the final event, timeouts,
  connection errors, status failed/cancelled/..., empty completed text
                            TransientError: retry.max_retries retries, full-jitter backoff
  401/403 -> AuthError; other 4xx -> LLMError (never retried)
  incomplete + EMPTY text   with a cap: re-asked ONCE in the same attempt at max_tokens x
                            incomplete_bump (both attempts billed); still empty -> finish "length" with
                            empty text (the agent's parse retry handles it). Without a cap: TransientError.

Rate limits (Adaption's looks like a shared ~450-500 requests/min budget; recovery ~90 s): per-client
llm.adaption.max_concurrency, a process-wide in-flight cap shared by every Adaption client
(llm.adaption.global_max_concurrency, default 300), and a process-wide token bucket over request starts
(llm.adaption.rpm, default 380/min, burst llm.adaption.rpm_burst = 60; any 60 s window <= burst + rpm).
Both are PER PROCESS: N processes sharing one key need rpm / N each. Cache, single-flight and the budget hooks are
OpenAICompatClient's. Spend: its OWN GlobalLedger (runs/adaption_ledger.jsonl,
llm.adaption.global_cap_usd) plus the run's shared RunBudget (llm.budget_usd).
Key: secrets.load_adaption_key() (key file first, else env ADAPTION_API_KEY), held only as a
SecretStr, registered with redact(); never logged, cached or recorded.

Imports: config, llm.base, llm.budget, llm.cache, llm.openai_compat, llm.secrets, httpx.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from swarm.config import REPO_ROOT, AdaptionCfg, RetryCfg
from swarm.llm.base import (
    AuthError,
    BudgetExceeded,
    ChatRequest,
    ChatResponse,
    LLMError,
    RateLimited,
    TransientError,
    Usage,
)
from swarm.llm.budget import GlobalLedger, RunBudget, estimate_cost
from swarm.llm.cache import ResponseCache
from swarm.llm.openai_compat import OpenAICompatClient, Recorder
from swarm.llm.secrets import ADAPTION_ENV, load_adaption_key, redact

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.prod.adaptionlabs.ai/api/v1"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_LEDGER = REPO_ROOT / "runs" / "adaption_ledger.jsonl"
DEFAULT_CAP_USD = 3.0
FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")

# process-wide cap shared by every AdaptionClient: {id(loop): Semaphore}
_GLOBAL_SEM: dict[int, asyncio.Semaphore] = {}


def _global_sem(cap: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    sem = _GLOBAL_SEM.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        _GLOBAL_SEM.clear()                   # drop semaphores of dead loops
        _GLOBAL_SEM[loop_id] = sem
    return sem


class RpmBucket:
    """Token bucket over request STARTS (every POST attempt: retries and re-asks included): refills at
    rpm/60 per second up to `burst` tokens. Any 60 s window then sees at most burst + rpm starts.
    acquire() is FIFO (one waiter at a time holds the lock while it sleeps for the next token)."""

    def __init__(self, rpm: float, burst: int, clock: Any = time.monotonic) -> None:
        self.rate = rpm / 60.0
        self.burst = max(1, burst)
        self.tokens = float(self.burst)
        self.clock = clock
        self.last = clock()
        self.lock = asyncio.Lock()
        self.waited_s = 0.0                        # total time callers spent waiting (for reports)

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now

    async def acquire(self) -> None:
        async with self.lock:
            self._refill()
            while self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                self.waited_s += wait
                await asyncio.sleep(wait)
                self._refill()
            self.tokens -= 1.0


# process-wide RPM bucket shared by every AdaptionClient: {id(loop): RpmBucket}
_RPM: dict[int, RpmBucket] = {}


def _rpm_bucket(rpm: float, burst: int) -> RpmBucket:
    loop_id = id(asyncio.get_running_loop())
    b = _RPM.get(loop_id)
    if b is None:
        b = RpmBucket(rpm, burst)
        _RPM.clear()
        _RPM[loop_id] = b
    return b


class _Both:
    """async-with over two semaphores (global first, then per-client)."""

    def __init__(self, outer: asyncio.Semaphore, inner: asyncio.Semaphore) -> None:
        self.outer, self.inner = outer, inner

    async def __aenter__(self) -> None:
        await self.outer.acquire()
        try:
            await self.inner.acquire()
        except BaseException:
            self.outer.release()
            raise

    async def __aexit__(self, *exc: Any) -> None:
        self.inner.release()
        self.outer.release()


def _reasoning_text(output: list[Any]) -> str | None:
    """Concatenate the text of `reasoning` output items (content or summary parts), if any."""
    parts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for k in ("content", "summary"):
            for p in item.get(k) or []:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p, str):
                    parts.append(p)
    return "\n".join(parts) or None


def _message_text(output: list[Any]) -> str:
    """Fallback when output_text is absent: join output_text parts of `message` items."""
    parts: list[str] = []
    for item in output or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for p in item.get("content") or []:
                if isinstance(p, dict) and p.get("type") in ("output_text", "text") and isinstance(p.get("text"), str):
                    parts.append(p["text"])
    return "".join(parts)


def _usage_cost(data: dict[str, Any]) -> tuple[int, int, int, float | None]:
    u = data.get("usage") or {}
    det = u.get("output_tokens_details") or {}
    cost = u.get("cost")
    try:
        cost_f = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_f = None
    return (int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
            int(det.get("reasoning_tokens") or 0), cost_f)


def _retry_after(r: httpx.Response) -> float | None:
    ra = r.headers.get("retry-after")
    try:
        return float(ra) if ra is not None else None
    except ValueError:
        return None


def _prompt_tokens_est(body: dict[str, Any]) -> float:
    """Estimated prompt tokens of a Responses body (chars / 3.6: measured on the sweep's GLM prompts)."""
    n = len(str(body.get("instructions") or ""))
    inp = body.get("input")
    if isinstance(inp, list):
        n += sum(len(str(m.get("content") or "")) for m in inp if isinstance(m, dict))
    elif isinstance(inp, str):
        n += len(inp)
    return n / 3.6


class AdaptionClient(OpenAICompatClient):
    """Paid Adaption backend. Construct with enabled=True (mirrors llm.adaption.enabled)."""

    backend = "adaption"
    path = "/responses"
    key_hint = "Adaption key file / ADAPTION_API_KEY"

    def __init__(self, *, enabled: bool = False, api_key: SecretStr | None = None,
                 base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, max_concurrency: int = 32,
                 price_in_per_mtok: float = 1.0, price_out_per_mtok: float = 4.0,
                 uncapped_out_tokens: int = 8000, incomplete_bump: float = 2.0,
                 reasoning_effort: str | None = "low", stream: bool = True, read_timeout_s: float = 300.0,
                 global_max_concurrency: int = 300, rpm: float | None = 380.0, rpm_burst: int = 60,
                 max_retries_429: int = 8,
                 rate_limit_max_delay_s: float = 120.0, ledger: GlobalLedger | None = None,
                 run_budget: RunBudget | None = None, retry: RetryCfg | None = None,
                 cache: ResponseCache | None = None, recorder: Recorder | None = None, run_id: str = "",
                 transport: httpx.AsyncBaseTransport | None = None, sleep: Any = None) -> None:
        if not enabled:
            raise LLMError("Adaption is disabled: set llm.adaption.enabled=true to use it")
        if api_key is None:
            api_key = load_adaption_key()
        if api_key is None:
            raise LLMError(f"Adaption key not found: provide the key file or set {ADAPTION_ENV}")
        super().__init__(base_url=base_url, model=model, api_key=api_key, max_concurrency=max_concurrency,
                         retry=retry, cache=cache, recorder=recorder, run_id=run_id, transport=transport,
                         sleep=sleep)
        if reasoning_effort == "minimal":
            log.warning("adaption: reasoning effort 'minimal' is broken on Adaption (runaway reasoning); "
                        "sending 'low' instead")
            reasoning_effort = "low"
        self.reasoning_effort = reasoning_effort
        # realistic-framing: the effort changes completions but is not in the request, so it is part of the
        # cache key (swarm.llm.openai_compat: cache_model); and an optional cross-process RPM limiter
        self.cache_model = f"{model}@{reasoning_effort}" if reasoning_effort else model
        self.rate_limiter: Any = None             # swarm.llm.ratelimit.FileRateLimiter (llm.adaption.rpm)
        self.key_lanes: list[tuple[SecretStr | None, Any]] = []   # llm.adaption.extra_keys: (key, bucket) per key
        self._lane_rr = 0
        self.rpm_local: float | None = None        # llm.adaption.rpm_local (extra per-process cap with rpm_shared)
        self.call_deadline_s: float | None = None  # llm.adaption.call_deadline_s (total per-attempt deadline)
        self.stream = stream
        self.read_timeout_s = read_timeout_s
        self.global_max_concurrency = global_max_concurrency
        self.rpm, self.rpm_burst = rpm, rpm_burst
        self.max_retries_429 = max_retries_429
        self.rate_limit_max_delay_s = rate_limit_max_delay_s
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.uncapped_out = uncapped_out_tokens
        self.incomplete_bump = incomplete_bump
        self.ledger = ledger or GlobalLedger(DEFAULT_LEDGER, DEFAULT_CAP_USD)
        self.run_budget = run_budget
        self.fail_counts: Counter[str] = Counter()   # failure kind -> count (retried or not); for reports

    @property
    def incomplete_retries(self) -> int:
        """Re-asks after status=incomplete with empty text."""
        return self.fail_counts["incomplete_reask"]

    @classmethod
    def from_cfg(cls, cfg: AdaptionCfg, *, model: str | None = None, budget_usd: float | None = None,
                 reasoning_effort: str | None = None,
                 run_budget: RunBudget | None = None, ledger: GlobalLedger | None = None,
                 retry: RetryCfg | None = None, cache: ResponseCache | None = None,
                 recorder: Recorder | None = None, run_id: str = "",
                 api_key: SecretStr | None = None) -> AdaptionClient:
        """Build from cfg.llm.adaption; raises LLMError unless enabled. Relative ledger paths resolve
        against the repo root."""
        if ledger is None:
            p = Path(cfg.ledger_path)
            ledger = GlobalLedger(p if p.is_absolute() else REPO_ROOT / p, cfg.global_cap_usd)
        if run_budget is None and budget_usd is not None:
            run_budget = RunBudget(budget_usd)
        if api_key is None and cfg.key_file:       # a second key: by path only, no env fallback, never echoed
            api_key = load_adaption_key(cfg.key_file, env={})
            if api_key is None:
                raise LLMError("Adaption key_file is set but empty or missing (path not shown)")
        c = cls(enabled=cfg.enabled, api_key=api_key, base_url=cfg.base_url, model=model or cfg.model,
                   max_concurrency=cfg.max_concurrency, price_in_per_mtok=cfg.price_in_per_mtok,
                   price_out_per_mtok=cfg.price_out_per_mtok, uncapped_out_tokens=cfg.uncapped_out_tokens,
                   incomplete_bump=cfg.incomplete_bump, reasoning_effort=reasoning_effort or cfg.reasoning_effort,
                   stream=cfg.stream, read_timeout_s=cfg.read_timeout_s,
                   global_max_concurrency=cfg.global_max_concurrency, rpm=cfg.rpm, rpm_burst=cfg.rpm_burst,
                   max_retries_429=cfg.max_retries_429,
                   rate_limit_max_delay_s=cfg.rate_limit_max_delay_s, ledger=ledger, run_budget=run_budget,
                   retry=retry, cache=cache, recorder=recorder, run_id=run_id)
        c.rpm_local = cfg.rpm_local
        c.call_deadline_s = cfg.call_deadline_s
        if cfg.rpm and cfg.rpm_shared:             # one adaptive token bucket file for every sweep process
            from swarm.llm.ratelimit import FileRateLimiter

            sp, lp = Path(cfg.rpm_state_path), Path(cfg.rpm_log_path)
            c.rate_limiter = FileRateLimiter(sp if sp.is_absolute() else REPO_ROOT / sp, cfg.rpm, cfg.rpm_burst,
                                             adaptive=cfg.rpm_adaptive, ceiling=cfg.rpm_ceiling,
                                             raise_every_s=cfg.rpm_raise_every_s, tpm=cfg.tpm,
                                             tpm_burst=cfg.tpm_burst, tpm_ceiling=cfg.tpm_ceiling,
                                             log_path=lp if lp.is_absolute() else REPO_ROOT / lp)
        c.key_lanes = []
        if cfg.extra_keys:                         # several keys for one run: (key, bucket) lanes, primary first
            from swarm.llm.ratelimit import FileRateLimiter

            if c.rate_limiter is None:
                raise LLMError("llm.adaption.extra_keys needs rpm + rpm_shared (one bucket per key)")
            c.key_lanes.append((c._api_key, c.rate_limiter))
            for ek in cfg.extra_keys:
                k = load_adaption_key(ek.get("key_file"), env={}) if ek.get("key_file") else load_adaption_key()
                if k is None:
                    raise LLMError("an llm.adaption.extra_keys key file is empty or missing (path not shown)")
                sp, lp = Path(str(ek["rpm_state_path"])), Path(str(ek["rpm_log_path"]))
                lim = FileRateLimiter(sp if sp.is_absolute() else REPO_ROOT / sp, cfg.rpm, cfg.rpm_burst,
                                      adaptive=cfg.rpm_adaptive, ceiling=cfg.rpm_ceiling,
                                      raise_every_s=cfg.rpm_raise_every_s, tpm=cfg.tpm, tpm_burst=cfg.tpm_burst,
                                      tpm_ceiling=cfg.tpm_ceiling, log_path=lp if lp.is_absolute() else REPO_ROOT / lp)
                c.key_lanes.append((k, lim))
        return c

    async def _pick_lane(self, cost: float) -> int:
        """llm.adaption.extra_keys: the first key (rotating start) whose bucket admits this POST now; else wait for
        the soonest one. Takes the request + `cost` prompt tokens from that key's bucket only."""
        n = len(self.key_lanes)
        while True:
            start = self._lane_rr
            self._lane_rr = (self._lane_rr + 1) % n
            waits = []
            for j in range(n):
                i = (start + j) % n
                w = self.key_lanes[i][1]._take(cost)
                if w == 0.0:
                    return i
                waits.append(w)
            await self._sleep(max(0.05, min(waits)))

    # ------------------------------------------------------------------ concurrency / retry policy
    def _loop_state(self) -> tuple[Any, httpx.AsyncClient]:
        sem, client = super()._loop_state()
        return _Both(_global_sem(self.global_max_concurrency), sem), client

    def _max_retries(self, e: LLMError) -> int:
        return self.max_retries_429 if isinstance(e, RateLimited) else self.retry.max_retries

    def _retry_delay(self, e: LLMError, attempt: int) -> float:
        if isinstance(e, RateLimited):            # exp backoff, full jitter on top of the Retry-After floor
            exp = min(self.rate_limit_max_delay_s, self.retry.base_delay_s * 2 ** attempt)
            floor = min(self.rate_limit_max_delay_s, e.retry_after_s or 0.0)
            return min(self.rate_limit_max_delay_s, floor + exp * self._jitter.random())
        return self._backoff(attempt)

    # ------------------------------------------------------------------ request
    def body(self, req: ChatRequest, max_output_tokens: int | None = None) -> dict[str, Any]:
        """The JSON body sent to /responses (no headers, no key)."""
        system = [m.content for m in req.messages if m.role == "system"]
        rest = [{"role": m.role, "content": m.content} for m in req.messages if m.role != "system"]
        b: dict[str, Any] = {"model": self.model, "input": rest if rest else "",
                             "temperature": req.temperature, "store": False}
        cap = max_output_tokens or req.max_tokens
        if cap is not None:
            b["max_output_tokens"] = int(cap)
        if system:
            b["instructions"] = "\n\n".join(system)
        if self.reasoning_effort:
            b["reasoning"] = {"effort": self.reasoning_effort}
        if self.stream:
            b["stream"] = True
        return b

    def _bumped(self, req: ChatRequest) -> int | None:
        if req.max_tokens is None:
            return None
        return max(req.max_tokens + 1, int(req.max_tokens * self.incomplete_bump))

    def _fail(self, kind: str, exc: LLMError) -> LLMError:
        self.fail_counts[kind] += 1
        return exc

    async def _check_status(self, r: httpx.Response) -> None:
        """Map an HTTP error status to an LLMError (reads the body only for the message)."""
        sc = r.status_code
        if sc < 400:
            return
        if sc in (401, 403):
            raise self._fail("auth", AuthError(f"HTTP {sc} from adaption: check the API key ({self.key_hint})"))
        if sc == 429:
            raise self._fail("429", RateLimited("HTTP 429 from adaption", retry_after_s=_retry_after(r)))
        if sc >= 500:
            kind = "504" if sc == 504 else "5xx"
            raise self._fail(kind, TransientError(f"HTTP {sc} from adaption"))
        await r.aread()
        raise self._fail("4xx", LLMError(f"HTTP {sc} from adaption: {redact(r.text[:300])}"))

    async def _read_sse(self, r: httpx.Response) -> dict[str, Any]:
        """Consume an SSE stream; return the final response object (+ accumulated deltas)."""
        text: list[str] = []
        reasoning: list[str] = []
        final: dict[str, Any] | None = None
        saw_data = False
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except ValueError:
                raise self._fail("nonjson", TransientError("non-JSON SSE event")) from None
            saw_data = True
            t = ev.get("type")
            if t == "response.output_text.delta":
                text.append(str(ev.get("delta") or ""))
            elif t in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                reasoning.append(str(ev.get("delta") or ""))
            elif t in FINAL_EVENTS:
                resp = ev.get("response")
                final = resp if isinstance(resp, dict) else {}
                final.setdefault("status", t.removeprefix("response."))
            elif t == "error":
                err = ev.get("error") if isinstance(ev.get("error"), dict) else ev
                msg = redact(str(err.get("message") or err.get("code") or "")[:200])
                raise self._fail("stream_error", TransientError(f"adaption stream error: {msg}"))
        if final is None:
            kind = "stream_cut" if saw_data else "nonjson"
            raise self._fail(kind, TransientError("adaption stream ended without a final event"
                                                  if saw_data else "non-SSE response body"))
        final["_stream_text"] = "".join(text)
        final["_stream_reasoning"] = "".join(reasoning)
        return final

    async def _post_raw(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        if self.key_lanes:                         # several keys: this POST's key = the first bucket that admits it
            i = await self._pick_lane(_prompt_tokens_est(body))
            key, lim = self.key_lanes[i]
            try:
                coro = self._post_raw_inner(client, body, api_key=key)
                res = await (asyncio.wait_for(coro, self.call_deadline_s) if self.call_deadline_s else coro)
                res["_key_lane"] = i
                return res
            except TimeoutError:
                raise self._fail("deadline", TransientError(
                    f"call exceeded the {self.call_deadline_s:.0f} s deadline")) from None
            except RateLimited:
                lim.on_429()
                raise
        if self.rate_limiter is not None:          # llm.adaption.rpm_shared: ONE adaptive bucket file for every
            if self.rpm_local:                     # a lower-priority lane: its own cap first, then the shared one
                await _rpm_bucket(self.rpm_local, self.rpm_burst).acquire()
            await self.rate_limiter.acquire(_prompt_tokens_est(body))   # each POST (retries, re-asks too) spends
                                                                         # one request + its prompt tokens
        elif self.rpm:                             # else adaption-test's per-process bucket over request starts
            await _rpm_bucket(self.rpm, self.rpm_burst).acquire()
        try:
            if self.call_deadline_s:
                try:
                    return await asyncio.wait_for(self._post_raw_inner(client, body), self.call_deadline_s)
                except TimeoutError:
                    raise self._fail("deadline", TransientError(
                        f"call exceeded the {self.call_deadline_s:.0f} s deadline")) from None
            return await self._post_raw_inner(client, body)
        except RateLimited:
            if self.rate_limiter is not None:      # adaptive RPM: a 429 burst lowers the shared rate
                self.rate_limiter.on_429()
            raise

    async def _post_raw_inner(self, client: httpx.AsyncClient, body: dict[str, Any], *,
                              api_key: SecretStr | None = None) -> dict[str, Any]:
        timeout = httpx.Timeout(self.read_timeout_s, connect=30.0)
        url = f"{self.base_url}{self.path}"
        headers = self._headers()
        if api_key is not None:                    # extra_keys lane: that key (never logged)
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        try:
            async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as r:
                await self._check_status(r)
                if "text/event-stream" in r.headers.get("content-type", ""):
                    return await self._read_sse(r)
                await r.aread()
                try:
                    data = r.json()
                except ValueError:
                    raise self._fail("nonjson", TransientError("non-JSON response body")) from None
                if not isinstance(data, dict):
                    raise self._fail("nonjson", TransientError("non-object JSON body"))
                return data
        except httpx.TimeoutException as e:
            raise self._fail("timeout", TransientError(f"timeout ({type(e).__name__})")) from None
        except httpx.TransportError as e:
            raise self._fail("conn", TransientError(f"connection error ({type(e).__name__})")) from None

    async def _post_once(self, client: httpx.AsyncClient, req: ChatRequest,
                         body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = await self._post_raw(client, body if body is not None else self.body(req))
        if self._empty_incomplete(data):
            bumped = self._bumped(req)
            if bumped is None:                         # no cap: nothing to raise; retry as transient
                raise self._fail("incomplete_empty", TransientError("incomplete with empty text (no cap)"))
            self.fail_counts["incomplete_reask"] += 1
            first = _usage_cost(data)
            data = await self._post_raw(client, self.body(req, bumped))
            data["_prior_usage"] = first               # the first attempt was billed too
        return data

    @staticmethod
    def _text(data: dict[str, Any]) -> str:
        t = data.get("output_text")
        if isinstance(t, str) and t:
            return t
        return _message_text(data.get("output") or []) or str(data.get("_stream_text") or "")

    def _empty_incomplete(self, data: dict[str, Any]) -> bool:
        return data.get("status") == "incomplete" and not self._text(data).strip()

    # ------------------------------------------------------------------ response
    def _to_response(self, req: ChatRequest, data: dict[str, Any], latency: float,
                     retries: int) -> ChatResponse:
        status = data.get("status")
        if status not in ("completed", "incomplete"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise self._fail("failed_status",
                             TransientError(f"adaption status={status!r}: {redact(str(msg or '')[:200])}"))
        text = self._text(data)
        reasoning = _reasoning_text(data.get("output") or []) or (data.get("_stream_reasoning") or None)
        if status == "completed" and not text.strip():
            raise self._fail("empty", TransientError("empty completion"))
        finish = "stop" if status == "completed" else "length"
        pin, pout, pr, pcost = _usage_cost(data)
        prior = data.get("_prior_usage")
        if prior:
            pin, pout, pr = pin + prior[0], pout + prior[1], pr + prior[2]
            pcost = None if pcost is None and prior[3] is None else (pcost or 0.0) + (prior[3] or 0.0)
        usage = Usage(prompt_tokens=pin, completion_tokens=pout, reasoning_tokens=pr)
        cost = pcost if pcost is not None else (pin * self.price_in + pout * self.price_out) / 1e6
        return ChatResponse(call_id=req.call_id, text=text, reasoning=reasoning, finish_reason=finish,
                            usage=usage, cost_usd=cost, latency_s=latency, cache_hit=False, retries=retries,
                            backend=self.backend, model=str(data.get("model") or self.model),
                            provider="adaption" if "_key_lane" not in data else f"adaption:k{data['_key_lane']}")

    # ------------------------------------------------------------------ budget
    def _estimate(self, req: ChatRequest) -> float:
        est = estimate_cost(req, self.price_in, self.price_out, uncapped_out=self.uncapped_out)
        bumped = self._bumped(req)
        if bumped is not None:                         # worst case covers the incomplete re-ask too
            est += bumped * self.price_out / 1e6
        return est

    async def _before_call(self, req: ChatRequest) -> Any:
        est = self._estimate(req)
        self.ledger.reserve(est)
        if self.run_budget is not None:
            try:
                self.run_budget.reserve(est)
            except BudgetExceeded:
                self.ledger.release(est)
                raise
        return est

    async def _after_call(self, req: ChatRequest, token: Any, resp: ChatResponse | None) -> None:
        est = float(token or 0.0)
        actual = resp.cost_usd if resp is not None else 0.0
        if resp is not None:
            self.ledger.record(run_id=self.run_id, call_id=req.call_id, model=self.model,
                               cost_usd=actual, estimate=est, role=req.meta.role)
        else:
            self.ledger.release(est)
        if self.run_budget is not None:
            self.run_budget.settle(est, actual)
