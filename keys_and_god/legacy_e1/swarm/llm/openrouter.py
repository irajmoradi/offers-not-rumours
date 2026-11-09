"""OpenRouter preset (https://openrouter.ai/api/v1). OFF by default (`llm.openrouter.enabled`).

- Key: env OPENROUTER_API_KEY, else the raw-key file (`load_key()`; default
  secrets.DEFAULT_OPENROUTER_KEY_FILE). The key lives only inside the client as a SecretStr; it
  is registered with secrets.redact() and is never printed, logged, cached or recorded.
- `req.thinking` maps to `reasoning: {enabled: bool}`; usage/cost via `usage: {include: true}`.
  With `reasoning_effort` set (models whose reasoning is mandatory, e.g. openai/gpt-oss-*, reject
  enabled=false with HTTP 400) it sends `reasoning: {effort: <value>}` instead. The reasoning comes
  back in message.reasoning; only message.content is parsed as the agent's JSON.
  `provider_sort` adds `provider: {sort: <value>}` (e.g. "price": cheapest provider first, with
  OpenRouter's normal fallbacks). `provider_order` / `allow_fallbacks` add `provider: {order: [...],
  allow_fallbacks: <bool>}` (pin providers). The serving provider is recorded in ChatResponse.provider.
- Spend caps (swarm/llm/budget.py, re-exported here), both checked just before each call is sent
  against a worst-case estimate:
    global  `GlobalLedger` over runs/openrouter_ledger.jsonl (append-only, fcntl.flock), default $8
    run     `RunBudget` (llm.budget_usd), shared by every client of one run
  Exceeding either raises BudgetExceeded. Actual cost (provider-reported, else the price table)
  is appended to the ledger after the call.

Imports: config, llm.base, llm.cache, llm.openai_compat, llm.secrets (key loading is WP0's).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from swarm.config import DEFAULT_OPENROUTER_MODEL, REPO_ROOT, OpenRouterCfg, RetryCfg
from swarm.llm.base import BudgetExceeded, ChatRequest, ChatResponse, LLMError, Usage
from swarm.llm.budget import (  # noqa: F401  (re-exported: the old home of the budget layer)
    DEFAULT_LEDGER,
    GLOBAL_CAP_USD,
    GlobalLedger,
    RunBudget,
    estimate_cost,
)
from swarm.llm.cache import ResponseCache
from swarm.llm.openai_compat import OpenAICompatClient, Recorder
from swarm.llm.secrets import DEFAULT_OPENROUTER_KEY_FILE, OPENROUTER_ENV, load_openrouter_key

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def load_key(key_file: str | Path | None = DEFAULT_OPENROUTER_KEY_FILE, *,
             env: Mapping[str, str] | None = None) -> SecretStr | None:
    """Env OPENROUTER_API_KEY first, else the raw-key file. Returns a masked SecretStr or None."""
    return load_openrouter_key(key_file, env=env, env_var=OPENROUTER_ENV)


class OpenRouterClient(OpenAICompatClient):
    """Paid OpenRouter backend. Construct with enabled=True (mirrors llm.openrouter.enabled)."""

    backend = "openrouter"

    def __init__(self, *, enabled: bool = False, api_key: SecretStr | None = None,
                 base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_OPENROUTER_MODEL,
                 max_concurrency: int = 8, price_in_per_mtok: float = 1.0, price_out_per_mtok: float = 4.0,
                 ledger: GlobalLedger | None = None, run_budget: RunBudget | None = None,
                 retry: RetryCfg | None = None, cache: ResponseCache | None = None,
                 recorder: Recorder | None = None, run_id: str = "",
                 transport: httpx.AsyncBaseTransport | None = None, sleep: Any = None,
                 reasoning_effort: str | None = None, provider_sort: str | None = None,
                 provider_order: list[str] | None = None, allow_fallbacks: bool | None = None) -> None:
        if not enabled:
            raise LLMError("OpenRouter is disabled: set llm.openrouter.enabled=true to use it")
        if api_key is None:
            api_key = load_key()
        if api_key is None:
            raise LLMError(f"OpenRouter key not found: set {OPENROUTER_ENV} or provide the key file")
        super().__init__(base_url=base_url, model=model, api_key=api_key, max_concurrency=max_concurrency,
                         retry=retry, cache=cache, recorder=recorder, run_id=run_id, transport=transport,
                         sleep=sleep)
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.ledger = ledger or GlobalLedger()
        self.run_budget = run_budget
        self.reasoning_effort = reasoning_effort
        self.provider_sort = provider_sort
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks

    @classmethod
    def from_cfg(cls, cfg: OpenRouterCfg, *, model: str | None = None, budget_usd: float | None = None,
                 run_budget: RunBudget | None = None, retry: RetryCfg | None = None,
                 cache: ResponseCache | None = None, recorder: Recorder | None = None,
                 run_id: str = "", api_key: SecretStr | None = None) -> OpenRouterClient:
        """Build from cfg.llm.openrouter; raises LLMError unless enabled. Relative ledger paths resolve against the repo root."""
        ledger_path = Path(cfg.ledger_path)
        if not ledger_path.is_absolute():
            ledger_path = REPO_ROOT / ledger_path
        if run_budget is None and budget_usd is not None:
            run_budget = RunBudget(budget_usd)
        return cls(enabled=cfg.enabled, api_key=api_key, base_url=cfg.base_url, model=model or cfg.model,
                   max_concurrency=cfg.max_concurrency, price_in_per_mtok=cfg.price_in_per_mtok,
                   price_out_per_mtok=cfg.price_out_per_mtok,
                   ledger=GlobalLedger(ledger_path, cfg.global_cap_usd), run_budget=run_budget,
                   retry=retry, cache=cache, recorder=recorder, run_id=run_id,
                   reasoning_effort=cfg.reasoning_effort, provider_sort=cfg.provider_sort,
                   provider_order=cfg.provider_order, allow_fallbacks=cfg.allow_fallbacks)

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h["X-Title"] = "swarm-keys-and-god"
        return h

    def _extra_body(self, req: ChatRequest) -> dict[str, Any]:
        reasoning: dict[str, Any] = ({"effort": self.reasoning_effort} if self.reasoning_effort
                                     else {"enabled": bool(req.thinking)})
        body: dict[str, Any] = {"reasoning": reasoning, "usage": {"include": True}}
        prov: dict[str, Any] = {}
        if self.provider_sort:
            prov["sort"] = self.provider_sort
        if self.provider_order:
            prov["order"] = list(self.provider_order)
        if self.allow_fallbacks is not None:
            prov["allow_fallbacks"] = self.allow_fallbacks
        if prov:
            body["provider"] = prov
        return body

    def _cost(self, data: dict[str, Any], usage: Usage) -> float:
        reported = (data.get("usage") or {}).get("cost")
        if reported is not None:
            try:
                return float(reported)
            except (TypeError, ValueError):
                pass
        return (usage.prompt_tokens * self.price_in + usage.completion_tokens * self.price_out) / 1e6

    async def _before_call(self, req: ChatRequest) -> Any:
        est = estimate_cost(req, self.price_in, self.price_out)
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
                               cost_usd=actual, estimate=est)
        else:
            self.ledger.release(est)
        if self.run_budget is not None:
            self.run_budget.settle(est, actual)
