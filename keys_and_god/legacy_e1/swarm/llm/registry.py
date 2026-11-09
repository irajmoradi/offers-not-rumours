"""build_client(cfg, role) -> LLMClient: the one place that picks a backend for a role.

    client = build_client(cfg, "agent", run_id=rid, recorder=writer.llm_call)   # vllm by default

- `fake`: a FakeLLM (or the `fake=` instance you pass, for tests).
- `vllm`: VLLMClient from cfg.llm.vllm (model = roles.<role>.model or the backend default).
- `openrouter`: refused unless cfg.llm.openrouter.enabled; global $ cap via the ledger, per-run cap
  via `run_budget` (create one RunBudget per run and pass it to every build_client call).
The response cache is opened from cfg.llm.cache (relative paths resolve against the repo root).
Clients are cheap; build one per role per run.

Imports: config, llm.* (no game/agents).
"""

from __future__ import annotations

from pathlib import Path

from swarm.config import REPO_ROOT, RunConfig
from swarm.llm.base import FakeLLM, LLMClient, LLMError
from swarm.llm.cache import ResponseCache
from swarm.llm.openai_compat import Recorder
from swarm.llm.openrouter import OpenRouterClient, RunBudget
from swarm.llm.vllm import VLLMClient

_CACHES: dict[tuple[str, str], ResponseCache] = {}


def open_cache(cfg: RunConfig) -> ResponseCache | None:
    """Shared ResponseCache for cfg.llm.cache (None when mode=off)."""
    c = cfg.llm.cache
    if c.mode == "off":
        return None
    p = Path(c.path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    k = (str(p), c.mode)
    if k not in _CACHES:
        _CACHES[k] = ResponseCache(p, c.mode)
    return _CACHES[k]


def build_client(cfg: RunConfig, role: str = "agent", *, run_id: str = "",
                 recorder: Recorder | None = None, run_budget: RunBudget | None = None,
                 fake: FakeLLM | None = None, cache: ResponseCache | None | bool = True) -> LLMClient:
    """cache=True opens cfg.llm.cache; pass a ResponseCache to share one, or None/False for none."""
    backend, model = cfg.model_for(role)
    rc = cache if isinstance(cache, ResponseCache) else (open_cache(cfg) if cache is True else None)
    if backend == "fake":
        return fake if fake is not None else FakeLLM(model=model)
    if backend == "vllm":
        return VLLMClient.from_cfg(cfg.llm.vllm, model=model, retry=cfg.llm.retry, cache=rc,
                                   recorder=recorder, run_id=run_id)
    if backend == "openrouter":
        if not cfg.llm.openrouter.enabled:
            raise LLMError(f"roles.{role}.backend=openrouter but llm.openrouter.enabled is false")
        return OpenRouterClient.from_cfg(cfg.llm.openrouter, model=model, budget_usd=cfg.llm.budget_usd,
                                         run_budget=run_budget, retry=cfg.llm.retry, cache=rc,
                                         recorder=recorder, run_id=run_id)
    raise LLMError(f"unknown backend {backend!r}")
