"""Local vLLM preset (http://localhost:8000/v1, Qwen/Qwen3-8B; see infra/README.md).

- `req.thinking` maps to `chat_template_kwargs.enable_thinking` (Qwen3 thinks by default, so we
  always send the flag explicitly).
- Native CoT comes back in `message.reasoning` (the server runs the qwen3 reasoning parser).
- Price is 0; the seed is passed through, so with the cache a rerun is exact.

Imports: config, llm.base, llm.cache, llm.openai_compat, httpx.
"""

from __future__ import annotations

from typing import Any

import httpx

from swarm.config import RetryCfg, VLLMCfg
from swarm.llm.base import ChatRequest
from swarm.llm.cache import ResponseCache
from swarm.llm.openai_compat import OpenAICompatClient, Recorder

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_TOKENS = 700


class VLLMClient(OpenAICompatClient):
    """Async client for the local vLLM server. Implements LLMClient."""

    backend = "vllm"

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 max_concurrency: int = 64, retry: RetryCfg | None = None,
                 cache: ResponseCache | None = None, recorder: Recorder | None = None, run_id: str = "",
                 transport: httpx.AsyncBaseTransport | None = None, sleep: Any = None) -> None:
        super().__init__(base_url=base_url, model=model, api_key=None, max_concurrency=max_concurrency,
                         retry=retry, cache=cache, recorder=recorder, run_id=run_id, transport=transport,
                         sleep=sleep)

    @classmethod
    def from_cfg(cls, cfg: VLLMCfg, *, model: str | None = None, retry: RetryCfg | None = None,
                 cache: ResponseCache | None = None, recorder: Recorder | None = None,
                 run_id: str = "") -> VLLMClient:
        """Build from cfg.llm.vllm (model override = roles.<role>.model)."""
        return cls(base_url=cfg.base_url, model=model or cfg.model, max_concurrency=cfg.max_concurrency,
                   retry=retry, cache=cache, recorder=recorder, run_id=run_id)

    def _extra_body(self, req: ChatRequest) -> dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": bool(req.thinking)}}


async def health(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> bool:
    """True if the server answers GET /health with 200."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{root}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
