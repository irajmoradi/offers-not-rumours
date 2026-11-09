"""Rate-limit probe on the MAIN Adaption key: one tiny streamed /responses call per model, printing the HTTP status,
every rate-limit-ish response header (names containing ratelimit / limit / retry / quota / request-id) and the
first 400 chars of a non-200 body (redacted). The key is loaded by path (load_adaption_key) and never printed.
    PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python infra/sweep100/probe_429.py glm-53-flash deepseek-v4-flash:big
(a ":big" suffix sends a ~16k-token sweep-sized prompt)
Cost: ~30 prompt + <=16 output tokens per call (not written to the sweep ledger; < $0.001 total)."""
import asyncio
import sys
import time

import httpx

from swarm.llm.adaption import AdaptionClient
from swarm.llm.secrets import load_adaption_key, redact
from swarm.schema import CallMeta, ChatMessage, ChatRequest

# big: a sweep-sized prompt (~16k tokens of neutral filler) to test a token-based limit at the size the sweep sends
FILLER = "The facility schedule lists rooms, dates and booking codes for each department. " * 1100 + "\n"
KEEP = ("ratelimit", "limit", "retry", "quota", "request-id", "x-request", "cf-ray", "server", "date")


async def probe(model: str, big: bool = False) -> None:
    key = load_adaption_key()
    c = AdaptionClient(enabled=True, api_key=key, model=model, max_concurrency=1)
    req = ChatRequest(call_id=f"probe:{model}", messages=[ChatMessage(role="user", content=(FILLER if big else "") + "Reply with the word OK.")],
                      max_tokens=16, temperature=0.0, seed=1, meta=CallMeta(role="agent"))
    body = c.body(req)
    body["reasoning"] = {"effort": "low"}
    url = c.base_url.rstrip("/") + c.path
    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as h:
        async with h.stream("POST", url, json=body, headers=c._headers()) as r:
            hdr = {k: v for k, v in r.headers.items() if any(s in k.lower() for s in KEEP)}
            text = ""
            if r.status_code != 200:
                await r.aread()
                text = redact(r.text[:400])
            else:
                async for _ in r.aiter_bytes():
                    pass
    print(f"{time.strftime('%H:%M:%S')} {model}{' BIG ~16k tokens' if big else ''}: HTTP {r.status_code} in {time.time() - t0:.1f}s")
    print("  headers:", {k: redact(v) for k, v in hdr.items()})
    if text:
        print("  body:", text)
    await c.aclose()


async def main() -> None:
    for m in sys.argv[1:] or ["glm-53-flash", "deepseek-v4-flash"]:
        await probe(m.removesuffix(":big"), big=m.endswith(":big"))


if __name__ == "__main__":
    asyncio.run(main())
