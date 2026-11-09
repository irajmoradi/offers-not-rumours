"""Tiny concurrency benchmark for the local vLLM server.

Usage:
  python bench.py                 # 32 concurrent requests, 256 max tokens, thinking off
  python bench.py -n 64 -c 64 --max-tokens 512 --thinking
Only needs the `openai` package (installed in the `swarm` env alongside vllm).
"""
import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

PROMPTS = [
    "Explain in a few paragraphs why the sky is blue.",
    "Write a short story about a robot learning to garden.",
    "List ten considerations when deploying an AI agent with tool access.",
    "Summarize the main arguments for and against AI model open-sourcing.",
]


async def one(client, model, i, max_tokens, thinking):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)] + f" (request {i})"}],
        max_tokens=max_tokens,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return time.perf_counter() - t0, r.usage.completion_tokens, r.usage.prompt_tokens


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="default: first model served")
    ap.add_argument("-n", "--num", type=int, default=32, help="total requests")
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    a = ap.parse_args()

    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)
    model = a.model or (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(a.concurrency)

    async def guarded(i):
        async with sem:
            return await one(client, model, i, a.max_tokens, a.thinking)

    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(i) for i in range(a.num)))
    wall = time.perf_counter() - t0

    lats = sorted(r[0] for r in res)
    out_toks = sum(r[1] for r in res)
    in_toks = sum(r[2] for r in res)
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    print(f"model={model} requests={a.num} concurrency={a.concurrency} max_tokens={a.max_tokens} thinking={a.thinking}")
    print(f"wall={wall:.2f}s  prompt_tokens={in_toks}  completion_tokens={out_toks}")
    print(f"aggregate output throughput = {out_toks / wall:.1f} tok/s  ({a.num / wall:.2f} req/s)")
    print(f"latency p50={statistics.median(lats):.2f}s  p90={p(0.9):.2f}s  max={lats[-1]:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
