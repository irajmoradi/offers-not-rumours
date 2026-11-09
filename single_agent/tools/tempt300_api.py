"""Run the tempt300 scenes unsteered on Adaption-hosted models via the Responses API (streaming SSE). No GPU.

usage: python -I tempt300_api.py <states.json> <out.jsonl> <key_file> [--concurrency 8] [--budget 5.0]
                                 [--pilot 3] [--arms all|name,name]
Key: read ONLY from <key_file> (a KEY=value line; the pt_live_... token is extracted). Never printed or logged.
Request: system -> "instructions", user turn -> "input"; "reasoning": {"effort": X} when set; stream=true; no max_output_tokens.
Retries: 429 -> Retry-After (or exp backoff) + jitter; 5xx / network -> up to 4 tries. Budget: stops launching new calls
once cumulative usage cost >= budget - 0.20 USD. Resumable: skips (arm, state) pairs already in out.jsonl.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

import httpx

URL = "https://api.prod.adaptionlabs.ai/api/v1/responses"
ARMS = [("glm-53-flash@low", "glm-53-flash", "low"), ("glm-53-flash@high", "glm-53-flash", "high"),
        ("deepseek-v4-flash@high", "deepseek-v4-flash", "high"), ("gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it", None),
        ("gemma-4-31b-it", "gemma-4-31b-it", None), ("qwen3.8-27b@low", "qwen3.8-27b", "low")]


def load_key(path):
    if path in ("", "-", "env"):                      # release: read ADAPTION_API_KEY from the environment
        v = os.environ.get("ADAPTION_API_KEY", "").strip()
        if not v:
            raise SystemExit("set ADAPTION_API_KEY or pass a key file path")
        return v
    with open(path) as f:
        for line in f:
            m = re.search(r"(pt_live_[A-Za-z0-9_\-]+)", line)
            if m:
                return m.group(1)
    raise SystemExit("no pt_live_ token found in key file")


def find_cost(obj):
    """usage cost may sit at response.usage.cost or similar; search a few known keys."""
    if isinstance(obj, dict):
        for k in ("cost", "total_cost", "cost_usd"):
            if isinstance(obj.get(k), (int, float)):
                return float(obj[k])
        for v in obj.values():
            c = find_cost(v)
            if c is not None:
                return c
    return None


def output_text(resp):
    out = []
    for item in resp.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    out.append(c["text"])
    return "".join(out)


PAUSE_FILE = None
THROTTLE = {"n429": 0, "sem": None}


async def one_call(client, key, arm, st, sem, budget, stop_at, sample=0, temperature=None):
    name, model, effort = arm
    while PAUSE_FILE and os.path.exists(PAUSE_FILE):          # auto-pause (e.g. during the v1 re-test)
        await asyncio.sleep(10)
    if THROTTLE["n429"] > 20 and THROTTLE["sem"] is not None:  # back off to the lower concurrency after repeated 429s
        async with THROTTLE["sem"]:
            return await _one_call(client, key, arm, st, sem, budget, stop_at, sample, temperature)
    return await _one_call(client, key, arm, st, sem, budget, stop_at, sample, temperature)


async def _one_call(client, key, arm, st, sem, budget, stop_at, sample=0, temperature=None):
    name, model, effort = arm
    body = {"model": model, "instructions": st["system"], "input": [{"role": "user", "content": st["user"]}],
            "stream": True}
    if effort:
        body["reasoning"] = {"effort": effort}
    if temperature is not None:
        body["temperature"] = temperature
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    tries, err = 0, None
    async with sem:
        if budget["cost"] >= stop_at:
            return None
        while tries < 6:
            tries += 1
            t0 = time.time()
            deltas, final, ev_err = [], None, None
            try:
                async with client.stream("POST", URL, json=body, headers=headers, timeout=httpx.Timeout(600.0, connect=30.0)) as r:
                    if r.status_code == 429:
                        THROTTLE["n429"] += 1
                        ra = r.headers.get("retry-after")
                        wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(60, 2 ** tries)
                        await asyncio.sleep(wait + random.uniform(0, 1.5))
                        err = "429"
                        continue
                    if r.status_code >= 500:
                        err = f"http {r.status_code}"
                        await asyncio.sleep(min(30, 2 ** tries) + random.uniform(0, 1.5))
                        continue
                    if r.status_code != 200:
                        txt = (await r.aread()).decode(errors="replace")[:300]
                        return dict(arm=name, state_id=st["state_id"], sample=sample, error=f"http {r.status_code}: {txt}",
                                    latency_s=time.time() - t0)
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            ev = json.loads(data)
                        except ValueError:
                            continue
                        typ = ev.get("type", "")
                        if typ == "response.output_text.delta":
                            deltas.append(ev.get("delta", ""))
                        elif typ in ("response.completed", "response.incomplete", "response.failed"):
                            final = ev
                            if typ != "response.completed":
                                ev_err = typ
                            break
                        elif typ == "error":
                            ev_err = str(ev.get("message") or ev.get("error"))[:300]
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                err = f"{type(e).__name__}"
                await asyncio.sleep(min(30, 2 ** tries) + random.uniform(0, 1.5))
                continue
            lat = time.time() - t0
            resp = (final or {}).get("response", {}) if final else {}
            text = output_text(resp) or "".join(deltas)
            usage = resp.get("usage") or {}
            cost = find_cost(final) if final else None
            if cost:
                budget["cost"] += cost
            budget["n"] += 1
            return dict(arm=name, model=model, effort=effort, state_id=st["state_id"], sample=sample,
                        temperature=temperature, text=text, usage=usage,
                        cost=cost, latency_s=round(lat, 2), status=resp.get("status"), event_error=ev_err, tries=tries)
        return dict(arm=name, state_id=st["state_id"], sample=sample, error=f"gave up after {tries} tries ({err})")


async def main_async(args):
    key = load_key(args.key_file)
    states = json.load(open(args.states))
    arms = ARMS if args.arms == "all" else [a for a in ARMS if a[0] in args.arms.split(",")]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                if "error" not in r:
                    done.add((r["arm"], r["state_id"], r.get("sample", 0)))
            except ValueError:
                pass
    todo = [(a, s, k) for a in arms for s in (states[:args.pilot] if args.pilot else states)
            for k in range(args.samples) if (a[0], s["state_id"], k) not in done]
    budget = {"cost": float(args.spent), "n": 0}
    stop_at = args.budget - 0.20
    sem = asyncio.Semaphore(args.concurrency)
    global PAUSE_FILE
    PAUSE_FILE = args.pause_file or None
    THROTTLE["sem"] = asyncio.Semaphore(args.backoff_concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=args.concurrency + 4)) as client:
        tasks = [asyncio.create_task(one_call(client, key, a, s, sem, budget, stop_at, k, args.temperature))
                 for a, s, k in todo]
        with open(args.out, "a") as fo:
            k = 0
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                if rec is None:
                    continue
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                k += 1
                if k % 50 == 0 or k == len(todo):
                    print(f"{k}/{len(todo)} done, cost so far ${budget['cost']:.3f}, {time.time()-t0:.0f}s", flush=True)
    print(f"FINISHED calls={budget['n']} cost=${budget['cost']:.3f} wall={time.time()-t0:.0f}s"
          + ("  [BUDGET STOP]" if budget["cost"] >= stop_at else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("key_file")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--spent", type=float, default=0.0, help="cost already spent (pilot) counted against the budget")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--pause-file", default="")
    ap.add_argument("--backoff-concurrency", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
