"""Plan C additions: global API concurrency semaphore (API_CONCURRENCY), 429/error counting,
MixedBackend model buckets sent concurrently instead of one after another, periodic tally save.
Harness-side patch for the reviewed swarm_llm modules (no game rules/prompts changed):
 - per-call OpenRouter cost tally (usage.cost) + hard cap
 - per-model request extras (reasoning effort / larger token budget for reasoning models)
 - optional separate judge backend for interview() verdicts (keeps the agent-model judge too)"""
import os, json, threading, time, random
COST = {"usd": 0.0, "calls": 0, "missing_cost": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "empty": 0, "failed": 0, "rate_limited_429": 0, "retried_errors": 0, "by_model": {}, "by_arm": {}}
_SEM = threading.BoundedSemaphore(int(os.environ.get("API_CONCURRENCY", "64")))
_LOCK = threading.Lock()
# reasoning models: keep reasoning short and give room for the JSON after it
MODEL_EXTRAS = {
    "openai/gpt-oss-20b": dict(extra_body={"reasoning": {"effort": "low"}}, min_tokens=900),
    "openai/gpt-5-mini": dict(extra_body={"reasoning": {"effort": "minimal"}}, min_tokens=1200),
}
class CostCapExceeded(BaseException):
    pass

def install(cap, out_dir, temperature_overrides=None):
    import backends as bk
    temperature_overrides = temperature_overrides or {}
    def _one(self, system, user, seed):
        if COST["usd"] > cap:
            raise CostCapExceeded(f"cost cap {cap} exceeded: {COST['usd']:.4f}")
        ex = MODEL_EXTRAS.get(self.model, {})
        kwargs = dict(model=self.model, max_tokens=max(self.max_new_tokens, ex.get("min_tokens", 0)),
                      temperature=temperature_overrides.get(self.model, self.temperature),
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      extra_body=dict({"usage": {"include": True}}, **ex.get("extra_body", {})))
        if self.model.startswith("openai/gpt-5"):
            kwargs.pop("temperature")   # gpt-5 family accepts only the default temperature
        for attempt in range(5):
            try:
                with _SEM:
                    r = self.client.chat.completions.create(**kwargs)
                u = r.usage
                c = getattr(u, "cost", None) if u is not None else None
                if c is None and u is not None and getattr(u, "model_extra", None):
                    c = u.model_extra.get("cost")
                txt = (r.choices[0].message.content or "") if r.choices else ""
                with _LOCK:
                    COST["calls"] += 1
                    m = COST["by_model"].setdefault(self.model, {"usd": 0.0, "calls": 0, "empty": 0})
                    m["calls"] += 1
                    if u is not None:
                        COST["prompt_tokens"] += u.prompt_tokens or 0
                        COST["completion_tokens"] += u.completion_tokens or 0
                    if c is None:
                        COST["missing_cost"] += 1
                    else:
                        COST["usd"] += float(c); m["usd"] += float(c)
                    if not txt.strip():
                        COST["empty"] += 1; m["empty"] += 1
                    a = COST["by_arm"].setdefault(str(getattr(self, "arm", "untagged")),
                                                  {"usd": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
                    a["calls"] += 1
                    if c is not None: a["usd"] += float(c)
                    if u is not None:
                        a["prompt_tokens"] += u.prompt_tokens or 0; a["completion_tokens"] += u.completion_tokens or 0
                return txt
            except Exception as e:
                with _LOCK:
                    COST["retried_errors"] += 1
                    if "429" in str(e) or "rate" in str(e).lower():
                        COST["rate_limited_429"] += 1
                if attempt == 4:
                    with _LOCK:
                        COST["failed"] += 1
                    print("API call failed after retries:", self.model, type(e).__name__, str(e)[:160], flush=True)
                    return "{}"
                time.sleep((2 ** attempt) + random.random())
    if os.environ.get("SMOKE") == "1":   # offline smoke test: no network, mock replies
        _mock = bk.MockBackend()
        import re as _re
        _rng = random.Random(0)
        _p_self = float(os.environ.get("SMOKE_SELFGIVE", "0"))   # emulate the 7B self-give failure
        def _one(self, system, user, seed):
            with _LOCK:
                COST["calls"] += 1
            out = _mock.chat([(system, user)])[0]
            m = _re.search(r"(?:for user|You are agent) (\d+)", user)
            if _p_self and m and '"action": "give"' in out and "INTERVIEW" not in user:
                with _LOCK:
                    hit = _rng.random() < _p_self
                if hit:
                    d = json.loads(out); d["to"] = int(m.group(1)); out = json.dumps(d)
                    with _LOCK:
                        COST["smoke_selfgive_injected"] = COST.get("smoke_selfgive_injected", 0) + 1
            return out
    bk.APIBackend._one = _one
    def _mixed_chat(self, pairs, seed=0):   # same routing/output as MixedBackend.chat, buckets in parallel
        from concurrent.futures import ThreadPoolExecutor
        if not pairs:
            return []
        buckets = {}
        for idx, (sysm, usr) in enumerate(pairs):
            buckets.setdefault(self._route(usr), []).append((idx, (sysm, usr)))
        out = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=max(1, len(buckets))) as ex:
            futs = {b: ex.submit(self.backends[b].chat, [p for _, p in items], seed) for b, items in buckets.items()}
            for b, items in buckets.items():
                for (idx, _), r in zip(items, futs[b].result()):
                    out[idx] = r
        return ["" if o is None else o for o in out]
    bk.MixedBackend.chat = _mixed_chat
    def save():
        with open(os.path.join(out_dir, "cost_tally.json"), "w") as f:
            json.dump(COST, f, indent=1)
    import atexit; atexit.register(save)
    return COST, save

JUDGE_SYS = "You are a strict, careful judge."
class JudgeRouter:
    """Sends interview() judge prompts to a separate judge backend; everything else to the
    agent backend. Also runs the agent backend's own verdicts for comparison (stored, not returned)."""
    def __init__(self, agent_backend, judge_backend, keep_agent_judge=True):
        self.a, self.j, self.keep = agent_backend, judge_backend, keep_agent_judge
        self.name = getattr(agent_backend, "name", "agent")
        self.agent_verdicts = []
        if hasattr(agent_backend, "model_for"):
            self.model_for = agent_backend.model_for
    def chat(self, pairs, seed=0):
        if pairs and all(s.startswith(JUDGE_SYS) for s, _ in pairs):
            if self.keep:
                self.agent_verdicts.append(self.a.chat(pairs, seed=seed))
            return self.j.chat(pairs, seed=seed)
        return self.a.chat(pairs, seed=seed)
