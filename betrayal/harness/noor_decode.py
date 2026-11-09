"""Noor spec item 2: decoding for EVERY agent call and the LLM forger: temperature 0.1 (set on the backend) and top_k 5,
with OpenRouter provider.require_parameters=true so only providers that honour top_k may serve a call.
apply(backend) wraps that backend's OpenAI client `chat.completions.create` and merges into extra_body:
    {"top_k": 5, "provider": {"require_parameters": True}}
(costpatch's {"usage": {"include": True}} is kept). It also tallies the provider that served each call
(PROVIDERS[model][provider] = calls), saved by the driver. Nothing else about the request changes.
Error bodies: OpenRouter may answer HTTP 200 with {"error": ...} and no choices (seen in the smoke: DeepInfra Gemma
"Provider timed out", code 504). costpatch would score that as an EMPTY reply (an idle move) without retrying. Here such
a response is retried up to PROVIDER_RETRIES times (counted in PROVIDER_ERRORS[model][code]); if it still fails, an
exception is raised so costpatch retries again and finally counts the call as FAILED (the driver's failed-call guard)."""
import threading, collections, time, random
TOP_K = 5
PROVIDER_RETRIES = 6
PROVIDERS = collections.defaultdict(collections.Counter)
PROVIDER_ERRORS = collections.defaultdict(collections.Counter)
_L = threading.Lock()

def apply(backend, top_k=TOP_K):
    comp = backend.client.chat.completions
    orig = comp.create
    model = backend.model
    def create(**kw):
        eb = dict(kw.pop("extra_body", None) or {})
        eb["top_k"] = top_k
        eb["provider"] = dict(eb.get("provider") or {}, require_parameters=True)
        for attempt in range(PROVIDER_RETRIES):
            r = orig(extra_body=eb, **kw)
            err = getattr(r, "error", None)
            if err is None and getattr(r, "model_extra", None):
                err = r.model_extra.get("error")
            if r.choices and err is None:
                break
            with _L:   # OpenRouter can return HTTP 200 with an error body and no choices (e.g. provider 504 timeout)
                PROVIDER_ERRORS[model][str((err or {}).get("code") if isinstance(err, dict) else err)] += 1
            time.sleep(1.5 * (attempt + 1) + random.random())
        else:
            raise RuntimeError(f"provider error after {PROVIDER_RETRIES} tries: {str(err)[:120]}")   # costpatch retries, then counts it failed
        prov = getattr(r, "provider", None)
        if prov is None and getattr(r, "model_extra", None):
            prov = r.model_extra.get("provider")
        with _L:
            PROVIDERS[model][str(prov)] += 1
        return r
    comp.create = create
    backend.noor_decode = dict(temperature=backend.temperature, top_k=top_k, require_parameters=True)
    return backend

# ---- quality stop rule (lead, 2026-11-09, pre-registered in prespec_noor.md section 7): per model, stop the run if the
# failed+empty share exceeds 5% over any 200 consecutive calls. "empty" = costpatch returned an empty reply; "failed" =
# costpatch gave up after retries (it returns "{}"). Counts are saved as QUALITY[model].
class QualityStop(BaseException):
    pass
QUALITY = collections.defaultdict(lambda: dict(calls=0, empty=0, failed=0, worst_window_bad=0, tripped=False))
def guard(backend, window=200, max_rate=0.05):
    from collections import deque
    orig = backend._one
    model = backend.model
    win = deque(maxlen=window)
    def _one(system, user, seed):
        q = QUALITY[model]
        if q["tripped"]:
            raise QualityStop(f"quality stop: {model} failed+empty > {max_rate:.0%} over {window} calls")
        txt = orig(system, user, seed)
        bad_e, bad_f = not (txt or "").strip(), txt == "{}"
        with _L:
            q["calls"] += 1; q["empty"] += int(bad_e); q["failed"] += int(bad_f)
            win.append(int(bad_e or bad_f))
            q["worst_window_bad"] = max(q["worst_window_bad"], sum(win))
            if len(win) == window and sum(win) > max_rate * window:
                q["tripped"] = True
        if q["tripped"]:
            raise QualityStop(f"quality stop: {model} failed+empty > {max_rate:.0%} over {window} calls")
        return txt
    backend._one = _one
    return backend
