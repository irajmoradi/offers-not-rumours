"""Adaption Labs backend for the Plan C v2 harness (arm F). Synchronous, thread-safe, with the same
`chat(pairs, seed) -> [text]` interface as Jennifer's backends.

Written fresh for this harness. Request/stream handling follows keys_and_god/swarm/llm/adaption.py
(read, not imported: that one is async and tied to another framework). Her package's adaption_api.py
was read as untrusted reference and is not imported either.

POST {BASE}/responses with stream=true (a non-streamed call gets an HTML 504 at about 60 s). Body:
model, instructions (system), input (user), temperature, store=false, reasoning.effort (omitted for
"default"), and no max_output_tokens (a cap causes empty "incomplete" replies on reasoning models).
The SSE stream is read to the final event (response.completed / incomplete / failed). Text comes from
output_text, else the message items, else the accumulated deltas. Cost comes from response.usage.cost.

Limits, all process-wide:
 - concurrency semaphore ADAPTION_MAX_CONCURRENCY (default 48)
 - request-start token bucket ADAPTION_RPM (default 380/min, burst 60). Adaption's limit looks like
   a shared 450-500 rpm with about 90 s recovery.
 - token bucket ADAPTION_TPM (default 1.5M tokens/min, on an estimate of prompt chars/4)
 - 300 s read timeout. Retries: 429 / 5xx / timeouts / a stream cut off before the final event, up to
   5 attempts with jittered backoff. 401/403/other 4xx are never retried.
Spend: hard cap ADAPTION_CAP_USD on the summed usage.cost (default 3.0). It is recorded in costpatch's
COST under by_model / by_arm and COST['adaption_usd']. It is NOT added to COST['usd'], which tracks the
OpenRouter key.
Key: set by run_arm_f.py (from env ADAPTION_API_KEY, or a hidden getpass prompt) into adaption_backend.KEY; otherwise
read from the file at ADAPTION_KEY_FILE. It is never printed, logged or written to any output.
"""
import os, json, time, random, threading
import requests
import costpatch

BASE = os.environ.get("ADAPTION_BASE_URL", "https://api.prod.adaptionlabs.ai/api/v1").rstrip("/")
FINAL = ("response.completed", "response.incomplete", "response.failed")
_SEM = threading.BoundedSemaphore(int(os.environ.get("ADAPTION_MAX_CONCURRENCY", "8")))
_LOCK = threading.Lock()

class _Bucket:
    def __init__(self, per_min, burst):
        self.rate, self.burst, self.tokens, self.last = per_min / 60.0, float(burst), float(burst), time.monotonic()
        self.lock = threading.Lock()
    def take(self, n=1.0):
        while True:
            with self.lock:
                now = time.monotonic(); self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate); self.last = now
                if self.tokens >= n:
                    self.tokens -= n; return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 2.0))
RPM = _Bucket(float(os.environ.get("ADAPTION_RPM", "380")), int(os.environ.get("ADAPTION_RPM_BURST", "60")))

class AdaptiveTPM:
    """Per-key tokens-per-minute bucket (Adaption limits PER USER over a 60 s token window).
    Same adaptive rule as swarm_framing's FileRateLimiter, in-process and thread-safe:
      start at ADAPTION_TPM (default 150k/min), burst = 10 s worth.
      A 429 burst (>= ADAPTION_BURST_429 within 60 s, default 1: a per-user window that still rejects after a full 60 s
      pause means the rate is too high; at most one drop per 60 s) cuts the rate by 25% (floor ADAPTION_TPM_FLOOR, 40k)
      and empties the bucket.
      Every 10 min with ZERO 429s the rate goes up 10% (ceiling ADAPTION_TPM_CEIL, 400k).
    Cost per request = estimated prompt tokens (chars/4) + 300 expected output tokens. It is corrected after each
    reply with the real usage (input+output tokens), so the bucket tracks the tokens actually used.
    A 429 with Retry-After also sets a GLOBAL pause, so no thread starts a request until it ends."""
    def __init__(self):
        self.tpm = float(os.environ.get("ADAPTION_TPM", "150000"))
        self.ceil = float(os.environ.get("ADAPTION_TPM_CEIL", "400000")); self.floor = float(os.environ.get("ADAPTION_TPM_FLOOR", "40000"))
        self.tokens = self.tpm / 6; self.last = time.monotonic(); self.lock = threading.Lock()
        self.pause_until = 0.0; self.n429_window = []; self.last_drop = 0.0; self.last_raise = time.monotonic(); self.any429_since_raise = False
        self.log_path = os.environ.get("ADAPTION_LIMITER_LOG") or os.path.join(os.environ.get("OUT", "."), "adaption_limiter.jsonl")
        self.n429_total = 0
        self._log(dict(event="start", tpm=self.tpm))
    def _log(self, ev):
        ev = dict(ts=round(time.time(), 2), **ev)
        try:
            with open(self.log_path, "a") as f: f.write(json.dumps(ev) + "\n")
        except OSError:
            pass
    def _refill(self, now):
        self.tokens = min(self.tpm / 6, self.tokens + (now - self.last) * self.tpm / 60.0); self.last = now
        if now - self.last_raise >= 600:
            if not self.any429_since_raise and self.tpm < self.ceil:
                old = self.tpm; self.tpm = min(self.ceil, self.tpm * 1.10); self._log(dict(event="raise_tpm", tpm=round(self.tpm), frm=round(old)))
            self.last_raise, self.any429_since_raise = now, False
    def acquire(self, cost):
        while True:
            with self.lock:
                now = time.monotonic()
                if now < self.pause_until:
                    wait = self.pause_until - now
                else:
                    self._refill(now)
                    need = min(cost, self.tpm / 6)
                    if self.tokens >= need:
                        self.tokens -= cost; return
                    wait = (need - self.tokens) / (self.tpm / 60.0)
            time.sleep(min(wait, 5.0) + random.random() * 0.5)
    def correct(self, estimated, actual):
        with self.lock:
            if actual: self.tokens -= (actual - estimated)
            m = int(time.time() // 60)
            if getattr(self, "_m", None) != m:
                if getattr(self, "_m", None) is not None:
                    self._log(dict(event="minute", ok_calls=self._ok, tokens=self._tok, tpm=round(self.tpm)))
                self._m, self._ok, self._tok = m, 0, 0
            self._ok += 1; self._tok += int(actual or estimated)
    def on_429(self, retry_after):
        with self.lock:
            now = time.monotonic(); self.n429_total += 1; self.any429_since_raise = True
            self.pause_until = max(self.pause_until, now + retry_after + random.random() * 5)
            self.n429_window = [t for t in self.n429_window if now - t <= 60] + [now]
            ev = dict(event="429", tpm=round(self.tpm), retry_after=retry_after)
            if len(self.n429_window) >= int(os.environ.get("ADAPTION_BURST_429", "1")) and now - self.last_drop >= 60:
                old = self.tpm; self.tpm = max(self.floor, self.tpm * 0.75); self.tokens = min(self.tokens, 0.0)
                self.last_drop = now; self.n429_window = []; ev.update(drop_tpm_to=round(self.tpm), frm=round(old))
            self._log(ev)
TPM = AdaptiveTPM()

KEY = None   # set by run_arm_f.py from env ADAPTION_API_KEY or getpass (never printed, never written anywhere)

def _load_key():
    if KEY:
        return KEY
    path = os.environ.get("ADAPTION_KEY_FILE", "")
    if (not path or not os.path.isfile(path)) and os.environ.get("ADAPTION_API_KEY", "").strip():
        return os.environ["ADAPTION_API_KEY"].strip()          # release: env var fallback
    if not path or not os.path.isfile(path):
        raise SystemExit("Set ADAPTION_KEY_FILE (a path) or ADAPTION_API_KEY; the key is never printed.")
    raw = open(path).read().strip()
    import re as _re
    _toks = _re.findall(r"pt_live_[A-Za-z0-9_\-]+", raw)
    if _toks:
        return _toks[-1]
    if "=" in raw and "\n" not in raw:
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
    return raw

class AuthOrClientError(Exception):
    pass

class _RateLimited(Exception):
    def __init__(self, retry_after):
        super().__init__(f"HTTP 429 (retry after {retry_after:.0f}s)"); self.retry_after = retry_after

class AdaptionBackend:
    def __init__(self, model, effort=None, arm="F", temperature=0.7, max_workers=8):
        self.model, self.name, self.effort, self.arm = model, model, effort, arm
        self.temperature, self.max_workers = temperature, max_workers
        self._key = _load_key()
        self.s = requests.Session()
    def _headers(self):
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    def body(self, system, user):
        b = {"model": self.model, "instructions": system, "input": user, "temperature": self.temperature,
             "store": False, "stream": True}
        if self.effort and self.effort != "default":
            b["reasoning"] = {"effort": self.effort}
        return b
    @staticmethod
    def parse_sse(lines):
        text, final = [], None
        for line in lines:
            if isinstance(line, bytes): line = line.decode("utf-8", "replace")
            if not line or not line.startswith("data:"): continue
            p = line[5:].strip()
            if not p or p == "[DONE]": continue
            ev = json.loads(p); t = ev.get("type")
            if t == "response.output_text.delta": text.append(str(ev.get("delta") or ""))
            elif t in FINAL:
                final = ev.get("response") if isinstance(ev.get("response"), dict) else {}
                final.setdefault("status", t.split(".", 1)[1])
            elif t == "error":
                raise RuntimeError("adaption stream error")
        if final is None:
            raise RuntimeError("adaption stream ended without a final event")
        out = final.get("output_text")
        if not out:
            parts = [c.get("text", "") for it in (final.get("output") or []) if isinstance(it, dict) and it.get("type") == "message"
                     for c in (it.get("content") or []) if isinstance(c, dict) and c.get("type") in ("output_text", "text")]
            out = "".join(parts) or "".join(text)
        return out, (final.get("usage") or {})
    def _one(self, system, user):
        cap = float(os.environ.get("ADAPTION_CAP_USD", "3.0"))
        est = (len(system) + len(user)) / 4.0 + 300.0
        n429 = nother = 0
        while True:
            if costpatch.COST.get("adaption_usd", 0.0) > cap:
                raise costpatch.CostCapExceeded(f"adaption cap {cap} exceeded")
            RPM.take(1); TPM.acquire(est)
            try:
                with _SEM:
                    r = self.s.post(f"{BASE}/responses", data=json.dumps(self.body(system, user)), headers=self._headers(),
                                    stream=True, timeout=(30, 300))
                    sc = r.status_code
                    if sc in (401, 403) or (400 <= sc < 500 and sc != 429):
                        msg = r.text[:200] if sc not in (401, 403) else "auth"
                        r.close(); raise AuthOrClientError(f"HTTP {sc} from adaption ({self.model}): {msg}")
                    _rl = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
                    if _rl and (sc == 429 or not getattr(TPM, "_hdr_logged", 0) < 3 or True):
                        if getattr(TPM, "_hdr_n", 0) < 20:
                            TPM._hdr_n = getattr(TPM, "_hdr_n", 0) + 1; TPM._log(dict(event="ratelimit_headers", status=sc, headers=_rl))
                    if sc == 429:
                        try: ra = float(r.headers.get("retry-after") or 60)
                        except ValueError: ra = 60.0
                        r.close(); raise _RateLimited(max(60.0, ra))
                    if sc >= 500:
                        r.close(); raise RuntimeError(f"HTTP {sc}")
                    if "text/event-stream" in r.headers.get("content-type", ""):
                        txt, usage = self.parse_sse(r.iter_lines())
                    else:
                        d = r.json(); txt, usage = d.get("output_text") or "", d.get("usage") or {}
                TPM.correct(est, int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0))
                c = usage.get("cost")
                with costpatch._LOCK:
                    C = costpatch.COST; C["adaption_calls"] = C.get("adaption_calls", 0) + 1
                    if c is not None: C["adaption_usd"] = C.get("adaption_usd", 0.0) + float(c)
                    m = C["by_model"].setdefault(f"adaption:{self.model}", {"usd": 0.0, "calls": 0, "empty": 0})
                    m["calls"] += 1; m["usd"] += float(c or 0)
                    if not (txt or "").strip(): m["empty"] += 1
                    a = C["by_arm"].setdefault(str(self.arm), {"usd": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
                    a["calls"] += 1; a["usd"] += float(c or 0)
                    a["prompt_tokens"] += int(usage.get("input_tokens") or 0); a["completion_tokens"] += int(usage.get("output_tokens") or 0)
                return txt or ""
            except AuthOrClientError:
                raise
            except _RateLimited as e:
                n429 += 1; TPM.on_429(e.retry_after)
                with costpatch._LOCK:
                    C = costpatch.COST; C["adaption_429"] = C.get("adaption_429", 0) + 1
                    mk = time.strftime("%H:%M"); C.setdefault("adaption_429_per_min", {}); C["adaption_429_per_min"][mk] = C["adaption_429_per_min"].get(mk, 0) + 1
                if n429 > 8:
                    with costpatch._LOCK:
                        costpatch.COST["adaption_failed"] = costpatch.COST.get("adaption_failed", 0) + 1
                    print("Adaption call failed: 429 after 8 retries:", self.model, flush=True)
                    return "{}"
                continue          # TPM.acquire() waits out the global Retry-After pause (>= 60 s + jitter)
            except Exception as e:
                nother += 1
                with costpatch._LOCK:
                    costpatch.COST["adaption_retries"] = costpatch.COST.get("adaption_retries", 0) + 1
                attempt = nother - 1
                if attempt == 4:
                    with costpatch._LOCK:
                        costpatch.COST["adaption_failed"] = costpatch.COST.get("adaption_failed", 0) + 1
                    print("Adaption call failed after retries:", self.model, type(e).__name__, str(e)[:120], flush=True)
                    return "{}"
                time.sleep(min(60, (2 ** attempt) + random.random() * 2))
    def chat(self, pairs, seed=0):
        from concurrent.futures import ThreadPoolExecutor
        if not pairs: return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda p: self._one(p[0], p[1]), pairs))

def make_adaption_mixed(specs, arm="F", backend_cls=None):
    """specs: [(model, effort), ...]. Agent i is played by specs[i % len(specs)], routed by her MixedBackend._route."""
    import backends as bk
    m = bk.MixedBackend.__new__(bk.MixedBackend)
    m.models = [s[0] for s in specs]
    m.backends = [(backend_cls or AdaptionBackend)(mod, eff, arm=arm) for mod, eff in specs]
    m.assignment = None
    m.name = "adaption-mixed(" + ",".join(m.models) + ")"
    return m
