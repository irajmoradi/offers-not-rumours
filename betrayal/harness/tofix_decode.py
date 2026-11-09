"""TOFIX: provider-timeout fix for the OpenRouter call path (default routing: NO top_k, NO provider.require_parameters).
Bug (out/empty_reply_audit.md): OpenRouter can answer HTTP 200 with {"error": {"message": "Provider timed out", "code": 504}}
and no `choices`. costpatch._one turns that into "" (an EMPTY reply -> idle move) and never retries or counts it failed.
Fix (logic reused from noor_decode.py, minus its decoding changes):
  apply(backend): wraps that backend's OpenAI client `chat.completions.create`. An error body (no choices, or an `error`
    field on the response or on choices[0]) is retried up to PROVIDER_RETRIES times with backoff; if it still fails, a
    RuntimeError is raised, so costpatch retries the whole call (5 attempts) and finally counts it FAILED (returns "{}").
    The request itself is passed through unchanged (costpatch's extra_body {"usage": {"include": True}} only).
  install_guard(bk, COST): wraps APIBackend._one (after costpatch.install) to record, per backend `arm` tag (one tag per
    branch in run_tofix.py), every logical call's outcome: first attempt was an error body (= would have been idle under
    the old path), recovered, failed after all retries, other exceptions, empty successful text. It also enforces the
    pre-registered stop rule: if more than 5% of the last 200 logical calls failed (> 10 of 200), every further call
    raises FailRateStop and the run stops (no relaunch).
Nothing here changes game rules or prompts; with no backend wrapped (offline DetMock) the game is her code."""
import threading, collections, time, random

PROVIDER_RETRIES = 6
FAIL_WINDOW, FAIL_MAX_IN_WINDOW = 200, 10          # stop if > 10 failed of the last 200 logical calls (> 5%)
_L = threading.Lock()
_TL = threading.local()
STATS = collections.defaultdict(lambda: collections.Counter())      # arm -> counters
ERR_CODES = collections.defaultdict(collections.Counter)            # arm -> error code -> error bodies
PROVIDERS = collections.defaultdict(collections.Counter)            # arm -> provider -> served calls
WINDOW = collections.deque(maxlen=FAIL_WINDOW)
STOP = {"tripped": False, "why": None, "at_call": None}
N_CALLS = [0]

class FailRateStop(BaseException):
    pass

def _err_of(r):
    err = getattr(r, "error", None)
    if err is None and getattr(r, "model_extra", None):
        err = r.model_extra.get("error")
    if err is None and r.choices:
        c0 = r.choices[0]
        err = getattr(c0, "error", None)
        if err is None and getattr(c0, "model_extra", None):
            err = c0.model_extra.get("error")
    return err

def _st():
    s = getattr(_TL, "s", None)
    if s is None:
        s = _TL.s = dict(invocations=0, error_bodies=0, first_error_body=None, ok=None, exc=0)
    return s

def apply(backend):
    comp = backend.client.chat.completions
    orig = comp.create
    arm = lambda: str(getattr(backend, "arm", "untagged"))
    def create(**kw):
        s = _st(); s["invocations"] += 1
        err = None
        try:
            for attempt in range(PROVIDER_RETRIES):
                r = orig(**kw)
                err = _err_of(r)
                if r.choices and err is None:
                    break
                s["error_bodies"] += 1
                if s["first_error_body"] is None: s["first_error_body"] = True
                code = str(err.get("code") if isinstance(err, dict) else err)[:40]
                with _L: ERR_CODES[arm()][code] += 1
                time.sleep(1.5 * (attempt + 1) + random.random())
            else:
                raise RuntimeError(f"provider error after {PROVIDER_RETRIES} tries: {str(err)[:120]}")
        except Exception:
            s["ok"] = False
            if s["first_error_body"] is None: s["first_error_body"] = False
            s["exc"] += 1
            raise
        if s["first_error_body"] is None: s["first_error_body"] = False
        s["ok"] = True
        prov = getattr(r, "provider", None)
        if prov is None and getattr(r, "model_extra", None):
            prov = r.model_extra.get("provider")
        with _L: PROVIDERS[arm()][str(prov)] += 1
        return r
    comp.create = create
    backend.tofix = dict(provider_retries=PROVIDER_RETRIES, top_k=None, require_parameters=False)
    return backend

def install_guard(bk, COST):
    orig_one = bk.APIBackend._one
    def _one(self, system, user, seed):
        if STOP["tripped"]:
            raise FailRateStop(STOP["why"])
        _TL.s = None; s = _st()
        txt = orig_one(self, system, user, seed)
        a = str(getattr(self, "arm", "untagged"))
        failed = s["ok"] is False          # (offline mock: no create invocations -> never failed)
        with _L:
            c = STATS[a]; N_CALLS[0] += 1
            c["calls"] += 1
            c["create_invocations"] += s["invocations"]
            c["error_bodies"] += s["error_bodies"]
            if s["first_error_body"]: c["first_attempt_error_body"] += 1        # old path: this call would be idle
            if s["error_bodies"]: c["calls_with_any_error_body"] += 1
            if s["exc"] and s["error_bodies"] == 0: c["calls_with_other_exception"] += 1
            if failed: c["failed"] += 1
            elif s["error_bodies"] or s["exc"]: c["recovered"] += 1
            if not failed and not (txt or "").strip(): c["empty_success"] += 1     # genuine empty completion (kept as her code)
            WINDOW.append(1 if failed else 0)
            if sum(WINDOW) > FAIL_MAX_IN_WINDOW and not STOP["tripped"]:
                STOP.update(tripped=True, at_call=N_CALLS[0],
                            why=f"fail-rate stop: {sum(WINDOW)} of the last {len(WINDOW)} calls failed (> 5%)")
                print("TOFIX", STOP["why"], flush=True)
        return txt
    bk.APIBackend._one = _one

def snapshot():
    with _L:
        return dict(stats={a: dict(c) for a, c in STATS.items()}, error_codes={a: dict(c) for a, c in ERR_CODES.items()},
                    providers={a: dict(c) for a, c in PROVIDERS.items()}, stop=dict(STOP), window_failed=int(sum(WINDOW)),
                    window_len=len(WINDOW), total_calls=N_CALLS[0])
