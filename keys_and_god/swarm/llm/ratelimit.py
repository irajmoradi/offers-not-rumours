"""Cross-process, adaptive requests-per-minute limiter (token bucket in a shared state file).

Adaption Labs enforces a per-key request budget (bursts of ~450-850/min were seen; the sustained rate is
unknown). Every run process of a sweep shares one bucket file, so the budget holds globally however many
runs / games are in flight:

    lim = FileRateLimiter("runs/adaption_rpm.json", rpm=380, adaptive=True, ceiling=900,
                          log_path="runs/adaption_rpm_log.jsonl")
    await lim.acquire()          # before EVERY HTTP attempt (retries included); cache hits never call it
    lim.on_429()                 # the client reports each HTTP 429

State {"tokens", "ts", "rpm", "last_adjust", "n429", "first429"}: refilled at rpm/60 tokens per second up to
`burst`, read and written under an exclusive flock (microseconds per request). When the bucket is empty,
acquire() sleeps until the next token is due (plus jitter, so waiters do not wake in lockstep).
A "ceiling" key in the state file lowers the ceiling live for every process (no restart needed).
Adaptive mode (lead, 2026-11-08): start at `rpm`; every `raise_every_s` (10 min) with ZERO 429s, raise the rate
by 10% up to `ceiling` (900); on a 429 BURST (>= burst_429 429s within window_429_s) drop it by 25% (not below
`floor`), at most once per window_429_s (a 429 storm must not cascade to the floor: the first live run fell
460 -> 60 within one second on 97 429s). Every change is appended to `log_path` (jsonl: ts, rpm, event), which the live results table reads
for the effective RPM. The in-flight cap is separate (the client's semaphore); 429 backoff stays the
client's job. Imports: stdlib only.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import random
import time
from pathlib import Path
from typing import Any


class FileRateLimiter:
    def __init__(self, path: str | Path, rpm: float, burst: float | None = None, *, adaptive: bool = False,
                 tpm: float | None = None, tpm_burst: float | None = None, tpm_ceiling: float | None = None,
                 tpm_floor: float = 300_000.0,
                 ceiling: float = 900.0, floor: float = 60.0, raise_every_s: float = 600.0,
                 raise_factor: float = 1.10, drop_factor: float = 0.75, burst_429: int = 3,
                 window_429_s: float = 60.0, log_path: str | Path | None = None,
                 clock=time.time, sleep=asyncio.sleep, seed: int | None = None) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.path = Path(path)
        self.rpm0 = float(rpm)
        self.burst_cfg = burst
        self.adaptive, self.ceiling, self.floor = adaptive, float(ceiling), float(floor)
        self.raise_every_s, self.raise_factor, self.drop_factor = raise_every_s, raise_factor, drop_factor
        self.burst_429, self.window_429_s = burst_429, window_429_s
        self.log_path = Path(log_path) if log_path else None
        # input-tokens-per-minute bucket (Adaption's binding limit: 429 storms at >= ~1.75M prompt tokens/min)
        self.tpm0 = float(tpm) if tpm else None
        self.tpm_burst = float(tpm_burst if tpm_burst is not None else (tpm or 0) / 6)
        self.tpm_ceiling = float(tpm_ceiling if tpm_ceiling is not None else (tpm or 0))
        self.tpm_floor = tpm_floor
        self.clock, self.sleep = clock, sleep
        self._rng = random.Random(seed)
        self.waited_s = 0.0                            # total time spent waiting (for reports)
        self.acquired = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _burst(self, rpm: float) -> float:
        return float(self.burst_cfg if self.burst_cfg is not None else max(1.0, rpm / 6))   # ~10 s of traffic

    # ------------------------------------------------------------------ shared state
    def _locked(self, fn: Any) -> Any:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.read(fd, 4096)
            try:
                st = json.loads(raw) if raw else {}
            except ValueError:
                st = {}
            now = self.clock()
            st.setdefault("rpm", self.rpm0)
            st.setdefault("ts", now)
            st.setdefault("tokens", self._burst(st["rpm"]))
            st.setdefault("last_adjust", now)
            st.setdefault("n429", 0)
            st.setdefault("first429", None)
            if self.tpm0:
                st.setdefault("tpm", self.tpm0)
                st.setdefault("ttokens", self.tpm_burst)
            out, events = fn(st, now)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(st).encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        for ev in events:
            self._log(ev)
        return out

    def _log(self, ev: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
        except OSError:
            pass

    def _take(self, cost: float = 0.0) -> float:
        """Take one request token (and `cost` input tokens of the TPM bucket) if available: 0.0; else the seconds
        until both are due (nothing taken)."""
        def fn(st: dict[str, Any], now: float) -> tuple[float, list[dict[str, Any]]]:
            events = []
            ceiling = min(self.ceiling, float(st.get("ceiling", self.ceiling)))   # state file may lower it live
            if st["rpm"] > ceiling:
                events.append({"ts": now, "rpm": round(ceiling, 1), "event": "ceiling", "from": round(st["rpm"], 1)})
                st["rpm"] = ceiling
            if self.adaptive and now - st["last_adjust"] >= self.raise_every_s:
                if st["n429"] == 0 and st["rpm"] < ceiling:
                    old = st["rpm"]
                    st["rpm"] = min(ceiling, st["rpm"] * self.raise_factor)
                    events.append({"ts": now, "rpm": round(st["rpm"], 1), "event": "raise", "from": round(old, 1)})
                if self.tpm0 and st["n429"] == 0 and st["tpm"] < self.tpm_ceiling:
                    old = st["tpm"]
                    st["tpm"] = min(self.tpm_ceiling, st["tpm"] * self.raise_factor)
                    events.append({"ts": now, "rpm": round(st["rpm"], 1), "tpm": round(st["tpm"]), "event": "raise_tpm",
                                   "from": round(old)})
                st["last_adjust"], st["n429"], st["first429"] = now, 0, None
            rate = st["rpm"] / 60.0
            dt = max(0.0, now - st["ts"])
            tokens = min(self._burst(st["rpm"]), st["tokens"] + dt * rate)
            wait = 0.0 if tokens >= 1.0 else (1.0 - tokens) / rate
            tt = None
            if self.tpm0:
                trate = st["tpm"] / 60.0
                tt = min(self.tpm_burst, st["ttokens"] + dt * trate)
                need = min(cost, self.tpm_burst)        # a request larger than the burst waits for a full bucket
                if tt < need:
                    wait = max(wait, (need - tt) / trate)
            if wait == 0.0:
                tokens -= 1.0
                if tt is not None:
                    tt -= cost
            st["tokens"], st["ts"] = tokens, now
            if tt is not None:
                st["ttokens"] = tt
            return wait, events
        return self._locked(fn)

    def on_429(self) -> None:
        """One HTTP 429 seen by a client: a burst (>= burst_429 within window_429_s) drops the rate by 25%;
        any 429 in a 10-minute window blocks that window's raise."""
        def fn(st: dict[str, Any], now: float) -> tuple[None, list[dict[str, Any]]]:
            events = []
            st["n429"] += 1
            if st["first429"] is None or now - st["first429"] > self.window_429_s:
                st["first429"], st["burst"] = now, 0
            st["burst"] = st.get("burst", 0) + 1
            cooled = now - float(st.get("last_drop") or 0.0) >= self.window_429_s   # at most one drop per window
            if self.adaptive and st["burst"] >= self.burst_429 and cooled:
                st["last_drop"] = now
                old = st["rpm"]
                if self.tpm0:                                  # the binding limit is TPM: drop the token rate
                    oldt = st["tpm"]
                    st["tpm"] = max(self.tpm_floor, st["tpm"] * self.drop_factor)
                    st["ttokens"] = min(st["ttokens"], 0.0)
                    events.append({"ts": now, "rpm": round(st["rpm"], 1), "tpm": round(st["tpm"]), "event": "drop_429_tpm",
                                   "from": round(oldt)})
                else:
                    st["rpm"] = max(self.floor, st["rpm"] * self.drop_factor)
                    st["tokens"] = min(st["tokens"], 0.0)      # brief pause: refill from zero
                    events.append({"ts": now, "rpm": round(st["rpm"], 1), "event": "drop_429", "from": round(old, 1)})
                st["last_adjust"], st["first429"], st["burst"] = now, None, 0
            return None, events
        self._locked(fn)

    def current_rpm(self) -> float:
        return float(self._locked(lambda st, now: (st["rpm"], [])))

    async def acquire(self, cost: float = 0.0) -> None:
        """cost: the request's estimated input tokens (TPM bucket; 0 = request count only)."""
        while True:
            wait = self._take(cost)
            if wait <= 0.0:
                self.acquired += 1
                return
            w = wait + self._rng.uniform(0.0, min(1.0, wait))
            self.waited_s += w
            await self.sleep(w)


def effective_rpm(log_path: str | Path, default: float | None = None) -> float | None:
    """The latest rate in a limiter log (for reports), else default."""
    p = Path(log_path)
    last = default
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                last = float(json.loads(ln)["rpm"])
            except (ValueError, KeyError):
                continue
    return last
