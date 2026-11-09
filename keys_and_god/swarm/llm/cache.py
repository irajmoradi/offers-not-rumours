"""On-disk LLM response cache (stdlib sqlite3, WAL mode; one file shared by runs and processes).

    key = cache_key(backend, model, req)
        = sha256(canonical_json({backend, model, messages, max_tokens, temperature, top_p, seed,
                                 thinking, response_format, stop}))

`ChatRequest.sampling()` supplies every field that changes the completion (call_id and meta are
excluded), so changing max_tokens or the seed never serves a stale answer.

Modes (config `llm.cache.mode`):
  off         no reads, no writes
  read_write  default
  read_only   strict rerun: a miss raises CacheMiss (this is the replay backend)
  write_only  never reads, always writes

Imports: llm.base, rng (stdlib sqlite3). Side effects: the cache file only.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

from swarm.llm.base import ChatRequest, ChatResponse, LLMError
from swarm.rng import canonical_json, sha256_hex

CacheMode = Literal["off", "read_write", "read_only", "write_only"]


class CacheMiss(LLMError):
    """read_only mode and the key is not in the cache."""


def cache_key(backend: str, model: str, req: ChatRequest) -> str:
    """Deterministic key over (backend, model, messages, seed, sampling params)."""
    return sha256_hex(canonical_json({"backend": backend, "model": model, **req.sampling()}))


class ResponseCache:
    """sqlite table cache(key TEXT PRIMARY KEY, response_json TEXT, created REAL)."""

    def __init__(self, path: str | Path, mode: CacheMode = "read_write") -> None:
        self.path = Path(path)
        self.mode: CacheMode = mode
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if mode != "off":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, "
                               "response_json TEXT NOT NULL, created REAL NOT NULL)")
            self._conn.commit()

    @property
    def readable(self) -> bool:
        """True when this mode serves hits."""
        return self.mode in ("read_write", "read_only")

    @property
    def writable(self) -> bool:
        """True when this mode stores responses."""
        return self.mode in ("read_write", "write_only")

    def get(self, key: str) -> ChatResponse | None:
        """Return the cached response, None on a miss. Raises CacheMiss on a miss in read_only."""
        if not self.readable or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            if self.mode == "read_only":
                raise CacheMiss(f"cache miss in read_only mode: {key[:16]}")
            return None
        return ChatResponse.model_validate_json(row[0])

    def put(self, key: str, resp: ChatResponse) -> None:
        """Store a response under key (no-op unless writable)."""
        if not self.writable or self._conn is None:
            return
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache (key, response_json, created) VALUES (?,?,?)",
                               (key, resp.model_dump_json(), time.time()))
            self._conn.commit()

    def __len__(self) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])

    def close(self) -> None:
        """Close the sqlite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
