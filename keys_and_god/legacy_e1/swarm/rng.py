"""Deterministic seeds and ids. Never use hash(), uuid, time or launch order for anything logged.

- derive_seed(base, *parts, branch=..., replica=...) -> int in [0, 2**63)
- make_id(*parts) -> 16 hex chars
- make_msg_id / make_call_id / make_entry_id / make_intervention_id: the only id schemes.

`branch` and `replica` are always folded into seeds. DECIDED (lead): forks use common random numbers.
A (intervention) and B (control) share the same branch and replica, so their seeds and cache keys are
identical and any divergence is attributable to God. Only the noise replica B' gets a different
`replica`, so its cache keys differ from B's (REVIEW_v2 4.5).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

SEED_BITS = 63
MAIN_BRANCH = "main"


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, pydantic models dumped in json mode."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def sha256_hex(data: str | bytes) -> str:
    """Hex sha256 of a str (utf-8) or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _digest(parts: tuple[Any, ...]) -> bytes:
    # Type-tagged so that 1 and "1" differ, and ("a","b") != ("ab",).
    return hashlib.sha256(canonical_json([[type(p).__name__, _plain(p)] for p in parts]).encode()).digest()


def derive_seed(base: int, *parts: Any, branch: str = MAIN_BRANCH, replica: int | str = 0) -> int:
    """Stable across processes and platforms. Example:
    derive_seed(cfg.run.seed, game_idx, "turn", agent, round, attempt, branch=b, replica=r)
    """
    d = _digest((int(base), "branch", str(branch), "replica", str(replica), *parts))
    return int.from_bytes(d[:8], "big") & ((1 << SEED_BITS) - 1)


def make_rng(base: int, *parts: Any, branch: str = MAIN_BRANCH, replica: int | str = 0) -> random.Random:
    """A private random.Random; never use the global `random` module."""
    return random.Random(derive_seed(base, *parts, branch=branch, replica=replica))


def make_id(*parts: Any, n: int = 16) -> str:
    """Deterministic short hex id from typed parts (sha256 prefix)."""
    return _digest(parts).hex()[:n]


def make_game_id(run_seed: int, game_idx: int, config_hash: str, *, branch: str = MAIN_BRANCH,
                 replica: int | str = 0) -> str:
    """Independent of wall clock, so a replay/rerun gets the same game_id."""
    return "g" + make_id("game", run_seed, game_idx, config_hash, branch, str(replica), n=12)


def make_msg_id(game_id: str, msg_index: int) -> str:
    """msg_index = position in GameState.messages. Same scheme for real and forged DMs, so the id
    carries no provenance (REVIEW_v2 4.1.1)."""
    return "m" + make_id("msg", game_id, msg_index)


def make_call_id(game_id: str | None, role: str, agent: str | None, round: int | None,
                 attempt: int, purpose: str = "turn") -> str:
    """purpose examples: "turn", "grep_followup", "god_step:3", "compact", "judge:<unit>"."""
    return "c" + make_id("call", game_id, role, agent, round, attempt, purpose)


def make_entry_id(agent: str, kind: str, round: int) -> str:
    """MemoryEntry id: one entry per (agent, kind, round)."""
    return f"{agent}:{kind}:{round}"


def make_intervention_id(game_id: str, round: int, n: int) -> str:
    """n = index of the God power within the game (0, 1, ...)."""
    return "i" + make_id("intervention", game_id, round, n, n=12)
