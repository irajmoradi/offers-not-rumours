"""Context policies (WP13-lite): `full_truncate` (default) and `window`. Both PURE and synchronous.

Budget algorithm (shared, `fit`):
  limit = floor(budget * (1 - safety_margin)) - reserve_tokens       (reserve = system prompt etc.)
  1. Pinned facts are always kept (they live on bundle.pinned; counted as their JSON).
  2. Current-round items are kept. If pinned + current > limit, the oldest current log lines, then
     the oldest current board posts, are dropped. Inbox DMs and notices are never dropped; if it
     still does not fit, ContextOverflow is raised.
  3. History (ledger + memory notes) fills the rest NEWEST FIRST, as a contiguous suffix: the first
     item that does not fit stops the fill (so a larger budget includes a superset: monotone).
  Result: bundle.tokens_total <= limit, always.

Ordering: history items are sorted by round; within a round, ledger items keep ledger order and
memory notes (scratchpad / replayed cot) come after them.

`window(K)`: history is first restricted to rounds > round - K (memory notes are NOT round-windowed:
the scratchpad is the persistent store; its length is set by reasoning.scratchpad.window upstream),
then `per_kind_caps` keeps the newest n items of each capped kind, then the budget fill applies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from swarm.config import ContextCfg
from swarm.context.base import ContextOverflow, TokenCounter, memory_items
from swarm.rng import canonical_json, sha256_hex
from swarm.schema import (
    AgentId,
    CompactionRecord,
    ContextBundle,
    ContextSection,
    HistoryItem,
    MemoryView,
    PinnedFacts,
)

DROPPABLE_CURRENT = ("log_line", "board_msg")   # drop order when the current round alone overflows


def pinned_tokens(pinned: PinnedFacts, counter: TokenCounter) -> int:
    return counter.count(pinned.model_dump_json())


def effective_limit(budget: int, safety_margin: float, reserve_tokens: int) -> int:
    return math.floor(budget * (1 - safety_margin)) - reserve_tokens


def order_history(ledger: Sequence[HistoryItem], mem: Sequence[HistoryItem]) -> list[HistoryItem]:
    keyed = [((it.round, 0, i), it) for i, it in enumerate(ledger)]
    keyed += [((it.round, 1, i), it) for i, it in enumerate(mem)]
    return [it for _, it in sorted(keyed, key=lambda p: p[0])]


def fit(*, agent: AgentId, round: int, policy: str, policy_hash: str, history: Sequence[HistoryItem],
        pre_dropped: Sequence[str], current: Sequence[HistoryItem], pinned: PinnedFacts, budget: int,
        limit: int, counter: TokenCounter) -> ContextBundle:
    tok = {it.item_id: counter.count(it.text) for it in [*history, *current]}
    p_tok = pinned_tokens(pinned, counter)
    dropped: list[str] = list(pre_dropped)

    cur = list(current)
    used = p_tok + sum(tok[i.item_id] for i in cur)
    for kind in DROPPABLE_CURRENT:
        while used > limit:
            victim = next((i for i in cur if i.kind == kind), None)
            if victim is None:
                break
            cur.remove(victim)
            dropped.append(victim.item_id)
            used -= tok[victim.item_id]
    if used > limit:
        raise ContextOverflow(f"pinned + current = {used} tokens > limit {limit} (budget {budget})")

    kept: list[HistoryItem] = []
    room = limit - used
    hist = list(history)
    stop = len(hist)
    for j in range(len(hist) - 1, -1, -1):
        t = tok[hist[j].item_id]
        if t > room:
            break
        room -= t
        stop = j
    kept = hist[stop:]
    dropped = [*(i.item_id for i in hist[:stop]), *dropped]

    w_tok = sum(tok[i.item_id] for i in kept)
    c_tok = sum(tok[i.item_id] for i in cur)
    items = {i.item_id: i.model_copy(update={"tokens": tok[i.item_id]}) for i in [*kept, *cur]}
    sections = [
        ContextSection(name="pinned", item_ids=[], tokens=p_tok),
        ContextSection(name="window", item_ids=[i.item_id for i in kept], tokens=w_tok),
        ContextSection(name="current", item_ids=[i.item_id for i in cur], tokens=c_tok),
    ]
    return ContextBundle(agent=agent, round=round, policy=policy, policy_hash=policy_hash, pinned=pinned,
                         sections=sections, items=items, dropped_item_ids=dropped,
                         tokens_total=p_tok + w_tok + c_tok, budget=budget)


class _Base:
    name = "base"

    def __init__(self, *, safety_margin: float = 0.05, reserve_tokens: int = 0) -> None:
        self.safety_margin = safety_margin
        self.reserve_tokens = reserve_tokens

    def params(self) -> dict[str, Any]:
        return {"safety_margin": self.safety_margin, "reserve_tokens": self.reserve_tokens}

    @property
    def policy_hash(self) -> str:
        return sha256_hex(canonical_json({"name": self.name, **self.params()}))

    def limit(self, budget: int) -> int:
        return effective_limit(budget, self.safety_margin, self.reserve_tokens)

    def due_for_compaction(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem]) -> bool:
        return False

    async def compact(self, **kwargs: object) -> CompactionRecord | None:
        return None


class FullTruncatePolicy(_Base):
    """Pinned + the whole ledger and notes, oldest dropped first until it fits."""

    name = "full_truncate"

    def build(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem],
              current: Sequence[HistoryItem], memory: Sequence[MemoryView], pinned: PinnedFacts,
              budget: int, counter: TokenCounter) -> ContextBundle:
        history = order_history(ledger, memory_items(agent, memory, counter))
        return fit(agent=agent, round=round, policy=self.name, policy_hash=self.policy_hash, history=history,
                   pre_dropped=[], current=current, pinned=pinned, budget=budget, limit=self.limit(budget),
                   counter=counter)


class WindowPolicy(_Base):
    """Pinned + the last `window_rounds` rounds of ledger (+ notes), per-kind caps, then the budget."""

    name = "window"

    def __init__(self, *, window_rounds: int = 3, per_kind_caps: dict[str, int] | None = None,
                 safety_margin: float = 0.05, reserve_tokens: int = 0) -> None:
        super().__init__(safety_margin=safety_margin, reserve_tokens=reserve_tokens)
        self.window_rounds = window_rounds
        self.per_kind_caps = dict(per_kind_caps or {})

    def params(self) -> dict[str, Any]:
        return {**super().params(), "window_rounds": self.window_rounds,
                "per_kind_caps": dict(sorted(self.per_kind_caps.items()))}

    def build(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem],
              current: Sequence[HistoryItem], memory: Sequence[MemoryView], pinned: PinnedFacts,
              budget: int, counter: TokenCounter) -> ContextBundle:
        lo = round - self.window_rounds
        recent = [it for it in ledger if it.round > lo]
        pre = [it.item_id for it in ledger if it.round <= lo]
        history = order_history(recent, memory_items(agent, memory, counter))
        capped: list[HistoryItem] = []
        seen: dict[str, int] = {}
        for it in reversed(history):
            cap = self.per_kind_caps.get(it.kind)
            seen[it.kind] = seen.get(it.kind, 0) + 1
            if cap is not None and seen[it.kind] > cap:
                pre.append(it.item_id)
                continue
            capped.append(it)
        capped.reverse()
        return fit(agent=agent, round=round, policy=self.name, policy_hash=self.policy_hash, history=capped,
                   pre_dropped=pre, current=current, pinned=pinned, budget=budget, limit=self.limit(budget),
                   counter=counter)


POLICIES = {"full_truncate": FullTruncatePolicy, "window": WindowPolicy}


def make_policy(cfg: ContextCfg, *, reserve_tokens: int = 0) -> FullTruncatePolicy | WindowPolicy:
    """cfg.context -> policy. `reserve_tokens`: tokens of the fixed system prompt + framing, so the
    whole prompt stays within context.budget_tokens."""
    if cfg.policy == "full_truncate":
        return FullTruncatePolicy(safety_margin=cfg.safety_margin, reserve_tokens=reserve_tokens)
    if cfg.policy == "window":
        return WindowPolicy(window_rounds=cfg.window_rounds, per_kind_caps=cfg.per_kind_caps,
                            safety_margin=cfg.safety_margin, reserve_tokens=reserve_tokens)
    raise ValueError(f"context policy {cfg.policy!r} is not implemented")
