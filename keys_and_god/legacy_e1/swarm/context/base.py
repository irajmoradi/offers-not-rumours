"""Context contract (WP0, FROZEN signatures; WP13 implements real policies).

Data flow per agent turn:
    obs      = game.observe(...)                       # the ONLY path from truth to agent
    current  = items_from_observation(obs, counter)    # the ONLY way history items are created
    memory   = MemoryStore.views(state, agent, "scratchpad", ...)  (+ "cot" if cot_replay_window > 0)
    bundle   = policy.build(agent=..., round=..., ledger=ledger[agent], current=current,
                            memory=memory, pinned=obs.pinned, budget=..., counter=counter)
    view     = bundle.to_view(grep)                    # PromptView -> templates
    ... after resolve: ledger[agent] += current + own_action_items(...)

The ledger (dict[AgentId, list[HistoryItem]]) is engine-owned and lives OUTSIDE GameState; it can be
rebuilt from the `turn` events (which log `obs`). Policies never see truth fields: only HistoryItem
text and MemoryView text.

Item text formats below are the frozen defaults; WP13 may change wording only via the lead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from swarm.rng import make_entry_id, sha256_hex
from swarm.schema import (
    Action,
    AgentId,
    CompactionRecord,
    ContextBundle,
    ContextSection,
    HistoryItem,
    HistoryKind,
    LogLine,
    MemoryView,
    Observation,
    PinnedFacts,
)


# ------------------------------------------------------------------ protocols
@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens of a text (approx, words, or a real tokenizer)."""
    name: str

    def count(self, text: str) -> int:
        """Token count of text."""
        ...


@runtime_checkable
class ContextPolicy(Protocol):
    """Decides which history items an agent sees within its token budget."""
    name: str

    def build(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem],
              current: Sequence[HistoryItem], memory: Sequence[MemoryView], pinned: PinnedFacts,
              budget: int, counter: TokenCounter) -> ContextBundle:
        """PURE and synchronous. Never drops pinned facts or current-round inbox DMs."""
        ...

    def due_for_compaction(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem]) -> bool:
        """Whether compact() should run for this agent now."""
        ...

    async def compact(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem],
                      memory: Sequence[MemoryView], pinned: PinnedFacts, counter: TokenCounter,
                      seed: int) -> CompactionRecord | None:
        """Summarize aged history; returns None when nothing to do."""
        ...


class ContextOverflow(Exception):
    """pinned + current alone exceed the budget."""


# ------------------------------------------------------------------ counters
class WordCounter:
    """Whitespace word count (tests: exact and fast)."""
    name = "words"

    def count(self, text: str) -> int:
        """Token count of text."""
        return len(text.split())


class ApproxCounter:
    """ceil(chars / 3.5): rough token estimate without a tokenizer."""
    name = "approx"

    def count(self, text: str) -> int:
        """Token count of text."""
        return math.ceil(len(text) / 3.5)


# ------------------------------------------------------------------ item builders (the choke point)
def render_log_line(line: LogLine) -> str:
    """Default text for one LogLine (identity-hidden fields render as 'an agent')."""
    a = line.actor or "an agent"
    t = line.target or "an agent"
    if line.kind == "give":
        return f"{a} gave {t} a key for info {line.info_id}"
    if line.kind == "defect":
        verb = "burned" if line.ok is not False else "tried and FAILED to burn"
        return f"{a} {verb} info {line.info_id} of {t}"
    if line.kind == "board_post":
        return f"{a} posted on the board" + (f": {line.text}" if line.text else "")
    if line.kind == "dm_meta":
        return f"{a} messaged {t}"
    return f"{a} -> {t}: {line.text or ''}"


def _item(agent: AgentId, round: int, kind: HistoryKind, n: int, text: str, ref: str | None,
          counter: TokenCounter | None) -> HistoryItem:
    return HistoryItem(item_id=f"{agent}:{round}:{kind}:{n}", agent=agent, round=round, kind=kind,
                       text=text, ref=ref, tokens=counter.count(text) if counter else 0)


def items_from_observation(obs: Observation, counter: TokenCounter | None = None) -> list[HistoryItem]:
    """History items for one Observation: the only constructor of ledger items."""
    a, r = obs.agent, obs.round
    out: list[HistoryItem] = []
    for n, m in enumerate(obs.inbox):
        out.append(_item(a, r, "inbox_msg", n, f"DM from {m.sender}: {m.text}", m.msg_id, counter))
    for n, m in enumerate(obs.board):
        out.append(_item(a, r, "board_msg", n, f"BOARD {m.sender}: {m.text}", m.msg_id, counter))
    for n, line in enumerate(obs.log):
        out.append(_item(a, r, "log_line", n, render_log_line(line), None, counter))
    for n, nt in enumerate(obs.notices):
        out.append(_item(a, r, "notice", n, f"NOTICE: {nt.text}", None, counter))
    return out


def render_action(action: Action) -> str:
    """Default text for one of the agent's own actions."""
    k = action.kind
    if k == "send_message":
        return f"You messaged {action.to}: {action.text}"
    if k == "post_board":
        return f"You posted on the board: {action.text}"
    if k == "give_key":
        return f"You gave {action.to} your key for info {action.info_id}"
    if k == "defect":
        return f"You tried to burn info {action.info_id} of {action.target}"
    if k == "grep_log":
        return f"You searched the log for '{action.pattern}'"
    return "You did nothing"


def own_action_items(agent: AgentId, round: int, actions: Sequence[Action],
                     results: Sequence[tuple[bool, str | None]],
                     counter: TokenCounter | None = None) -> list[HistoryItem]:
    """results[i] = (ok, reason) from the matching action_result."""
    out = []
    for n, (act, (ok, reason)) in enumerate(zip(actions, results, strict=True)):
        text = render_action(act) + ("" if ok else f" (failed: {reason or 'invalid'})")
        out.append(_item(agent, round, "own_action", n, text, None, counter))
    return out


def memory_items(agent: AgentId, memory: Sequence[MemoryView],
                 counter: TokenCounter | None = None) -> list[HistoryItem]:
    """History items for the agent's memory views (scratchpad / cot / summary)."""
    label = {"scratchpad": "Your note", "cot": "Your thought", "summary": "Your summary"}
    return [_item(agent, m.round, m.kind, 0, f"{label[m.kind]} (round {m.round}): {m.text}",
                  make_entry_id(agent, m.kind, m.round), counter) for m in memory]


# ------------------------------------------------------------------ PassthroughPolicy
class PassthroughPolicy:
    """Includes everything, never drops, never compacts. For WP1/WP3 integration before WP13 lands."""

    name = "passthrough"

    def build(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem],
              current: Sequence[HistoryItem], memory: Sequence[MemoryView], pinned: PinnedFacts,
              budget: int, counter: TokenCounter) -> ContextBundle:
        """Assemble the context bundle for one turn (pure)."""
        mem = memory_items(agent, memory, counter)
        window = sorted([*ledger, *mem], key=lambda it: (it.round, it.item_id))
        cur = list(current)

        def tok(it: HistoryItem) -> int:
            return it.tokens or counter.count(it.text)

        pinned_tokens = counter.count(pinned.model_dump_json())
        sections = [
            ContextSection(name="pinned", item_ids=[], tokens=pinned_tokens),
            ContextSection(name="window", item_ids=[i.item_id for i in window], tokens=sum(map(tok, window))),
            ContextSection(name="current", item_ids=[i.item_id for i in cur], tokens=sum(map(tok, cur))),
        ]
        items = {i.item_id: i for i in [*window, *cur]}
        return ContextBundle(agent=agent, round=round, policy=self.name, policy_hash=sha256_hex(self.name),
                             pinned=pinned, sections=sections, items=items,
                             tokens_total=sum(s.tokens for s in sections), budget=budget)

    def due_for_compaction(self, *, agent: AgentId, round: int, ledger: Sequence[HistoryItem]) -> bool:
        """Whether compact() should run for this agent now."""
        return False

    async def compact(self, **kwargs: object) -> CompactionRecord | None:
        """Summarize aged history; returns None when nothing to do."""
        return None
