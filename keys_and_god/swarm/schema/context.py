"""Context records (WP13). Ledger = dict[AgentId, list[HistoryItem]], held by the engine OUTSIDE
GameState, built only by context.base.items_from_observation(obs)."""

from __future__ import annotations

from typing import Literal

from swarm.schema.base import AgentId, Record
from swarm.schema.views import GrepResult, PinnedFacts, PromptView, RenderItem, RenderSection

HistoryKind = Literal["inbox_msg", "board_msg", "log_line", "notice", "own_action", "action_result",
                      "cot", "scratchpad", "summary", "retrieved"]


class HistoryItem(Record):
    """One unit of an agent's private context history (ledger). Built only from Observations."""
    item_id: str                       # f"{agent}:{round}:{kind}:{n}"
    agent: AgentId
    round: int                         # round the agent OBSERVED / produced it
    kind: HistoryKind
    text: str                          # exactly what the agent is shown
    ref: str | None = None             # msg_id / entry_id / event pointer (truth; never rendered)
    tokens: int = 0


class ContextSection(Record):
    """Ids and token count of one section of a built context."""
    name: Literal["pinned", "summary", "retrieved", "window", "current"]
    item_ids: list[str]
    tokens: int


class ContextBundle(Record):
    """A built agent context: pinned facts + included items per section (truth side)."""
    agent: AgentId
    round: int
    policy: str
    policy_hash: str
    pinned: PinnedFacts
    sections: list[ContextSection]
    items: dict[str, HistoryItem]      # included items only
    dropped_item_ids: list[str] = []
    tokens_total: int
    budget: int

    def to_view(self, grep: GrepResult | None = None) -> PromptView:
        """Text-only PromptView for templates (drops refs and ids)."""
        return PromptView(
            agent=self.agent, round=self.round, pinned=self.pinned, grep=grep,
            sections=[RenderSection(name=s.name, items=[
                RenderItem(round=self.items[i].round, kind=self.items[i].kind, text=self.items[i].text)
                for i in s.item_ids]) for s in self.sections],
        )


class CompactionRecord(Record):
    """One summary compaction of an agent's history (WP13)."""
    compaction_id: str
    agent: AgentId
    round: int
    policy: str
    summarizer: Literal["self", "cheap", "extractive"]
    covers_rounds: tuple[int, int]
    source_item_ids: list[str]
    prev_summary_id: str | None = None
    summary_entry_id: str
    summary_text: str
    tokens_before: int
    tokens_after: int
    llm_call_id: str | None = None
