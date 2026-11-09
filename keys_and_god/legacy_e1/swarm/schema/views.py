"""AGENT-VISIBLE side. Built ONLY by game/observe.py (field by field) and context/. No field in
TRUTH_FIELDS and no truth model may appear here (tests/test_agent_view.py)."""

from __future__ import annotations

from typing import Literal

from swarm.schema.base import AgentId, InfoId, MemoryKind, Record


class AgentView(Record):
    """Marker base. Anything a prompt template may receive must be an AgentView."""


class KeyView(AgentView):
    """A key as the agent sees it: info id and key number only."""
    info_id: InfoId
    key: int


class AgentMessageView(AgentView):
    """No round, no provenance. msg_id is an opaque hash (same scheme for real and forged)."""

    msg_id: str
    channel: Literal["dm", "board"]
    sender: AgentId
    receiver: AgentId | None = None
    text: str


class LogLine(AgentView):
    """One view_log line. actor/target are None when the identity is hidden."""

    round: int
    kind: Literal["give", "defect", "board_post", "dm_meta", "dm"]
    actor: AgentId | None = None
    target: AgentId | None = None
    info_id: InfoId | None = None
    text: str | None = None
    ok: bool | None = None             # defects: hit (True) / wrong (False)


class NoticeView(AgentView):
    """A system notice as the agent sees it."""
    kind: str
    text: str


class MemoryView(AgentView):
    """A memory entry as the agent sees it: current text only, no tamper flags."""
    kind: MemoryKind
    round: int
    text: str


class PinnedFacts(AgentView):
    """Rebuilt from GameState every turn; never from history."""

    agent: AgentId
    round: int
    max_rounds: int
    needs_open: list[InfoId]
    needs_burned: list[InfoId]
    needs_lost: list[InfoId] = []
    holdings: list[KeyView]
    can_give: list[InfoId]
    score: float
    peers: list[AgentId]
    won: bool = False


class Observation(AgentView):
    """This turn's deltas plus pinned facts. History comes from the context policy (ledger)."""

    agent: AgentId
    round: int
    turn_index: int
    pinned: PinnedFacts
    inbox: list[AgentMessageView] = []     # DMs delivered since the previous observation
    board: list[AgentMessageView] = []     # board posts since the previous observation
    log: list[LogLine] = []                # view_log() result for this turn
    notices: list[NoticeView] = []


class GrepResult(AgentView):
    """Result of the agent's grep_log action (subset of its visible log history)."""
    pattern: str
    hits: list[LogLine]
    truncated: bool = False


class RenderItem(AgentView):
    """One pre-rendered history line for templates."""
    round: int
    kind: str
    text: str


class RenderSection(AgentView):
    """A named group of history lines for templates (prompt order, s17.6)."""
    name: Literal["pinned", "summary", "retrieved", "window", "current"]
    items: list[RenderItem]


class PromptView(AgentView):
    """What turn templates render (from ContextBundle.to_view()). Text only."""

    agent: AgentId
    round: int
    pinned: PinnedFacts
    sections: list[RenderSection]
    grep: GrepResult | None = None


# Field names that must never appear in an AgentView (recursively) unless reveal_provenance.
TRUTH_FIELDS: frozenset[str] = frozenset({
    "forged", "true_sender", "intervention_id", "tampered", "original_text", "tampered_by",
    "tamper_round", "taint", "copy_id", "origin", "ref", "cause", "source_item_ids",
    "release_turn", "deliver_from_turn",
})
