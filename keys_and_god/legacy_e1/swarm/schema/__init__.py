"""Shared records (WP0, FROZEN). Change only via the team lead.

A package since the refactor (plan/REFACTOR_PLAN.md R1): one module per family below, every name
re-exported here, so `from swarm.schema import X` and `import swarm.schema as S` are unchanged.
  base (Record, aliases)  timing (turn-index contract)  state (truth)  views (agent-visible)
  actions (agent + God)   context (ledger items)        llm_io         events (+ sinks, Snapshot)
  labels                  jsonl

Three families, kept apart on purpose:
  1. TRUTH side (engine/analysis only): GameState, AgentState, KeyCopy, Message, MemoryEntry,
     ActionRecord, Notice, Event + payloads, LLMCall, HistoryItem, ContextBundle, Label.
  2. AGENT-VISIBLE side: subclasses of `AgentView` (Observation, PinnedFacts, KeyView,
     AgentMessageView, LogLine, NoticeView, MemoryView, GrepResult, PromptView, ...). They contain
     no field in TRUTH_FIELDS and no truth model (enforced by tests/test_agent_view.py). Prompt
     templates receive ONLY AgentView objects.
  3. Actions (agent -> engine) and GodActions, as tagged unions on `kind`.

Mutability: GameState / AgentState are mutable (rules mutate them in place; the engine deep-copies at
round boundaries only). Everything else is frozen. Ledgers (agent context history) live OUTSIDE GameState.

Time: rounds are 1-based (round 0 = setup). Within a game every agent-turn slot has a global
turn_index = (round - 1) * n_agents + position_in_order (slots of skipped agents are still counted).
See `turn_index`, `log_visible`, `defect_release_turn`, `release_visible` below: they ARE the
view_log timing contract.
"""

from __future__ import annotations

from swarm import SCHEMA_VERSION
from swarm.schema.actions import (  # noqa: F401
    ACTION_ADAPTER,
    ACTION_KINDS,
    GOD_ACTION_ADAPTER,
    Action,
    ActivityCheck,
    AgentOutput,
    CotControl,
    DefectAction,
    GiveKey,
    GodAction,
    GrepLog,
    MessageCheck,
    MessageFraud,
    Noop,
    PostBoard,
    SendMessage,
    Wait,
)
from swarm.schema.base import (  # noqa: F401
    GOD,
    SYSTEM,
    AgentId,
    GameId,
    InfoId,
    MemoryKind,
    MutableState,
    Phase,
    Record,
    RunId,
)
from swarm.schema.context import (  # noqa: F401
    CompactionRecord,
    ContextBundle,
    ContextSection,
    HistoryItem,
    HistoryKind,
)
from swarm.schema.events import (  # noqa: F401
    ActionResultData,
    Actor,
    AgentOutcome,
    BurnData,
    CompactionData,
    ContextBuiltData,
    Event,
    EventBody,
    EventData,
    EventSink,
    EventType,
    GameEndData,
    GameStartData,
    GodPowerData,
    GodToolData,
    KeyTransferData,
    MemoryOverwriteData,
    MemorySink,
    MemoryWriteData,
    MessageData,
    NoticeData,
    Outcome,
    RoundEndData,
    Snapshot,
    TurnData,
    WinData,
    make_event,
)
from swarm.schema.jsonl import (  # noqa: F401
    read_jsonl,
    read_jsonl_raw,
    to_jsonl_line,
    write_jsonl,
)
from swarm.schema.labels import (  # noqa: F401
    Label,
    MonitorScore,
)
from swarm.schema.llm_io import (  # noqa: F401
    CallMeta,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LLMCall,
    Role,
    Usage,
)
from swarm.schema.state import (  # noqa: F401
    ActionRecord,
    AgentState,
    GameState,
    KeyCopy,
    MemoryEntry,
    Message,
    Notice,
    Taint,
)
from swarm.schema.timing import (  # noqa: F401
    defect_release_turn,
    log_visible,
    release_visible,
    turn_index,
)
from swarm.schema.views import (  # noqa: F401
    TRUTH_FIELDS,
    AgentMessageView,
    AgentView,
    GrepResult,
    KeyView,
    LogLine,
    MemoryView,
    NoticeView,
    Observation,
    PinnedFacts,
    PromptView,
    RenderItem,
    RenderSection,
)

__all__ = [
    "SCHEMA_VERSION", "AgentId", "InfoId", "GameId", "RunId", "GOD",
    "SYSTEM", "MemoryKind", "Phase", "Record", "MutableState", "turn_index",
    "log_visible", "defect_release_turn", "release_visible", "KeyCopy", "AgentState", "Message",
    "Taint", "MemoryEntry", "ActionRecord", "Notice", "GameState", "AgentView",
    "KeyView", "AgentMessageView", "LogLine", "NoticeView", "MemoryView", "PinnedFacts",
    "Observation", "GrepResult", "RenderItem", "RenderSection", "PromptView", "TRUTH_FIELDS",
    "SendMessage", "PostBoard", "GiveKey", "DefectAction", "GrepLog", "Noop",
    "Action", "ACTION_ADAPTER", "ACTION_KINDS", "AgentOutput", "ActivityCheck", "MessageCheck",
    "Wait", "MessageFraud", "CotControl", "GodAction", "GOD_ACTION_ADAPTER", "HistoryKind",
    "HistoryItem", "ContextSection", "ContextBundle", "CompactionRecord", "Role", "ChatMessage",
    "CallMeta", "ChatRequest", "Usage", "ChatResponse", "LLMCall", "GameStartData",
    "TurnData", "ActionResultData", "MessageData", "KeyTransferData", "BurnData", "MemoryWriteData",
    "MemoryOverwriteData", "GodToolData", "GodPowerData", "NoticeData", "ContextBuiltData", "CompactionData",
    "WinData", "RoundEndData", "AgentOutcome", "Outcome", "GameEndData", "EventData",
    "EventType", "Actor", "EventBody", "Event", "make_event", "EventSink",
    "MemorySink", "Snapshot", "Label", "MonitorScore", "to_jsonl_line", "write_jsonl",
    "read_jsonl", "read_jsonl_raw",
]
