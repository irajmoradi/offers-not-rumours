"""Events (events.jsonl): one typed payload per event type, discriminated by `type`; the EventSink
protocol, the in-memory MemorySink and the per-round Snapshot."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from swarm import SCHEMA_VERSION
from swarm.schema.actions import Action, GodAction
from swarm.schema.base import AgentId, GameId, InfoId, MemoryKind, Phase, Record, RunId
from swarm.schema.context import CompactionRecord
from swarm.schema.state import AgentState, GameState, MemoryEntry, Message, Notice
from swarm.schema.views import Observation


class GameStartData(Record):
    """Payload: world summary and initial inventories."""
    type: Literal["game_start"] = "game_start"
    config_hash: str
    n_agents: int
    max_rounds: int
    order: list[AgentId]
    agents: dict[AgentId, AgentState]  # initial inventories and needs
    branch: str = "main"
    replica: str = "0"


class TurnData(Record):
    """Payload: one agent turn (observation, prompt hashes, parsed output)."""
    type: Literal["turn"] = "turn"
    agent: AgentId
    obs: Observation                   # logged so ledgers can be rebuilt from events
    obs_hash: str
    prompt_set_hash: str | None = None
    prompt_hash: str | None = None
    llm_call_ids: list[str] = []
    cot: str | None = None
    note: str | None = None
    actions: list[Action] = []
    parse_ok: bool = True
    retries: int = 0
    errors: list[str] = []


class ActionResultData(Record):
    """Payload: outcome of one action."""
    type: Literal["action_result"] = "action_result"
    agent: AgentId
    action: Action
    ok: bool
    reason: str | None = None          # "no_such_key", "target_won", "not_a_peer", ...
    n_hits: int | None = None          # grep_log only


class MessageData(Record):
    """Payload: a DM/board message with full ground truth."""
    type: Literal["message"] = "message"
    message: Message


class KeyTransferData(Record):
    """Payload: a key copy changed hands (give / defect gain)."""
    type: Literal["key_transfer"] = "key_transfer"
    copy_id: str
    info_id: InfoId
    from_agent: AgentId | None          # None for defect_mint
    to_agent: AgentId
    # defect_penalty: wrong_defect_penalty=lose_held_key, the wrong defector's spare goes to the target
    # trade: game.trade (offer_trade settled)
    cause: Literal["give", "defect_mint", "defect_steal", "defect_penalty", "trade"]


class BurnData(Record):
    """Payload: a defect attempt and its effects."""
    type: Literal["burn"] = "burn"
    actor: AgentId
    target: AgentId
    info_id: InfoId
    correct: bool
    # actor's penalty (wrong defects only); lose_held_need: a held need was lost (lost_info_id); when
    # that penalty finds no held need it falls back to "lose_need" (the lowest-id open need)
    # score_penalty: the actor's score dropped by game.defect.score_penalty (no need lost); lose_held_key:
    # the actor's spare key went to the target (a key_transfer cause defect_penalty); with no spare it
    # falls back to "score_penalty"
    penalty: Literal["none", "lose_need", "delay", "lose_held_need", "score_penalty", "lose_held_key"]
    lost_info_id: InfoId | None = None
    delayed_until_round: int | None = None
    gain: Literal["mint", "steal", "none"]
    release_turn: int | None = None    # None in public / victim_only


class EdgeAddedData(Record):
    """Payload: game.network.connect, a successful connect by `actor` to `target` (round/turn on the
    event). mutual=True: the edge already existed because `target` connected to `actor` earlier in
    this round's resolution, before `actor` could see it (both pay; still one edge)."""
    type: Literal["edge_added"] = "edge_added"
    actor: AgentId
    target: AgentId
    mutual: bool = False
    skip_until_round: int              # the actor takes no turns while round <= this (cost_rounds)


class TradeOfferData(Record):
    """Payload: game.trade, a valid offer_trade at the proposer's slot (the target is asked next)."""
    type: Literal["trade_offer"] = "trade_offer"
    offer_id: str
    proposer: AgentId
    target: AgentId
    give: list[InfoId]                 # the proposer's keys
    want: list[InfoId]                 # the target's keys


class TradeReplyData(Record):
    """Payload: the target's accept / reject (one LLM call; unparseable or error = reject)."""
    type: Literal["trade_reply"] = "trade_reply"
    offer_id: str
    proposer: AgentId
    target: AgentId
    accept: bool
    parse_ok: bool = True
    errors: list[str] = []
    llm_call_ids: list[str] = []
    cot: str | None = None


class TradeExecutedData(Record):
    """Payload: an accepted trade swapped atomically. completes_need_*: open needs that side newly
    holds through this trade."""
    type: Literal["trade_executed"] = "trade_executed"
    offer_id: str
    proposer: AgentId
    target: AgentId
    give: list[InfoId]
    want: list[InfoId]
    copies_to_target: list[str]
    copies_to_proposer: list[str]
    completes_need_proposer: list[InfoId] = []
    completes_need_target: list[InfoId] = []


class TradeFailedData(Record):
    """Payload: an accepted trade that could not execute (holdings changed / a side left)."""
    type: Literal["trade_failed"] = "trade_failed"
    offer_id: str
    proposer: AgentId
    target: AgentId
    reason: str                        # stale_holdings | party_left
    missing_proposer: list[InfoId] = []
    missing_target: list[InfoId] = []


class MemoryWriteData(Record):
    """Payload: agent-authored memory write."""
    type: Literal["memory_write"] = "memory_write"
    entry: MemoryEntry


class MemoryOverwriteData(Record):
    """Ground-truth label source for tampering."""

    type: Literal["memory_overwrite"] = "memory_overwrite"
    entry_id: str
    agent: AgentId
    kind: MemoryKind
    round: int
    old_text: str | None               # None if the entry was created by the overwrite
    new_text: str
    mode: Literal["replace", "append"]
    by: Literal["GOD"] = "GOD"
    intervention_id: str
    now_round: int
    created: bool = False


class GodToolData(Record):
    """Payload: one free God observation call (digest only)."""
    type: Literal["god_tool"] = "god_tool"
    tool: Literal["activity_check", "message_check", "wait"]
    args: dict[str, Any] = {}
    result_digest: str = ""
    result_size: int = 0
    ceiling_hit: bool = False


class GodPowerData(Record):
    """Payload: one God power attempt (cost, budget, ok)."""
    type: Literal["god_power"] = "god_power"
    power: Literal["message_fraud", "cot_control", "reveal_needs"]
    intervention_id: str | None = None  # None when ok=False
    action: GodAction
    cost: int
    budget_left: int
    ok: bool
    reason: str | None = None


class GodThoughtData(Record):
    """Payload (god.kind=llm only): one step of the LLM God's tool loop, for analysis. `text` is God's
    short stated rationale (the JSON "why"), `action` the parsed action name, `note` what the loop did
    with it (e.g. invalid_json, step_limit, budget_spent). God-only: never rendered to any agent."""
    type: Literal["god_thought"] = "god_thought"
    step: int
    text: str = ""
    action: str | None = None
    args: dict[str, Any] = {}
    note: str | None = None
    llm_call_ids: list[str] = []


class NoticeData(Record):
    """Payload: a notice queued for an agent."""
    type: Literal["notice"] = "notice"
    notice: Notice


class ContextBuiltData(Record):
    """Payload: token accounting of one built context."""
    type: Literal["context_built"] = "context_built"
    agent: AgentId
    policy: str
    tokens_by_section: dict[str, int]
    tokens_total: int
    budget: int
    included_item_ids: list[str]
    dropped_count: int = 0


class CompactionData(Record):
    """Payload: a compaction record."""
    type: Literal["compaction"] = "compaction"
    record: CompactionRecord


class WinData(Record):
    """Payload: an agent won."""
    type: Literal["win"] = "win"
    agent: AgentId
    score: float


class RoundEndData(Record):
    """Payload: end-of-round state hash and scores."""
    type: Literal["round_end"] = "round_end"
    state_hash: str
    scores: dict[AgentId, float]
    welfare: float


class AgentOutcome(Record):
    """Final result for one agent. score = the raw needs score (held open needs / needs; locked at the
    win under lock_and_stay). game.score (game-fixes) adds, only when on (otherwise None and left out
    of every dump): key_bonus = held_key_bonus * keys held at game end, private = score + key_bonus
    (held_key_bonus > 0); shared = score + team_share * mean score of all agents (team_share > 0)."""
    score: float
    won_round: int | None = None
    burned: list[InfoId] = []
    lost: list[InfoId] = []
    key_bonus: float | None = None
    private: float | None = None
    shared: float | None = None        # game.score.team_share: raw + a * mean raw of all agents
    community_bonus: float | None = None   # game.score.community_bonus: B or 0 (showcase)
    delivery_credit: float | None = None   # game.score.delivery_credit: phi * credited deliveries (showcase)

    @model_serializer(mode="wrap")
    def _drop_unset_scores(self, handler: SerializerFunctionWrapHandler) -> Any:
        d = handler(self)
        if isinstance(d, dict):
            for k in ("key_bonus", "private", "shared", "community_bonus", "delivery_credit"):
                if d.get(k, 0) is None:
                    d.pop(k)
        return d


class Outcome(Record):
    """Final result of one game."""
    game_id: GameId
    rounds_played: int
    end_reason: Literal["max_rounds", "all_done", "budget_exceeded", "error"]
    per_agent: dict[AgentId, AgentOutcome]
    welfare: float                     # mean RAW score (unchanged meaning)
    win_rate: float
    # game.score (game-fixes) only, else None and left out of every dump: mean private score
    # (held_key_bonus) / mean team score (team_share)
    welfare_private: float | None = None
    welfare_shared: float | None = None
    welfare_community: float | None = None   # game.score.community_bonus: mean(raw + community_bonus)
    delivery_credit_total: float | None = None   # game.score.delivery_credit: sum of all credits paid

    @model_serializer(mode="wrap")
    def _drop_unset_welfare(self, handler: SerializerFunctionWrapHandler) -> Any:
        d = handler(self)
        if isinstance(d, dict):
            for k in ("welfare_private", "welfare_shared", "welfare_community", "delivery_credit_total"):
                if d.get(k, 0) is None:
                    d.pop(k)
        return d


class GameEndData(Record):
    """Payload: the game outcome."""
    type: Literal["game_end"] = "game_end"
    outcome: Outcome


EventData = Annotated[
    GameStartData | TurnData | ActionResultData | MessageData | KeyTransferData | BurnData
    | MemoryWriteData | MemoryOverwriteData | GodToolData | GodPowerData | GodThoughtData | NoticeData
    | ContextBuiltData | CompactionData | WinData | RoundEndData | GameEndData | EdgeAddedData
    | TradeOfferData | TradeReplyData | TradeExecutedData | TradeFailedData,
    Field(discriminator="type"),
]
EventType = Literal["game_start", "turn", "action_result", "message", "key_transfer", "burn",
                    "memory_write", "memory_overwrite", "god_tool", "god_power", "god_thought", "notice",
                    "context_built", "compaction", "win", "round_end", "game_end", "edge_added",
                    "trade_offer", "trade_reply", "trade_executed", "trade_failed"]
Actor = AgentId | Literal["GOD", "SYSTEM"]


class EventBody(Record):
    """What rules/powers/engine produce. The EventSink stamps v, run_id, game_id, seq."""

    round: int
    turn_index: int | None = None
    phase: Phase
    actor: Actor
    data: EventData


class Event(Record):
    """One line of events.jsonl: stamped envelope + typed payload."""
    v: int = SCHEMA_VERSION
    run_id: RunId
    game_id: GameId
    seq: int
    round: int
    turn_index: int | None = None
    phase: Phase
    type: EventType
    actor: Actor
    data: EventData

    @model_validator(mode="after")
    def _type_matches(self) -> Event:
        if self.data.type != self.type:
            raise ValueError(f"event type {self.type!r} != payload type {self.data.type!r}")
        return self


def make_event(body: EventBody, *, run_id: RunId, game_id: GameId, seq: int) -> Event:
    """Stamp an EventBody into an Event."""
    return Event(run_id=run_id, game_id=game_id, seq=seq, round=body.round, turn_index=body.turn_index,
                 phase=body.phase, type=body.data.type, actor=body.actor, data=body.data)


class EventSink(Protocol):
    """Anything that stamps and stores events (RunWriter, MemorySink)."""
    def emit(self, body: EventBody) -> Event:
        """Stamp and store one event body; returns the Event."""
        ...


class MemorySink:
    """In-memory EventSink for tests and fakes. seq starts at 0 and is contiguous."""

    def __init__(self, run_id: RunId = "test", game_id: GameId = "g0") -> None:
        self.run_id, self.game_id = run_id, game_id
        self.events: list[Event] = []

    def emit(self, body: EventBody) -> Event:
        """Stamp and store one event body; returns the Event."""
        ev = make_event(body, run_id=self.run_id, game_id=self.game_id, seq=len(self.events))
        self.events.append(ev)
        return ev


class Snapshot(Record):
    """One line of snapshots.jsonl (end of each round, plus round 0). No ledgers."""

    v: int = SCHEMA_VERSION
    run_id: RunId
    game_id: GameId
    round: int
    state: GameState
