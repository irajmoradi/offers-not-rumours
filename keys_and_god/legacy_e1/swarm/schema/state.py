"""TRUTH side (engine and analysis only): world and game state. Never rendered to agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import SerializerFunctionWrapHandler, model_serializer

from swarm.rng import canonical_json, sha256_hex
from swarm.schema.base import AgentId, GameId, InfoId, MemoryKind, MutableState, Record


class KeyCopy(Record):
    """The rivalrous unit. copy_id and origin are engine-internal (never shown to agents)."""

    copy_id: str
    info_id: InfoId
    key: int
    origin: Literal["initial", "defect_mint"] = "initial"


class AgentState(MutableState):
    """One agent's truth state: inventory, needs, burns, penalties, win, last observation slot."""
    agent: AgentId
    keys: list[KeyCopy]
    needs: list[InfoId]
    burned: list[InfoId] = []          # needs burned by a correct defect against this agent
    lost: list[InfoId] = []            # needs lost as wrong-defect penalty (lose_need)
    delayed_until_round: int = -1      # wrong-defect penalty `delay`: no turns while round <= this
    won_round: int | None = None
    last_obs_turn: int = -1            # turn_index of this agent's previous observation
    # game.network (added after the freeze; both None = no network = the original rules). contacts:
    # the agent's graph neighbours (sorted). known: game.network.mode=known_ids only, the names it
    # knows (starts as its contacts, grows as it learns names; sorted). contacts set + known None =
    # mode contacts. Both are left out of every dump while None, so state hashes, snapshots and
    # game_start events of no-network games are byte-identical to before.
    contacts: list[AgentId] | None = None
    known: list[AgentId] | None = None

    @model_serializer(mode="wrap")
    def _drop_unset_network(self, handler: SerializerFunctionWrapHandler) -> Any:
        d = handler(self)
        if isinstance(d, dict):
            for k in ("contacts", "known"):
                if d.get(k, 0) is None:
                    d.pop(k)
        return d


class Message(Record):
    """A DM or board post. Ground truth for fraud lives here (forged, true_sender, intervention_id).
    Forged DMs are stamped exactly like real ones (same msg_id scheme, round/turn of the slot they
    are delivered in), so nothing agent-visible distinguishes them."""

    msg_id: str
    round: int
    turn_index: int
    channel: Literal["dm", "board"]
    sender_shown: AgentId
    true_sender: AgentId | Literal["GOD"]
    receiver: AgentId | None = None    # None for board
    text: str
    forged: bool = False
    intervention_id: str | None = None


class Taint(Record):
    """Hidden provenance mark: which intervention influenced an entry, and how."""
    intervention_id: str
    via: Literal["direct", "derived", "exposed"]


class MemoryEntry(Record):
    """CoT / scratchpad / summary entry. One per (agent, kind, round); entry_id = rng.make_entry_id."""

    entry_id: str
    agent: AgentId
    kind: MemoryKind
    round: int
    text: str
    tampered: bool = False
    original_text: str | None = None   # agent-authored text before the FIRST overwrite
    tampered_by: Literal["GOD"] | None = None
    tamper_round: int | None = None
    taint: list[Taint] = []
    covers_rounds: tuple[int, int] | None = None   # summaries only
    source_item_ids: list[str] = []                # summaries only


class ActionRecord(Record):
    """Truth-side history of actions view_log may render (built by rules, read by observe)."""

    turn_index: int
    round: int
    kind: Literal["give", "defect", "board_post", "dm"]
    actor: AgentId | Literal["GOD"]
    target: AgentId | None = None
    info_id: InfoId | None = None
    msg_id: str | None = None
    ok: bool = True                    # defect: correct (True) or wrong (False)
    release_turn: int | None = None    # defects only (see defect_release_turn)


class Notice(Record):
    """System notification for one agent, delivered at its next observation after turn_index."""

    agent: AgentId
    round: int
    turn_index: int
    kind: Literal["gift_received", "burned", "defect_attempt", "defect_result", "need_lost",
                  "delayed", "won", "left", "system"]   # left: a winner left (winner_policy sit_out)
    text: str
    ref: str | None = None             # truth pointer (event/msg id); never rendered
    deliver_from_turn: int | None = None   # None = next observation; else first obs with t >= this


class GameState(MutableState):
    """The whole game's truth state (mutable; snapshotted per round; no ledgers)."""
    game_id: GameId
    round: int = 0
    turn_index: int = 0
    order: list[AgentId]
    agents: dict[AgentId, AgentState]
    messages: list[Message] = []
    memory: list[MemoryEntry] = []
    history: list[ActionRecord] = []
    notices: list[Notice] = []
    god_budget_left: int = 0
    n_interventions: int = 0           # index for make_intervention_id
    done: bool = False

    def state_hash(self) -> str:
        """sha256 of the canonical JSON of the full state."""
        return sha256_hex(canonical_json(self))
