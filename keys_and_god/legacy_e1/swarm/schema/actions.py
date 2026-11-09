"""Actions (agent -> engine; rules.apply(state, actor, action, cfg)) and God actions (observation
tools are free with a safety ceiling per wake; powers cost budget). Tagged unions on `kind`."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from swarm.schema.base import AgentId, InfoId, MemoryKind, Record


class SendMessage(Record):
    """Private DM to one agent."""
    kind: Literal["send_message"] = "send_message"
    to: AgentId
    text: str


class PostBoard(Record):
    """Public board post."""
    kind: Literal["post_board"] = "post_board"
    text: str


class GiveKey(Record):
    """Give one held copy of info_id to another agent."""
    kind: Literal["give_key"] = "give_key"
    to: AgentId
    info_id: InfoId


class DefectAction(Record):
    """Burn target's need info_id (penalty if target does not need it)."""
    kind: Literal["defect"] = "defect"
    target: AgentId
    info_id: InfoId


class GrepLog(Record):
    """Case-insensitive literal substring over the agent's visible log history. Max once per turn."""

    kind: Literal["grep_log"] = "grep_log"
    pattern: str = Field(min_length=1, max_length=80)
    rounds: int | None = None          # look back this many rounds (None = all)


class Noop(Record):
    """Do nothing."""
    kind: Literal["noop"] = "noop"


Action = Annotated[SendMessage | PostBoard | GiveKey | DefectAction | GrepLog | Noop,
                   Field(discriminator="kind")]
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)
ACTION_KINDS = ("send_message", "post_board", "give_key", "defect", "grep_log", "noop")


class AgentOutput(Record):
    """Parsed result of one agent turn (from LLMAgent or ScriptedAgent)."""
    cot: str | None = None
    note: str | None = None            # scratchpad write (free; no action slot)
    actions: list[Action] = []
    parse_ok: bool = True
    retries: int = 0
    errors: list[str] = []
    llm_call_ids: list[str] = []


class ActivityCheck(Record):
    """God tool: recent activity summary (free)."""
    kind: Literal["activity_check"] = "activity_check"
    last_n_rounds: int = 1


class MessageCheck(Record):
    """God tool: filter/grep the message log (free)."""
    kind: Literal["message_check"] = "message_check"
    sender: AgentId | None = None
    receiver: AgentId | None = None
    contains: str | None = None
    last_n_rounds: int | None = None


class Wait(Record):
    """God tool: sleep until n more agent turns or the next wake (free)."""
    kind: Literal["wait"] = "wait"
    n_turns: int | None = None         # None = until next wake


class MessageFraud(Record):
    """God power: deliver a DM to receiver that appears to come from sender."""
    kind: Literal["message_fraud"] = "message_fraud"
    sender: AgentId                    # shown sender (impersonated)
    receiver: AgentId
    text: str


class CotControl(Record):
    """God power: overwrite/append agent's memory entry of a round (default: scratchpad)."""
    kind: Literal["cot_control"] = "cot_control"
    agent: AgentId
    target: MemoryKind = "scratchpad"
    round: int
    text: str
    mode: Literal["replace", "append"] = "replace"


GodAction = Annotated[ActivityCheck | MessageCheck | Wait | MessageFraud | CotControl,
                      Field(discriminator="kind")]
GOD_ACTION_ADAPTER: TypeAdapter[GodAction] = TypeAdapter(GodAction)
