"""God's observation tools (WP4): activity_check, message_check, wait.

They are free (no budget). One `GodView` is created per wake. It counts calls and stops at
`tool_call_safety_ceiling` (default 50, D11): calls 1..ceiling run normally; the next call logs one
`god_tool` event with ceiling_hit=True, returns None, and marks the view exhausted (the wake should
end). Every call logs one `god_tool` event (args, result digest and size); the events are collected
in `view.events` for the engine to emit.

God sees the truth side (Message, ActionRecord), never agent prompts.
"""

from __future__ import annotations

from typing import Any

from swarm.rng import canonical_json, sha256_hex
from swarm.schema import (
    GOD,
    ActionRecord,
    ActivityCheck,
    EventBody,
    GameState,
    GodToolData,
    Message,
    MessageCheck,
    Wait,
)

DEFAULT_CEILING = 50


class GodView:
    def __init__(self, state: GameState, *, ceiling: int = DEFAULT_CEILING) -> None:
        self.state = state
        self.ceiling = ceiling
        self.calls = 0
        self.exhausted = False
        self.events: list[EventBody] = []
        self.waits: list[int | None] = []

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, tool: str, args: dict[str, Any], result: Any, ceiling_hit: bool = False) -> None:
        payload = canonical_json(result) if result is not None else ""
        data = GodToolData(tool=tool, args=args, result_digest=sha256_hex(payload)[:16] if payload else "",
                           result_size=len(result) if isinstance(result, list) else 0,
                           ceiling_hit=ceiling_hit)
        self.events.append(EventBody(round=self.state.round, turn_index=self.state.turn_index, phase="god",
                                     actor=GOD, data=data))

    def _admit(self, tool: str, args: dict[str, Any]) -> bool:
        if self.exhausted:
            return False
        if self.calls >= self.ceiling:
            self.exhausted = True
            self._log(tool, args, None, ceiling_hit=True)
            return False
        self.calls += 1
        return True

    # ------------------------------------------------------------------ tools
    def activity_check(self, last_n_rounds: int = 1) -> list[ActionRecord] | None:
        """Actions (give, defect, board_post, dm) of the last n rounds, incl. the current one."""
        args = {"last_n_rounds": last_n_rounds}
        if not self._admit("activity_check", args):
            return None
        lo = self.state.round - last_n_rounds
        out = [a for a in self.state.history if a.round > lo]
        self._log("activity_check", args, out)
        return out

    def message_check(self, sender: str | None = None, receiver: str | None = None,
                      contains: str | None = None, last_n_rounds: int | None = None) -> list[Message] | None:
        """Filter all messages (DMs and board). `sender` matches the shown sender; `contains` is a
        case-insensitive literal substring."""
        args = {k: v for k, v in {"sender": sender, "receiver": receiver, "contains": contains,
                                  "last_n_rounds": last_n_rounds}.items() if v is not None}
        if not self._admit("message_check", args):
            return None
        needle = contains.lower() if contains else None
        lo = None if last_n_rounds is None else self.state.round - last_n_rounds
        out = [m for m in self.state.messages
               if (sender is None or m.sender_shown == sender)
               and (receiver is None or m.receiver == receiver)
               and (needle is None or needle in m.text.lower())
               and (lo is None or m.round > lo)]
        self._log("message_check", args, out)
        return out

    def wait(self, n_turns: int | None = None) -> bool:
        """Records that God waits (until the next wake, or n more agent turns)."""
        args = {} if n_turns is None else {"n_turns": n_turns}
        if not self._admit("wait", args):
            return False
        self.waits.append(n_turns)
        self._log("wait", args, None)
        return True

    def run(self, action: ActivityCheck | MessageCheck | Wait) -> Any:
        """Dispatch a tool GodAction (as an LLM God would emit it)."""
        if isinstance(action, ActivityCheck):
            return self.activity_check(action.last_n_rounds)
        if isinstance(action, MessageCheck):
            return self.message_check(action.sender, action.receiver, action.contains, action.last_n_rounds)
        if isinstance(action, Wait):
            return self.wait(action.n_turns)
        raise TypeError(f"not a God tool: {action.kind}")
