"""Whitelist projection of finished event logs for the rule-based watchtower.

Only message contents/stamps and observed public log lines are evidence. Raw
burns, transfers, actions, memory, God events and event provenance are ignored.
No runlog dependency is needed; callers can pass Events or decoded JSONL rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from swarm.schema import AgentMessageView, Event, LogLine

EventLike = Event | Mapping[str, Any]


def _get(record: Any, name: str) -> Any:
    """Read exactly one whitelisted field without serializing a truth record."""
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


@dataclass(frozen=True)
class PublicTurn:
    """One observed agent turn; no event sequence or provenance is retained."""

    game_id: str
    agent: str
    round: int
    turn_index: int

    @property
    def unit_id(self) -> str:
        """Stable per-agent/round identifier within a game."""
        return f"{self.agent}:{self.round}"


@dataclass(frozen=True)
class PublicMessage:
    """Visible message plus its public delivery slot (also used for forgeries)."""

    game_id: str
    round: int
    turn_index: int
    message: AgentMessageView


@dataclass(frozen=True)
class PublicAction:
    """A give/defect that actually appeared in an agent's visible log."""

    game_id: str
    turn_index: int
    line: LogLine


@dataclass(frozen=True)
class PublicTrace:
    """Immutable evidence accepted directly by both detectors."""

    turns: tuple[PublicTurn, ...]
    messages: tuple[PublicMessage, ...]
    actions: tuple[PublicAction, ...]


def public_trace(events: Iterable[EventLike] | PublicTrace) -> PublicTrace:
    """Project visible traffic; never inspect truth flags or event actor/phase/seq.

    Actions are reconstructed ONLY from ``turn.data.obs.log``. Their slot is
    recovered from the observed (actor, round) turn, since LogLine has no slot.
    Hidden identities and actions with no corresponding turn are unscorable.
    Messages use their own public round/turn stamps, never the event envelope's.
    Repeated observations of the same log line/message are deduplicated.
    """
    if isinstance(events, PublicTrace):
        return events
    turns: dict[tuple[str, str, int], PublicTurn] = {}
    messages: dict[tuple[str, str], PublicMessage] = {}
    lines: set[tuple[str, LogLine]] = set()
    for event in events:
        kind = _get(event, "type")
        if kind not in {"turn", "message"}:
            continue
        game_id = _get(event, "game_id")
        data = _get(event, "data")
        if kind == "turn":
            obs = _get(data, "obs")
            turn = PublicTurn(game_id, _get(obs, "agent"), _get(obs, "round"),
                              _get(obs, "turn_index"))
            turns[(game_id, turn.agent, turn.round)] = turn
            for raw in _get(obs, "log"):
                # Rebuild field by field rather than dumping a larger record.
                line = LogLine(**{field: _get(raw, field) for field in LogLine.model_fields})
                if line.kind in {"give", "defect"} and line.actor is not None:
                    lines.add((game_id, line))
        else:
            raw = _get(data, "message")
            message = AgentMessageView(
                msg_id=_get(raw, "msg_id"), channel=_get(raw, "channel"),
                sender=_get(raw, "sender_shown"), receiver=_get(raw, "receiver"),
                text=_get(raw, "text"),
            )
            messages[(game_id, message.msg_id)] = PublicMessage(
                game_id, _get(raw, "round"), _get(raw, "turn_index"), message,
            )
    actions = []
    for game_id, line in lines:
        origin = turns.get((game_id, line.actor, line.round))
        if origin is not None:
            actions.append(PublicAction(game_id, origin.turn_index, line))
    return PublicTrace(
        tuple(sorted(turns.values(), key=lambda t: (t.game_id, t.turn_index, t.agent))),
        tuple(sorted(messages.values(), key=lambda m: (m.game_id, m.turn_index, m.message.msg_id))),
        tuple(sorted(actions, key=lambda a: (a.game_id, a.turn_index, a.line.model_dump_json()))),
    )
