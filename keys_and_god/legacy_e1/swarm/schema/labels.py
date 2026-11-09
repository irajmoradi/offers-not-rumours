"""Labels and monitor scores (derived from events, never primary)."""

from __future__ import annotations

from typing import Any, Literal

from swarm.schema.base import AgentId, GameId, Record


class Label(Record):
    """Derived ground-truth label for evaluation (from events)."""
    game_id: GameId
    label_kind: Literal["forged_message", "tampered_memory", "tampered_turn"]
    unit_id: str
    agent: AgentId
    round: int
    positive: bool
    intervention_id: str | None = None


class MonitorScore(Record):
    """One monitor score for one unit."""
    monitor: str
    game_id: GameId
    unit_kind: str
    unit_id: str
    agent: AgentId
    round: int
    score: float
    detail: dict[str, Any] = {}
