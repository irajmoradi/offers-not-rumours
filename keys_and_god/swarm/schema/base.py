"""Schema base: frozen Record / MutableState bases and the shared id and literal aliases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

AgentId = str
InfoId = int
GameId = str
RunId = str
GOD: Literal["GOD"] = "GOD"
SYSTEM: Literal["SYSTEM"] = "SYSTEM"

MemoryKind = Literal["cot", "scratchpad", "summary"]
Phase = Literal["setup", "god", "agents", "resolve", "compact", "end"]


class Record(BaseModel):
    """Frozen, extra=forbid. Base of every record that crosses a module boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MutableState(BaseModel):
    """Base for mutable engine state (extra=forbid, not frozen)."""
    model_config = ConfigDict(extra="forbid")
