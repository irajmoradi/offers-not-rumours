"""MemoryStore (WP0, FROZEN): the ONLY interface to agent CoT, scratchpad and summaries, and the
ONLY mind-control path (God's overwrite).

Entries live in `GameState.memory` (so snapshots/forks carry them). Every method takes the state
explicitly; the store itself holds only config (no hidden reference to a possibly stale state copy).
Mutating methods mutate `state.memory` in place and return the EventBody to emit.

Rules:
- One entry per (agent, kind, round), entry_id = rng.make_entry_id(agent, kind, round).
- Agent writes are truncated to the configured caps. Writing to a round that already has an entry
  (e.g. God pre-created it, or a second note in the same turn) APPENDS "\n" + text and keeps the
  entry's tamper flags and taint.
- overwrite(): replace (default) or append; a missing round is created. Sets tampered=True,
  tampered_by="GOD", tamper_round=now, keeps the FIRST agent-authored text in original_text and adds
  Taint(intervention_id, "direct"). Only adversary.powers.apply_cot_control may call it.
- Agents see MemoryView (kind, round, text): never the flags.
"""

from __future__ import annotations

from typing import Literal

from swarm.config import ReasoningCfg
from swarm.rng import make_entry_id
from swarm.schema import (
    GOD,
    AgentId,
    EventBody,
    GameState,
    MemoryEntry,
    MemoryKind,
    MemoryOverwriteData,
    MemoryView,
    MemoryWriteData,
    Phase,
    Taint,
)


class MemoryStore:
    """Stateless access to GameState.memory, configured with reasoning caps."""
    def __init__(self, cfg: ReasoningCfg | None = None) -> None:
        self.cfg = cfg or ReasoningCfg()

    # ------------------------------------------------------------------ reads
    def _index(self, state: GameState, entry_id: str) -> int | None:
        for i, e in enumerate(state.memory):
            if e.entry_id == entry_id:
                return i
        return None

    def get(self, state: GameState, agent: AgentId, kind: MemoryKind, round: int) -> MemoryEntry | None:
        """The entry for (agent, kind, round), or None."""
        i = self._index(state, make_entry_id(agent, kind, round))
        return None if i is None else state.memory[i]

    def entries(self, state: GameState, agent: AgentId, kind: MemoryKind) -> list[MemoryEntry]:
        """All entries of one agent and kind, oldest first."""
        return sorted((e for e in state.memory if e.agent == agent and e.kind == kind), key=lambda e: e.round)

    def window(self, state: GameState, agent: AgentId, kind: MemoryKind, upto_round: int,
               n: int | None) -> list[MemoryEntry]:
        """Entries with round <= upto_round, oldest first, the last n (None = all, 0 = none)."""
        es = [e for e in self.entries(state, agent, kind) if e.round <= upto_round]
        if n is None:
            return es
        return es[-n:] if n > 0 else []

    def views(self, state: GameState, agent: AgentId, kind: MemoryKind, upto_round: int,
              n: int | None) -> list[MemoryView]:
        """Agent-visible projection of window(): current text only."""
        return [MemoryView(kind=e.kind, round=e.round, text=e.text)
                for e in self.window(state, agent, kind, upto_round, n)]

    # ------------------------------------------------------------------ agent writes
    def _cap(self, kind: MemoryKind, text: str) -> str:
        if kind == "scratchpad":
            return text[: self.cfg.scratchpad.max_chars]
        if kind == "cot":
            return text[: self.cfg.max_cot_chars]
        return text

    def write(self, state: GameState, agent: AgentId, kind: MemoryKind, round: int, text: str, *,
              phase: Phase = "agents") -> tuple[MemoryEntry, EventBody]:
        """Agent-authored write (capped). Appends if the round already has an entry. Returns (entry, event body)."""
        text = self._cap(kind, text)
        eid = make_entry_id(agent, kind, round)
        i = self._index(state, eid)
        if i is None:
            entry = MemoryEntry(entry_id=eid, agent=agent, kind=kind, round=round, text=text)
            state.memory.append(entry)
        else:
            old = state.memory[i]
            entry = old.model_copy(update={"text": old.text + "\n" + text})
            state.memory[i] = entry
        body = EventBody(round=state.round, turn_index=state.turn_index, phase=phase, actor=agent,
                         data=MemoryWriteData(entry=entry))
        return entry, body

    # ------------------------------------------------------------------ God overwrite
    def overwrite(self, state: GameState, agent: AgentId, kind: MemoryKind, round: int, text: str, *,
                  intervention_id: str, now_round: int, mode: Literal["replace", "append"] = "replace",
                  by: Literal["GOD"] = GOD) -> tuple[MemoryEntry, EventBody]:
        """God's mind-control write (replace/append; creates if missing). Returns (entry, memory_overwrite body)."""
        if round > now_round:
            raise ValueError("cannot overwrite a future round")
        eid = make_entry_id(agent, kind, round)
        i = self._index(state, eid)
        taint = Taint(intervention_id=intervention_id, via="direct")
        if i is None:
            entry = MemoryEntry(entry_id=eid, agent=agent, kind=kind, round=round, text=text,
                                tampered=True, original_text=None, tampered_by=by, tamper_round=now_round,
                                taint=[taint])
            state.memory.append(entry)
            old_text, created = None, True
        else:
            old = state.memory[i]
            old_text, created = old.text, False
            new_text = text if mode == "replace" else old.text + "\n" + text
            original = old.original_text if old.tampered else old.text
            taints = list(old.taint) + ([taint] if taint not in old.taint else [])
            entry = old.model_copy(update={"text": new_text, "tampered": True, "original_text": original,
                                           "tampered_by": by, "tamper_round": now_round, "taint": taints})
            state.memory[i] = entry
        body = EventBody(
            round=state.round, turn_index=state.turn_index, phase="god", actor=GOD,
            data=MemoryOverwriteData(entry_id=eid, agent=agent, kind=kind, round=round, old_text=old_text,
                                     new_text=entry.text, mode=mode, by=by, intervention_id=intervention_id,
                                     now_round=now_round, created=created),
        )
        return entry, body
