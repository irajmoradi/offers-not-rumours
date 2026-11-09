"""End-to-end truth-leak oracle (ARCHITECTURE_v2 section 10, REVIEW_v2 4.1.3).

Scans every agent-bound prompt (ChatRequest with meta.role in {agent, summarizer}) for:
  - truth/provenance field names: true_sender, original_text, intervention_id, tampered_by, ...
    anywhere (TRUTH_NEEDLES), and forged / tampered / taint in serialized key-value form
    (TRUTH_WORD_FIELDS, e.g. `"forged": true`), case-insensitive;
  - canaries learned from the event stream as it is emitted (time-aware, so a note that was legit
    in a prompt BEFORE it was tampered is not flagged):
      * every intervention id (god_power / message / memory_overwrite events),
      * the replaced original text of a `replace` memory_overwrite (if >= min_canary_len chars), as
        rendered for that round's memory entry ("(round r): <text>"),
  - extra canaries added by the caller (`add_canary`), optionally scoped to some agents and a round
    range, e.g. an unreleased defect info id for non-victims, or a God canary token (make_canary);
  - PROMPT_BANNED_WORDS (D14) in system messages (scope "system", default) or all messages ("all").
    Agent speech can legitimately say "god", so the default only checks the templated system part.
The literal word "GOD" is never a canary by itself; "god" is only a banned word under its scope.

    oracle = LeakOracle()
    sink = oracle.wrap(sink)               # observes every emitted event
    ... llm middleware / FakeLLM: oracle.assert_clean(request)   (or hits = oracle.check(request))

With reveal_provenance=true (ablation) construct LeakOracle(enabled=False).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from swarm.schema import (
    ChatRequest,
    Event,
    EventBody,
    EventSink,
    GodPowerData,
    MemoryOverwriteData,
    MessageData,
)

PROMPT_BANNED_WORDS: frozenset[str] = frozenset(
    {"god", "adversary", "attacker", "forged", "forgery", "tamper", "injected"})
# Identifier-like field names: any occurrence is a leak.
TRUTH_NEEDLES: tuple[str, ...] = (
    "true_sender", "tampered_by", "tamper_round", "original_text", "intervention_id",
    "source_item_ids", "copy_id", "deliver_from_turn", "release_turn",
)
# Plain-word field names: flagged only in a serialized key/value form (`"forged": true`,
# `tampered=False`, `taint: [...]`), so natural speech ("a forged signature") is not a hit.
TRUTH_WORD_FIELDS: tuple[str, ...] = ("forged", "tampered", "taint")
SCANNED_ROLES: frozenset[str] = frozenset({"agent", "summarizer"})


class TruthLeak(AssertionError):
    pass


@dataclass(frozen=True)
class Canary:
    needle: str
    kind: str                                   # "intervention_id" | "original_text" | "custom" | ...
    agents: frozenset[str] | None = None        # None = every viewer
    from_round: int | None = None               # active for requests with meta.round >= this
    until_round: int | None = None              # ... and < this (None = forever)
    case_sensitive: bool = True

    def applies(self, agent: str | None, round: int | None) -> bool:
        if self.agents is not None and agent not in self.agents:
            return False
        if round is not None:
            if self.from_round is not None and round < self.from_round:
                return False
            if self.until_round is not None and round >= self.until_round:
                return False
        return True


@dataclass(frozen=True)
class LeakHit:
    kind: str                                   # "truth_field" | "banned_word" | canary kind
    needle: str
    call_id: str
    message_index: int
    snippet: str


def _word_re(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _kv_re(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(word)
                      + r"""["']?\s*[:=]\s*(?:true|false|none|null|\d|["'\[{(])""", re.IGNORECASE)


_TRUTH_RES = [(w, _word_re(w)) for w in TRUTH_NEEDLES] + [(w, _kv_re(w)) for w in TRUTH_WORD_FIELDS]
_BANNED_RES = [(w, _word_re(w)) for w in sorted(PROMPT_BANNED_WORDS)]


def _snip(text: str, start: int, end: int, pad: int = 40) -> str:
    return text[max(0, start - pad): end + pad].replace("\n", " ")


def scan_text(text: str, canaries: Sequence[Canary] = (), *, truth_fields: bool = True,
              banned_words: bool = False, call_id: str = "", message_index: int = 0) -> list[LeakHit]:
    """Pure scan of one string. Canary scoping is NOT applied here (pass pre-filtered canaries)."""
    hits: list[LeakHit] = []

    def add(kind: str, needle: str, m: re.Match[str] | tuple[int, int]) -> None:
        s, e = (m.start(), m.end()) if isinstance(m, re.Match) else m
        hits.append(LeakHit(kind, needle, call_id, message_index, _snip(text, s, e)))

    if truth_fields:
        for w, rx in _TRUTH_RES:
            if (m := rx.search(text)) is not None:
                add("truth_field", w, m)
    if banned_words:
        for w, rx in _BANNED_RES:
            if (m := rx.search(text)) is not None:
                add("banned_word", w, m)
    low = None
    for c in canaries:
        if c.case_sensitive:
            i = text.find(c.needle)
        else:
            low = low if low is not None else text.lower()
            i = low.find(c.needle.lower())
        if i >= 0:
            add(c.kind, c.needle, (i, i + len(c.needle)))
    return hits


def scan_request(req: ChatRequest, canaries: Sequence[Canary] = (), *,
                 banned_scope: str = "system", roles: Iterable[str] = SCANNED_ROLES) -> list[LeakHit]:
    """Scan every message of an agent-bound request. Non-agent roles (god, judge) return []."""
    if req.meta.role not in set(roles):
        return []
    active = [c for c in canaries if c.applies(req.meta.agent, req.meta.round)]
    hits: list[LeakHit] = []
    for i, msg in enumerate(req.messages):
        banned = banned_scope == "all" or (banned_scope == "system" and msg.role == "system")
        hits += scan_text(msg.content, active, banned_words=banned, call_id=req.call_id, message_index=i)
    return hits


@dataclass
class LeakOracle:
    enabled: bool = True
    banned_scope: str = "system"                # "system" | "all" | "none"
    min_canary_len: int = 12
    canaries: list[Canary] = field(default_factory=list)
    hits: list[LeakHit] = field(default_factory=list)
    n_checked: int = 0

    # ------------------------------------------------------------------ learning
    def add_canary(self, canary: Canary) -> None:
        if canary.needle and canary not in self.canaries:
            self.canaries.append(canary)

    def observe(self, ev: Event | EventBody) -> None:
        """Feed every event, in emission order."""
        d = ev.data
        if isinstance(d, GodPowerData) and d.intervention_id:
            self.add_canary(Canary(d.intervention_id, "intervention_id"))
        elif isinstance(d, MessageData) and d.message.intervention_id:
            self.add_canary(Canary(d.message.intervention_id, "intervention_id"))
        elif isinstance(d, MemoryOverwriteData):
            self.add_canary(Canary(d.intervention_id, "intervention_id"))
            if d.mode == "replace" and d.old_text and len(d.old_text.strip()) >= self.min_canary_len:
                # The agent's pre-tamper note must never be shown again AS THAT ROUND'S ENTRY. The
                # needle is anchored to how context.base.memory_items renders it ("... (round r): text"),
                # so the same text in another round's note, or in a DM the agent wrote, is no hit.
                needle = f"(round {d.round}): {d.old_text.strip()}"
                self.add_canary(Canary(needle, "original_text", from_round=ev.round))

    def observe_all(self, events: Iterable[Event | EventBody]) -> None:
        for ev in events:
            self.observe(ev)

    def wrap(self, sink: EventSink) -> EventSink:
        return _OracleSink(sink, self)

    # ------------------------------------------------------------------ checking
    def check(self, req: ChatRequest) -> list[LeakHit]:
        if not self.enabled:
            return []
        self.n_checked += 1
        hits = scan_request(req, self.canaries, banned_scope=self.banned_scope)
        self.hits += hits
        return hits

    def assert_clean(self, req: ChatRequest) -> None:
        hits = self.check(req)
        if hits:
            raise TruthLeak("; ".join(f"{h.kind}:{h.needle!r} in {h.call_id}[{h.message_index}]: "
                                      f"...{h.snippet}..." for h in hits[:5]))


class _OracleSink:
    def __init__(self, inner: EventSink, oracle: LeakOracle) -> None:
        self.inner, self.oracle = inner, oracle

    def emit(self, body: EventBody) -> Event:
        ev = self.inner.emit(body)
        self.oracle.observe(ev)
        return ev

    def __getattr__(self, name: str):        # pass through e.g. MemorySink.events
        return getattr(self.inner, name)
