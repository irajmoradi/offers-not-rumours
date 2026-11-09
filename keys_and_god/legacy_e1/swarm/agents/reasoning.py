"""cot_mode handling (pure). All reasoning-mode logic lives here; the engine is mode-agnostic.

| cot_mode  | request thinking | CoT source                             |
|-----------|------------------|----------------------------------------|
| none      | False            | none                                   |
| explicit  | False            | the reply's "thought" JSON field       |
| native    | True             | ChatResponse.reasoning (vLLM / OR)     |

Imports: nothing from swarm. Pure.
"""

from __future__ import annotations

from typing import Literal

CotMode = Literal["none", "native", "explicit"]


def request_thinking(cot_mode: CotMode) -> bool:
    """Value for ChatRequest.thinking."""
    return cot_mode == "native"


def wants_thought_field(cot_mode: CotMode) -> bool:
    """True when the reply's "thought" field is the CoT (explicit mode)."""
    return cot_mode == "explicit"


def extract_cot(cot_mode: CotMode, thought: str | None, reasoning: str | None) -> str | None:
    """The CoT to store as MemoryEntry(kind="cot"); None if the mode has none."""
    if cot_mode == "explicit":
        return thought or None
    if cot_mode == "native":
        return (reasoning or "").strip() or None
    return None
