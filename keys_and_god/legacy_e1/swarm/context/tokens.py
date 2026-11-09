"""Token counters (WP13-lite). `make_counter(cfg.context.tokenizer)`.

"words"  -> WordCounter (tests), "approx" -> ApproxCounter (ceil(len/3.5), the cheap estimate).
"hf:<model>" -> ApproxCounter for now (demo: no tokenizer load); swap in a real tokenizer later.
"""

from __future__ import annotations

from swarm.context.base import ApproxCounter, TokenCounter, WordCounter


def make_counter(name: str) -> TokenCounter:
    if name == "words":
        return WordCounter()
    if name == "approx" or name.startswith("hf:"):
        return ApproxCounter()
    raise ValueError(f"unknown tokenizer {name!r}")
