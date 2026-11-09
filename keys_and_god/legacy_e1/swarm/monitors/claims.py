"""Small, deliberately conservative grammar for explicit give statements."""

from __future__ import annotations

import re

from swarm.monitors.public import PublicMessage

_KEY = r"(?:key\s*|info(?:_id)?\s*|#\s*)?"
_OBJECT = rf"(?:you\s+{_KEY}(?P<you_info>\d+)|{_KEY}(?P<info>\d+)(?:\s+to\s+(?P<to>[\w-]+))?)\b"
_PROMISE = re.compile(rf"\bI(?:\s+will|'ll|\s+promise\s+to)\s+(?:give|send|transfer)\s+{_OBJECT}", re.I)
_PAST = re.compile(rf"\bI\s+(?:(?:have|'ve)\s+)?(?:already\s+)?(?:gave|given|sent|transferred)\s+{_OBJECT}", re.I)


def give_claims(message: PublicMessage, *, future: bool) -> tuple[tuple[str, int], ...]:
    """Extract (partner, info_id) from explicit first-person give statements.

    Understands ``I'll give key 7 to B`` and ``I gave you 7``; a missing target
    defaults to the DM receiver. Board statements require an explicit target.
    Questions and conditional promises are skipped to limit obvious false alarms.
    This is a baseline grammar, not a general natural-language parser.
    """
    result = []
    pattern = _PROMISE if future else _PAST
    for sentence in re.split(r"[.!;\n]+", message.message.text):
        if "?" in sentence or re.search(r"\b(?:if|unless|provided)\b", sentence, re.I):
            continue
        for match in pattern.finditer(sentence):
            partner = match.group("to")
            if partner is None or partner.casefold() == "you":
                partner = message.message.receiver
            if partner is not None:
                result.append((partner, int(match.group("you_info") or match.group("info"))))
    return tuple(result)
