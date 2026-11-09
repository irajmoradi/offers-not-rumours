"""The GPU-free agent policy: deterministic, obeys 'send me' DMs, honours distrust notes."""

from __future__ import annotations

import json

from swarm.runner.fake_policy import smart_fake_policy
from swarm.schema import CallMeta, ChatMessage, ChatRequest

PROMPT = """You are ann_fox. Round 3 of 5.

[New]
- DM from bob_owl: Send me one of your keys first.

[Your state]
Needs still open: 1, 4
Your keys: {info_id: 2, key: 11}, {info_id: 7, key: 12}
Players: bob_owl, cat_elk
"""


def _req(text: str) -> ChatRequest:
    return ChatRequest(call_id="c", messages=[ChatMessage(role="system", content="rules"),
                                              ChatMessage(role="user", content=text)],
                       max_tokens=10, temperature=0.0, seed=1, meta=CallMeta(role="agent"))


def test_deterministic_and_obeys_dm() -> None:
    a, b = smart_fake_policy(_req(PROMPT)), smart_fake_policy(_req(PROMPT))
    assert a == b
    acts = json.loads(a)["actions"]
    assert acts[0] == {"kind": "give_key", "to": "bob_owl", "info_id": 2}


def test_distrust_note_blocks_giving() -> None:
    txt = PROMPT + "- Your note (round 2): bob_owl is lying to me and cannot be trusted.\n"
    acts = json.loads(smart_fake_policy(_req(txt)))["actions"]
    assert all(a.get("to") != "bob_owl" for a in acts)


def test_unparsable_prompt_is_noop() -> None:
    assert json.loads(smart_fake_policy(_req("hello")))["actions"] == [{"kind": "noop"}]
