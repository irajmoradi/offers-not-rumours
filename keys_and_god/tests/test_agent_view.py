"""Type-level guarantee: agent-visible models carry no truth fields and embed no truth models."""

from __future__ import annotations

import typing

from pydantic import BaseModel

import swarm.schema as S
from swarm.context.base import PassthroughPolicy, WordCounter, items_from_observation, memory_items
from swarm.memory import MemoryStore
from tests.fixtures.fixed_state import INTERVENTION_IDS, ORIGINAL_NOTE, TAMPERED_NOTE

EXPECTED_VIEWS = {"Observation", "PinnedFacts", "KeyView", "AgentMessageView", "LogLine", "NoticeView",
                  "MemoryView", "GrepResult", "PromptView", "RenderSection", "RenderItem"}
TRUTH_MODELS = (S.GameState, S.AgentState, S.KeyCopy, S.Message, S.MemoryEntry, S.ActionRecord, S.Notice,
                S.Event, S.EventBody, S.HistoryItem, S.ContextBundle, S.Taint, S.LLMCall, S.Label)


def _all_subclasses(cls):
    out = set()
    for c in cls.__subclasses__():
        out |= {c} | _all_subclasses(c)
    return out


def _models_in(tp) -> set[type]:
    found = set()
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        found.add(tp)
    for a in typing.get_args(tp):
        found |= _models_in(a)
    return found


def test_expected_classes_are_agent_views():
    names = {c.__name__ for c in _all_subclasses(S.AgentView)}
    assert EXPECTED_VIEWS <= names


def test_no_truth_fields_in_any_agent_view():
    for cls in _all_subclasses(S.AgentView):
        bad = set(cls.model_fields) & S.TRUTH_FIELDS
        assert not bad, f"{cls.__name__} exposes truth fields {bad}"
        for name, f in cls.model_fields.items():
            for m in _models_in(f.annotation):
                assert issubclass(m, S.AgentView), f"{cls.__name__}.{name} embeds non-view model {m.__name__}"


def test_truth_fields_list_is_meaningful():
    assert {"forged", "true_sender"} <= set(S.Message.model_fields)
    assert {"tampered", "original_text", "taint"} <= set(S.MemoryEntry.model_fields)
    assert {"copy_id", "origin"} <= set(S.KeyCopy.model_fields)
    for m in TRUTH_MODELS:
        assert not issubclass(m, S.AgentView)


def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(_keys(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


def test_prompt_view_from_tampered_state_has_no_truth(fixed_state):
    """Runtime spot-check on the WP0 path: MemoryStore.views -> context items -> PromptView."""
    mem = MemoryStore().views(fixed_state, "amber_fox", "scratchpad", upto_round=2, n=None)
    pinned = S.PinnedFacts(agent="amber_fox", round=2, max_rounds=5, needs_open=[3, 4], needs_burned=[],
                           holdings=[S.KeyView(info_id=k.info_id, key=k.key)
                                     for k in fixed_state.agents["amber_fox"].keys],
                           can_give=[1, 2], score=0.0, peers=list(fixed_state.order))
    forged = fixed_state.messages[1]
    obs = S.Observation(agent="amber_fox", round=2, turn_index=8, pinned=pinned,
                        inbox=[S.AgentMessageView(msg_id=forged.msg_id, channel="dm", sender=forged.sender_shown,
                                                  receiver=forged.receiver, text=forged.text)])
    bundle = PassthroughPolicy().build(agent="amber_fox", round=2, ledger=[],
                                       current=items_from_observation(obs), memory=mem, pinned=pinned,
                                       budget=1000, counter=WordCounter())
    view = bundle.to_view()
    blob = view.model_dump_json()
    assert TAMPERED_NOTE in blob and ORIGINAL_NOTE not in blob
    assert not any(i in blob for i in INTERVENTION_IDS)
    assert not (_keys(view.model_dump()) & S.TRUTH_FIELDS)
    assert "GOD" not in blob
    assert memory_items("amber_fox", mem)[0].ref == "amber_fox:scratchpad:1"
