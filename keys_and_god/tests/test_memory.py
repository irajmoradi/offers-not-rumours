from __future__ import annotations

import pytest

from swarm.config import ReasoningCfg, ScratchpadCfg
from swarm.memory import MemoryStore
from swarm.schema import GameState, MemoryOverwriteData, MemoryWriteData, Snapshot, Taint
from tests.fixtures.fixed_state import COT_IID, ORIGINAL_NOTE, TAMPERED_NOTE


def _state() -> GameState:
    return GameState(game_id="g", round=3, order=["a", "b"], agents={})


def test_write_get_window():
    st, m = _state(), MemoryStore()
    for r in (1, 2, 3):
        e, body = m.write(st, "a", "scratchpad", r, f"note {r}")
        assert isinstance(body.data, MemoryWriteData) and body.actor == "a"
    m.write(st, "b", "scratchpad", 1, "other")
    assert m.get(st, "a", "scratchpad", 2).text == "note 2"
    assert [e.round for e in m.window(st, "a", "scratchpad", upto_round=3, n=2)] == [2, 3]
    assert [e.round for e in m.window(st, "a", "scratchpad", upto_round=2, n=None)] == [1, 2]
    assert m.window(st, "a", "scratchpad", upto_round=3, n=0) == []
    assert m.get(st, "a", "cot", 1) is None


def test_write_truncates():
    m = MemoryStore(ReasoningCfg(scratchpad=ScratchpadCfg(max_chars=5), max_cot_chars=3))
    st = _state()
    assert m.write(st, "a", "scratchpad", 1, "abcdefgh")[0].text == "abcde"
    assert m.write(st, "a", "cot", 1, "abcdefgh")[0].text == "abc"


def test_overwrite_replace_keeps_original_and_flags():
    st, m = _state(), MemoryStore()
    m.write(st, "a", "scratchpad", 2, "orig")
    e, body = m.overwrite(st, "a", "scratchpad", 2, "evil", intervention_id="i1", now_round=3)
    assert (e.text, e.tampered, e.original_text, e.tampered_by, e.tamper_round) == ("evil", True, "orig", "GOD", 3)
    assert e.taint == [Taint(intervention_id="i1", via="direct")]
    d = body.data
    assert isinstance(d, MemoryOverwriteData)
    assert (d.old_text, d.new_text, d.created, d.intervention_id, body.actor, body.phase) == \
        ("orig", "evil", False, "i1", "GOD", "god")
    # window returns the overwritten text; agent views hide flags
    assert m.window(st, "a", "scratchpad", 3, None)[0].text == "evil"
    v = m.views(st, "a", "scratchpad", 3, None)[0]
    assert v.model_dump() == {"kind": "scratchpad", "round": 2, "text": "evil"}
    assert len(st.memory) == 1


def test_overwrite_append_and_second_tamper():
    st, m = _state(), MemoryStore()
    m.write(st, "a", "scratchpad", 1, "orig")
    m.overwrite(st, "a", "scratchpad", 1, "x", intervention_id="i1", now_round=2, mode="append")
    assert m.get(st, "a", "scratchpad", 1).text == "orig\nx"
    e, _ = m.overwrite(st, "a", "scratchpad", 1, "y", intervention_id="i2", now_round=3)
    assert e.text == "y" and e.original_text == "orig"
    assert [t.intervention_id for t in e.taint] == ["i1", "i2"]


def test_overwrite_missing_round_creates():
    st, m = _state(), MemoryStore()
    e, body = m.overwrite(st, "a", "scratchpad", 3, "planted", intervention_id="i1", now_round=3)
    assert e.tampered and e.original_text is None and body.data.created and body.data.old_text is None
    # agent writing later in the same round appends, keeping the tamper flags
    e2, _ = m.write(st, "a", "scratchpad", 3, "my note")
    assert e2.text == "planted\nmy note" and e2.tampered and e2.taint == e.taint


def test_overwrite_future_round_rejected():
    with pytest.raises(ValueError):
        MemoryStore().overwrite(_state(), "a", "scratchpad", 5, "x", intervention_id="i", now_round=3)


def test_memory_survives_snapshot_roundtrip(fixed_state):
    e = MemoryStore().get(fixed_state, "amber_fox", "scratchpad", 1)
    assert e.text == TAMPERED_NOTE and e.original_text == ORIGINAL_NOTE and e.taint[0].intervention_id == COT_IID
    snap = Snapshot(run_id="r", game_id="g", round=2, state=fixed_state)
    back = Snapshot.model_validate_json(snap.model_dump_json())
    assert back.state.memory == fixed_state.memory
    assert MemoryStore().get(back.state, "amber_fox", "scratchpad", 1) == e


def test_deep_copy_isolates():
    st, m = _state(), MemoryStore()
    m.write(st, "a", "scratchpad", 1, "x")
    cp = st.model_copy(deep=True)
    m.overwrite(cp, "a", "scratchpad", 1, "y", intervention_id="i", now_round=3)
    assert m.get(st, "a", "scratchpad", 1).text == "x"
    assert m.get(cp, "a", "scratchpad", 1).text == "y"
