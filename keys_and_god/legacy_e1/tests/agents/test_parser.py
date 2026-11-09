"""Golden cases for swarm.agents.parser.parse_reply."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.agents.parser import parse_reply
from swarm.llm.base import MALFORMED_REPLY, NOOP_REPLY
from swarm.schema import DefectAction, GiveKey, GrepLog, Noop, PostBoard, SendMessage

# (name, text, expected actions, expected ok, expected thought, expected note, error substrings)
GOLDEN = [
    ("clean",
     '{"thought": "t", "note": "n", "actions": [{"kind": "send_message", "to": "blue_owl", "text": "hi"}]}',
     [SendMessage(to="blue_owl", text="hi")], True, "t", "n", []),
    ("fenced",
     'Here you go:\n```json\n{"actions": [{"kind": "give_key", "to": "b", "info_id": 3}]}\n```',
     [GiveKey(to="b", info_id=3)], True, None, None, []),
    ("think_block",
     '<think>I should burn {x}</think>\n{"actions": [{"kind": "defect", "target": "c", "info_id": 7}]}',
     [DefectAction(target="c", info_id=7)], True, None, None, ["think_leak"]),
    ("unclosed_think_json_inside",
     '<think>\n{"actions": [{"kind": "noop"}]}',
     [Noop()], True, None, None, ["think_leak"]),
    ("string_info_id",
     '{"actions": [{"kind": "give_key", "to": "b", "info_id": "info 12"}]}',
     [GiveKey(to="b", info_id=12)], True, None, None, []),
    ("kind_and_field_aliases",
     '{"actions": [{"type": "message", "recipient": "@b", "message": "deal?"}, {"action": "burn", "player": "c", "info": 4}]}',
     [SendMessage(to="b", text="deal?"), DefectAction(target="c", info_id=4)], True, None, None, []),
    ("trailing_comma_python_literals",
     '{"thought": "x", "actions": [{"kind": "post_board", "text": "offering 5",},], "flag": True}',
     [PostBoard(text="offering 5")], True, "x", None, []),
    ("too_many_actions",
     '{"actions": [{"kind":"post_board","text":"a"},{"kind":"post_board","text":"b"},{"kind":"post_board","text":"c"}]}',
     [PostBoard(text="a"), PostBoard(text="b")], True, None, None, ["too_many_actions:1"]),
    ("bare_list",
     '[{"kind": "noop"}]', [Noop()], True, None, None, []),
    ("empty_actions_is_noop",
     '{"thought": "wait", "actions": []}', [Noop()], True, "wait", None, []),
    ("noop_dropped_when_real_action",
     '{"actions": [{"kind": "noop"}, {"kind": "give_key", "to": "b", "info_id": 1}]}',
     [GiveKey(to="b", info_id=1)], True, None, None, []),
    ("one_invalid_one_valid",
     '{"actions": [{"kind": "give_key", "to": "b"}, {"kind": "post_board", "text": "hi"}]}',
     [PostBoard(text="hi")], True, None, None, ["invalid_action:0:give_key:info_id"]),
    ("unknown_kind_only",
     '{"actions": [{"kind": "fly", "to": "moon"}]}',
     [Noop()], False, None, None, ["unknown_kind:0:fly", "all_actions_invalid"]),
    ("no_json",
     "I will give my key to blue_owl.", [Noop()], False, None, None, ["no_json"]),
    ("truncated_json",
     MALFORMED_REPLY, [Noop()], False, None, None, ["bad_json"]),
    ("no_actions_key",
     '{"thought": "hmm"}', [Noop()], False, "hmm", None, ["no_actions_key"]),
    ("single_action_object",
     '{"action": {"kind": "give_key", "to": "b", "info_id": 2}}',
     [GiveKey(to="b", info_id=2)], True, None, None, []),
    ("duplicate_grep",
     '{"actions": [{"kind": "grep_log", "pattern": "b"}, {"kind": "grep_log", "pattern": "c"}]}',
     [GrepLog(pattern="b")], True, None, None, ["duplicate_grep:1"]),
    ("braces_inside_strings",
     '{"actions": [{"kind": "send_message", "to": "b", "text": "my key is {info: 3}"}]}',
     [SendMessage(to="b", text="my key is {info: 3}")], True, None, None, []),
    ("prose_then_two_objects_prefers_actions",
     'Example {"x": 1} and my move {"actions": [{"kind": "noop"}]}',
     [Noop()], True, None, None, []),
    ("fake_noop_reply", NOOP_REPLY, [Noop()], True, None, None, []),
    ("note_non_string",
     '{"note": {"plan": "trade"}, "actions": [{"kind": "noop"}]}',
     [Noop()], True, None, '{"plan": "trade"}', []),
]


@pytest.mark.parametrize("name,text,actions,ok,thought,note,errs", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden(name, text, actions, ok, thought, note, errs):
    r = parse_reply(text, actions_per_turn=2)
    assert r.actions == actions
    assert r.ok is ok
    assert r.thought == thought
    assert r.note == note
    for e in errs:
        assert e in r.errors, (e, r.errors)


def test_grep_free_does_not_use_slot():
    t = ('{"actions": [{"kind": "grep_log", "pattern": "b"}, {"kind": "post_board", "text": "a"}, '
         '{"kind": "post_board", "text": "b"}]}')
    assert len(parse_reply(t, actions_per_turn=2, grep_free=True).actions) == 3
    assert len(parse_reply(t, actions_per_turn=2, grep_free=False).actions) == 2


def test_scratchpad_off_ignores_note_and_thought_flag():
    r = parse_reply('{"thought": "t", "note": "n", "actions": []}', scratchpad=False, want_thought=False)
    assert r.note is None and r.thought is None


def test_text_truncated():
    r = parse_reply('{"actions": [{"kind": "post_board", "text": "' + "x" * 50 + '"}]}', max_text_chars=10)
    assert r.actions == [PostBoard(text="x" * 10)]


@given(st.text(max_size=300))
def test_never_raises_and_never_empty(text):
    r = parse_reply(text)
    assert r.actions
    assert r.ok or r.actions == [Noop()]


# ------------------------------------------------------------------ salvage (reasoning.salvage_truncated)
CUT_AFTER_ONE_ACTION = ('{"thought": "need 3", "note": "owl", "actions": [{"kind": "give_key", "to": "blue_owl", '
                        '"info_id": 1}, {"kind": "send_message", "to": "blue_owl", "text": "sent you 1, could y')
CUT_IN_THOUGHT = '{"thought": "I need keys for 3 and 4. Blue owl said earlier that it might have'


def test_salvage_keeps_complete_actions_of_a_cut_reply():
    from swarm.agents.parser import parse_reply
    from swarm.schema import GiveKey

    off = parse_reply(CUT_AFTER_ONE_ACTION)
    assert not off.ok and "bad_json" in off.errors                 # default: unchanged behaviour
    on = parse_reply(CUT_AFTER_ONE_ACTION, salvage=True)
    assert on.ok and on.actions == [GiveKey(to="blue_owl", info_id=1)]
    assert on.thought == "need 3" and on.note == "owl" and "salvaged_truncated" in on.errors
    assert "bad_json" not in on.errors


def test_salvage_nothing_when_cut_inside_thought_or_no_actions():
    from swarm.agents.parser import parse_reply

    for text in (CUT_IN_THOUGHT, '{"thought": "x", "actions": [{"kind": "give_key", "to": "bl'):
        r = parse_reply(text, salvage=True)
        assert not r.ok and "salvaged_truncated" not in r.errors


def test_salvage_does_not_touch_valid_replies():
    from swarm.agents.parser import parse_reply

    ok = '{"thought": "t", "actions": [{"kind": "noop"}]}'
    assert parse_reply(ok, salvage=True).errors == parse_reply(ok).errors == []
