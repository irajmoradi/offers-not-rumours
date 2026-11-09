from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from swarm.schema import (
    ACTION_ADAPTER,
    GOD_ACTION_ADAPTER,
    ActionResultData,
    AgentOutcome,
    AgentOutput,
    BurnData,
    CallMeta,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompactionData,
    CompactionRecord,
    ContextBuiltData,
    CotControl,
    DefectAction,
    Event,
    EventBody,
    GameEndData,
    GameStartData,
    GiveKey,
    GodPowerData,
    GodToolData,
    KeyTransferData,
    Label,
    LLMCall,
    MemorySink,
    MemoryWriteData,
    MessageData,
    MessageFraud,
    MonitorScore,
    Noop,
    NoticeData,
    Observation,
    Outcome,
    PinnedFacts,
    RoundEndData,
    SendMessage,
    Snapshot,
    TurnData,
    Usage,
    WinData,
    defect_release_turn,
    log_visible,
    make_event,
    read_jsonl,
    release_visible,
    turn_index,
    write_jsonl,
)


def _pinned() -> PinnedFacts:
    return PinnedFacts(agent="a", round=2, max_rounds=5, needs_open=[3], needs_burned=[], holdings=[],
                       can_give=[], score=0.0, peers=["b"])


def _all_payloads(fixed_state):
    m_forged = fixed_state.messages[1]
    entry = fixed_state.memory[0]
    obs = Observation(agent="a", round=2, turn_index=4, pinned=_pinned())
    outcome = Outcome(game_id="g", rounds_played=5, end_reason="max_rounds",
                      per_agent={"a": AgentOutcome(score=1.0, won_round=3)}, welfare=1.0, win_rate=1.0)
    rec = CompactionRecord(compaction_id="c1", agent="a", round=4, policy="rolling_summary",
                           summarizer="extractive", covers_rounds=(1, 3), source_item_ids=["a:1:log_line:0"],
                           summary_entry_id="a:summary:4", summary_text="s", tokens_before=10, tokens_after=2)
    return [
        GameStartData(config_hash="h", n_agents=4, max_rounds=5, order=["a"], agents=fixed_state.agents),
        TurnData(agent="a", obs=obs, obs_hash="x", actions=[SendMessage(to="b", text="hi"), Noop()]),
        ActionResultData(agent="a", action=GiveKey(to="b", info_id=3), ok=False, reason="no_such_key"),
        MessageData(message=m_forged),
        KeyTransferData(copy_id="3-0", info_id=3, from_agent="a", to_agent="b", cause="give"),
        BurnData(actor="a", target="b", info_id=3, correct=True, penalty="none", gain="mint", release_turn=9),
        MemoryWriteData(entry=entry),
        GodToolData(tool="message_check", args={"sender": "a"}, result_digest="d", result_size=3),
        GodPowerData(power="message_fraud", intervention_id="i1", cost=1, budget_left=2, ok=True,
                     action=MessageFraud(sender="a", receiver="b", text="t")),
        NoticeData(notice=fixed_state.notices[0]),
        ContextBuiltData(agent="a", policy="full_truncate", tokens_by_section={"window": 3}, tokens_total=3,
                         budget=100, included_item_ids=["x"]),
        CompactionData(record=rec),
        WinData(agent="a", score=1.0),
        RoundEndData(state_hash=fixed_state.state_hash(), scores={"a": 0.5}, welfare=0.5),
        GameEndData(outcome=outcome),
    ]


def test_every_event_type_roundtrips_jsonl(tmp_path, fixed_state):
    sink = MemorySink(run_id="r", game_id="g")
    for d in _all_payloads(fixed_state):
        sink.emit(EventBody(round=2, turn_index=4, phase="agents", actor="a", data=d))
    p = tmp_path / "events.jsonl"
    assert write_jsonl(p, sink.events) == len(sink.events)
    back = list(read_jsonl(p, Event))
    assert back == sink.events
    assert [e.seq for e in back] == list(range(len(back)))
    assert {e.type for e in back} >= {"turn", "burn", "memory_write", "game_end"}


def test_event_type_must_match_payload():
    with pytest.raises(ValidationError):
        Event(run_id="r", game_id="g", seq=0, round=1, phase="end", type="win", actor="SYSTEM",
              data=RoundEndData(state_hash="h", scores={}, welfare=0.0))


def test_make_event_stamps():
    ev = make_event(EventBody(round=1, phase="end", actor="SYSTEM", data=WinData(agent="a", score=1)),
                    run_id="r", game_id="g", seq=5)
    assert (ev.type, ev.seq, ev.v) == ("win", 5, 1)


def test_records_frozen_state_mutable(fixed_state):
    with pytest.raises(ValidationError):
        fixed_state.messages[0].text = "x"  # type: ignore[misc]
    fixed_state.agents["amber_fox"].needs.append(9)   # state is mutable
    fixed_state.round = 3
    assert fixed_state.round == 3


def test_extra_forbid():
    with pytest.raises(ValidationError):
        SendMessage(to="b", text="x", bogus=1)  # type: ignore[call-arg]


def test_action_union_parses_by_kind():
    a = ACTION_ADAPTER.validate_python({"kind": "defect", "target": "b", "info_id": "7"})
    assert isinstance(a, DefectAction) and a.info_id == 7
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"kind": "fly"})
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"kind": "grep_log", "pattern": "x" * 81})
    g = GOD_ACTION_ADAPTER.validate_python({"kind": "cot_control", "agent": "a", "round": 2, "text": "t"})
    assert isinstance(g, CotControl) and g.target == "scratchpad" and g.mode == "replace"


def test_other_records_roundtrip(tmp_path, fixed_state):
    req = ChatRequest(call_id="c1", messages=[ChatMessage(role="user", content="hi")], max_tokens=10,
                      temperature=0.0, seed=1, meta=CallMeta(role="agent", agent="a"))
    recs = [
        LLMCall(run_id="r", backend="fake", model="fake", request=req,
                response=ChatResponse(call_id="c1", text="ok", usage=Usage(prompt_tokens=1))),
        Snapshot(run_id="r", game_id="g", round=2, state=fixed_state),
        Label(game_id="g", label_kind="forged_message", unit_id="m", agent="a", round=2, positive=True),
        MonitorScore(monitor="random", game_id="g", unit_kind="msg", unit_id="m", agent="a", round=2, score=.5),
        AgentOutput(cot="c", note="n", actions=[Noop()], llm_call_ids=["c1"]),
    ]
    for r in recs:
        p = tmp_path / f"{type(r).__name__}.jsonl"
        write_jsonl(p, [r])
        assert list(read_jsonl(p, type(r))) == [r]
    snap = list(read_jsonl(tmp_path / "Snapshot.jsonl", Snapshot))[0]
    assert snap.state.state_hash() == fixed_state.state_hash()


def test_request_sampling_excludes_meta():
    req = ChatRequest(call_id="c", messages=[], max_tokens=5, temperature=0.1, seed=3,
                      meta=CallMeta(role="god"))
    assert "meta" not in req.sampling() and "call_id" not in req.sampling()
    assert req.sampling()["max_tokens"] == 5


# ---------------------------------------------------------------- timing contract
def test_simultaneous_release_is_d_plus_1_plus_delay():
    n, delay, d = 6, 3, 2
    u = turn_index(d, 4, n)
    rel = defect_release_turn(u, delay, n)
    obs_round = [r for r in range(1, 20) if turn_index(r, 0, n) >= rel][0]
    assert obs_round == d + 1 + delay


@given(n=st.integers(2, 12), delay=st.integers(0, 4), rnd=st.integers(1, 5), pos=st.data())
def test_each_viewer_sees_release_exactly_once(n, delay, rnd, pos):
    p = pos.draw(st.integers(0, n - 1))
    u = turn_index(rnd, p, n)
    rel = defect_release_turn(u, delay, n)
    for q in range(n):           # every viewer position, observing every round (sequential)
        seen = 0
        prev = -1
        for r in range(1, rnd + delay + 4):
            t = turn_index(r, q, n)
            seen += release_visible(rel, t, prev)
            prev = t
        assert seen == 1
    # delay 0 == normal log rule for every viewer
    if delay == 0:
        for q in range(n):
            for r in range(1, rnd + 3):
                t = turn_index(r, q, n)
                assert release_visible(rel, t, t - n) == log_visible(u, t, n)


def test_queued_release_for_skipped_viewer():
    n, u = 4, turn_index(1, 0, 4)
    rel = defect_release_turn(u, 1, n)        # = 5
    # viewer at pos 1 observes at t=1, then is delayed for two rounds, then observes at t=13
    assert not release_visible(rel, 1, -1)
    assert release_visible(rel, 13, 1)
    assert not release_visible(rel, 17, 13)
