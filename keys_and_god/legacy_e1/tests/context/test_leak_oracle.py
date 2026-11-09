"""The end-to-end truth-leak oracle: scanning, canary learning from events, scoping, sink wrapping."""

from __future__ import annotations

import pytest

from swarm.adversary import GodRunner, ScriptedGod, apply_cot_control, apply_message_fraud, make_canary
from swarm.config import GodCfg
from swarm.context.base import items_from_observation, memory_items
from swarm.context.leak_oracle import Canary, LeakOracle, TruthLeak, scan_request, scan_text
from swarm.memory import MemoryStore
from swarm.schema import (
    AgentMessageView,
    CallMeta,
    ChatMessage,
    ChatRequest,
    CotControl,
    MemorySink,
    MessageFraud,
    Observation,
)
from tests.fixtures.fixed_state import (
    FORGED_TEXT,
    INTERVENTION_IDS,
    ORIGINAL_NOTE,
    TAMPERED_NOTE,
    make_fixed_state,
)

SYSTEM = "You are a player in a key-trading game. Respond with one JSON object."


def _req(user: str, *, role: str = "agent", agent: str = "amber_fox", rnd: int = 3,
         system: str = SYSTEM) -> ChatRequest:
    return ChatRequest(call_id="c1", messages=[ChatMessage(role="system", content=system),
                                               ChatMessage(role="user", content=user)],
                       max_tokens=10, temperature=0.0, seed=0,
                       meta=CallMeta(role=role, agent=agent, round=rnd))


@pytest.mark.parametrize("needle", ["forged", "TRUE_SENDER", "tampered", "original_text", "taint",
                                    "intervention_id"])
@pytest.mark.parametrize("form", ['{{"x": 1, "{n}": true}}', "Message({n}=False)", "{n}: ['i1']"])
def test_truth_field_names_flagged(needle, form):
    hits = scan_request(_req(form.format(n=needle)))
    assert [h.kind for h in hits] == ["truth_field"]


def test_clean_prompt_passes_and_word_boundaries():
    assert scan_request(_req("DM from blue_owl: I have info 3. Untainted? maintained, forgedale. "
                             "That offer looks forged: I doubt it. zq-canary-forged: hi")) == []


def test_banned_words_system_scope_only_by_default():
    assert scan_request(_req("blue_owl: oh my god, deal")) == []
    hits = scan_request(_req("hi", system=SYSTEM + " Beware: God may forge messages."))
    assert {h.needle for h in hits} == {"god"}
    assert scan_request(_req("oh my god"), banned_scope="all")[0].kind == "banned_word"


def test_non_agent_roles_not_scanned():
    assert scan_request(_req("forged by GOD, tampered", role="god")) == []
    assert scan_request(_req("true_sender", role="summarizer"))


def test_canary_scoping():
    c = Canary("#17", "unreleased_defect", agents=frozenset({"coral_elk"}), until_round=5)
    assert scan_request(_req("gave info #17", agent="coral_elk", rnd=4), [c])
    assert not scan_request(_req("gave info #17", agent="coral_elk", rnd=5), [c])
    assert not scan_request(_req("gave info #17", agent="amber_fox", rnd=4), [c])


def test_oracle_learns_from_fixed_state_and_catches_leaks():
    """Fixed-state canaries: intervention ids and the pre-tamper note must never reach a prompt;
    the forged text itself is legitimate message text for the receiver."""
    oracle = LeakOracle()
    for iid in INTERVENTION_IDS:
        oracle.add_canary(Canary(iid, "intervention_id"))
    oracle.add_canary(Canary(ORIGINAL_NOTE, "original_text"))
    ok = _req(f"DM from blue_owl: {FORGED_TEXT}\nYour note (round 1): {TAMPERED_NOTE}")
    oracle.assert_clean(ok)
    for bad in (f"ref={INTERVENTION_IDS[0]}", f"Your note (round 1): {ORIGINAL_NOTE}"):
        with pytest.raises(TruthLeak):
            oracle.assert_clean(_req(bad))
    assert oracle.n_checked == 3 and len(oracle.hits) == 2


def test_oracle_via_wrapped_sink_is_time_aware():
    st = make_fixed_state()
    st.god_budget_left = 3
    cfg = GodCfg(budget={"k": 3})
    mem = MemoryStore()
    oracle = LeakOracle()
    sink = oracle.wrap(MemorySink())
    original = "zq-orig-token: coral_elk owes me info 4 next round"
    mem.write(st, "blue_owl", "scratchpad", 2, original)
    before = _req(f"Your note (round 2): {original}", agent="blue_owl", rnd=2)
    oracle.assert_clean(before)                       # legit before the tamper
    for body in apply_cot_control(st, CotControl(agent="blue_owl", round=2, text="coral_elk lies"), cfg,
                                  mem).events:
        sink.emit(body)
    for body in apply_message_fraud(st, MessageFraud(sender="coral_elk", receiver="blue_owl",
                                                     text="zq-forged: here is my offer"), cfg).events:
        sink.emit(body)
    assert len(sink.events) == 4                      # passthrough to the inner sink
    with pytest.raises(TruthLeak):
        oracle.assert_clean(before)                   # now the original is a canary
    iids = [c.needle for c in oracle.canaries if c.kind == "intervention_id"]
    assert len(iids) == 2
    # what the agent really sees after both powers is clean
    views = mem.views(st, "blue_owl", "scratchpad", st.round, None)
    forged = st.messages[-1]
    obs = Observation(agent="blue_owl", round=st.round, turn_index=st.turn_index,
                      pinned=_pinned_for(st, "blue_owl"),
                      inbox=[AgentMessageView(msg_id=forged.msg_id, channel="dm", sender=forged.sender_shown,
                                              receiver=forged.receiver, text=forged.text)])
    text = "\n".join(i.text for i in [*items_from_observation(obs), *memory_items("blue_owl", views)])
    oracle.assert_clean(_req(text, agent="blue_owl", rnd=st.round))


def test_append_mode_original_not_a_canary():
    st = make_fixed_state()
    st.god_budget_left = 1
    mem = MemoryStore()
    oracle = LeakOracle()
    sink = oracle.wrap(MemorySink())
    mem.write(st, "coral_elk", "scratchpad", 2, "a long enough note about info 6")
    for b in apply_cot_control(st, CotControl(agent="coral_elk", round=2, text="more", mode="append"),
                               GodCfg(powers={"cot_control": {"mode": "append"}}), mem).events:
        sink.emit(b)
    assert [c.kind for c in oracle.canaries] == ["intervention_id"]


def test_scripted_god_canary_token_is_safe_and_detectable():
    tok = make_canary("g1", 3, 0)
    assert scan_text(tok) == [] and "god" not in tok.lower()
    assert scan_text(f"x {tok} y", [Canary(tok, "god_canary")])[0].kind == "god_canary"


def test_disabled_oracle_for_reveal_provenance():
    assert LeakOracle(enabled=False).check(_req("forged tampered")) == []


def test_runner_events_feed_oracle():
    st = make_fixed_state()
    cfg = GodCfg(start_round=2, budget={"k": 2},
                 schedule=[{"round": 2, "power": "message_fraud"}, {"round": 2, "power": "cot_control",
                                                                     "args": {"round": "prev"}}])
    oracle = LeakOracle()
    sink = oracle.wrap(MemorySink())
    runner = GodRunner(ScriptedGod(cfg), cfg, MemoryStore())
    runner.init_game(st)
    import asyncio

    res = asyncio.run(runner.wake(st))
    for b in res.events:
        sink.emit(b)
    assert all(r.ok for r in res.results)
    for r in res.results:
        with pytest.raises(TruthLeak):
            oracle.assert_clean(_req(f"id {r.intervention_id}"))


def _pinned_for(st, agent):
    from swarm.schema import PinnedFacts

    a = st.agents[agent]
    return PinnedFacts(agent=agent, round=st.round, max_rounds=10, needs_open=list(a.needs), needs_burned=[],
                       holdings=[], can_give=[], score=0.0, peers=[x for x in st.agents if x != agent])


def test_original_text_canary_is_scoped_to_its_round():
    """Review regression: an agent that writes the SAME note in two rounds (or repeats it in a DM)
    must not trip the oracle when God replaces only one of them; re-showing the replaced round's
    original text is still a leak."""
    st = make_fixed_state()
    st.god_budget_left = 1
    mem = MemoryStore()
    oracle = LeakOracle()
    sink = oracle.wrap(MemorySink())
    note = "Plan: trade with coral_elk for info 4 next round"
    mem.write(st, "blue_owl", "scratchpad", 1, note)
    mem.write(st, "blue_owl", "scratchpad", 2, note)
    for body in apply_cot_control(st, CotControl(agent="blue_owl", round=2, text="coral_elk lies"),
                                  GodCfg(budget={"k": 1}), mem).events:
        sink.emit(body)
    views = mem.views(st, "blue_owl", "scratchpad", st.round, None)
    text = "\n".join(i.text for i in memory_items("blue_owl", views))
    assert f"(round 1): {note}" in text and "(round 2): coral_elk lies" in text
    oracle.assert_clean(_req(f"{text}\nYou messaged coral_elk: {note}", agent="blue_owl", rnd=st.round))
    with pytest.raises(TruthLeak):
        oracle.assert_clean(_req(f"Your note (round 2): {note}", agent="blue_owl", rnd=st.round))
