"""WP13-lite: full_truncate / window budget, pinned, current inbox, monotonicity."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.config import ContextCfg
from swarm.context.base import ApproxCounter, ContextOverflow, WordCounter
from swarm.context.policies import (
    FullTruncatePolicy,
    WindowPolicy,
    effective_limit,
    make_policy,
    pinned_tokens,
)
from swarm.context.tokens import make_counter
from swarm.schema import HistoryItem, KeyView, MemoryView, PinnedFacts

W = WordCounter()
KINDS = ["inbox_msg", "board_msg", "log_line", "notice", "own_action"]


def _pinned(rnd: int = 5) -> PinnedFacts:
    return PinnedFacts(agent="amber_fox", round=rnd, max_rounds=40, needs_open=[3, 4], needs_burned=[9],
                       holdings=[KeyView(info_id=1, key=1111), KeyView(info_id=2, key=2222)], can_give=[1, 2],
                       score=0.33, peers=["blue_owl", "coral_elk"])


def _item(rnd: int, kind: str, n: int, words: int) -> HistoryItem:
    return HistoryItem(item_id=f"amber_fox:{rnd}:{kind}:{n}", agent="amber_fox", round=rnd, kind=kind,
                       text=" ".join(f"w{rnd}{n}{j}" for j in range(words)))


ledgers = st.lists(st.tuples(st.integers(1, 9), st.sampled_from(KINDS), st.integers(1, 60)), max_size=60)


def _ledger(spec) -> list[HistoryItem]:
    spec = sorted(spec, key=lambda t: t[0])
    return [_item(r, k, n, w) for n, (r, k, w) in enumerate(spec)]


def _current(n_inbox=2, n_log=3, words=5) -> list[HistoryItem]:
    return ([_item(10, "inbox_msg", i, words) for i in range(n_inbox)]
            + [_item(10, "log_line", i, words) for i in range(n_log)])


MEM = [MemoryView(kind="scratchpad", round=r, text=f"note round {r} " * 3) for r in (2, 6, 9)]
POLICIES = [FullTruncatePolicy(), WindowPolicy(window_rounds=3, per_kind_caps={"log_line": 4})]


def _build(p, ledger, budget, current=None, memory=MEM, counter=W):
    return p.build(agent="amber_fox", round=10, ledger=ledger, current=_current() if current is None else current,
                   memory=memory, pinned=_pinned(), budget=budget, counter=counter)


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.name)
@given(spec=ledgers, budget=st.integers(300, 16000))
def test_budget_never_exceeded(policy, spec, budget):
    ledger = _ledger(spec)
    b = _build(policy, ledger, budget)
    assert b.tokens_total <= budget * (1 - policy.safety_margin)
    assert b.tokens_total == sum(s.tokens for s in b.sections)
    included = {i for s in b.sections for i in s.item_ids}
    assert included.isdisjoint(b.dropped_item_ids)
    assert included | set(b.dropped_item_ids) >= {i.item_id for i in ledger}


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.name)
@given(spec=ledgers, b1=st.integers(300, 8000), extra=st.integers(0, 8000))
def test_window_monotone(policy, spec, b1, extra):
    ledger = _ledger(spec)
    a = set(_build(policy, ledger, b1).items)
    b = set(_build(policy, ledger, b1 + extra).items)
    assert a <= b


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.name)
def test_pinned_and_current_inbox_never_dropped(policy):
    ledger = _ledger([(r, "board_msg", 200) for r in range(1, 10)])
    cur = _current(n_inbox=3, n_log=40, words=20)
    p_tok = pinned_tokens(_pinned(), W)
    inbox_tok = 3 * 20
    budget = int((p_tok + inbox_tok + 25) / 0.95) + 1     # room for inbox plus ~1 log line
    b = _build(policy, ledger, budget, current=cur)
    assert b.pinned == _pinned()                           # holdings, needs, burned, score all kept
    cur_ids = b.sections[2].item_ids
    assert all(f"amber_fox:10:inbox_msg:{i}" in cur_ids for i in range(3))
    assert any("log_line" in i for i in b.dropped_item_ids)  # current log lines dropped first
    assert b.tokens_total <= budget * 0.95
    assert b.sections[1].item_ids == []                    # no history room left


def test_overflow_when_pinned_plus_inbox_too_big():
    cur = [_item(10, "inbox_msg", i, 400) for i in range(3)]
    with pytest.raises(ContextOverflow):
        _build(FullTruncatePolicy(), [], 300, current=cur)


def test_full_truncate_drops_oldest_first_and_keeps_order():
    ledger = _ledger([(r, "board_msg", 50) for r in range(1, 10)])
    b = _build(FullTruncatePolicy(), ledger, 300, memory=[])
    kept = b.sections[1].item_ids
    rounds = [b.items[i].round for i in kept]
    assert rounds == sorted(rounds) and rounds[-1] == 9 and rounds[0] > 1
    assert b.dropped_item_ids[0] == ledger[0].item_id


def test_full_truncate_keeps_everything_when_it_fits():
    ledger = _ledger([(r, k, 5) for r in range(1, 10) for k in KINDS])
    b = _build(FullTruncatePolicy(), ledger, 14000)
    assert b.dropped_item_ids == []
    assert len(b.sections[1].item_ids) == len(ledger) + len(MEM)
    # memory notes come after the ledger items of their round
    ids = b.sections[1].item_ids
    assert ids.index("amber_fox:6:scratchpad:0") > ids.index("amber_fox:6:own_action:29")


def test_window_rounds_and_caps():
    ledger = _ledger([(r, "log_line", 3) for r in range(1, 10) for _ in range(3)])
    b = _build(WindowPolicy(window_rounds=3, per_kind_caps={"log_line": 4}), ledger, 14000)
    hist = [b.items[i] for i in b.sections[1].item_ids]
    logs = [h for h in hist if h.kind == "log_line"]
    assert len(logs) == 4 and all(h.round >= 8 for h in logs)
    assert {h.round for h in hist if h.kind == "scratchpad"} == {2, 6, 9}   # notes not round-windowed


def test_pure_and_deterministic():
    ledger = _ledger([(r, "board_msg", 30) for r in range(1, 10)])
    snapshot = [i.model_copy() for i in ledger]
    for p in POLICIES:
        assert _build(p, ledger, 900) == _build(p, ledger, 900)
    assert ledger == snapshot


def test_make_policy_and_counters():
    assert make_policy(ContextCfg()).name == "full_truncate"
    w = make_policy(ContextCfg(policy="window", window_rounds=2, per_kind_caps={"cot": 0}))
    assert isinstance(w, WindowPolicy) and w.window_rounds == 2
    assert w.policy_hash != make_policy(ContextCfg(policy="window")).policy_hash
    assert isinstance(make_counter("hf:Qwen/Qwen3-8B"), ApproxCounter)
    assert make_counter("words").count("a b c") == 3
    assert effective_limit(14000, 0.05, 1000) == 12300
    ledger = _ledger([(r, "board_msg", 400) for r in range(1, 10)])
    b = make_policy(ContextCfg(), reserve_tokens=1000).build(
        agent="amber_fox", round=10, ledger=ledger, current=_current(), memory=MEM, pinned=_pinned(),
        budget=14000, counter=ApproxCounter())
    assert b.tokens_total <= 12300
