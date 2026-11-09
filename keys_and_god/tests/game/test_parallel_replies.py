"""game.trade.parallel_replies: concurrent reply calls, settled in priority order (simultaneous mode)."""

from __future__ import annotations

import asyncio
import json

from swarm.analysis.invariants import check_run
from swarm.analysis.trades import summarize_run
from swarm.game import trade
from swarm.game.engine import play_game
from swarm.schema import GiveKey, SendMessage, TradeOffer
from tests.game.helpers import FOUR, fresh_game, run_cfg
from tests.game.test_trade import Trader
from tests.runner.conftest import FAKE_OVERRIDES, _run

# ann (1,2 | needs 3,5)  bob (3,4 | 1,6)  cat (5,6 | 2,4)  dan (7,8 | 1,3); priority order ann, bob, cat, dan
NO_CONFLICT = {(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3]), SendMessage(to="cat", text="hi")],
               (1, "cat"): [TradeOffer(target="dan", give=[6], want=[7])],
               (2, "bob"): [TradeOffer(target="cat", give=[4], want=[5])],
               (2, "dan"): [TradeOffer(target="ann", give=[6], want=[2])]}


def _cfg(parallel: bool, rounds: int = 2):
    return run_cfg(game={"turn_mode": "simultaneous", "max_rounds": rounds,
                         "trade": {"enabled": True, "parallel_replies": parallel}})


def _holdings(eng):
    return {a: sorted(k.info_id for k in s.keys) for a, s in eng.state.agents.items()}


def _bodies(eng, *types):
    return sorted(json.dumps(e.data.model_dump(mode="json"), sort_keys=True)
                  for e in eng.sink.events if e.type in types)


def test_default_off_and_hash_neutral():
    assert run_cfg().game.trade.parallel_replies is False
    off = _cfg(False).config_hash()
    assert off == run_cfg(game={"turn_mode": "simultaneous", "max_rounds": 2,
                                "trade": {"enabled": True}}).config_hash()
    assert _cfg(True).config_hash() != off


def test_no_conflict_same_outcome_as_serial():
    out_s, eng_s = play_game(_cfg(False), fresh_game(FOUR), Trader(NO_CONFLICT, accept={"bob", "dan", "cat", "ann"}))
    out_p, eng_p = play_game(_cfg(True), fresh_game(FOUR), Trader(NO_CONFLICT, accept={"bob", "dan", "cat", "ann"}))
    assert _holdings(eng_s) == _holdings(eng_p)
    assert out_s.model_dump(mode="json") == out_p.model_dump(mode="json")
    # same events (as a multiset); only their order in the stream differs
    for t in ("action_result", "trade_offer", "trade_reply", "trade_executed", "key_transfer", "message"):
        assert _bodies(eng_s, t) == _bodies(eng_p, t), t
    # trade-phase events come after every action of their round, stamped at its last resolved slot
    for r in (1, 2):
        acts = [e.turn_index for e in eng_p.sink.events if e.type == "action_result" and e.round == r]
        reps = [e.turn_index for e in eng_p.sink.events if e.type == "trade_reply" and e.round == r]
        assert reps and len(set(reps)) == 1 and reps[0] >= max(acts)
    assert len([e for e in eng_p.sink.events if e.type == "trade_executed"]) == 4


def test_replies_run_concurrently():
    state = {"cur": 0, "max": 0}

    class Async(Trader):
        async def reply_trade(self, ask):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1
            self.asks.append(ask)
            return trade.TradeAnswer(accept=False)

    script = {(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])],
              (1, "cat"): [TradeOffer(target="dan", give=[6], want=[7])],
              (1, "dan"): [TradeOffer(target="ann", give=[8], want=[2])]}
    play_game(_cfg(True, 1), fresh_game(FOUR), Async(script))
    assert state["max"] == 3
    state["max"] = 0
    play_game(_cfg(False, 1), fresh_game(FOUR), Async(script))
    assert state["max"] == 1


def test_later_action_resolves_before_settlement_deterministically():
    """ann (first) trades 1<->3 with bob; bob's own s0 actions give 3 and 1. Serial: the trade settled
    first (3 gone: no_such_key; 1 received but round-locked: key_locked). Parallel: bob's actions resolve
    before the settlement (3 pending-locked: key_locked; 1 not yet held: no_such_key). Both fail either way
    and the trade still executes; the parallel result is the same on every run."""
    script = {(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])],
              (1, "bob"): [GiveKey(to="dan", info_id=3), GiveKey(to="cat", info_id=1)]}

    def reasons(parallel):
        _, eng = play_game(_cfg(parallel, 1), fresh_game(FOUR), Trader(script, accept={"bob"}))
        return ([e.data.reason for e in eng.sink.events if e.type == "action_result" and e.data.agent == "bob"],
                _holdings(eng))

    serial, par = reasons(False), reasons(True)
    assert serial[0] == ["no_such_key", "key_locked"]
    assert par[0] == ["key_locked", "no_such_key"]
    assert serial[1] == par[1] and par[1]["ann"] == [2, 3] and par[1]["bob"] == [1, 4]
    assert reasons(True) == par


def test_reply_prompts_use_trade_phase_start_state():
    """bob gets two offers in one round (different keys). Serial: the second ask shows bob's holdings after
    the first trade; parallel: both asks show the pre-trade holdings. Settlement is unaffected (the keys
    are disjoint and locked), so the final holdings match."""
    script = {(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])],
              (1, "cat"): [TradeOffer(target="bob", give=[6], want=[4])]}
    ps, pp = Trader(script, accept={"bob"}), Trader(script, accept={"bob"})
    _, eng_s = play_game(_cfg(False, 1), fresh_game(FOUR), ps)
    _, eng_p = play_game(_cfg(True, 1), fresh_game(FOUR), pp)
    held = lambda ask: sorted(i for i, _ in ask.holdings)                       # noqa: E731
    assert [held(a) for a in ps.asks] == [[3, 4], [1, 4]]
    assert [held(a) for a in pp.asks] == [[3, 4], [3, 4]]
    assert _holdings(eng_s) == _holdings(eng_p) and _holdings(eng_p)["bob"] == [1, 6]


def test_sequential_mode_unaffected():
    seq = {"turn_mode": "sequential", "max_rounds": 2}
    a = run_cfg(game={**seq, "trade": {"enabled": True}})
    b = run_cfg(game={**seq, "trade": {"enabled": True, "parallel_replies": True}})
    _, ea = play_game(a, fresh_game(FOUR), Trader(NO_CONFLICT, accept={"bob", "dan", "cat", "ann"}))
    _, eb = play_game(b, fresh_game(FOUR), Trader(NO_CONFLICT, accept={"bob", "dan", "cat", "ann"}))
    strip = lambda eng: [(e.type, e.turn_index, json.dumps(e.data.model_dump(mode="json"), sort_keys=True))  # noqa: E731
                         for e in eng.sink.events if e.type != "game_start"]
    assert strip(ea) == strip(eb)


def test_fake_e2e_parallel_trade_run_holds_invariants(tmp_path):
    ov = [*FAKE_OVERRIDES, "game.trade.enabled=true", "game.turn_mode=simultaneous",
          "game.trade.parallel_replies=true"]
    run = _run(tmp_path, "trade_par_e2e", ov)
    tot = summarize_run(run)["total"]
    assert tot["offers"] > 0 and tot["replies"] == tot["offers"]
    assert check_run(run) == []
