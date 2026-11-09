"""game.trade (plan/TRADE_DESIGN.md): defaults byte-identical to 9a14abe (config hashes, system prompts,
event streams), accept / reject / stale holdings, simultaneous-mode locks, the per-turn limit,
contacts-only, lock_and_stay winners, the parser gate, LLMAgent.reply_trade, the fake policy, the
metrics helper and a FakeLLM end-to-end run that holds every invariant."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from swarm.agents.llm_agent import LLMAgent, system_context
from swarm.agents.parser import parse_reply
from swarm.analysis.invariants import check_run
from swarm.analysis.trades import summarize, summarize_run
from swarm.config import REPO_ROOT
from swarm.game import trade
from swarm.game.engine import play_game
from swarm.game.resolve import resolve_turn
from swarm.llm.base import FakeLLM
from swarm.prompts import load_prompt_set
from swarm.rng import canonical_json
from swarm.runner.fake_policy import smart_fake_policy
from swarm.runner.run import load_experiment
from swarm.schema import CallMeta, ChatMessage, ChatRequest, GiveKey, PostBoard, SendMessage, TradeOffer
from tests.game.helpers import FOUR, Scripted, fresh_game, mk_state, run_cfg
from tests.runner.conftest import FAKE_OVERRIDES, _run

# config_hash + sha256(system prompt) at 9a14abe (before game.trade)
BEFORE = {
    "g1_v2_costly_sim": ("7621efabf3e206e5ba64da902b9dc02ebc844f2ae769c56e5a3f7d92174b5b63",
                         "3e126a2c82abfb2c8ef26e919a71502f709a916808f0d20055d9481d65c45618"),
    "grid_oss_sim_k2": ("15f60423be9731158c2ea9e95947800474162f34990f778fd201af584334eb0f",
                        "8a9b6ee4477ef8eb4926a06efadb7c1ca747c5c0f9c03447d29ddb2c0bd8da25"),
    "net40_er3_contacts_sim": ("407874f17fb83cdf21fb41c4ca7013f9cfaa1699b3d1998f6a56898cb0642d9a",
                               "92f7e7ad5deeff10a5745b627e2aa89b45c6cd96a1d75f5c2f3ca7bd42901513"),
    "base": ("b442ad7af4a87da8bb2ff289c1e794cd7b98f44742c8a7066966921cfad35ee1",
             "e42f9f8ff387fc965a55aa74ff02b2b9911b8d3f5d37e52e18fcd288812475a7"),
}
# sha256 of the scripted game's event stream (MemorySink, canonical JSON) at 9a14abe
EVENTS_BEFORE = {"sequential": "cb113db853b77bcd499512f20554e1c6d2340c4fd2fb01cd920358575c03bebb",
                 "simultaneous": "0380403604a0efd640ed1360235d0706fb93a64a4b6d4db251244a1b0b6ca7b8"}
SCRIPT = {(1, "ann"): [GiveKey(to="bob", info_id=1), SendMessage(to="cat", text="hi")],
          (1, "bob"): [GiveKey(to="ann", info_id=3), PostBoard(text="need 6")],
          (1, "cat"): [GiveKey(to="ann", info_id=5), GiveKey(to="bob", info_id=6)],
          (2, "cat"): [GiveKey(to="dan", info_id=2)]}


def _sys(cfg) -> str:
    return load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))


def _cfg(mode="sequential", **game):
    return run_cfg(game={"turn_mode": mode, "trade": {"enabled": True}, **game})


def _types(eng, *types):
    return [e for e in eng.sink.events if e.type in types]


class Trader(Scripted):
    """Scripted actions + a scripted trade reply (accept per target, default reject)."""

    def __init__(self, script, accept=(), on_reply=None):
        super().__init__(script)
        self.accept, self.asks, self.on_reply = set(accept), [], on_reply

    def reply_trade(self, ask):
        self.asks.append(ask)
        if self.on_reply is not None:
            self.on_reply(ask)
        return trade.TradeAnswer(accept=ask.target in self.accept, llm_call_ids=[f"fake-{ask.offer_id}"])


# ------------------------------------------------------------------ defaults: byte-identical
@pytest.mark.parametrize("name", sorted(BEFORE))
def test_existing_configs_keep_hash_and_prompt(name):
    cfg = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", [])[0]
    assert (cfg.config_hash(), hashlib.sha256(_sys(cfg).encode()).hexdigest()) == BEFORE[name]
    explicit = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml",
                               ["game.trade.enabled=false", "game.trade.max_offers_per_turn=1"])[0]
    assert explicit.config_hash() == BEFORE[name][0]
    on = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml", ["game.trade.enabled=true"])[0]
    assert on.config_hash() != BEFORE[name][0]
    assert "offer_trade" not in _sys(cfg) and "offer_trade" in _sys(on)


@pytest.mark.parametrize("mode", sorted(EVENTS_BEFORE))
def test_trade_off_event_stream_is_byte_identical(mode):
    _, eng = play_game(run_cfg(game={"turn_mode": mode}), fresh_game(FOUR), Scripted(SCRIPT))
    digest = hashlib.sha256(canonical_json([e.model_dump(mode="json") for e in eng.sink.events]).encode())
    assert digest.hexdigest() == EVENTS_BEFORE[mode]


def test_example_config_is_e4_plus_trade():
    e4 = load_experiment(REPO_ROOT / "configs" / "g1_v2_costly_sim.yaml", [])[0]
    tr = load_experiment(REPO_ROOT / "configs" / "g1_v2_costly_trade_sim.yaml", [])[0]
    assert tr.game.trade.enabled and tr.game.model_copy(update={"trade": e4.game.trade}) == e4.game
    assert tr.god == e4.god and tr.roles == e4.roles and tr.run.seed == e4.run.seed


def test_parser_accepts_offer_trade_only_when_enabled():
    reply = '{"actions": [{"kind": "offer_trade", "target": "bob", "give": 1, "want": ["3", 4]}]}'
    off = parse_reply(reply)
    assert not off.ok and off.errors[0].startswith("unknown_kind:0:offer_trade")
    on = parse_reply(reply, trade=True)
    assert on.ok and on.actions == [TradeOffer(target="bob", give=[1], want=[3, 4])]
    assert parse_reply('{"actions": [{"kind": "swap", "to": "bob", "give": [1], "want": [3]}]}', trade=True).ok


# ------------------------------------------------------------------ accept / reject / stale
def test_accept_swaps_atomically_and_tells_both():
    p = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])]}, accept={"bob"})
    out, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p)
    st = eng.state
    assert sorted(k.info_id for k in st.agents["ann"].keys) == [2, 3]
    assert sorted(k.info_id for k in st.agents["bob"].keys) == [1, 4]
    ask = p.asks[0]
    assert (ask.proposer, ask.target, ask.give, ask.want, ask.needs_open) == ("ann", "bob", [1], [3], [1, 6])
    ex = _types(eng, "trade_executed")[0].data
    assert ex.completes_need_proposer == [3] and ex.completes_need_target == [1]
    assert [e.data.cause for e in _types(eng, "key_transfer")] == ["trade", "trade"]
    seq = [e.type for e in eng.sink.events if e.type.startswith("trade")]
    assert seq == ["trade_offer", "trade_reply", "trade_executed"]
    notes = {n.agent: n.text for n in st.notices if n.kind == "trade"}
    assert "accepted your trade" in notes["ann"] and "You accepted ann's trade" in notes["bob"]
    assert len({e.turn_index for e in _types(eng, "trade_offer", "trade_reply", "trade_executed")}) == 1


def test_reject_moves_nothing_and_tells_both():
    p = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1, 2], want=[3])]})
    _, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p)
    assert sorted(k.info_id for k in eng.state.agents["ann"].keys) == [1, 2]
    assert not _types(eng, "trade_executed", "trade_failed", "key_transfer")
    assert _types(eng, "trade_reply")[0].data.accept is False
    texts = [n.text for n in eng.state.notices if n.kind == "trade"]
    assert any("bob rejected your trade offer" in t for t in texts)
    assert any("You rejected ann's trade offer" in t for t in texts)


def test_stale_holdings_fail_with_notice_to_both():
    cfg = _cfg()
    st = mk_state(FOUR)
    res = resolve_turn(st, "ann", [TradeOffer(target="bob", give=[1], want=[3])], None, cfg.game,
                       locks=trade.Locks())
    (o,) = res.offers
    bob = st.agents["bob"]                            # bob's key 3 leaves before the reply lands
    st.agents["cat"].keys.append(bob.keys.pop(0))
    evs, moved = trade.settle(st, o, trade.TradeAnswer(accept=True), cfg.game)
    assert not moved
    failed = next(e.data for e in evs if e.data.type == "trade_failed")
    assert failed.reason == "stale_holdings" and failed.missing_target == [3] and failed.missing_proposer == []
    assert sorted(k.info_id for k in st.agents["ann"].keys) == [1, 2]
    assert {n.agent for n in st.notices if n.kind == "trade" and "did not happen" in n.text} == {"ann", "bob"}


def test_offer_refusals_cost_no_reply():
    p = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[6])],     # bob lacks 6
                (1, "bob"): [TradeOffer(target="cat", give=[9], want=[5])],     # bob lacks 9
                (1, "cat"): [TradeOffer(target="dan", give=[5], want=[5])]})    # overlap
    _, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p)
    reasons = [e.data.reason for e in _types(eng, "action_result")]
    assert reasons == ["target_lacks_key", "no_such_key", "invalid_offer"]
    assert p.asks == [] and not _types(eng, "trade_offer")


def test_no_reply_policy_rejects():
    _, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR),
                       Scripted({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])]}))
    r = _types(eng, "trade_reply")[0].data
    assert not r.accept and not r.parse_ok and r.errors == ["no_reply_policy"]


# ------------------------------------------------------------------ limits / locks / network
def test_one_offer_per_turn_and_listed_keys_locked_for_the_turn():
    p = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3]),
                             TradeOffer(target="bob", give=[2], want=[4])]}, accept={"bob"})
    _, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p)
    assert [e.data.reason for e in _types(eng, "action_result")] == [None, "offer_limit"]
    assert len(p.asks) == 1
    p2 = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3]), GiveKey(to="cat", info_id=1)]})
    _, eng2 = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p2)
    assert [e.data.reason for e in _types(eng2, "action_result")] == [None, "key_locked"]
    assert sorted(k.info_id for k in eng2.state.agents["ann"].keys) == [1, 2]     # rejected; nothing moved
    two = _cfg(max_rounds=1, trade={"enabled": True, "max_offers_per_turn": 2})
    _, eng3 = play_game(two, fresh_game(FOUR), Trader(p.script, accept={"bob"}))
    assert len(_types(eng3, "trade_executed")) == 2


def test_simultaneous_traded_keys_locked_for_the_round():
    """ann (first in priority) trades 1<->3 with bob. bob's s0 choices: give 3 to dan (gone: no_such_key)
    and give 1 (just received) to cat (locked for the round). Next round the lock is gone."""
    script = {(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])],
              (1, "bob"): [GiveKey(to="dan", info_id=3), GiveKey(to="cat", info_id=1)],
              (2, "bob"): [GiveKey(to="cat", info_id=1)]}
    p = Trader(script, accept={"bob"})
    _, eng = play_game(_cfg("simultaneous", max_rounds=2), fresh_game(FOUR), p)
    res = [(e.round, e.data.agent, e.data.action.kind, e.data.reason) for e in _types(eng, "action_result")]
    assert (1, "bob", "give_key", "no_such_key") in res and (1, "bob", "give_key", "key_locked") in res
    assert (2, "bob", "give_key", None) in res
    texts = [n.text for n in eng.state.notices if n.agent == "bob"]
    assert any("changed hands in a trade" in t for t in texts)
    assert any("resolved in turn order" in t for t in texts)        # the generic notice, for no_such_key


def test_simultaneous_target_replies_from_current_state():
    """bob's s0 give of 3 resolves before ann's offer (bob first in priority): the offer is refused
    at ann's slot because bob no longer holds 3, without a reply call."""
    script = {(1, "bob"): [GiveKey(to="dan", info_id=3)],
              (1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])]}
    p = Trader(script, accept={"bob"})
    cfg = _cfg("simultaneous", max_rounds=1)
    st = fresh_game(FOUR)
    st.order = ["bob", "ann", "cat", "dan"]
    _, eng = play_game(cfg, st, p)
    assert [e.data.reason for e in _types(eng, "action_result") if e.data.agent == "ann"] == ["target_lacks_key"]
    assert p.asks == []


def test_contacts_only():
    cfg = _cfg(network={"topology": "ring", "mode": "contacts"})
    st = fresh_game(FOUR)
    for a, nb in {"ann": ["bob"], "bob": ["ann", "cat"], "cat": ["bob", "dan"], "dan": ["cat"]}.items():
        st.agents[a].contacts = nb
    p = Trader({(1, "ann"): [TradeOffer(target="cat", give=[1], want=[5])]}, accept={"cat"})
    _, eng = play_game(cfg.model_copy(update={"game": cfg.game.model_copy(update={"max_rounds": 1})}), st, p)
    assert [e.data.reason for e in _types(eng, "action_result")] == ["not_connected"]
    assert p.asks == [] and any("not connected" in n.text for n in eng.state.notices if n.agent == "ann")


def test_locked_winner_can_trade_and_keeps_its_score():
    spec = {"ann": ([1, 2], [3]), "bob": ([3, 4, 9], [1]), "cat": ([5], [2]), "dan": ([6], [4])}
    script = {(1, "ann"): [GiveKey(to="bob", info_id=1)],                 # bob wins in round 1
              (2, "ann"): [TradeOffer(target="bob", give=[2], want=[3])]}
    p = Trader(script, accept={"bob"})
    cfg = _cfg(max_rounds=2, winner_policy="lock_and_stay")
    _, eng = play_game(cfg, fresh_game(spec), p)
    assert eng.state.agents["bob"].won_round == 1 and p.asks[0].won and p.asks[0].locked
    assert len(_types(eng, "trade_executed")) == 1 and eng.scores()["bob"] == 1.0
    assert eng.state.agents["ann"].won_round == 2


# ------------------------------------------------------------------ LLM / fake policy / metrics
def _ask(**kw):
    base = dict(offer_id="o3-0", proposer="ann", target="bob", give=[1], want=[3], round=1, max_rounds=6,
                needs_open=[1, 6], holdings=[(3, 1003), (4, 1004)], score=0.0, won=False, locked=False,
                key_bonus=0.05)
    return trade.TradeAsk(**{**base, **kw})


@pytest.mark.parametrize("text,accept,ok", [('{"thought": "yes", "accept": true}', True, True),
                                            ('{"accept": "no"}', False, True), ("sure!", False, False)])
def test_llm_agent_reply_trade(text, accept, ok):
    import asyncio

    fake = FakeLLM(script=[text])
    ag = LLMAgent("bob", fake, _cfg())
    ans = asyncio.run(ag.reply_trade(_ask(), game_id="g1"))
    assert (ans.accept, ans.parse_ok) == (accept, ok) and len(ans.llm_call_ids) == 1
    req = fake.requests[0]
    assert req.messages[0].content == ag.system_text and "offer_trade" in ag.system_text
    user = req.messages[1].content
    assert "TRADE OFFER from ann" in user and "You would give: info #3" in user and "+0" in user
    assert req.meta.purpose == "trade_reply:o3-0" and ag.last.prompt_hash is None    # turn trace untouched


def test_fake_policy_replies_and_offers():
    ag = LLMAgent("bob", FakeLLM(), _cfg())
    user = load_prompt_set("agent/v1").render("trade_reply.j2", ask=dataclasses.asdict(_ask()),
                                              cot_mode="explicit", delta=0)

    def req(u):
        return ChatRequest(call_id="c", messages=[ChatMessage(role="system", content=ag.system_text),
                                                  ChatMessage(role="user", content=u)], max_tokens=10,
                           temperature=0.0, seed=1, meta=CallMeta(role="agent"))
    assert json.loads(smart_fake_policy(req(user)))["accept"] is True
    bad = user.replace("You would give: info #3", "You would give: info #6")    # gives away its need 6
    assert json.loads(smart_fake_policy(req(bad)))["accept"] is False


def test_summarize_counts_and_central_nodes():
    p = Trader({(1, "ann"): [TradeOffer(target="bob", give=[1], want=[3])],
                (1, "cat"): [TradeOffer(target="dan", give=[5], want=[9])]}, accept={"bob"})
    _, eng = play_game(_cfg(max_rounds=1), fresh_game(FOUR), p)
    evs = [e.model_dump(mode="json") for e in eng.sink.events]
    star = {"bob": ["ann", "cat", "dan"], "ann": ["bob"], "cat": ["bob"], "dan": ["bob"]}
    s = summarize(evs, contacts=star)
    assert (s["offers"], s["offers_refused"], s["executed"], s["keys_moved"], s["need_completing"]) == (1, 1, 1, 2, 1)
    assert s["accept_rate"] == 1.0 and s["bridge_trades"] == 1 and s["hub_trades"] == 1
    assert (s["need_both"], s["need_one"], s["need_none"]) == (1, 0, 0)
    one = [e if e["type"] != "trade_executed" else {**e, "data": {**e["data"], "completes_need_target": []}}
           for e in evs]
    s1 = summarize(one)
    assert (s1["need_both"], s1["need_one"], s1["need_none"]) == (0, 1, 0)
    assert summarize(evs)["network"] is False


def test_fake_e2e_trade_run_holds_invariants(tmp_path):
    ov = [*FAKE_OVERRIDES, "game.trade.enabled=true", "game.turn_mode=simultaneous"]
    run = _run(tmp_path, "trade_e2e", ov)
    tot = summarize_run(run)["total"]
    assert tot["offers"] > 0 and tot["replies"] == tot["offers"]
    assert check_run(run) == []
