"""game.trade (plan/TRADE_DESIGN.md; off by default): the atomic n-for-k key swap `offer_trade`.

    evs = offer(state, actor, TradeOffer, cfg, oid)    # at the proposer's slot (game.resolve)
    ask = make_ask(state, TradeOfferData, cfg)         # what the target sees in its reply sub-turn
    evs, moved = settle(state, TradeOfferData, TradeAnswer, cfg)   # after the reply (game.engine)

offer: refusals (state unchanged, action_result ok=False, a "trade" or network notice to the
proposer): trade_disabled, actor_left, unknown_agent / self_target, not_connected / unknown_contact,
recipient_left (sit_out), invalid_offer (a side empty or over max_keys_per_side, an id twice, give and
want overlap), no_such_key (the proposer lacks a give key), target_lacks_key. A valid offer emits
action_result ok + trade_offer and moves nothing. Locks and offer_limit are game.resolve's.
settle: reject -> trade_reply + a notice to both. Accept -> re-validate (both still hold every listed
key and neither left, else trade_failed + a notice to both) and swap every copy at once: key_transfer
(cause "trade") + an ActionRecord of kind "give" per copy (the view_log shows the two gives), then
trade_executed, a notice to both, and the win check for both. Never touches a needs list, so
lock_and_stay winners trade freely and copy counts are conserved (solvability).
Pure apart from mutating `state`. Imports: config, schema, game.network, game.rules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from swarm.config import GameCfg
from swarm.schema import (
    ActionRecord,
    AgentId,
    EventBody,
    GameState,
    InfoId,
    KeyTransferData,
    TradeExecutedData,
    TradeFailedData,
    TradeOffer,
    TradeOfferData,
    TradeReplyData,
)

from . import rules

Events = list[EventBody]
Locks = Counter          # (agent, info_id) -> number of that agent's copies locked by a trade
REASONS = ("trade_disabled", "invalid_offer", "target_lacks_key", "offer_limit", "key_locked")


@dataclass(frozen=True)
class TradeAsk:
    """What the target is shown in its reply sub-turn (agent-visible facts only: its own state)."""
    offer_id: str
    proposer: AgentId
    target: AgentId
    give: list[InfoId]                 # the proposer gives (the target would get)
    want: list[InfoId]                 # the proposer wants (the target would give)
    round: int
    max_rounds: int
    needs_open: list[InfoId]
    holdings: list[tuple[InfoId, int]]
    score: float
    won: bool
    locked: bool                       # won under lock_and_stay
    key_bonus: float


@dataclass
class TradeAnswer:
    """The target's reply (LLMAgent.reply_trade / a scripted reply_trade)."""
    accept: bool
    parse_ok: bool = True
    cot: str | None = None
    errors: list[str] = field(default_factory=list)
    llm_call_ids: list[str] = field(default_factory=list)


def _ids(xs: Sequence[InfoId]) -> str:
    return ", ".join(f"info #{i}" for i in xs) or "nothing"


def _counts(state: GameState, agent: AgentId) -> Counter:
    return Counter(kc.info_id for kc in state.agents[agent].keys)


def missing(state: GameState, agent: AgentId, infos: Sequence[InfoId]) -> list[InfoId]:
    """Listed infos the agent does not hold (multiplicity counted)."""
    have, out = _counts(state, agent), []
    for i in infos:
        if have[i] > 0:
            have[i] -= 1
        else:
            out.append(i)
    return out


def locked_out(state: GameState, agent: AgentId, infos: Sequence[InfoId], locks: Locks | None) -> list[InfoId]:
    """Listed infos the agent holds but cannot use because trade locks cover them (game.resolve)."""
    if not locks:
        return []
    have = _counts(state, agent)
    want = Counter(infos)
    return sorted(i for i, k in want.items() if have[i] >= k and have[i] - locks[(agent, i)] < k)


def lock(locks: Locks, agent: AgentId, infos: Sequence[InfoId], sign: int = 1) -> None:
    for i in infos:
        locks[(agent, i)] += sign
        if locks[(agent, i)] <= 0:
            del locks[(agent, i)]


def offer_text(o: TradeOfferData) -> str:
    return f"your {_ids(o.give)} for {o.target}'s {_ids(o.want)}"


def _refuse(state: GameState, actor: AgentId, a: TradeOffer, reason: str, why: str) -> Events:
    return [*rules._fail(state, actor, a, reason),
            rules.notice(state, actor, "trade", f"Your trade offer to {a.target} was not made: {why}")]


def offer(state: GameState, actor: AgentId, a: TradeOffer, cfg: GameCfg, oid: str) -> Events:
    """Validate an offer at the proposer's slot (module docstring). First event = action_result."""
    tc = cfg.trade
    if not tc.enabled:
        return rules._fail(state, actor, a, "trade_disabled")
    if actor not in state.agents:
        return rules._fail(state, actor, a, "unknown_actor")
    if rules.departed(state, actor, cfg):
        return rules._fail(state, actor, a, "actor_left")
    err = rules._peer_error(state, actor, a.target)
    if err:
        return rules._fail(state, actor, a, err)
    blocked = rules._network_fail(state, actor, a, a.target)
    if blocked is not None:
        return blocked
    if rules.departed(state, a.target, cfg):
        return rules._left_fail(state, actor, a, a.target, "recipient_left")
    n = tc.max_keys_per_side
    if (not a.give or not a.want or len(a.give) > n or len(a.want) > n or len(set(a.give)) < len(a.give)
            or len(set(a.want)) < len(a.want) or set(a.give) & set(a.want)):
        return _refuse(state, actor, a, "invalid_offer",
                       f"each side must list 1 to {n} different info ids, and no info id on both sides.")
    if missing(state, actor, a.give):
        return _refuse(state, actor, a, "no_such_key",
                       f"you do not hold a key for {_ids(missing(state, actor, a.give))}.")
    lack = missing(state, a.target, a.want)
    if lack:
        return _refuse(state, actor, a, "target_lacks_key", f"{a.target} does not hold a key for {_ids(lack)}.")
    data = TradeOfferData(offer_id=oid, proposer=actor, target=a.target, give=list(a.give), want=list(a.want))
    return [rules._ok(state, actor, a), rules._ev(state, actor, data)]


def make_ask(state: GameState, o: TradeOfferData, cfg: GameCfg) -> TradeAsk:
    """The target's view for its reply: the offer plus its CURRENT open needs, keys and score."""
    s = state.agents[o.target]
    return TradeAsk(offer_id=o.offer_id, proposer=o.proposer, target=o.target, give=list(o.give),
                    want=list(o.want), round=state.round, max_rounds=cfg.max_rounds,
                    needs_open=rules.open_needs(state, o.target),
                    holdings=[(kc.info_id, kc.key) for kc in s.keys],
                    score=rules.score(state, o.target, cfg), won=s.won_round is not None,
                    locked=s.won_round is not None and rules.score_locked(cfg),
                    key_bonus=cfg.score.held_key_bonus if cfg.score.held_key_scope == "all" else 0.0)


def _newly_held(state: GameState, agent: AgentId, got: Sequence[InfoId]) -> list[InfoId]:
    held = rules.held_infos(state, agent)
    return sorted({i for i in got if i in rules.open_needs(state, agent) and i not in held})


def _move(state: GameState, src: AgentId, dst: AgentId, infos: Sequence[InfoId]) -> tuple[list[str], Events]:
    ids, evs = [], []
    for i in infos:
        keys = state.agents[src].keys
        kc = keys.pop(next(j for j, k in enumerate(keys) if k.info_id == i))
        state.agents[dst].keys.append(kc)
        state.history.append(ActionRecord(turn_index=state.turn_index, round=state.round, kind="give",
                                          actor=src, target=dst, info_id=i))
        evs.append(rules._ev(state, src, KeyTransferData(copy_id=kc.copy_id, info_id=i, from_agent=src,
                                                         to_agent=dst, cause="trade")))
        ids.append(kc.copy_id)
    return ids, evs


def settle(state: GameState, o: TradeOfferData, ans: TradeAnswer, cfg: GameCfg) -> tuple[Events, bool]:
    """Apply the target's answer (module docstring). -> (events, keys_moved)."""
    p, t = o.proposer, o.target
    evs: Events = [rules._ev(state, t, TradeReplyData(
        offer_id=o.offer_id, proposer=p, target=t, accept=ans.accept, parse_ok=ans.parse_ok,
        errors=list(ans.errors), llm_call_ids=list(ans.llm_call_ids), cot=ans.cot))]
    if not ans.accept:
        evs += [rules.notice(state, p, "trade", f"{t} rejected your trade offer ({offer_text(o)}). Nothing moved."),
                rules.notice(state, t, "trade", f"You rejected {p}'s trade offer (their {_ids(o.give)} for "
                                                f"your {_ids(o.want)}). Nothing moved.")]
        return evs, False
    gone = rules.departed(state, p, cfg) or rules.departed(state, t, cfg)
    mp, mt = missing(state, p, o.give), missing(state, t, o.want)
    if gone or mp or mt:
        reason = "party_left" if gone else "stale_holdings"
        evs.append(rules._ev(state, "SYSTEM", TradeFailedData(offer_id=o.offer_id, proposer=p, target=t,
                                                              reason=reason, missing_proposer=mp,
                                                              missing_target=mt)))
        why = "a player has left the game" if gone else "the listed keys are no longer all held"
        for who, other in ((p, t), (t, p)):
            evs.append(rules.notice(state, who, "trade", f"The trade between you and {other} was accepted but "
                                                         f"did not happen: {why}. Nothing moved."))
        return evs, False
    need_p, need_t = _newly_held(state, p, o.want), _newly_held(state, t, o.give)
    to_t, ev1 = _move(state, p, t, o.give)
    to_p, ev2 = _move(state, t, p, o.want)
    evs += ev1 + ev2
    evs.append(rules._ev(state, "SYSTEM", TradeExecutedData(
        offer_id=o.offer_id, proposer=p, target=t, give=list(o.give), want=list(o.want),
        copies_to_target=to_t, copies_to_proposer=to_p, completes_need_proposer=need_p,
        completes_need_target=need_t)))
    evs += [rules.notice(state, p, "trade", f"{t} accepted your trade: you gave {_ids(o.give)} and got "
                                            f"{_ids(o.want)}."),
            rules.notice(state, t, "trade", f"You accepted {p}'s trade: you gave {_ids(o.want)} and got "
                                            f"{_ids(o.give)}.")]
    for ag in (p, t):
        evs += rules.update_win(state, ag, cfg)
    return evs, True


def key_locked_text(infos: Sequence[InfoId]) -> str:
    return (f"Your action with {_ids(infos)} was not done: those keys are part of a trade offer this turn"
            " or changed hands in a trade this round.")
