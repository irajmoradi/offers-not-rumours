"""Turn resolution (split out of the engine loop): apply ONE agent turn's actions at the current slot.

    res = resolve_turn(state, agent, actions, grep, cfg)      # mutates state like rules.apply
    -> TurnResolution(events=[EventBody, ...] in emission order, results=[(ok, reason)] per action)

Per action, in order:
  grep_log        free (no slot); ok iff the engine ran the turn's grep (`grep` is its result); only the
                  first grep of a turn can succeed ("grep_unavailable" otherwise)
  board-blocked   under game.board.post_costs_turn, every action except the turn's first post_board,
                  grep_log and noop fails with board.BOARD_WHOLE_TURN, using no slot
  connect-blocked under game.network.connect.enabled, a turn with a connect is the connect: the turn's
                  first connect is applied and every other action except grep_log and noop fails with
                  network.CONNECT_WHOLE_TURN, using no slot (also when the connect itself is refused).
                  With board.post_costs_turn too, the EARLIER of the first post_board / first connect
                  takes the turn and the other is blocked with that one's reason.
  slot limit      beyond game.actions_per_turn -> "action_limit"
  otherwise       rules.apply; in simultaneous mode a failed give_key / defect also queues a "system"
                  notice (conflict_text): the state may have changed since the agent observed it
                  (not for winner_policy=sit_out departures: rules already sent a "left" notice, or
                  the actor itself has left, "actor_left"; nor for game.network refusals, which
                  already sent their own notice)
  sit_out mid-turn win: once an action of this turn (a minted defect) or, in simultaneous mode, an
                  earlier agent's give in this round's resolution completes the agent's needs, it has
                  left: every remaining action fails "actor_left" (it does not execute) and the agent
                  gets one "left" notice listing them (rules.midturn_text)
game.trade (only when the engine passes `locks`): offer_trade beyond game.trade.max_offers_per_turn
                  fails "offer_limit"; a give_key / offer_trade of a key covered by a trade lock
                  (listed in an offer earlier this turn, or moved by a trade this round in
                  simultaneous mode) fails "key_locked", each with a "trade" notice; a valid offer locks
                  its listed keys and is returned in `offers` for the engine's reply sub-turn (game.trade)
After the loop: one notice for board-blocked (or connect-blocked) actions, then the post_skip_rounds
skip (if a post succeeded). The engine emits `events` in order, so the event stream is the one the old inline loop
produced. Pure apart from mutating `state` (no I/O). Imports: config, schema, game.rules, game.board.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from swarm.config import GameCfg
from swarm.schema import (
    Action,
    ActionResultData,
    AgentId,
    ConnectAction,
    DefectAction,
    EventBody,
    GameState,
    GiveKey,
    GrepLog,
    GrepResult,
    Noop,
    PostBoard,
    TradeOffer,
    TradeOfferData,
)

from . import board, network, rules, trade

Result = tuple[bool, str | None]
# sit_out departures: the actor already got a "left" notice (or has itself left), so no conflict notice;
# game.network refusals: rules already sent a "not connected" / "don't know that agent" notice
_NO_CONFLICT_NOTICE = (*rules.LEFT_REASONS, "actor_left", *network.REASONS, *rules.OWN_NOTICE_REASONS,
                       *trade.REASONS)


@dataclass
class TurnResolution:
    """Event bodies to emit (in order) and one (ok, reason) per action (for own-action ledger items)."""
    events: list[EventBody] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    offers: list[TradeOfferData] = field(default_factory=list)   # game.trade: valid offers, in order


def _result_event(state: GameState, agent: AgentId, act: Action, ok: bool, reason: str | None,
                  n_hits: int | None = None) -> EventBody:
    return EventBody(round=state.round, turn_index=state.turn_index, phase="resolve", actor=agent,
                     data=ActionResultData(agent=agent, action=act, ok=ok, reason=reason, n_hits=n_hits))


def conflict_text(act: Action, reason: str | None) -> str:
    """Notice for a failed give/defect in simultaneous mode (the state may have changed since the
    agent observed it: actions resolve in turn-priority order)."""
    what = (f"give_key of info #{act.info_id} to {act.to}" if isinstance(act, GiveKey)
            else f"defect on {act.target} for info #{act.info_id}")
    return (f"Your {what} failed ({reason or 'invalid'}). Actions this round resolved in turn order, "
            "so players before you may have changed the state you saw.")


def connect_whole_turn_text(blocked: Sequence[Action]) -> str:
    """Notice for actions dropped because a connect took the whole turn (game.network.connect)."""
    kinds = ", ".join(a.kind for a in blocked)
    return f"Your other action(s) this turn ({kinds}) were not done: connecting uses your whole turn."


def resolve_turn(state: GameState, agent: AgentId, actions: Sequence[Action], grep: GrepResult | None,
                 cfg: GameCfg, *, locks: trade.Locks | None = None) -> TurnResolution:
    """Apply one turn's actions at slot state.turn_index; see the module docstring for the rules."""
    out = TurnResolution()
    used, grep_left = 0, grep
    post_at = board.whole_turn_post(actions, cfg.board)
    conn_at = (next((i for i, a in enumerate(actions) if isinstance(a, ConnectAction)), None)
               if cfg.network.connect.enabled else None)
    if post_at is not None and conn_at is not None:       # the earlier one takes the whole turn
        post_at, conn_at = (post_at, None) if post_at < conn_at else (None, conn_at)
    whole_at, whole_reason = ((post_at, board.BOARD_WHOLE_TURN) if post_at is not None
                              else (conn_at, network.CONNECT_WHOLE_TURN))
    blocked: list[Action] = []
    left_skipped: list[Action] = []                       # sit_out: actions after the agent won and left
    posted = False
    for i, act in enumerate(actions):
        if isinstance(act, GrepLog):                      # free: no action slot
            ok = grep_left is not None
            reason = None if ok else "grep_unavailable"
            out.events.append(_result_event(state, agent, act, ok, reason,
                                            n_hits=len(grep_left.hits) if grep_left is not None else None))
            out.results.append((ok, reason))
            grep_left = None                              # a second grep_log in one turn fails
            continue
        if whole_at is not None and i != whole_at and not isinstance(act, Noop):
            out.events.append(_result_event(state, agent, act, False, whole_reason))
            out.results.append((False, whole_reason))               # uses no slot
            blocked.append(act)
            continue
        res = _apply_one(state, agent, act, used, cfg, out.events, locks, out.offers)
        out.results.append(res)
        if res == (False, "actor_left"):
            left_skipped.append(act)
        used += 1
        posted = posted or (res[0] and isinstance(act, PostBoard))
        if (not res[0] and cfg.turn_mode == "simultaneous" and isinstance(act, (GiveKey, DefectAction))
                and res[1] not in _NO_CONFLICT_NOTICE):
            out.events.append(rules.notice(state, agent, "system", conflict_text(act, res[1])))
    if blocked:
        text = (board.whole_turn_text(blocked) if whole_reason == board.BOARD_WHOLE_TURN
                else connect_whole_turn_text(blocked))
        out.events.append(rules.notice(state, agent, "system", text))
    if left_skipped:
        out.events.append(rules.notice(state, agent, "left",
                                       rules.midturn_text(left_skipped)))
    if posted and cfg.board.post_skip_rounds > 0:
        out.events += board.skip_after_post(state, agent, cfg.board.post_skip_rounds)
    return out


def _trade_gate(state: GameState, agent: AgentId, act: Action, cfg: GameCfg, locks: trade.Locks,
                offers: list[TradeOfferData]) -> list[EventBody] | None:
    """game.trade: offer_limit / key_locked refusals (None = go ahead)."""
    if isinstance(act, TradeOffer):
        if len(offers) >= cfg.trade.max_offers_per_turn:
            n = cfg.trade.max_offers_per_turn
            return [_result_event(state, agent, act, False, "offer_limit"),
                    rules.notice(state, agent, "trade", f"Your trade offer to {act.target} was not made: you can "
                                                        f"make at most {n} offer{'s' if n != 1 else ''} per turn.")]
        if act.target not in state.agents:
            return None
        bad = trade.locked_out(state, agent, act.give, locks) + trade.locked_out(state, act.target, act.want, locks)
    elif isinstance(act, GiveKey):
        bad = trade.locked_out(state, agent, [act.info_id], locks)
    else:
        return None
    if not bad:
        return None
    return [_result_event(state, agent, act, False, "key_locked"),
            rules.notice(state, agent, "trade", trade.key_locked_text(bad))]


def _apply_one(state: GameState, agent: AgentId, act: Action, used: int, cfg: GameCfg,
               events: list[EventBody], locks: trade.Locks | None = None,
               offers: list[TradeOfferData] | None = None) -> Result:
    if used >= cfg.actions_per_turn:
        events.append(_result_event(state, agent, act, False, "action_limit"))
        return False, "action_limit"
    if locks is not None and offers is not None:              # game.trade on
        gate = None if rules.departed(state, agent, cfg) else _trade_gate(state, agent, act, cfg, locks, offers)
        if gate is not None:
            events += gate
            return False, gate[0].data.reason
        if isinstance(act, TradeOffer):
            evs = trade.offer(state, agent, act, cfg, f"o{state.turn_index}-{len(offers)}")
            events += evs
            o = next((e.data for e in evs if isinstance(e.data, TradeOfferData)), None)
            if o is not None:
                offers.append(o)
                trade.lock(locks, agent, o.give)
                trade.lock(locks, o.target, o.want)
            return evs[0].data.ok, evs[0].data.reason
    _, evs = rules.apply(state, agent, act, cfg)
    events += evs
    first = evs[0].data if evs else None
    if isinstance(first, ActionResultData):
        return first.ok, first.reason
    return True, None
