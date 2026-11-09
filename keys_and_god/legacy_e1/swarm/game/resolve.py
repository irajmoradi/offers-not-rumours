"""Turn resolution (split out of the engine loop): apply ONE agent turn's actions at the current slot.

    res = resolve_turn(state, agent, actions, grep, cfg)      # mutates state like rules.apply
    -> TurnResolution(events=[EventBody, ...] in emission order, results=[(ok, reason)] per action)

Per action, in order:
  grep_log        free (no slot); ok iff the engine ran the turn's grep (`grep` is its result); only the
                  first grep of a turn can succeed ("grep_unavailable" otherwise)
  board-blocked   under game.board.post_costs_turn, every action except the turn's first post_board,
                  grep_log and noop fails with board.BOARD_WHOLE_TURN, using no slot
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
After the loop: one notice for board-blocked actions, then the post_skip_rounds skip (if a post
succeeded). The engine emits `events` in order, so the event stream is the one the old inline loop
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
    DefectAction,
    EventBody,
    GameState,
    GiveKey,
    GrepLog,
    GrepResult,
    Noop,
    PostBoard,
)

from . import board, network, rules

Result = tuple[bool, str | None]
# sit_out departures: the actor already got a "left" notice (or has itself left), so no conflict notice;
# game.network refusals: rules already sent a "not connected" / "don't know that agent" notice
_NO_CONFLICT_NOTICE = (*rules.LEFT_REASONS, "actor_left", *network.REASONS)


@dataclass
class TurnResolution:
    """Event bodies to emit (in order) and one (ok, reason) per action (for own-action ledger items)."""
    events: list[EventBody] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)


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


def resolve_turn(state: GameState, agent: AgentId, actions: Sequence[Action], grep: GrepResult | None,
                 cfg: GameCfg) -> TurnResolution:
    """Apply one turn's actions at slot state.turn_index; see the module docstring for the rules."""
    out = TurnResolution()
    used, grep_left = 0, grep
    post_at = board.whole_turn_post(actions, cfg.board)
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
        if post_at is not None and i != post_at and not isinstance(act, Noop):
            out.events.append(_result_event(state, agent, act, False, board.BOARD_WHOLE_TURN))
            out.results.append((False, board.BOARD_WHOLE_TURN))     # uses no slot
            blocked.append(act)
            continue
        res = _apply_one(state, agent, act, used, cfg, out.events)
        out.results.append(res)
        if res == (False, "actor_left"):
            left_skipped.append(act)
        used += 1
        posted = posted or (res[0] and isinstance(act, PostBoard))
        if (not res[0] and cfg.turn_mode == "simultaneous" and isinstance(act, (GiveKey, DefectAction))
                and res[1] not in _NO_CONFLICT_NOTICE):
            out.events.append(rules.notice(state, agent, "system", conflict_text(act, res[1])))
    if blocked:
        out.events.append(rules.notice(state, agent, "system", board.whole_turn_text(blocked)))
    if left_skipped:
        out.events.append(rules.notice(state, agent, "left",
                                       rules.midturn_text(left_skipped)))
    if posted and cfg.board.post_skip_rounds > 0:
        out.events += board.skip_after_post(state, agent, cfg.board.post_skip_rounds)
    return out


def _apply_one(state: GameState, agent: AgentId, act: Action, used: int, cfg: GameCfg,
               events: list[EventBody]) -> Result:
    if used >= cfg.actions_per_turn:
        events.append(_result_event(state, agent, act, False, "action_limit"))
        return False, "action_limit"
    _, evs = rules.apply(state, agent, act, cfg)
    events += evs
    first = evs[0].data if evs else None
    if isinstance(first, ActionResultData):
        return first.ok, first.reason
    return True, None
