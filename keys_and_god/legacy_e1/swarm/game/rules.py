"""Game rules (WP1a): `apply(state, actor, action, cfg) -> (state, event_bodies)`.

No I/O, no LLM, no randomness. The state is mutated IN PLACE (deep copies happen only at round
boundaries, in the engine) unless `copy=True`. `apply` never raises on agent input: an invalid action
returns a single `action_result` event with ok=False and leaves the state unchanged. An INTERNAL error
(a KeyError/ValueError/IndexError inside a rule) is logged at ERROR with its traceback and becomes a
failed action "error:<Type>"; with SWARM_STRICT_RULES=1 (the test suite) it is re-raised.

Timing convention: every record created here is stamped with `state.turn_index` (the slot of the
acting agent). A record is visible to observations with index STRICTLY greater than its stamp
(see game/observe.py). The engine wakes God with `state.turn_index` = the slot about to be played,
so forged messages carry the wake slot (CONTRACTS §2) and reach the receiver at its next observation
after that slot.

Rules (ARCHITECTURE_v2 §1, D1-D4, D12, D13):
- send_message / post_board: always succeed for a valid peer; text truncated to max_message_chars.
- give_key: the actor must hold a copy of info_id; one copy moves to `to`. Only Give transfers keys.
- defect(target, info_id): CORRECT iff info_id is one of target's open needs (not burned, not lost).
  Correct: the target's need is burned; the actor gains per `defect_gain` (mint a new copy / steal a
  copy from another holder / none). Wrong: the actor pays `wrong_defect_penalty` (lose_need: its
  first open need is lost permanently; delay: no turns until round + delay_n). A target that has
  won cannot be defected against (ok=False, no penalty).
- Win (D3): holds a key for every open need and has at least one open need. won_round is sticky.
  Score = held open needs / len(needs) (burned and lost needs count 0).
- game.winner_policy = "sit_out" (winners leave the game): a winner DEPARTS at the moment it wins
  (`departed`). It takes no further action (any later action of its own fails "actor_left", e.g. the
  rest of a turn, or a simultaneous-mode turn chosen before it won), and nobody can reach it:
  send_message / give_key to it fail "recipient_left", defect on it fails "target_left", each with a
  "left" notice to the actor; defect_gain="steal" never takes from it. Its keys leave with it: they
  stay recorded on its AgentState (copy conservation) but are out of play, so its score is LOCKED at
  the value it had when it won. Every other agent gets a "left" notice. "continue" (the default) is
  unchanged: winners keep acting and only defect-on-a-winner is refused.
- game.network (game/network.py; off by default): send_message / give_key / defect aimed at an agent
  the actor may not address fail "not_connected" (mode contacts: not a graph neighbour) or
  "unknown_contact" (mode known_ids: a name it does not know), state unchanged, with a "system"
  notice to the actor. Checked after unknown_agent / self_target and before the sit_out checks.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from swarm.config import GameCfg
from swarm.game import network
from swarm.rng import make_msg_id
from swarm.schema import (
    GOD,
    Action,
    ActionRecord,
    ActionResultData,
    AgentId,
    BurnData,
    DefectAction,
    EventBody,
    GameState,
    GiveKey,
    GrepLog,
    InfoId,
    KeyCopy,
    KeyTransferData,
    Message,
    MessageData,
    Noop,
    Notice,
    NoticeData,
    Phase,
    PostBoard,
    SendMessage,
    WinData,
    defect_release_turn,
)

Events = list[EventBody]
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ scoring helpers
def open_needs(state: GameState, agent: AgentId) -> list[InfoId]:
    """Needs that are neither burned nor lost."""
    s = state.agents[agent]
    return [i for i in s.needs if i not in s.burned and i not in s.lost]


def held_infos(state: GameState, agent: AgentId) -> set[InfoId]:
    return {kc.info_id for kc in state.agents[agent].keys}


def score(state: GameState, agent: AgentId) -> float:
    """Held open needs / total needs (D3)."""
    s = state.agents[agent]
    if not s.needs:
        return 0.0
    held = held_infos(state, agent)
    return sum(1 for i in open_needs(state, agent) if i in held) / len(s.needs)


def is_winning(state: GameState, agent: AgentId) -> bool:
    on = open_needs(state, agent)
    return bool(on) and set(on) <= held_infos(state, agent)


def departed(state: GameState, agent: AgentId, cfg: GameCfg) -> bool:
    """Has `agent` left the game? (winner_policy = "sit_out" and it has won; always False otherwise)."""
    return cfg.winner_policy == "sit_out" and state.agents[agent].won_round is not None


LEFT_RULE = ("When you hold keys for all your needs you win and leave the game immediately; you take no "
             "more turns and any keys you still hold leave with you. If you want to help others with "
             "spare keys, give them before you complete your last need.")
MIDTURN_RULE = ("If your last need is completed during a turn, the rest of that turn's actions do not "
                "happen.")


def _action_phrase(a: Action) -> str:
    if isinstance(a, GiveKey):
        return f"give_key of info #{a.info_id} to {a.to}"
    if isinstance(a, SendMessage):
        return f"message to {a.to}"
    if isinstance(a, DefectAction):
        return f"defect on {a.target} for info #{a.info_id}"
    if isinstance(a, PostBoard):
        return "board post"
    return a.kind


def midturn_text(skipped: Sequence[Action]) -> str:
    """Notice to an agent that won (and left) mid-turn: which of its actions did not happen."""
    return ("You won and left the game during this turn, so these actions did not happen: "
            + "; ".join(_action_phrase(a) for a in skipped) + ".")


def left_text(agent: AgentId) -> str:
    """The notice every other agent gets when `agent` wins and leaves (winner_policy = "sit_out")."""
    return f"{agent} has won and left the game; it can no longer trade."


def is_done(state: GameState, agent: AgentId) -> bool:
    """Done = has won, or has no open need left (nothing more to gain)."""
    return state.agents[agent].won_round is not None or not open_needs(state, agent)


def _ev(state: GameState, actor: str, data, phase: Phase = "resolve") -> EventBody:
    return EventBody(round=state.round, turn_index=state.turn_index, phase=phase, actor=actor, data=data)


def notice(state: GameState, agent: AgentId, kind, text: str, *, ref: str | None = None,
           deliver_from_turn: int | None = None) -> EventBody:
    """Queue a Notice for `agent` (stamped state.round / state.turn_index) and return its event body."""
    n = Notice(agent=agent, round=state.round, turn_index=state.turn_index, kind=kind, text=text,
               ref=ref, deliver_from_turn=deliver_from_turn)
    state.notices.append(n)
    return _ev(state, "SYSTEM", NoticeData(notice=n))


_notice = notice   # the original private name (rules-internal callers)


def update_win(state: GameState, agent: AgentId, cfg: GameCfg | None = None) -> Events:
    """Set won_round (sticky) if the agent now satisfies D3; returns win + notice events. Under
    winner_policy = "sit_out" the winner also leaves: its notice says so and every other agent gets a
    "left" notice (cfg None = the original "continue" behaviour)."""
    s = state.agents[agent]
    if s.won_round is not None or not is_winning(state, agent):
        return []
    s.won_round = state.round
    won = "You now hold a key for every one of your remaining needs. You have won."
    leaves = cfg is not None and departed(state, agent, cfg)
    if leaves:
        won += " You have left the game: you take no more turns and your keys leave with you."
    evs = [_ev(state, "SYSTEM", WinData(agent=agent, score=score(state, agent))),
           _notice(state, agent, "won", won)]
    if leaves:
        evs += [_notice(state, other, "left", left_text(agent))
                for other in sorted(state.agents) if other != agent]
    return evs


# ------------------------------------------------------------------ messages (shared with God)
def deliver_message(state: GameState, *, channel: str, sender_shown: AgentId, text: str,
                    receiver: AgentId | None = None, true_sender: str | None = None,
                    forged: bool = False, intervention_id: str | None = None,
                    phase: Phase = "resolve") -> tuple[Message, Events]:
    """The way an AGENT's DM/board post enters the game (also usable for forged messages).
    NOTE: adversary.powers.apply_message_fraud builds its Message itself with the same msg_id scheme
    and stamp (state.round, state.turn_index) but appends NO ActionRecord. That is invisible to
    agents (view_log renders messages from state.messages) but state.history differs for forgeries."""
    msg = Message(msg_id=make_msg_id(state.game_id, len(state.messages)), round=state.round,
                  turn_index=state.turn_index, channel=channel, sender_shown=sender_shown,
                  true_sender=true_sender or sender_shown, receiver=receiver, text=text,
                  forged=forged, intervention_id=intervention_id)
    state.messages.append(msg)
    state.history.append(ActionRecord(turn_index=state.turn_index, round=state.round,
                                      kind="dm" if channel == "dm" else "board_post",
                                      actor=sender_shown, target=receiver, msg_id=msg.msg_id))
    return msg, [_ev(state, GOD if forged else sender_shown, MessageData(message=msg), phase)]


# ------------------------------------------------------------------ apply
def _fail(state: GameState, actor: AgentId, action: Action, reason: str) -> Events:
    return [_ev(state, actor, ActionResultData(agent=actor, action=action, ok=False, reason=reason))]


def _ok(state: GameState, actor: AgentId, action: Action) -> EventBody:
    return _ev(state, actor, ActionResultData(agent=actor, action=action, ok=True))


def _name(actor: AgentId, cfg: GameCfg, *, defect: bool = False) -> str:
    ident = cfg.view_log.actor_identity
    if defect and cfg.view_log.defect_actor_identity is not None:
        ident = cfg.view_log.defect_actor_identity
    return actor if ident == "shown" else "Someone"


def apply(state: GameState, actor: AgentId, action: Action, cfg: GameCfg, *,
          copy: bool = False) -> tuple[GameState, Events]:
    """Apply one action of `actor`. Returns (state, event bodies). Never raises on agent input.
    The first event is always the action_result."""
    if copy:
        state = state.model_copy(deep=True)
    if actor not in state.agents:
        return state, _fail(state, actor, action, "unknown_actor")
    if departed(state, actor, cfg):              # sit_out: a winner takes no further action
        return state, _fail(state, actor, action, "actor_left")
    try:
        if isinstance(action, SendMessage):
            return state, _send(state, actor, action, cfg)
        if isinstance(action, PostBoard):
            if not cfg.board.enabled:
                return state, _fail(state, actor, action, "board_disabled")
            text = action.text[: cfg.max_message_chars]
            _, evs = deliver_message(state, channel="board", sender_shown=actor, text=text)
            return state, [_ok(state, actor, action), *evs]
        if isinstance(action, GiveKey):
            return state, _give(state, actor, action, cfg)
        if isinstance(action, DefectAction):
            return state, _defect(state, actor, action, cfg)
        if isinstance(action, Noop):
            return state, [_ok(state, actor, action)]
        if isinstance(action, GrepLog):
            return state, _fail(state, actor, action, "grep_handled_by_engine")
    except (KeyError, ValueError, IndexError) as e:  # defensive: rule closure
        # Not agent input (that is validated above): an internal bug, and the state may already be
        # partly mutated. Log it loudly; SWARM_STRICT_RULES=1 (set by the test suite) re-raises.
        log.error("rules.apply internal error on %s by %s: %r", action.kind, actor, e, exc_info=True)
        if os.environ.get("SWARM_STRICT_RULES") == "1":
            raise
        return state, _fail(state, actor, action, f"error:{type(e).__name__}")
    return state, _fail(state, actor, action, "unknown_action")


def _peer_error(state: GameState, actor: AgentId, other: AgentId) -> str | None:
    if other not in state.agents:
        return "unknown_agent"
    if other == actor:
        return "self_target"
    return None


LEFT_REASONS = ("recipient_left", "target_left")


_WHAT_FAILED = {"send_message": "Your message was not delivered.",
                "give_key": "Your give_key of info #{info} did not happen (you keep the key).",
                "defect": "Your defect had no effect (no penalty)."}


def _network_fail(state: GameState, actor: AgentId, action: Action, other: AgentId) -> Events | None:
    """game.network: refuse an action aimed at an agent the actor may not address (None = allowed)."""
    reason = network.reach_error(state, actor, other)
    if reason is None:
        return None
    what = _WHAT_FAILED.get(action.kind, "").format(info=getattr(action, "info_id", "?"))
    return [*_fail(state, actor, action, reason),
            _notice(state, actor, "system", network.reach_text(state, actor, other, reason, what))]


def _left_fail(state: GameState, actor: AgentId, action: Action, other: AgentId, reason: str) -> Events:
    """Failed action aimed at a departed winner: action_result + a "left" notice to the actor."""
    what = {"send_message": "Your message was not delivered.",
            "give_key": f"Your give_key of info #{getattr(action, 'info_id', '?')} did not happen "
                        "(you keep the key).",
            "defect": "Your defect had no effect (no penalty)."}.get(action.kind, "")
    return [*_fail(state, actor, action, reason),
            _notice(state, actor, "left", f"{left_text(other)} {what}".rstrip())]


def _send(state: GameState, actor: AgentId, a: SendMessage, cfg: GameCfg) -> Events:
    err = _peer_error(state, actor, a.to)
    if err:
        return _fail(state, actor, a, err)
    blocked = _network_fail(state, actor, a, a.to)
    if blocked is not None:
        return blocked
    if departed(state, a.to, cfg):
        return _left_fail(state, actor, a, a.to, "recipient_left")
    _, evs = deliver_message(state, channel="dm", sender_shown=actor, receiver=a.to,
                             text=a.text[: cfg.max_message_chars])
    return [_ok(state, actor, a), *evs]


def _give(state: GameState, actor: AgentId, a: GiveKey, cfg: GameCfg) -> Events:
    err = _peer_error(state, actor, a.to)
    if err:
        return _fail(state, actor, a, err)
    blocked = _network_fail(state, actor, a, a.to)
    if blocked is not None:
        return blocked
    if departed(state, a.to, cfg):
        return _left_fail(state, actor, a, a.to, "recipient_left")
    src = state.agents[actor]
    idx = next((j for j, kc in enumerate(src.keys) if kc.info_id == a.info_id), None)
    if idx is None:
        return _fail(state, actor, a, "no_such_key")
    kc = src.keys.pop(idx)
    state.agents[a.to].keys.append(kc)
    state.history.append(ActionRecord(turn_index=state.turn_index, round=state.round, kind="give",
                                      actor=actor, target=a.to, info_id=a.info_id))
    evs = [_ok(state, actor, a),
           _ev(state, actor, KeyTransferData(copy_id=kc.copy_id, info_id=kc.info_id, from_agent=actor,
                                             to_agent=a.to, cause="give")),
           _notice(state, a.to, "gift_received",
                   f"{_name(actor, cfg)} gave you a key for info #{kc.info_id} (key {kc.key}).")]
    return evs + update_win(state, a.to, cfg)


def _defect(state: GameState, actor: AgentId, a: DefectAction, cfg: GameCfg) -> Events:
    err = _peer_error(state, actor, a.target)
    if err:
        return _fail(state, actor, a, err)
    blocked = _network_fail(state, actor, a, a.target)
    if blocked is not None:
        return blocked
    tgt = state.agents[a.target]
    if departed(state, a.target, cfg):
        return _left_fail(state, actor, a, a.target, "target_left")
    if tgt.won_round is not None:
        return _fail(state, actor, a, "target_won")
    vl = cfg.view_log
    n = cfg.n_agents
    t_d = state.turn_index
    correct = a.info_id in open_needs(state, a.target)
    mode = vl.defect_visibility
    release = defect_release_turn(t_d, vl.defect_delay_rounds, n) \
        if mode in ("delayed", "victim_then_delayed") else None
    # victim-timed notices: next observation (victim_* / public) or at release (delayed)
    victim_dft = release if mode == "delayed" else None
    who = _name(actor, cfg, defect=True)
    me = state.agents[actor]
    evs: Events = [_ok(state, actor, a)]
    lost: InfoId | None = None
    delayed_until: int | None = None
    if correct:
        tgt.burned.append(a.info_id)
        evs.append(_notice(state, a.target, "burned",
                           f"{who} burned your need for info #{a.info_id}: you can never get info "
                           f"#{a.info_id} now.", ref=f"burn:{t_d}", deliver_from_turn=victim_dft))
        gain_text = ""
        if cfg.defect_gain == "mint":
            n_copies = sum(1 for s in state.agents.values() for k in s.keys if k.info_id == a.info_id)
            kc = KeyCopy(copy_id=f"m{a.info_id}-{t_d}-{n_copies}", info_id=a.info_id,
                         key=_key_value(state, a.info_id), origin="defect_mint")
            me.keys.append(kc)
            evs.append(_ev(state, actor, KeyTransferData(copy_id=kc.copy_id, info_id=a.info_id,
                                                         from_agent=None, to_agent=actor,
                                                         cause="defect_mint")))
            gain_text = f" You gained a key for info #{a.info_id} (key {kc.key})."
        elif cfg.defect_gain == "steal":
            holder = next((h for h in sorted(state.agents) if h != actor
                           and not departed(state, h, cfg)       # its keys left the game with it
                           and any(k.info_id == a.info_id for k in state.agents[h].keys)), None)
            if holder is not None:
                hk = state.agents[holder].keys
                kc = hk.pop(next(j for j, k in enumerate(hk) if k.info_id == a.info_id))
                me.keys.append(kc)
                evs.append(_ev(state, actor, KeyTransferData(copy_id=kc.copy_id, info_id=a.info_id,
                                                             from_agent=holder, to_agent=actor,
                                                             cause="defect_steal")))
                evs.append(_notice(state, holder, "system",
                                   f"{who} took your key for info #{a.info_id}.",
                                   ref=f"steal:{t_d}", deliver_from_turn=victim_dft))
                gain_text = f" You took a key for info #{a.info_id} (key {kc.key})."
        evs.append(_notice(state, actor, "defect_result",
                           f"Your burn of info #{a.info_id} on {a.target} was correct.{gain_text}"))
    else:
        evs.append(_notice(state, a.target, "defect_attempt",
                           f"{who} tried to burn info #{a.info_id} of yours; it had no effect on you.",
                           ref=f"burn:{t_d}", deliver_from_turn=victim_dft))
        if cfg.wrong_defect_penalty == "lose_need":
            on = open_needs(state, actor)
            if on:
                lost = on[0]
                me.lost.append(lost)
            pen = f" You permanently lost your need for info #{lost}." if lost is not None else ""
        else:
            delayed_until = state.round + cfg.delay_n
            me.delayed_until_round = delayed_until
            pen = f" You are delayed: no turns until after round {delayed_until}."
        evs.append(_notice(state, actor, "defect_result",
                           f"Your burn of info #{a.info_id} on {a.target} was wrong.{pen}"))
    state.history.append(ActionRecord(turn_index=t_d, round=state.round, kind="defect", actor=actor,
                                      target=a.target, info_id=a.info_id, ok=correct,
                                      release_turn=release))
    evs.append(_ev(state, actor, BurnData(
        actor=actor, target=a.target, info_id=a.info_id, correct=correct,
        penalty="none" if correct else cfg.wrong_defect_penalty, lost_info_id=lost,
        delayed_until_round=delayed_until, gain=cfg.defect_gain, release_turn=release)))
    for ag in (actor, a.target):
        evs += update_win(state, ag, cfg)
    return evs


def _key_value(state: GameState, info_id: InfoId) -> int:
    """The secret key value of an info (all copies share it). Minted copies reuse it."""
    for s in state.agents.values():
        for kc in s.keys:
            if kc.info_id == info_id:
                return kc.key
    return 0
