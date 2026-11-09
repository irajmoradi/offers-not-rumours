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
- game.network.connect (off by default): connect(target) adds the edge actor-target to the live graph
  (network.connect_error lists the refusals; each refusal also sends the actor a "system" notice and
  costs nothing). A success appends a "connect" ActionRecord, emits an edge_added event, sends the
  target a "connected" notice, and costs the actor its turns in rounds r+1..r+cost_rounds
  (delayed_until_round, like board.post_skip_rounds; skip notice now, resume notice at the first slot
  of round r+cost_rounds+1). The whole-turn rule is in game/resolve.py.
Game-fixes switches (CONTRACTS §9f; all default off = the rules above, byte-identical):
- winner_policy = "lock_and_stay": the score is LOCKED at the win (`score(..., cfg)` = open needs at
  the win / needs; nothing after the win can change the open needs, see below) and the winner STAYS:
  it keeps its turns and may give any key (including the ones it needed), its keys stay in play, it
  cannot be defected on ("target_won", as under continue) and it cannot defect itself ("actor_won",
  no penalty), so its own actions can never alter its locked score either.
- wrong_defect_penalty = "lose_held_need": a wrong defect loses the actor's lowest-id open need that it
  currently HOLDS a key for (the key stays with it as a spare; BurnData.penalty "lose_held_need"); if it
  holds none of its open needs, it loses its lowest-id open need as under lose_need (BurnData.penalty
  "lose_need"). So a wrong defect costs a held need whenever the actor has one, is never cheaper than
  under lose_need, and can never create a win (lose_need could burn the only missing need). A `delay`
  fallback was tried and rejected: turn delays are almost free here (receiving is passive), and in the
  scripted simulation a blind defector among cooperators then scored 1.00 instead of 0.23.
- wrong_defect_penalty = "score_penalty": a wrong defect adds game.defect.score_penalty (None = 1/needs)
  to AgentState.score_penalty (no floor), subtracted from the PRIVATE score (game.scoring) and the
  score the agent is shown, never from `score` / welfare; no need is lost, so it can never create a win. "lose_held_key": the actor's most-demanded spare (not one of its open needs) goes to
  the target (key_transfer cause defect_penalty); with no spare, score_penalty. No need is lost.
- game.defect.win_rule = "all_needs": a win needs every original need held (none burned or lost);
  is_done also counts an agent that holds all its open needs after a burn.
- game.view_log.defect_broadcast (gossip knob; off by default): `defect_broadcast` (called by the
  engine at each round start) sends EVERY agent an anonymous "system" notice about the defects of round
  r - delay_rounds (round_summary: one count notice if N > 0; per_defect: one notice per defect), with
  ref "defect_broadcast:r<r>[:k]". It never names attacker or victim and leaves every other notice and
  log line unchanged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

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
    ConnectAction,
    DefectAction,
    EdgeAddedData,
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
    TradeOffer,
    WinData,
    defect_release_turn,
    turn_index,
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


def score_locked(cfg: GameCfg | None) -> bool:
    """Is a winner's score locked at its win while it stays in play? (winner_policy lock_and_stay)"""
    return cfg is not None and cfg.winner_policy == "lock_and_stay"


def score(state: GameState, agent: AgentId, cfg: GameCfg | None = None) -> float:
    """Held open needs / total needs (D3). With a locked-score winner policy (score_locked(cfg)) a
    winner scores its open needs at the win / needs, whatever it gives away afterwards: its open needs
    cannot change after the win (it cannot be defected on and cannot defect)."""
    s = state.agents[agent]
    if not s.needs:
        return 0.0
    if s.won_round is not None and score_locked(cfg):
        return len(open_needs(state, agent)) / len(s.needs)
    held = held_infos(state, agent)
    return sum(1 for i in open_needs(state, agent) if i in held) / len(s.needs)


def penalty(state: GameState, agent: AgentId) -> float:
    """Score points taken by wrong defects under score_penalty / lose_held_key (0 if none). NOT part of
    `score` (the needs score: round scores, wins and Outcome.welfare stay needs-only); it is subtracted
    in the private score (game.scoring) and in the score the agent is shown (observe pinned.score)."""
    return state.agents[agent].score_penalty or 0.0


def win_all_needs(cfg: GameCfg | None) -> bool:
    """game.defect.win_rule = all_needs: a win needs every ORIGINAL need held (none burned or lost)."""
    return cfg is not None and cfg.defect.win_rule == "all_needs"


def is_winning(state: GameState, agent: AgentId, cfg: GameCfg | None = None) -> bool:
    """D3: holds a key for every open need and has one. win_rule all_needs: no need burned or lost."""
    on = open_needs(state, agent)
    if win_all_needs(cfg) and len(on) < len(state.agents[agent].needs):
        return False
    return bool(on) and set(on) <= held_infos(state, agent)


def wrong_score_penalty(state: GameState, agent: AgentId, cfg: GameCfg) -> float:
    """Score points one wrong defect costs under score_penalty (game.defect.score_penalty, None = one
    need of this agent = 1 / needs)."""
    if cfg.defect.score_penalty is not None:
        return cfg.defect.score_penalty
    return 1.0 / max(len(state.agents[agent].needs), 1)


def departed(state: GameState, agent: AgentId, cfg: GameCfg) -> bool:
    """Has `agent` left the game? (winner_policy = "sit_out" and it has won; always False otherwise)."""
    return cfg.winner_policy == "sit_out" and state.agents[agent].won_round is not None


LEFT_RULE = ("When you hold keys for all your needs you win and leave the game immediately; you take no "
             "more turns and any keys you still hold leave with you. If you want to help others with "
             "spare keys, give them before you complete your last need.")
MIDTURN_RULE = ("If your last need is completed during a turn, the rest of that turn's actions do not "
                "happen.")
LOCK_RULE = ("When you hold keys for all your needs you win, and your score is locked: nothing can "
             "lower it afterwards. You stay in the game: you keep your turns and can keep helping others "
             "by giving them keys, any of your keys, including the ones you needed. Nobody can defect on "
             "a player who has won, and a player who has won cannot defect.")
WON_LOCKED = (" Your score is now locked: nothing can lower it. You stay in the game and can keep helping "
              "others by giving them any of your keys.")



def _action_phrase(a: Action) -> str:
    if isinstance(a, GiveKey):
        return f"give_key of info #{a.info_id} to {a.to}"
    if isinstance(a, SendMessage):
        return f"message to {a.to}"
    if isinstance(a, DefectAction):
        return f"defect on {a.target} for info #{a.info_id}"
    if isinstance(a, PostBoard):
        return "board post"
    if isinstance(a, ConnectAction):
        return f"connect with {a.target}"
    if isinstance(a, TradeOffer):
        return f"trade offer to {a.target}"
    return a.kind


def midturn_text(skipped: Sequence[Action]) -> str:
    """Notice to an agent that won (and left) mid-turn: which of its actions did not happen."""
    return ("You won and left the game during this turn, so these actions did not happen: "
            + "; ".join(_action_phrase(a) for a in skipped) + ".")


def left_text(agent: AgentId) -> str:
    """The notice every other agent gets when `agent` wins and leaves (winner_policy = "sit_out")."""
    return f"{agent} has won and left the game; it can no longer trade."


def is_done(state: GameState, agent: AgentId, cfg: GameCfg | None = None) -> bool:
    """Done = has won, or has no open need left (nothing more to gain). win_rule all_needs: also an
    agent that holds every open need but lost or had burned one (it can no longer win; under the
    original rule it would have won here, so games end when they used to)."""
    on = open_needs(state, agent)
    if state.agents[agent].won_round is not None or not on:
        return True
    return win_all_needs(cfg) and set(on) <= held_infos(state, agent)


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
    if s.won_round is not None or not is_winning(state, agent, cfg):
        return []
    s.won_round = state.round
    won = "You now hold a key for every one of your remaining needs. You have won."
    leaves = cfg is not None and departed(state, agent, cfg)
    if leaves:
        won += " You have left the game: you take no more turns and your keys leave with you."
    elif score_locked(cfg):
        won += WON_LOCKED
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
        if isinstance(action, ConnectAction):
            return state, _connect(state, actor, action, cfg)
        if isinstance(action, TradeOffer):                    # game.trade (resolve passes the offer id)
            from swarm.game import trade
            return state, trade.offer(state, actor, action, cfg, f"o{state.turn_index}-0")
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
OWN_NOTICE_REASONS = ("actor_won",)   # rules already sent the actor a notice


_WHAT_FAILED = {"send_message": "Your message was not delivered.",
                "give_key": "Your give_key of info #{info} did not happen (you keep the key).",
                "defect": "Your defect had no effect (no penalty).",
                "offer_trade": "Your trade offer was not made."}


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
    if score_locked(cfg) and state.agents[actor].won_round is not None:   # its locked score must not move
        return [*_fail(state, actor, a, "actor_won"),
                _notice(state, actor, "system", "You have won, so you cannot defect (your score is locked). "
                                                "Your defect had no effect (no penalty).")]
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
    held_lost = False                    # lose_held_need: the lost need was one the actor held
    key_lost = False                     # lose_held_key: a spare went to the target
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
        elif cfg.wrong_defect_penalty == "lose_held_need":
            held = held_infos(state, actor)
            mine = [i for i in open_needs(state, actor) if i in held]
            on = open_needs(state, actor)
            if mine:
                lost, held_lost = mine[0], True
                pen = (f" You permanently lost your need for info #{lost}, one you held: you keep its key, "
                       "but it no longer counts for you.")
            elif on:                                   # holds none: the original lose_need
                lost = on[0]
                pen = (f" You held none of your needs, so you permanently lost your need for info "
                       f"#{lost}.")
            else:
                pen = ""
            if lost is not None:
                me.lost.append(lost)
        elif cfg.wrong_defect_penalty in ("score_penalty", "lose_held_key"):
            spare = (_penalty_key(state, actor, a.target)
                     if cfg.wrong_defect_penalty == "lose_held_key" else None)
            if spare is not None:                      # lose_held_key: the spare goes to the target
                hk = me.keys
                kc = hk.pop(next(j for j, k in enumerate(hk) if k.copy_id == spare.copy_id))
                tgt.keys.append(kc)
                key_lost = True
                evs.append(_ev(state, actor, KeyTransferData(copy_id=kc.copy_id, info_id=kc.info_id,
                                                             from_agent=actor, to_agent=a.target,
                                                             cause="defect_penalty")))
                evs.append(_notice(state, a.target, "system",
                                   f"You received a key for info #{kc.info_id} (key {kc.key}) from {who} as "
                                   "the penalty for its wrong burn attempt on you.",
                                   ref=f"burn:{t_d}", deliver_from_turn=victim_dft))
                pen = (f" You lost your key for info #{kc.info_id} (key {kc.key}): it went to {a.target}. "
                       "None of your needs is removed.")
            else:
                amt = wrong_score_penalty(state, actor, cfg)
                me.score_penalty = (me.score_penalty or 0.0) + amt
                pen = (f" Your score drops by {amt:.2f}; none of your needs is removed." if
                       cfg.wrong_defect_penalty == "score_penalty" else
                       f" You held no spare key, so your score drops by {amt:.2f}; none of your needs is "
                       "removed.")
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
        penalty=_penalty_label(correct, cfg, held_lost, key_lost), lost_info_id=lost,
        delayed_until_round=delayed_until, gain=cfg.defect_gain, release_turn=release)))
    for ag in (actor, a.target):
        evs += update_win(state, ag, cfg)
    return evs


BROADCAST_REF = "defect_broadcast"   # Notice.ref prefix of game.view_log.defect_broadcast notices


def broadcast_texts(n_defects: int, src_round: int, mode: str) -> list[str]:
    """The defect_broadcast notice texts for n_defects defects resolved in src_round (no names)."""
    if n_defects <= 0:
        return []
    if mode == "per_defect":
        return [f"A defect occurred in round {src_round}."] * n_defects
    noun = "defect" if n_defects == 1 else "defects"
    return [f"In round {src_round}, {n_defects} {noun} occurred somewhere in the game."]


def defect_broadcast(state: GameState, cfg: GameCfg) -> Events:
    """game.view_log.defect_broadcast: called by the engine at the start of round state.round (before
    any slot; state.turn_index = the previous round's last slot, so every notice is new at each
    agent's next observation). Announces the defects resolved in round state.round - delay_rounds
    (state.history, correct and wrong) to EVERY agent, whatever the network: departed winners too
    (they no longer observe, so it only sits in state.notices). [] when off or nothing to announce."""
    b = cfg.view_log.defect_broadcast
    src = state.round - b.delay_rounds
    if not b.enabled or src < 1:
        return []
    n_def = sum(1 for h in state.history if h.kind == "defect" and h.round == src)
    texts = broadcast_texts(n_def, src, b.mode)
    evs: Events = []
    for agent in sorted(state.agents):
        for k, text in enumerate(texts):
            ref = f"{BROADCAST_REF}:r{src}" + (f":{k}" if b.mode == "per_defect" else "")
            evs.append(_notice(state, agent, "system", text, ref=ref))
    return evs


REFUSAL_REF = "refusal_broadcast"


def refusal_broadcast(state: GameState, cfg: GameCfg, attacks: dict[str, Any]) -> Events:
    """game.view_log.refusal_broadcast (Noor tie-in): called by the engine at the start of round state.round
    (stamped like defect_broadcast). For every God DM (state.messages forged=True) delivered in round
    m = state.round - window_rounds whose receiver did not act on it in rounds m .. state.round - 1, each
    current contact of the receiver (not the receiver) gets the anonymous notice; one per viewer per round.
    attacks: intervention_id -> the applied GodAction (the engine records them from god_power events);
    a fraud's act = a give receiver -> shown sender, a tip's act = a defect receiver -> revealed target."""
    b = cfg.view_log.refusal_broadcast
    m = state.round - b.window_rounds
    if not b.enabled or m < 1:
        return []
    acts = {(h.kind, h.actor, h.target) for h in state.history if m <= h.round < state.round}
    viewers: dict[AgentId, str] = {}
    for msg in state.messages:
        if not msg.forged or msg.round != m or msg.receiver is None:
            continue
        a = attacks.get(msg.intervention_id or "")
        kind = getattr(a, "kind", None)
        if kind == "message_fraud":
            acted = ("give", msg.receiver, msg.sender_shown) in acts
        elif kind == "reveal_needs":
            acted = ("defect", msg.receiver, a.target) in acts
        else:
            continue
        if acted:
            continue
        for v in sorted(state.agents[msg.receiver].contacts or ()):
            if v != msg.receiver and v not in viewers:
                viewers[v] = msg
    return [_notice(state, v, "system", b.text.replace("{sender}", m.sender_shown), ref=f"{REFUSAL_REF}:{m.msg_id}")
            for v, m in sorted(viewers.items())]


NORM_REF = "norm_notice"


def norm_notice(state: GameState, cfg: GameCfg) -> Events:
    """game.view_log.norm_notice (Noor tie-in): the 'verify first' notice to every agent at the start of
    start_round (and every every_rounds after). [] when off or not due."""
    b = cfg.view_log.norm_notice
    r = state.round
    if not b.enabled or r < b.start_round:
        return []
    if r != b.start_round and (b.every_rounds is None or (r - b.start_round) % b.every_rounds):
        return []
    return [_notice(state, a, "system", b.text, ref=f"{NORM_REF}:r{r}") for a in sorted(state.agents)]


def _penalty_label(correct: bool, cfg: GameCfg, held_lost: bool, key_lost: bool = False) -> str:
    """BurnData.penalty: none (hit) / the configured penalty; lose_held_need that found no held need
    fell back to the original lose_need; lose_held_key without a spare fell back to score_penalty."""
    if correct:
        return "none"
    if cfg.wrong_defect_penalty == "lose_held_need" and not held_lost:
        return "lose_need"
    if cfg.wrong_defect_penalty == "lose_held_key" and not key_lost:
        return "score_penalty"
    return cfg.wrong_defect_penalty


def demand(state: GameState, info: InfoId, exclude: AgentId) -> int:
    """Agents other than `exclude` that have not won and miss `info` (an open need they hold no key for)."""
    return sum(1 for a, s in state.agents.items() if a != exclude and s.won_round is None
               and info in open_needs(state, a) and info not in held_infos(state, a))


def _penalty_key(state: GameState, actor: AgentId, target: AgentId) -> KeyCopy | None:
    """lose_held_key: the actor's MOST-DEMANDED spare (a key copy whose info is not one of its open
    needs; demand = other non-winners missing it; ties: lowest info id, then copy id), or None. The
    most-demanded spare is the actor's most valuable trade chip, so the penalty bites where a key is
    worth something; the choice is deterministic (rules use no randomness) and keeps the copy in play."""
    mine = set(open_needs(state, actor))
    spares = [k for k in state.agents[actor].keys if k.info_id not in mine]
    if not spares:
        return None
    return min(spares, key=lambda k: (-demand(state, k.info_id, actor), k.info_id, k.copy_id))


def _connect(state: GameState, actor: AgentId, a: ConnectAction, cfg: GameCfg) -> Events:
    """game.network.connect: add the edge actor-target (network module docstring for the refusals)."""
    gone = a.target in state.agents and departed(state, a.target, cfg)
    reason = network.connect_error(state, actor, a.target, cfg, departed=gone)
    if reason is not None:
        return [*_fail(state, actor, a, reason),
                _notice(state, actor, "left" if reason == "target_left" else "system",
                        network.connect_fail_text(a.target, reason, cfg))]
    mutual = not network.add_edge(state, actor, a.target)
    k = cfg.network.connect.cost_rounds
    me = state.agents[actor]
    last = state.round + k
    if k > 0:
        me.delayed_until_round = max(me.delayed_until_round, last)
    state.history.append(ActionRecord(turn_index=state.turn_index, round=state.round, kind="connect",
                                      actor=actor, target=a.target))
    now, resume = network.connect_cost_texts(actor, a.target, state.round, k, mutual)
    evs = [_ok(state, actor, a),
           _ev(state, actor, EdgeAddedData(actor=actor, target=a.target, mutual=mutual,
                                           skip_until_round=me.delayed_until_round))]
    if not mutual:
        evs.append(_notice(state, a.target, "connected", network.connected_text(actor)))
    evs.append(_notice(state, actor, "system", now))
    if resume is not None:
        evs.append(_notice(state, actor, "system", resume,
                           deliver_from_turn=turn_index(last + 1, 0, len(state.agents))))
    return evs


def _key_value(state: GameState, info_id: InfoId) -> int:
    """The secret key value of an info (all copies share it). Minted copies reuse it."""
    for s in state.agents.values():
        for kc in s.keys:
            if kc.info_id == info_id:
                return kc.key
    return 0
