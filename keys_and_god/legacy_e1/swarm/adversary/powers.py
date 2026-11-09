"""God's powers (WP4): Message Fraud and CoT Control, plus the budget K.

Every power goes through `apply_power` (or the two `apply_*` functions it dispatches to). They are
the ONLY code that spends `GameState.god_budget_left`, and `apply_cot_control` is the only caller of
`MemoryStore.overwrite`. Each call mutates `state` in place and returns a `PowerResult` holding the
EventBodies to emit. Those events carry the ground truth:
  - `message` (Message.forged=True, true_sender="GOD", intervention_id) for fraud,
  - `memory_overwrite` (old_text = the original, new_text, intervention_id) for CoT control,
  - `god_power` (the action incl. its target, cost, budget_left, ok/reason) for both.
A failed power (disabled, bad target, a cot_control mode other than the configured one, over
budget, a fraud naming an agent that has left the game under game.winner_policy=sit_out
(`winners_leave=True`: "sender_left" / "receiver_left"), a fraud along a non-edge of the
game.network graph in mode contacts ("not_connected"), ...) spends nothing, creates no intervention id,
and emits one `god_power` event with ok=False.

Indistinguishability (REVIEW_v2 leak 1): a forged DM is built exactly the way a real DM sent in the
slot about to be played would be: msg_id = make_msg_id(game_id, len(state.messages)), round =
state.round, turn_index = state.turn_index, text truncated to the same `max_message_chars`. So every
agent-visible field (and the hidden stamp) matches a real DM; only the truth fields differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm.config import GodCfg
from swarm.memory import MemoryStore
from swarm.rng import make_intervention_id, make_msg_id
from swarm.schema import (
    GOD,
    CotControl,
    EventBody,
    GameState,
    GodAction,
    GodPowerData,
    MemoryEntry,
    Message,
    MessageData,
    MessageFraud,
)


@dataclass(frozen=True)
class PowerResult:
    ok: bool
    reason: str | None
    intervention_id: str | None
    events: list[EventBody] = field(default_factory=list)
    message: Message | None = None       # fraud: the forged message
    entry: MemoryEntry | None = None     # cot_control: the tampered entry


# ------------------------------------------------------------------ budget
def init_budget(state: GameState, cfg: GodCfg) -> None:
    """Call once at game start (both scopes)."""
    state.god_budget_left = cfg.budget.k


def refresh_budget(state: GameState, cfg: GodCfg, last_refresh_round: int | None) -> int | None:
    """Call at every God wake. scope=round: reset to k at the first wake of each round.
    scope=game: no-op. Returns the new `last_refresh_round` for the caller to keep."""
    if cfg.budget.scope == "round" and state.round != last_refresh_round:
        state.god_budget_left = cfg.budget.k
        return state.round
    return last_refresh_round


# ------------------------------------------------------------------ helpers
def _body(state: GameState, data: object) -> EventBody:
    return EventBody(round=state.round, turn_index=state.turn_index, phase="god", actor=GOD, data=data)


def _fail(state: GameState, action: GodAction, power: str, reason: str) -> PowerResult:
    data = GodPowerData(power=power, intervention_id=None, action=action, cost=0,
                        budget_left=state.god_budget_left, ok=False, reason=reason)
    return PowerResult(ok=False, reason=reason, intervention_id=None, events=[_body(state, data)])


def _new_intervention(state: GameState, cost: int) -> str:
    iid = make_intervention_id(state.game_id, state.round, state.n_interventions)
    state.n_interventions += 1
    state.god_budget_left -= cost
    return iid


def _not_an_edge(state: GameState, sender: str, receiver: str) -> bool:
    """game.network mode contacts (AgentState.contacts set, known unset; game/network.py): agents
    only ever DM their graph neighbours, so God can only forge a DM along an edge. known_ids and no
    network put no limit on God."""
    r = state.agents[receiver]
    return r.contacts is not None and r.known is None and sender not in r.contacts


# ------------------------------------------------------------------ Message Fraud
def apply_message_fraud(state: GameState, action: MessageFraud, cfg: GodCfg, *,
                        max_message_chars: int = 500, winners_leave: bool = False) -> PowerResult:
    """Deliver a DM to `action.receiver` that appears to come from `action.sender`.

    Pass `max_message_chars = cfg.game.max_message_chars` so the forged text is truncated exactly
    like a real DM (otherwise its length could give it away). Pass `winners_leave =
    (cfg.game.winner_policy == "sit_out")`: then a sender or receiver that has won has left the game,
    and the fraud fails ("sender_left" / "receiver_left", nothing spent): a DM from an agent everyone
    was told has left would be impossible, and a departed receiver never reads it."""
    pcfg = cfg.powers.message_fraud
    if not pcfg.enabled:
        return _fail(state, action, "message_fraud", "power_disabled")
    if action.receiver not in state.agents:
        return _fail(state, action, "message_fraud", "unknown_receiver")
    if action.sender not in state.agents:
        return _fail(state, action, "message_fraud", "unknown_sender")
    if action.sender == action.receiver:
        return _fail(state, action, "message_fraud", "self_message")
    if _not_an_edge(state, action.sender, action.receiver):
        return _fail(state, action, "message_fraud", "not_connected")
    if winners_leave and state.agents[action.sender].won_round is not None:
        return _fail(state, action, "message_fraud", "sender_left")
    if winners_leave and state.agents[action.receiver].won_round is not None:
        return _fail(state, action, "message_fraud", "receiver_left")
    if not action.text.strip():
        return _fail(state, action, "message_fraud", "empty_text")
    if state.god_budget_left < pcfg.cost:
        return _fail(state, action, "message_fraud", "over_budget")

    iid = _new_intervention(state, pcfg.cost)
    msg = Message(msg_id=make_msg_id(state.game_id, len(state.messages)), round=state.round,
                  turn_index=state.turn_index, channel="dm", sender_shown=action.sender,
                  true_sender=GOD, receiver=action.receiver, text=action.text[:max_message_chars],
                  forged=True, intervention_id=iid)
    state.messages.append(msg)
    power = GodPowerData(power="message_fraud", intervention_id=iid, action=action, cost=pcfg.cost,
                         budget_left=state.god_budget_left, ok=True)
    return PowerResult(ok=True, reason=None, intervention_id=iid, message=msg,
                       events=[_body(state, power), _body(state, MessageData(message=msg))])


# ------------------------------------------------------------------ CoT Control
def apply_cot_control(state: GameState, action: CotControl, cfg: GodCfg, memory: MemoryStore) -> PowerResult:
    """Replace (or append to) the target agent's memory entry for `action.round` via
    MemoryStore.overwrite. A missing entry is created (the PDF lets God *add* CoT). The text is
    capped like an agent write of that kind, so its length gives nothing away."""
    pcfg = cfg.powers.cot_control
    if not pcfg.enabled:
        return _fail(state, action, "cot_control", "power_disabled")
    if action.target not in pcfg.targets:
        return _fail(state, action, "cot_control", "target_kind_disabled")
    if action.mode != pcfg.mode:                  # the edit mode is a config knob, not God's choice
        return _fail(state, action, "cot_control", "mode_disabled")
    if action.agent not in state.agents:
        return _fail(state, action, "cot_control", "unknown_agent")
    if action.round < 1:
        return _fail(state, action, "cot_control", "bad_round")
    if action.round > state.round:
        return _fail(state, action, "cot_control", "future_round")
    if state.god_budget_left < pcfg.cost:
        return _fail(state, action, "cot_control", "over_budget")

    text = action.text
    if action.target == "scratchpad":
        text = text[: memory.cfg.scratchpad.max_chars]
    elif action.target == "cot":
        text = text[: memory.cfg.max_cot_chars]
    iid = _new_intervention(state, pcfg.cost)
    entry, ow_body = memory.overwrite(state, action.agent, action.target, action.round, text,
                                      intervention_id=iid, now_round=state.round, mode=action.mode)
    power = GodPowerData(power="cot_control", intervention_id=iid, action=action, cost=pcfg.cost,
                         budget_left=state.god_budget_left, ok=True)
    return PowerResult(ok=True, reason=None, intervention_id=iid, entry=entry,
                       events=[_body(state, power), ow_body])


# ------------------------------------------------------------------ dispatch
def apply_power(state: GameState, action: GodAction, cfg: GodCfg, memory: MemoryStore, *,
                max_message_chars: int = 500, winners_leave: bool = False) -> PowerResult:
    if isinstance(action, MessageFraud):
        return apply_message_fraud(state, action, cfg, max_message_chars=max_message_chars,
                                   winners_leave=winners_leave)
    if isinstance(action, CotControl):
        return apply_cot_control(state, action, cfg, memory)
    raise TypeError(f"not a God power: {action.kind}")


def is_power(action: GodAction) -> bool:
    return isinstance(action, (MessageFraud, CotControl))
