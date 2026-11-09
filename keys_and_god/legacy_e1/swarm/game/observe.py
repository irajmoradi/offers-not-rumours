"""Agent observations (WP1b): the ONLY path from GameState (truth) to what an agent may know.

`observe(state, agent, cfg, turn_idx)` builds an `Observation` field by field from whitelisted view
models. No truth field (forged, true_sender, intervention ids, tamper flags, copy ids, refs, release
turns) is ever copied, and forged messages render exactly like real ones (the shown sender).

Timing (all by global turn index t; `prev` = AgentState.last_obs_turn, -1 before the first one):
- A record stamped at slot s is first visible to observations with index > s.
- inbox / board / notices: "new since your previous observation" = prev <= s < t.
  Notices with `deliver_from_turn` d use schema.release_visible(d, t, prev) instead (shown once).
- view_log: the rolling window schema.log_visible(s, t, window_turns) (window = n_agents by default)
  over give / board_post (+ dm_meta / dm under actions_msg_meta / full; never under actions_only).
  Message lines are rendered from state.messages (not ActionRecords) because God's forged DMs are
  appended by adversary.powers without an ActionRecord; this keeps forged DMs indistinguishable.
- Defects (D12): `public` -> in the rolling window like any action. `victim_only` -> never in the log.
  `delayed` -> every viewer gets the line once, at release (release_visible). `victim_then_delayed`
  -> every viewer except the victim gets it once at release (the victim got a notice at t_d+1).
  Wrong defects follow the same path (ok=False on the line).
- content="none": empty log, grep disabled.
- game.board.visible_delay_rounds = k: a board post stamped at slot s is treated as stamped at
  s + k * n_agents (k rounds later) everywhere an agent can see it (board view, view_log window,
  grep_log history). k = 0 is the original rule. (game.board.enabled = false: no posts exist, since
  post_board fails "board_disabled".)
- game.board.anonymous: board posts carry no author anywhere an agent can see them (board view
  sender = ANON_SENDER, board_post log lines with actor=None, which also covers grep_log and the
  context built from observations). Truth (Message.sender_shown / true_sender) is untouched.

- game.network (game/network.py; off = unchanged): pinned peers = the agent's contacts (mode contacts)
  or the names it knows, including those it learns from this very observation (known_ids), minus
  departed winners; give / defect / DM log lines only when local (network.line_visible).
  game.board.scope = neighbors: board view and board_post lines only for posts by graph neighbours.

`grep_log` searches the agent's VISIBLE log history: the union of the windows of all its past
observations (`obs_turns`, tracked by the engine) plus released defect lines, matched as a
case-insensitive literal substring of the rendered line (plus " #<info_id>").

The engine calls `mark_observed(state, agent, t)` after each observation; `observe` itself is pure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from swarm.config import GameCfg
from swarm.context.base import render_log_line
from swarm.game import network
from swarm.game.rules import departed, open_needs, score
from swarm.schema import (
    ActionRecord,
    AgentId,
    AgentMessageView,
    GameState,
    GrepResult,
    KeyView,
    LogLine,
    Message,
    NoticeView,
    Observation,
    PinnedFacts,
    log_visible,
    release_visible,
)

ANON_SENDER = "(anonymous)"   # board-view sender under game.board.anonymous


def _new_since(stamp: int, viewer_turn: int, prev: int) -> bool:
    return prev <= stamp < viewer_turn


def _shown_stamp(m: Message, cfg: GameCfg) -> int:
    """The slot from which agents treat message m as posted (board posts: + visible_delay_rounds)."""
    if m.channel == "board" and cfg.board.visible_delay_rounds:
        return m.turn_index + cfg.board.visible_delay_rounds * cfg.n_agents
    return m.turn_index


def _msg_view(m: Message, cfg: GameCfg) -> AgentMessageView:
    text = m.text
    if cfg.view_log.reveal_provenance and m.forged:          # ablation only
        text = "[FORGED] " + text
    sender = ANON_SENDER if (m.channel == "board" and cfg.board.anonymous) else m.sender_shown
    return AgentMessageView(msg_id=m.msg_id, channel=m.channel, sender=sender,
                            receiver=m.receiver, text=text)


def pinned_facts(state: GameState, agent: AgentId, cfg: GameCfg,
                 learned: set[AgentId] | frozenset[AgentId] = frozenset()) -> PinnedFacts:
    """Identity, needs, holdings (with key ids, D15), score. Rebuilt from state every turn.
    peers = the players still in the game (winner_policy=sit_out: departed winners are dropped);
    with game.network on, only those the agent may address (its contacts, or the names it knows
    plus `learned`, the names it learns from the observation being built)."""
    s = state.agents[agent]
    may = network.visible_peers(state, agent)
    pool = state.agents if may is None else may | set(learned)
    holdings = sorted((KeyView(info_id=k.info_id, key=k.key) for k in s.keys),
                      key=lambda kv: (kv.info_id, kv.key))
    return PinnedFacts(
        agent=agent, round=state.round, max_rounds=cfg.max_rounds,
        needs_open=list(open_needs(state, agent)), needs_burned=list(s.burned), needs_lost=list(s.lost),
        holdings=holdings, can_give=sorted({k.info_id for k in s.keys}), score=score(state, agent),
        peers=sorted(a for a in pool if a != agent and not departed(state, a, cfg)),
        won=s.won_round is not None,
    )


# ------------------------------------------------------------------ view_log
def _record_line(rec: ActionRecord, cfg: GameCfg) -> LogLine | None:
    """Render a give / defect ActionRecord as a LogLine under cfg.view_log."""
    vl = cfg.view_log
    if rec.kind == "defect":
        ident = vl.defect_actor_identity or vl.actor_identity
        return LogLine(round=rec.round, kind="defect",
                       actor=None if ident == "hidden" else rec.actor,
                       target=rec.target, info_id=rec.info_id, ok=rec.ok)
    if rec.kind == "give":
        return LogLine(round=rec.round, kind="give", actor=None if vl.actor_identity == "hidden" else rec.actor,
                       target=rec.target, info_id=rec.info_id)
    return None


def _message_line(msg: Message, cfg: GameCfg) -> LogLine | None:
    """Render a board post / DM as a LogLine (None under actions_only for DMs). Uses the SHOWN sender
    only, so a forged DM renders exactly like a real one."""
    vl = cfg.view_log
    sender = None if vl.actor_identity == "hidden" else msg.sender_shown
    if msg.channel == "board":
        return LogLine(round=msg.round, kind="board_post",
                       actor=None if cfg.board.anonymous else sender, text=msg.text)
    if vl.content == "actions_msg_meta":
        return LogLine(round=msg.round, kind="dm_meta", actor=sender, target=msg.receiver)
    if vl.content == "full":
        text = ("[FORGED] " + msg.text) if (vl.reveal_provenance and msg.forged) else msg.text
        return LogLine(round=msg.round, kind="dm", actor=sender, target=msg.receiver, text=text)
    return None                                             # actions_only: no message events at all


def _defect_shown_by_release(rec: ActionRecord, viewer: AgentId, cfg: GameCfg) -> bool:
    """Is this defect record shown to `viewer` via the release rule (not the rolling window)?"""
    mode = cfg.view_log.defect_visibility
    if mode == "delayed":
        return True
    if mode == "victim_then_delayed":
        return viewer != rec.target
    return False


def _line_shown(state: GameState, viewer: AgentId, m: Message, line: LogLine, cfg: GameCfg) -> bool:
    """game.network / board.scope filter for a message line (always True when both are off)."""
    if m.channel == "board":
        return network.sees_post(state, viewer, m.sender_shown, cfg)
    return network.line_visible(state, viewer, line.actor, line.target)


def _log_lines(state: GameState, viewer: AgentId, cfg: GameCfg,
               in_window: Callable[[int], bool], released: Callable[[int], bool]) -> list[LogLine]:
    """Every line `viewer` may see, given predicates for window records (by stamp) and for delayed
    defects (by release turn). Messages come from state.messages (God's forged DMs have no
    ActionRecord); gives / defects from state.history. Ordered by (turn_index, messages first, index)."""
    if cfg.view_log.content == "none":
        return []
    out: list[tuple[int, int, int, LogLine]] = []
    for idx, m in enumerate(state.messages):
        if in_window(_shown_stamp(m, cfg)):
            line = _message_line(m, cfg)
            if line is not None and _line_shown(state, viewer, m, line, cfg):
                out.append((_shown_stamp(m, cfg), 0, idx, line))
    mode = cfg.view_log.defect_visibility
    for idx, rec in enumerate(state.history):
        if rec.kind not in ("give", "defect"):
            continue
        if rec.kind == "give" or mode == "public":
            show = in_window(rec.turn_index)
        elif rec.release_turn is not None and _defect_shown_by_release(rec, viewer, cfg):
            show = released(rec.release_turn)
        else:
            show = False                                    # victim_only, or the victim's own burn
        if show:
            line = _record_line(rec, cfg)
            if line is not None and network.line_visible(state, viewer, line.actor, line.target):
                out.append((rec.turn_index, 1, idx, line))
    out.sort(key=lambda x: x[:3])
    return [ln for *_, ln in out]


def view_log(state: GameState, viewer: AgentId, cfg: GameCfg, viewer_turn: int,
             prev_obs_turn: int) -> list[LogLine]:
    """The log lines `viewer` sees in the observation at `viewer_turn` (D5/D12/D16)."""
    w = cfg.window_turns
    return _log_lines(state, viewer, cfg,
                      in_window=lambda s: log_visible(s, viewer_turn, w),
                      released=lambda rel: release_visible(rel, viewer_turn, prev_obs_turn))


# ------------------------------------------------------------------ observe
def observe(state: GameState, agent: AgentId, cfg: GameCfg, turn_idx: int | None = None) -> Observation:
    """Build the agent-visible Observation at global slot `turn_idx` (default state.turn_index).
    Pure: does not update last_obs_turn (call mark_observed)."""
    t = state.turn_index if turn_idx is None else turn_idx
    prev = state.agents[agent].last_obs_turn
    inbox = [_msg_view(m, cfg) for m in state.messages
             if m.channel == "dm" and m.receiver == agent and _new_since(m.turn_index, t, prev)]
    board = [_msg_view(m, cfg) for m in state.messages
             if m.channel == "board" and m.sender_shown != agent and _new_since(_shown_stamp(m, cfg), t, prev)
             and network.sees_post(state, agent, m.sender_shown, cfg)]
    notices = []
    for n in state.notices:
        if n.agent != agent:
            continue
        due = (release_visible(n.deliver_from_turn, t, prev) if n.deliver_from_turn is not None
               else _new_since(n.turn_index, t, prev))
        if due:
            notices.append(NoticeView(kind=n.kind, text=n.text))
    learned = (network.names_seen(agent, inbox, board, notices, state.agents)
               if state.agents[agent].known is not None else set())
    return Observation(agent=agent, round=state.round, turn_index=t,
                       pinned=pinned_facts(state, agent, cfg, learned),
                       inbox=inbox, board=board, log=view_log(state, agent, cfg, t, prev), notices=notices)


def mark_observed(state: GameState, agent: AgentId, turn_idx: int) -> None:
    """Record that `agent` observed at `turn_idx` (drives the 'new since last observation' rules)."""
    state.agents[agent].last_obs_turn = turn_idx


# ------------------------------------------------------------------ grep_log
def grep_text(line: LogLine) -> str:
    """The searchable text of a line: its rendering plus ' #<info_id>' (D15 key ids)."""
    s = render_log_line(line)
    return s + (f" #{line.info_id}" if line.info_id is not None else "")


def visible_history(state: GameState, viewer: AgentId, cfg: GameCfg, obs_turns: Sequence[int]) -> list[LogLine]:
    """Union of everything view_log showed `viewer` at the observations `obs_turns` (ascending, the
    current one included), with defect releases evaluated as of the latest observation."""
    if not obs_turns:
        return []
    w = cfg.window_turns
    now = max(obs_turns)
    return _log_lines(state, viewer, cfg,
                      in_window=lambda s: any(log_visible(s, t, w) for t in obs_turns),
                      released=lambda rel: rel <= now)


def grep_log(state: GameState, viewer: AgentId, cfg: GameCfg, pattern: str, obs_turns: Sequence[int],
             rounds: int | None = None) -> GrepResult:
    """Case-insensitive LITERAL substring search over the viewer's visible log history (D5).
    Patterns longer than grep_max_pattern_chars give no hits; results are capped at grep_max_results."""
    vl = cfg.view_log
    if not vl.grep_enabled or vl.content == "none" or not pattern or len(pattern) > vl.grep_max_pattern_chars:
        return GrepResult(pattern=pattern[: vl.grep_max_pattern_chars], hits=[])
    pat = pattern.lower()
    hits = [ln for ln in visible_history(state, viewer, cfg, obs_turns)
            if (rounds is None or ln.round > state.round - rounds) and pat in grep_text(ln).lower()]
    cap = vl.grep_max_results
    return GrepResult(pattern=pattern, hits=hits[-cap:], truncated=len(hits) > cap)
