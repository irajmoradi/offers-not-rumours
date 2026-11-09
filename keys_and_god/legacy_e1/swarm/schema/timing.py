"""The view_log / defect-visibility timing contract (CONTRACTS §2): pure arithmetic over global
turn indices; deterministic across forks."""

from __future__ import annotations


def turn_index(round: int, position: int, n_agents: int) -> int:
    """Global slot index of the agent at `position` in `round` (1-based round).
    Simultaneous mode: every agent observes at the round's first slot, (round-1)*n_agents."""
    return (round - 1) * n_agents + position


def log_visible(record_turn: int, viewer_turn: int, window_turns: int) -> bool:
    """Normal actions (give, board_post, dm meta, public defects): rolling window of the previous
    `window_turns` slots (default window_turns = n_agents, i.e. 'the last round')."""
    return viewer_turn - window_turns <= record_turn < viewer_turn


def defect_release_turn(defect_turn: int, delay_rounds: int, n_agents: int) -> int:
    """Delayed defects (modes delayed / victim_then_delayed; wrong defects too): non-victims may see
    the line at the first observation with turn_index >= release. With delay=0 this equals the
    normal 'next slot' rule. In simultaneous mode a defect in round d shows in round d+1+delay."""
    return defect_turn + 1 + delay_rounds * n_agents


def release_visible(release_turn: int, viewer_turn: int, viewer_prev_obs_turn: int) -> bool:
    """A released defect line is shown EXACTLY ONCE per viewer: at the first observation at or after
    release. Viewers who were delayed / sat out get it queued at their next observation.
    viewer_prev_obs_turn = AgentState.last_obs_turn before this observation (-1 if none)."""
    return viewer_prev_obs_turn < release_turn <= viewer_turn
