"""Public-board options (game.board; both default off = the original rules). Pure helpers used by
game.resolve; texts here are the agent-facing notices (frozen: they reach prompts).

  post_costs_turn   a post_board takes the whole turn: the turn's FIRST post_board is applied and every
                    other action of that turn (grep_log and noop excepted) fails with BOARD_WHOLE_TURN,
                    using no slot; the agent gets one "system" notice (whole_turn_text).
  post_skip_rounds  k > 0: a SUCCESSFUL post_board in round r -> no turns in rounds r+1..r+k (via
                    AgentState.delayed_until_round, like the wrong-defect delay), plus a skip notice
                    now and a resume notice delivered from the first slot of round r+k+1.
  anonymous         rendering only (game.observe).
Imports: config, schema, game.rules.
"""

from __future__ import annotations

from collections.abc import Sequence

from swarm.config import BoardCfg
from swarm.schema import Action, AgentId, EventBody, GameState, PostBoard, turn_index

from . import rules

BOARD_WHOLE_TURN = "board_post_uses_whole_turn"


def whole_turn_post(actions: Sequence[Action], board: BoardCfg) -> int | None:
    """Under post_costs_turn: index of the turn's first post_board (None when off or no post)."""
    if not board.post_costs_turn:
        return None
    return next((i for i, a in enumerate(actions) if isinstance(a, PostBoard)), None)


def whole_turn_text(blocked: Sequence[Action]) -> str:
    """Notice for actions dropped because a post_board took the whole turn (board.post_costs_turn)."""
    kinds = ", ".join(a.kind for a in blocked)
    return (f"Your other action(s) this turn ({kinds}) were not done: posting to the public board uses "
            "your whole turn. A direct message costs one action.")


def skip_after_post(state: GameState, agent: AgentId, k: int) -> list[EventBody]:
    """post_skip_rounds = k: set delayed_until_round and return the skip + resume notice events."""
    s = state.agents[agent]
    last = state.round + k
    s.delayed_until_round = max(s.delayed_until_round, last)
    resume = turn_index(last + 1, 0, len(state.agents))
    return [
        rules.notice(state, agent, "system",
                     f"You posted on the public board in round {state.round}, so you skip your turns "
                     f"in rounds {state.round + 1} to {last}. You can still receive keys and messages."),
        rules.notice(state, agent, "system",
                     f"Your {k} skipped turns (for your board post in round {state.round}) are over. "
                     "You can act again.", deliver_from_turn=resume)]
