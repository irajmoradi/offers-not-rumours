"""Keys & God game core (WP1a/WP1b): world generation, pure rules, observations, engine loop.

Public API:
    generate_world(cfg: GameCfg, seed, game_id="g0", names=None) -> GameState   (world.py)
    is_feasible(state) -> bool                                                   (world.py)
    apply(state, actor, action, cfg, *, copy=False) -> (GameState, list[EventBody])  (rules.py)
    deliver_message(state, *, channel, sender_shown, text, receiver=None, true_sender=None,
                    forged=False, intervention_id=None) -> (Message, events)     (rules.py; God fraud uses it)
    score / is_winning / is_done / open_needs                                    (rules.py)
    observe(state, agent, cfg, turn_idx=None) -> Observation                     (observe.py)
    view_log(...), grep_log(state, viewer, cfg, pattern, obs_turns, rounds=None) -> GrepResult
    GameEngine(cfg: RunConfig, state, decide, *, god_hook=None, sink=None, seed=None)  (engine.py)
    play_game(cfg, state, decide, **kw) -> (Outcome, GameEngine)
"""

from .engine import GameEngine, noop_decide, play_game
from .observe import grep_log, mark_observed, observe, view_log
from .rules import apply, deliver_message, is_done, is_winning, open_needs, score
from .world import generate_world, is_feasible

__all__ = [
    "GameEngine", "apply", "deliver_message", "generate_world", "grep_log", "is_done", "is_feasible",
    "is_winning", "mark_observed", "noop_decide", "observe", "open_needs", "play_game", "score", "view_log",
]
