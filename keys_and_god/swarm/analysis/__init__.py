"""Analysis (WP6, minimal): per-game outcomes and the paired-fork ripple metric. Pure functions."""

from swarm.analysis.outcomes import agent_table, game_summary, god_actions
from swarm.analysis.ripple import RippleResult, ripple

__all__ = ["RippleResult", "agent_table", "game_summary", "god_actions", "ripple"]
