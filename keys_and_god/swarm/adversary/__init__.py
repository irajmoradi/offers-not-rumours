"""God: observation tools, powers (budget-enforced), ScriptedGod, LLMGod stub (WP4)."""

from swarm.adversary.base import God, GodRunner, WakeResult, turns_into_round
from swarm.adversary.god_tools import GodView
from swarm.adversary.llm_god import LLMGod, make_god
from swarm.adversary.powers import (
    PowerResult,
    apply_cot_control,
    apply_message_fraud,
    apply_power,
    init_budget,
    refresh_budget,
)
from swarm.adversary.scripted_god import ScriptedGod, closest_to_winning, make_canary, recent_partner

__all__ = [
    "God", "GodRunner", "WakeResult", "turns_into_round", "GodView", "LLMGod", "make_god", "PowerResult",
    "apply_cot_control", "apply_message_fraud", "apply_power", "init_budget", "refresh_budget",
    "ScriptedGod", "closest_to_winning", "make_canary", "recent_partner",
]
