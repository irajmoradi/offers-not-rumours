"""config_hash (split out of swarm/config.py, which re-exports the constants; R6).

    RunConfig.config_hash() == config_hash(cfg): sha256 of the resolved config's GAME SEMANTICS.
It decides every game_id, agent seed and LLM cache key, so its rules are frozen:
  - `sweep` is excluded (each expanded cell has its own hash);
  - options added after the freeze are left out while at their defaults (game.board and its later
    keys, game.network and its later game.network.connect, game.view_log.defect_broadcast (while
    disabled), game.view_log.horizon (while "shown"), the game-fixes game.score (and its later
    showcase community / delivery knobs) / game.defect / god.powers.reveal_needs, game.trade, game.framing, god.llm (unless god.kind=llm),
    run.resample_from_round, reasoning.salvage_truncated, the OpenRouter provider knobs), so
    older configs keep their hash; run.log_prompt_trace (logging only) is never hashed;
  - operational knobs (caps, budget, concurrency, paths, retry, price estimates, run name/n_games)
    are hashed as the fixed values in HASH_OPERATIONAL (CODE_REVIEW #6), so changing them never
    re-keys a game.
Imports: rng only (the config is passed in).
"""

from __future__ import annotations

import copy
from typing import Any

from swarm.rng import canonical_json, sha256_hex

# game.network knobs added with the sbm / powerlaw_cluster / community_hubs topologies and
# key_placement: dropped from the hash at these defaults (older network configs keep their hash).
NEW_NETWORK_DEFAULTS: dict[str, Any] = {"communities": 4, "mixing": 0.1, "p_in": None, "p_out": None,
                                        "min_bridges": 4, "p_triangle": 0.5, "bridges": 2,
                                        "key_placement": "random", "local_hops": 2,
                                        "local_strength": 1.0, "far_min_hops": 3}
HASH_GLOBAL_CAP_USD = 2.0   # value config_hash() always uses for llm.openrouter.global_cap_usd
# Operational knobs (spend caps, concurrency, paths, retry, price estimates, run labels): config_hash()
# always hashes them as these fixed values (the original base.yaml defaults), so changing them never
# re-keys a game. Everything else (game, reasoning, roles, god, context, models, run.seed/branch/
# replica) is game semantics and stays in the hash.
HASH_OPERATIONAL: dict[tuple[str, ...], Any] = {
    ("run", "name"): "base",
    ("run", "n_games"): 1,
    ("run", "out_dir"): "runs",
    ("run", "max_concurrency_games"): 1,
    ("llm", "budget_usd"): 2.0,
    ("llm", "vllm", "base_url"): "http://localhost:8000/v1",
    ("llm", "vllm", "max_concurrency"): 64,
    ("llm", "openrouter", "base_url"): "https://openrouter.ai/api/v1",
    ("llm", "openrouter", "api_key_env"): "OPENROUTER_API_KEY",
    ("llm", "openrouter", "max_concurrency"): 8,
    ("llm", "openrouter", "global_cap_usd"): HASH_GLOBAL_CAP_USD,
    ("llm", "openrouter", "ledger_path"): "runs/openrouter_ledger.jsonl",
    ("llm", "openrouter", "price_in_per_mtok"): 1.0,
    ("llm", "openrouter", "price_out_per_mtok"): 4.0,
    ("llm", "cache"): {"mode": "read_write", "path": "cache/llm.sqlite"},
    ("llm", "retry"): {"max_retries": 5, "base_delay_s": 1.0, "max_delay_s": 30.0, "timeout_s": 120.0,
                       "timeout_thinking_s": 300.0},
}


def config_hash(cfg: Any) -> str:
    """sha256 of cfg's game semantics (see the module docstring)."""
    d = cfg.model_dump(mode="json", exclude={"sweep"})
    if not cfg.game.view_log.defect_broadcast.enabled:          # gossip knob; off = not hashed
        d["game"]["view_log"].pop("defect_broadcast")
    if not cfg.game.view_log.refusal_broadcast.enabled:         # Noor tie-in knob; off = not hashed
        d["game"]["view_log"].pop("refusal_broadcast", None)
    if not cfg.game.view_log.norm_notice.enabled:               # Noor tie-in knob; off = not hashed
        d["game"]["view_log"].pop("norm_notice", None)
    if cfg.game.board == type(cfg.game.board)():
        d["game"].pop("board")
    else:
        b = d["game"]["board"]
        for k, default in (("post_skip_rounds", 0), ("enabled", True),     # added later; same rule
                           ("visible_delay_rounds", 0), ("scope", "global")):
            if b[k] == default:
                b.pop(k)
    if cfg.game.network == type(cfg.game.network)():           # added later; board rule
        d["game"].pop("network")
    else:
        if cfg.game.network.connect == type(cfg.game.network.connect)():   # added later; board rule
            d["game"]["network"].pop("connect")
        elif not cfg.game.network.connect.require_known:                     # showcase knob; same rule
            d["game"]["network"]["connect"].pop("require_known")
        nw = d["game"]["network"]                               # later network knobs; same rule
        for k, default in NEW_NETWORK_DEFAULTS.items():
            if nw[k] == default:
                nw.pop(k)
    if not cfg.game.view_log.anonymize_noncontacts:                      # showcase v2 knob; board rule
        d["game"]["view_log"].pop("anonymize_noncontacts")
    if cfg.game.view_log.horizon == "shown":                             # hide-horizon knob; board rule
        d["game"]["view_log"].pop("horizon")
    sc = d["game"]["score"]
    if not cfg.game.score.community_bonus:                               # showcase knobs; board rule
        sc.pop("community_bonus")
        sc.pop("community_threshold")
        sc.pop("community_threshold_count")
        sc.pop("community_success_needs")
    else:
        for k in ("community_threshold_count", "community_success_needs"):
            if sc[k] is None:
                sc.pop(k)
    if not cfg.game.score.delivery_credit:
        sc.pop("delivery_credit")
    if cfg.game.score.held_key_scope == "all":
        sc.pop("held_key_scope")
    if sc == {"held_key_bonus": 0.0, "team_share": 0.0}:                 # game-fixes; board rule
        d["game"].pop("score")
    if cfg.game.defect == type(cfg.game.defect)():                       # game-fixes; board rule
        d["game"].pop("defect")
    if cfg.god.powers.reveal_needs == type(cfg.god.powers.reveal_needs)():   # game-fixes; board rule
        d["god"]["powers"].pop("reveal_needs")
    if cfg.god.kind != "llm":                                            # LLM God knobs: hashed only for it
        d["god"].pop("llm")
    if cfg.game.trade == type(cfg.game.trade)():                         # offer_trade; board rule
        d["game"].pop("trade")
    elif not cfg.game.trade.parallel_replies:                            # added later; board rule
        d["game"]["trade"].pop("parallel_replies")
    if cfg.game.framing == type(cfg.game.framing)():                     # realistic framing; board rule
        d["game"].pop("framing")
    else:
        if not cfg.game.framing.won_render_fix:                         # AUDIT2 I2 fix, added later; off = not hashed
            d["game"]["framing"].pop("won_render_fix")
        if cfg.game.framing.handoff_wording == "v1":                    # added later; the original = not hashed
            d["game"]["framing"].pop("handoff_wording")
    if d["run"]["resample_from_round"] is None:                 # added later; board rule
        d["run"].pop("resample_from_round")
    d["run"].pop("log_prompt_trace")                            # logging only: never hashed
    # Operational knobs are not game semantics: hash them as fixed values (HASH_OPERATIONAL), so
    # raising a cap, budget or concurrency, renaming a run or moving the cache keeps every hash,
    # game id, seed and cache key. Configs at the base defaults keep their pre-change hash
    # (the $2 -> $6 cap raise included). Past OpenRouter runs that pinned budget/prices/
    # concurrency are re-keyed once; they all predate the 8ebbee7 ledger fix, so their cached
    # prompts no longer match the current code anyway (CODE_REVIEW #6).
    for path, value in HASH_OPERATIONAL.items():
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = copy.deepcopy(value)
    if not cfg.reasoning.salvage_truncated:                  # added later; board rule
        d["reasoning"].pop("salvage_truncated")
    ad = d["llm"].pop("adaption")                 # Adaption backend (added later): only `enabled` and the
    if cfg.llm.adaption.enabled:                  # default model are semantic; caps/paths/prices/concurrency
        d["llm"]["adaption"] = {"enabled": True, "model": ad["model"],   # are operational (never hashed)
                                "reasoning_effort": ad["reasoning_effort"]}
        if ad.get("god_reasoning_effort") is not None and cfg.god.kind != "none":   # God's own effort
            d["llm"]["adaption"]["god_reasoning_effort"] = ad["god_reasoning_effort"]
    orr = d["llm"]["openrouter"]
    for k in ("reasoning_effort", "provider_sort", "provider_order", "allow_fallbacks",  # board rule
              "provider_max_price"):
        if orr[k] is None:
            orr.pop(k)
    orr.pop("call_deadline_s", None)                         # operational (a time guard): never hashed
    return sha256_hex(canonical_json(d))
