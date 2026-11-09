"""config_hash (split out of swarm/config.py, which re-exports the constants; R6).

    RunConfig.config_hash() == config_hash(cfg): sha256 of the resolved config's GAME SEMANTICS.
It decides every game_id, agent seed and LLM cache key, so its rules are frozen:
  - `sweep` is excluded (each expanded cell has its own hash);
  - options added after the freeze are left out while at their defaults (game.board and its later
    keys, game.network, run.resample_from_round, reasoning.salvage_truncated, the OpenRouter provider knobs), so
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
    orr = d["llm"]["openrouter"]
    for k in ("reasoning_effort", "provider_sort", "provider_order", "allow_fallbacks"):  # board rule
        if orr[k] is None:
            orr.pop(k)
    return sha256_hex(canonical_json(d))
