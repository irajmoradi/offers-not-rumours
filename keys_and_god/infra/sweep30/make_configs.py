"""Writes configs/sweep30_*.yaml: the sweep30 single-factor experiment (Iraj-approved via lead, 2026-11-08 ~18:45).

30 agents (sbm, 3 departments x 10), 30 rounds (hidden horizon), GLM-5.3-flash on OpenRouter (providers Relace,
Wafer, Darkbloom; no other fallbacks), worlds game_idx 0, 1, 4 (run.seed 1: world seeds depend only on run.seed and
game_idx, as in sweep60_or_glm). Every other setting = sweep60_or_glm scaled to N=30. 7 arms, one config / run dir each:

  A1 sweep30_a1_facility  facility prompts (agent/v2_facility), effort medium, no God          <- the reference
  A2 sweep30_a2_oldframe  old Keys & God prompts (agent/v1, God god/v2), framing none (= sweep60_or_oldframe)
  A3 sweep30_a3_minimal   A1 with agent reasoning effort minimal
  A4 sweep30_a4_low       A1 with agent reasoning effort low
  A5 sweep30_a5_high      A1 with agent reasoning effort high
  A6 sweep30_a6_k10       A1 + LLM God k10 powers per wake (sweep60_or_glm_k5's God: same model / effort medium /
  A7 sweep30_a7_k20       A1 + LLM God k20      providers / powers / prompts / reveal_needs; wakes r3..r28)

N=30 scaling: n_agents 30, max_rounds 30, God wake schedule r3..r28 (sweep60: r3..r78 = start 3, end max_rounds-2),
sbm communities 3 (10 per department, as 60 = 3 x 20), mean_degree 4 (per-node, size-free), mixing 0.1, min_bridges 4
(sweep60's 8 halved with N). Everything else (score, trade, connect, board, view_log, defect broadcast delay 3) as is.

Decisions (stated in the launch report):
  - game.framing.won_render_fix: TRUE for every facility arm (A1, A3-A7): a new clean experiment, so no AUDIT2 I2
    render bug. A2 has framing none (no facility render), so it is unaffected. Not directly comparable to the
    sweep60 facility runs (won_render_fix false) except sweep60_or_glm_i2fix.
  - God dose: god.llm.max_steps scales with k (2k: k10 -> 20, k20 -> 40). k5 used max_steps 10 (2 x 5) and spent all
    10 steps nearly every wake (~4 powers + ~6 free investigation tools), so max_steps 10 would cap k10 / k20 at about
    the k5 dose and the arms would not differ. max_steps is a God setting, so A6 / A7 still differ from A1 only in God.
    max_wake_tokens (40000) and max_wake_s (600) stay at k5's (defaults); a 40-step wake uses ~8k completion tokens.
  - Caches: A1 / A6 / A7 share cache/sweep30_a1_facility.sqlite (WAL, multi-process) and share game ids / agent seeds
    (runner.arms hashes the God-free config), so the God arms replay A1's rounds 1-2 when launched after A1 passed
    round 2 (the sweep60_or_glm / k5 pattern). A2-A5 have their own caches: reasoning effort is a client setting, not
    part of the cache key, so an effort arm must never read another effort's cache.
  - Spend: a SEPARATE ledger (llm.openrouter.ledger_path = .../openrouter_ledger_sweep30.jsonl, global_cap_usd 32 =
    A <= $20 + gpt-oss B <= $8 + the sweep60 soft-k1 runs <= $4; was 20 before the B family was added),
    so this experiment can never push the shared runs/openrouter_ledger.jsonl past the running drivers' --ledger-stop
    130; per-run caps (llm.budget_usd): A sum <= $20, B sum <= $8; driver --ledger-stop 32 on the sweep30 ledger.
  - gpt-oss family B0 / B1 / B5 / B10: B0 = A1 with agents openai/gpt-oss-120b (Plan A routing: provider_sort price,
    normal fallbacks), effort medium, max_tokens null; B1 / B5 / B10 = B0 + the same GLM God (god.llm is explicit and
    provider_sort null, so it never inherits the agents' routing) with k 1 / 5 / 10, max_steps max(10, 2k).
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swarm.runner.run import load_experiment  # noqa: E402
from swarm.runner.spec import build_specs, world_seed  # noqa: E402

LEDGER = f"{os.environ.get('SWARM_ROOT', '.')}/runs/openrouter_ledger_sweep30.jsonl"
MAIN_LEDGER = f"{os.environ.get('SWARM_ROOT', '.')}/runs/openrouter_ledger.jsonl"
TOTAL_CAP = 32.0   # sweep30 ledger: GLM A arms <= $20 + gpt-oss B arms <= $8 + soft-k1 (sweep60) <= $4 (lead, 18:55)
SCHEDULE = "\n".join(f"  - {{round: {r}, after_turns: 0, power: message_fraud, args: {{}}}}" for r in range(3, 29))
PROV = "provider_order: [Relace, Wafer, Darkbloom], allow_fallbacks: false"
GLM = ("z-ai/glm-5.3-flash", f"price_in_per_mtok: 0.0335, price_out_per_mtok: 0.083, {PROV}")
# gpt-oss-120b: Plan A routing (configs/sweep/base.yaml: provider_sort price, OpenRouter's normal fallbacks; its price
# estimates 0.03 / 0.15 are pre-call reservations only). Effort medium like A1; max_tokens null like A1 (single factor).
OSS = ("openai/gpt-oss-120b", "price_in_per_mtok: 0.03, price_out_per_mtok: 0.15, provider_sort: price")


def god_arm(label: str, k: int) -> str:
    """LLM God arm: k powers per wake, max_steps max(10, 2k) (sweep60 k1 / k5 used 10; k5 spent all 10 most wakes)."""
    return f"[{{label: {label}, god: {{kind: llm, budget: {{k: {k}}}, llm: {{max_steps: {max(10, 2 * k)}}}}}}}]"
FACILITY = "{mode: facility, stance: silent, handoff_variant: 0, won_render_fix: true}"
NOGOD = "[{label: nogod, god: {kind: none}}]"

# name: (arm, agent prompt set, framing, God prompt set, agent effort, runner.arms, run cap $, cache, agent model)
ARMS = {
    "sweep30_a1_facility": ("A1", "agent/v2_facility", FACILITY, "god/v2_facility", "medium", NOGOD, 2.0, "sweep30_a1_facility", GLM),
    "sweep30_a2_oldframe": ("A2", "agent/v1", "{mode: none}", "god/v2", "medium", NOGOD, 1.5, "sweep30_a2_oldframe", GLM),
    "sweep30_a3_minimal": ("A3", "agent/v2_facility", FACILITY, "god/v2_facility", "minimal", NOGOD, 2.0, "sweep30_a3_minimal", GLM),
    "sweep30_a4_low": ("A4", "agent/v2_facility", FACILITY, "god/v2_facility", "low", NOGOD, 2.0, "sweep30_a4_low", GLM),
    "sweep30_a5_high": ("A5", "agent/v2_facility", FACILITY, "god/v2_facility", "high", NOGOD, 3.0, "sweep30_a5_high", GLM),
    "sweep30_a6_k10": ("A6", "agent/v2_facility", FACILITY, "god/v2_facility", "medium",
                       god_arm("k10", 10), 3.5, "sweep30_a1_facility", GLM),
    "sweep30_a7_k20": ("A7", "agent/v2_facility", FACILITY, "god/v2_facility", "medium",
                       god_arm("k20", 20), 5.5, "sweep30_a1_facility", GLM),
    # gpt-oss family (Iraj via lead, 18:55): B0 = A1 with the agent model gpt-oss-120b; B1 / B5 / B10 = B0 + the GLM
    # LLM God k1 / k5 / k10 (God model and settings as the A k-arms). B God arms share B0's cache and game ids.
    "sweep30_b0_oss": ("B0", "agent/v2_facility", FACILITY, "god/v2_facility", "medium", NOGOD, 1.6, "sweep30_b0_oss", OSS),
    "sweep30_b1_oss_k1": ("B1", "agent/v2_facility", FACILITY, "god/v2_facility", "medium", god_arm("k1", 1), 1.8,
                          "sweep30_b0_oss", OSS),
    "sweep30_b5_oss_k5": ("B5", "agent/v2_facility", FACILITY, "god/v2_facility", "medium", god_arm("k5", 5), 2.0,
                          "sweep30_b0_oss", OSS),
    "sweep30_b10_oss_k10": ("B10", "agent/v2_facility", FACILITY, "god/v2_facility", "medium", god_arm("k10", 10), 2.6,
                            "sweep30_b0_oss", OSS),
}
A_CAPS, B_CAPS = 20.0, 8.0


def text(name: str, arm: str, agent_ps: str, framing: str, god_ps: str, effort: str, arms: str, cap: float,
         cache: str, model: tuple[str, str]) -> str:
    return f"""# GENERATED by infra/sweep30/make_configs.py: edit the generator, not this file.
# sweep30 arm {arm}: 30 agents x 30 rounds, agents {model[0]} (OpenRouter), worlds game_idx 0, 1, 4; agent prompts
# {agent_ps}, effort {effort}, arms {arms}. Own spend ledger (openrouter_ledger_sweep30.jsonl, cap ${TOTAL_CAP:.0f}).
extends: base.yaml
run: {{name: {name}, seed: 1, n_games: 20, max_concurrency_games: 3}}
reasoning: {{salvage_truncated: true}}
game:
  n_agents: 30
  max_rounds: 30
  turn_mode: simultaneous
  turn_order: reshuffle
  winner_policy: lock_and_stay
  wrong_defect_penalty: score_penalty
  defect: {{win_rule: all_needs}}
  trade: {{enabled: true, max_offers_per_turn: 1, max_keys_per_side: 3, parallel_replies: false}}
  network:
    topology: sbm
    communities: 3
    mean_degree: 4
    mixing: 0.1
    min_bridges: 4
    mode: contacts
    key_placement: mixed
    connect: {{enabled: true, cost_rounds: 0, require_known: true}}
  board: {{scope: neighbors}}
  view_log:
    anonymize_noncontacts: true
    horizon: round_only
    defect_broadcast: {{enabled: true, delay_rounds: 3, mode: round_summary}}
  score: {{held_key_bonus: 0.05, held_key_scope: initial, community_bonus: 0.15, community_threshold_count: 4,
          community_success_needs: 2, delivery_credit: 0.1}}
  framing: {framing}
god:
  kind: llm
  start_round: 3
  budget: {{k: 1, scope: round}}
  powers:
    reveal_needs: {{enabled: true, variant: mix, p_false: 0.5, max_receivers: 2}}
  llm: {{model: z-ai/glm-5.3-flash, reasoning_effort: medium, {PROV}, provider_sort: null, price_in_per_mtok: 0.0335,
    price_out_per_mtok: 0.083, max_steps: 10, max_tokens: null, powers: [message_fraud, reveal_needs],
    prompt_set: {god_ps}}}
  schedule:
{SCHEDULE}
runner:
  engine: real
  report: false
  leak_action: warn
  game_indices: [0, 1, 4]
  arms: {arms}
  game_stall_min: 30
  game_stall_restarts: 2
roles:
  agent: {{backend: openrouter, model: {model[0]}, prompt_set: {agent_ps}, max_tokens: null}}
  god: {{backend: openrouter, model: z-ai/glm-5.3-flash, max_tokens: null}}
llm:
  budget_usd: {cap}
  cache: {{mode: read_write, path: cache/{cache}.sqlite}}
  retry: {{max_retries: 6, base_delay_s: 2.0, max_delay_s: 60.0, timeout_s: 300.0, timeout_thinking_s: 300.0}}
  openrouter: {{enabled: true, model: {model[0]}, max_concurrency: 64, global_cap_usd: {TOTAL_CAP},
    ledger_path: {LEDGER},
    {model[1]}, reasoning_effort: {effort}, call_deadline_s: 900}}
"""


def flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def main() -> None:
    loaded = {}
    for name, a in ARMS.items():
        (ROOT / "configs" / f"{name}.yaml").write_text(text(name, *a), encoding="utf-8")
        cfg, opts = load_experiment(ROOT / "configs" / f"{name}.yaml")
        loaded[name] = (cfg, opts, build_specs(cfg, opts))
    BOOK = {"run.name", "run.max_concurrency_games", "llm.budget_usd", "llm.cache.path", "llm.cache.mode"}
    GOD = {"god.kind", "god.budget.k", "god.llm.max_steps"}
    MODEL = {"roles.agent.model", "llm.openrouter.model", "llm.openrouter.provider_order",     # the agent model and
             "llm.openrouter.allow_fallbacks", "llm.openrouter.provider_sort",               # its OpenRouter routing /
             "llm.openrouter.price_in_per_mtok", "llm.openrouter.price_out_per_mtok"}        # price estimates
    # arm -> (reference arm, the factor set it may differ in; for God arms: a subset of GOD, k checked below)
    want = {"sweep30_a1_facility": ("sweep30_a1_facility", set()),
            "sweep30_a2_oldframe": ("sweep30_a1_facility", {"game.framing.mode", "game.framing.won_render_fix",
                                    "roles.agent.prompt_set", "god.llm.prompt_set"}),  # framing none: stance/variant defaults silent/0
            "sweep30_a3_minimal": ("sweep30_a1_facility", {"llm.openrouter.reasoning_effort"}),
            "sweep30_a4_low": ("sweep30_a1_facility", {"llm.openrouter.reasoning_effort"}),
            "sweep30_a5_high": ("sweep30_a1_facility", {"llm.openrouter.reasoning_effort"}),
            "sweep30_a6_k10": ("sweep30_a1_facility", GOD), "sweep30_a7_k20": ("sweep30_a1_facility", GOD),
            "sweep30_b0_oss": ("sweep30_a1_facility", MODEL),
            "sweep30_b1_oss_k1": ("sweep30_b0_oss", GOD), "sweep30_b5_oss_k5": ("sweep30_b0_oss", GOD),
            "sweep30_b10_oss_k10": ("sweep30_b0_oss", GOD)}
    god_k = {"sweep30_a6_k10": 10, "sweep30_a7_k20": 20, "sweep30_b1_oss_k1": 1, "sweep30_b5_oss_k5": 5,
             "sweep30_b10_oss_k10": 10}
    # A game ids as committed in c208a19 (A1 / A2 restarted at 18:5x replay their cache: ids must not move)
    pinned = {"sweep30_a1_facility": ["g69c8359c6627", "g90a7a1bd66c0", "g48ee5153cf1f"],
              "sweep30_a2_oldframe": ["g97f6ada74645", "gce3b329d28ec", "g2c914f4d996a"]}
    ref_ws = [s.world_seed for g in loaded["sweep30_a1_facility"][2] for s in g]
    resolved_god = {}
    for name, (cfg, opts, groups) in loaded.items():
        assert cfg.run.name == name
        o = cfg.llm.openrouter
        assert o.ledger_path == LEDGER != MAIN_LEDGER and o.global_cap_usd == TOTAL_CAP, name
        assert cfg.game.n_agents == 30 and cfg.game.max_rounds == 30 and cfg.game.network.topology == "sbm"
        assert list(opts.game_indices) == [0, 1, 4] and len(groups) == 3 and all(len(g) == 1 for g in groups)
        assert cfg.roles.agent.max_tokens is None and cfg.game.trade.parallel_replies is False
        assert max(sp.round for sp in cfg.god.schedule) == 28 and min(sp.round for sp in cfg.god.schedule) == 3
        specs = [g[0] for g in groups]
        assert [s.world_seed for s in specs] == ref_ws == [world_seed(cfg, i) for i in (0, 1, 4)], name
        if name in pinned:
            assert [s.game_id for s in specs] == pinned[name], (name, [s.game_id for s in specs])
        ref_name, allowed = want[name]
        rcfg, _, rgroups = loaded[ref_name]
        for s, r in zip(specs, (g[0] for g in rgroups)):
            a, b = flat(r.cfg.model_dump(mode="json")), flat(s.cfg.model_dump(mode="json"))
            diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - BOOK
            if name in god_k:
                k = god_k[name]
                assert diff <= GOD and "god.kind" in diff, (name, sorted(diff))
                assert s.cfg.god.kind == "llm" and s.cfg.god.budget.k == k and s.cfg.god.llm.max_steps == max(10, 2 * k)
                assert s.game_id == r.game_id and cfg.llm.cache.path == rcfg.llm.cache.path, name   # paired prefix
                g = s.cfg.god_llm()
                resolved_god[name] = {kk: v for kk, v in g.items()} if isinstance(g, dict) else g
            else:
                assert diff == allowed, (name, sorted(diff))
                assert s.cfg.god.kind == "none"
            if s.cfg.game.framing.mode == "facility":
                assert s.cfg.game.framing.won_render_fix is True, name
        fresh = "" if not (ROOT / "runs" / name).exists() else "  [runs/ dir EXISTS]"
        print(f"{name}: cap ${cfg.llm.budget_usd:.2f}, vs {ref_name} differs in {sorted(diff) or '-'};",
              [(s.label, s.game_id) for s in specs], fresh)
    # the God is the same model / settings in every God arm (GLM-5.3-flash, medium, Relace/Wafer/Darkbloom), whatever
    # the agent model: the resolved god_llm() of the gpt-oss k10 arm equals the GLM k10 arm's
    assert resolved_god["sweep30_b10_oss_k10"] == resolved_god["sweep30_a6_k10"], resolved_god
    assert len({str(v) for v in resolved_god.values()}) == 1, resolved_god
    print("resolved God (all God arms):", resolved_god["sweep30_a6_k10"])
    nonshare = [c.llm.cache.path for n, (c, _, _) in loaded.items() if n not in god_k]
    assert len(set(nonshare)) == len(nonshare), "no-God arms need their own caches (effort is not in the cache key)"
    a_tot = sum(c.llm.budget_usd for n, (c, _, _) in loaded.items() if n.startswith("sweep30_a"))
    b_tot = sum(c.llm.budget_usd for n, (c, _, _) in loaded.items() if n.startswith("sweep30_b"))
    assert a_tot <= A_CAPS + 1e-9 and b_tot <= B_CAPS + 1e-9, (a_tot, b_tot)
    print(f"per-run caps: A ${a_tot:.2f} <= ${A_CAPS:.0f}, B ${b_tot:.2f} <= ${B_CAPS:.0f}; ledger {LEDGER} cap ${TOTAL_CAP:.0f}")


if __name__ == "__main__":
    main()
