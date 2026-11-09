# Arm F: Adaption mixed swarm (PREP ONLY; nothing has been run)

> **SUPERSEDED (13:00):** Jennifer chose a self-run package, so the key never touches this machine. Use `arm_f_package.zip` and its README. The Iraj-launch commands below are kept for reference only.

**What it is:** a separate robustness arm. Its results are **never pooled** with the Qwen-2.5 arms (A-E).
- It runs the same main-grid contrasts as arms A and D: attacker {D-Evil, insider} x defense {none, provenance,
  rep+corr+quar}, each with a clean twin that uses the same defense.
- It uses the same matched forks (shared round-1 prefix, branch seed 31_001+seed), the same instruments (1-5), and the
  same audit fixes (unparsed replies = missing; damage measured against the same-defense twin; Holm-corrected secondary
  claims only).

## Design

| | Setting |
|---|---|
| Agents (N=8) | 2 each of `deepseek-v4-flash` (effort none), `gemma-4-26b-a4b-it` (default), `gemma-4-31b-it` (default), `qwen3.8-27b` (none). Agent i plays model i mod 4. |
| Optional 5th model (N=10) | `glm-53-flash` (effort **low**; other values run away, see ADAPTION_MODELS.md). Set `ARM_F_MODELS=...,glm-53-flash:low` and `ARM_F_N=10`. |
| D-Evil and the judge | OpenRouter `qwen/qwen-2.5-7b-instruct` (as in A and D), so only the agents change. Tagged `F_d_evil` and `F_judge`. |
| Seeds | 10 (0-9). The 70/30 split for its H1-style analysis is train 0-6, test 7-9. Expect very few held-out positives, so this is descriptive, as for arm D at 10 seeds. |
| Topology, visibility | complete, private (same as D) |
| Outputs | `out/arm_F/`: games/agents/cells/events JSONL, `messages_arm_F.csv` (with `kind`), `cost_tally.json`. Figures and results via `analyze_v2.py` with `REMAP=F:A`, labelled "arm F (Adaption)". |

## Code (in `harness/`; Jennifer's package is unchanged)

`adaption_backend.py` is new and synchronous, with the same `chat(pairs, seed)` interface as her backends.
- **Request:** streaming SSE to `POST https://api.prod.adaptionlabs.ai/api/v1/responses`.
  - Body: model, instructions, input, temperature 0.7, store=false, and `reasoning.effort`, which is omitted for "default".
  - No `max_output_tokens`.
  - 300 s read timeout.
- **Rate limits,** process-wide:
  - concurrency 48
  - a request-start bucket at 380/min (burst 60)
  - a token bucket at 1.5M tokens/min
- **Retries:** 429, 5xx, timeouts and cut-off streams are retried up to 5 times with backoff. 401, 403 and other 4xx stop the run; the driver stops after 3 identical errors.
- **Cost:** `usage.cost` is summed into `cost_tally.json` (`adaption_usd`, `by_model["adaption:<model>"]`, `by_arm["F"]`), with a hard cap `ADAPTION_CAP_USD`. It is kept separate from the OpenRouter total.
- **Key:** read **only** from the file at `ADAPTION_KEY_FILE`. It is never printed, logged or put in the environment.
- **Reuse:** the request and stream logic follows `~/projects/swarm_adaption/swarm/llm/adaption.py`, which was read but not imported, because it is async and depends on another framework. Her package's `adaption_api.py` was read as untrusted reference and is not used.
- **Tested offline:** the SSE parser (deltas, final event, cut-off stream) and the missing-key-file check. No network was used.

`run_v2.py` now has `ARMS=F` (`arm_F`): mixed routing via her `MixedBackend._route`.

## Cost projection

The basis is per-call costs from `plan/ADAPTION_MODELS.md`, using the round-5 column (about 2.2k-token prompts).
- Our N=8 prompts average about 0.7k tokens (arm A: 836k prompt tokens / 1,212 calls), so these numbers are an **upper bound**.
- Calls per seed come from the v2 pilot: about 605 agent calls per seed (9 games + interview + battery).

**Average cost per 1k calls at the recommended efforts:**

| Model | $ / 1k calls |
|---|---|
| deepseek-v4-flash (none) | 0.087 |
| gemma-4-26b (default) | 0.196 |
| gemma-4-31b (default) | 0.191 |
| qwen3.8-27b (none) | 0.263 |
| **Mean, 4 models** | **0.184** |
| glm-53-flash (low) | 0.189 |

**Projection by option:**

| Option | Agent calls | Adaption $ (r5 upper bound) | Scaled to our prompt size (about 0.4x) | OpenRouter $ (D-Evil + judge) | Wall time |
|---|---|---|---|---|---|
| **N=8, 4 models, 10 seeds** | about 6,050 | **about $1.11** | about $0.45 | about $0.04 | about 20 min (the 380 rpm limit binds: at least 16 min) |
| N=10, + glm-53-flash, 10 seeds | about 7,600 | **about $1.40** | about $0.60 | about $0.05 | about 25 min |
| 1-seed pilot (N=8) | about 605 | about $0.11 | | | about 3-4 min |

**Recommended caps:**
- Pilot: `ADAPTION_CAP_USD=0.30`
- Full run: `ADAPTION_CAP_USD=2.50` (with the margin, if gemma/qwen outputs run longer than in the probe)
- OpenRouter: `COST_CAP=0.30`. The Plan C OpenRouter total stays within $8; it is $6.10 now.

**Billing unit:** unknown for Jennifer's account. The Adaption probe in `ADAPTION_MODELS.md` read `usage.cost` and reported it as dollars ($0.82 for 168 calls). Her package's `adaption_api.py` also exposes `credits_available()`, so the account may be billed in **credits**.
- The cap and the tally use `usage.cost` units as returned.
- If those are credits, the projection above holds in credits, not dollars.
- The pilot will show `adaption_usd` alongside the account's own figure, and Jennifer should check the units against her dashboard.

## Blockers

1. **No Adaption key file on this machine.** The path is configured as `ADAPTION_KEY_FILE=<your adaption_key_file>`, which doesn't exist yet. Jennifer or Iraj needs to create it out of band, with the key only as file contents (no `export`, never in chat). I will not ask for it in chat.
2. **First use of a new key** will probably be blocked by the permission system, so **Iraj launches it**, using the commands below.
3. **Model availability on her account is unverified.** If a model ID is not enabled, the run stops with "HTTP 4xx ... (model)". In that case, drop it from `ARM_F_MODELS`.

## Exact launch commands (for Iraj)

**1) Pilot.** 1 seed, about 3-4 minutes, Adaption cap 0.30, OpenRouter cap $0.10:
```bash
cd $PLANC_ROOT && PILOT=1 ARMS=F \
  ADAPTION_KEY_FILE='<your adaption_key_file>' ADAPTION_CAP_USD=0.30 \
  OUT=$PLANC_ROOT/out/arm_F_pilot COST_CAP=0.10 PAR_CELLS=10 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  $PY -s -P -u harness/launch.py "<your adaption_key_file>" \
  runs/v2run $PLANC_ROOT/harness/run_v2.py 2>&1 | tee runs/arm_F_pilot_log.txt
```

**2) Full arm F.** 10 seeds, N=8, about 20 minutes:
```bash
cd $PLANC_ROOT && ARMS=F SEEDS_F=10 \
  ADAPTION_KEY_FILE='<your adaption_key_file>' ADAPTION_CAP_USD=2.50 \
  OUT=$PLANC_ROOT/out/arm_F COST_CAP=0.30 PAR_CELLS=10 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  $PY -s -P -u harness/launch.py "<your adaption_key_file>" \
  runs/v2run $PLANC_ROOT/harness/run_v2.py 2>&1 | tee runs/arm_F_log.txt
```

**Optional 5-model version (N=10).** Add these to command 2:
`ARM_F_MODELS=deepseek-v4-flash:none,gemma-4-26b-a4b-it:default,gemma-4-31b-it:default,qwen3.8-27b:none,glm-53-flash:low ARM_F_N=10`

**3) Analysis** (no API calls):
```bash
cd $PLANC_ROOT && REMAP=F:A VERSION=armF $PY -I harness/analyze_v2.py \
  $PLANC_ROOT/out/arm_F $PLANC_ROOT/out/arm_F/figs
```

With `REMAP=F:A`, arm F goes through the main-grid code path. Its 70/30 split is seeds 0-6 / 7-9. Every label in its results is "arm F (Adaption)", not the main grid.
