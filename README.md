# Offers, Not Rumours: What Turns Agents in an LLM Swarm

Code, configs, pre-registrations and aggregate results for the paper *Offers, Not Rumours: What Turns Agents in an LLM
Swarm* (Noor, Lance Santiago, Iraj Moradi, Jenifer Nguyen, Mario Sepriano; Swarm Alignment Hackathon, Apart Research, November 2026). The paper itself is not
part of this repository. Section and appendix numbers below refer to it.

**Summary.** Does a message that fools one agent also corrupt a swarm of agents? We ran two cooperative
key-trading games and matched every attacked game to a clean twin. Messages that promised the receiver something
(bribes, forged offers) turned agents. Messages about other agents (rumours, forged accusations) did not. An insider's
bribe beat its lies in a head-to-head on Gemma-3-12B and in a pre-registered replication on Mistral-Small-3.2 and phi-4.
A forger in a 10-agent swarm harmed welfare only when it forged every round. Most of an insider's measured harm came from
withholding keys, and a larger swarm diluted that harm. No detector did better than reading the inbox. One resilience
instruction removed bought defections, including for bribe wordings it had not seen. Per-agent obedience rates from
single-scene tests did not predict swarm harm.

The study has three parts:

| Part | Folder | What it is | Main author |
|---|---|---|---|
| 1. Keys & God | [`keys_and_god/`](keys_and_god/) | A swarm trading game with a hidden "God" that forges DMs and overwrites notes. Every world is forked into God / no-God / re-sampled noise twin (B″). Includes the 10-agent dose runs, the 60-agent Game-vs-Realistic Env runs, the 100-agent sweep configs, and the chain-of-thought (CoT) suspicion judge. | Iraj Moradi |
| 2. Betrayal study | [`betrayal/`](betrayal/) | An 8-agent key-trading swarm with insider and outsider attackers. Every attacked branch has a clean twin. Includes the pre-registered Qwen study, the Gemma head-to-head, the follow-ups, dual control and the multi-model replication. | Jenifer Nguyen |
| 3. Single agents and the model's inside | [`single_agent/`](single_agent/) | The 300-scene single-agent test (tempt300) on hosted models, and activation geometry on local Gemma-2-2B, Qwen2.5-1.5B and SmolLM2-1.7B. | Noor |

## Layout

```
keys_and_god/          Keys & God engine (swarm/), configs/, infra/ (launchers), tests/, results/ (aggregate result notes)
  legacy_e1/           engine snapshot at commit cf2c46b, used for the 10-agent dose runs (E1, K=1..6, board off)
  cot_suspicion/       CoT judge (cot_judge.py), run scripts, aggregates (summary.json, per_round.csv; god_dose/)
  figs/                Keys & God figure scripts (*_viz.py)
betrayal/              pre-registrations (prespec_*.md, PREREG_*_SHA256.txt, HYPOTHESES.md), CHANGES.md, PLAN.md
  pkg/swarm_llm/       Jenifer Nguyen's game package (hash-pinned in pkg_module_sha256.txt)
  harness/             drivers (run_v2.py, run_jenfu.py, run_noor.py, run_closers.py, run_tofix.py), attack modules, analyses
  analysis/            cross-arm analyses (family-wide correction, size/key damage, ledger, timeouts, empty replies)
  runs/                launch scripts used for each arm
  out/                 aggregate outputs (results*.json, *.csv, READOUT*.md); out/viz_figs/ = paper figure scripts
single_agent/
  tools/               tempt300 scene generator, API runner, scorer; mechanism scripts (separate, cross_model*, act_control*, extract_acts)
  data/                aggregates: tempt300 summary/scores, act_control results, steering sweep counts, cosine tables
  vectors/             steering directions for Gemma-2-2B, Qwen2.5-1.5B, SmolLM2-1.7B + Gemma push specs; see vectors/README.md
  figs/                single-agent figure scripts
requirements/          pinned requirements per part (+ full lock files from the envs the paper used)
scripts/               secret_scan.py, check_prereg.sh
```

## Install

Two Python environments were used. Nothing here needs a GPU except the in-model mechanism scripts.

| Part | Python | Requirements |
|---|---|---|
| Keys & God, CoT judge, single-agent tools | 3.12 (paper: 3.12.16, conda env) | `requirements/keys_and_god.txt` (full freeze: `keys_and_god.lock.txt`); mechanism scripts add `requirements/single_agent.txt` |
| Betrayal study | 3.13 (paper: 3.13.12, plain venv) | `requirements/betrayal.txt` (= exact freeze `betrayal.lock.txt`) |
| Figures that `import viz` | 3.13 (any) | `requirements/figures.txt` (viz @ commit 9cda155) |

```bash
# Keys & God (+ CoT judge, single-agent tools)
conda env create -f environment.yml && conda activate swarm-p1
cd keys_and_god && pip install -e '.[dev]' && make test        # about 1000 offline tests, no network, ~1.5 min

# Betrayal study
python3.13 -m venv .venv-betrayal && .venv-betrayal/bin/pip install -r requirements/betrayal.txt

# Figure library (or: uv tool install "git+https://github.com/safety-research/viz@9cda155aabf95125c996bda51eacb259c659c78d")
pip install -r requirements/figures.txt
```

The betrayal drivers load Jenifer's package from a run directory, which must be a copy of the hash-pinned package:

```bash
cd betrayal && mkdir -p runs/v2run && cp pkg/swarm_llm/*.py runs/v2run/ && bash ../scripts/check_prereg.sh
```

## API setup

All agents are hosted models, reached through OpenRouter or the Adaption Labs Responses API. No script contains a key.
Each script reads the key from an environment variable, or from a key **file** whose path you give. Keys are never
printed or logged. Copy `.env.example` and fill in your own values:

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` or `OPENROUTER_KEY_FILE` | Keys & God (`llm.openrouter`), betrayal drivers (`harness/launch.py KEYFILE ...`; pass `-` as KEYFILE to use `OPENROUTER_API_KEY`), CoT judge |
| `ADAPTION_API_KEY` or `ADAPTION_KEY_FILE` | Keys & God `llm.adaption` (100-agent sweep, tie-in), betrayal stronger-model arms (`PROVIDER=adaption`), `single_agent/tools/tempt300_api.py` (pass `-` as the key-file argument to use the env var) |
| `SWARM_ROOT`, `PLANC_ROOT`, `LFM_ROOT` | Shell launchers and a few scripts: the folder the part runs from (`keys_and_god/`, `betrayal/`, `single_agent/`). The default is the current directory. |
| `PY` | Interpreter used by the shell launchers (default `python`) |

Some configs set `llm.adaption.key_file: null`. Those configs fall back to `ADAPTION_API_KEY`. The original 100-agent
lanes used extra keys (`extra_keys` / `infra/sweep100/lanes.py`). That was only to get more throughput; a single key
gives the same results.

**Models (all runs 7–9 November 2026).**

| Where | Model IDs |
|---|---|
| Keys & God, 10-agent dose runs | `openai/gpt-oss-120b`, reasoning effort low (OpenRouter, cheapest-provider routing) |
| Keys & God, 60-agent runs | `z-ai/glm-5.3-flash` (OpenRouter) for agents and LLM God |
| Keys & God, tie-in and 100-agent sweep | Adaption `gemma-4-31b-it`, `glm-53-flash` (low/high effort), `deepseek-v4-flash`, `qwen3.8-27b` |
| CoT suspicion judge | `typesafe/jev-1.13` (OpenRouter decisions API, `/api/alpha/decisions`) |
| Betrayal study | `qwen/qwen-2.5-7b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `google/gemma-3-12b-it`, `openai/gpt-oss-20b`, `microsoft/phi-4`, `mistralai/mistral-small-3.2-24b-instruct`, `google/gemma-3-27b-it` (OpenRouter, temperature 0.7, 160 max new tokens; multi-model replication: temperature 0.1, top_k 5) |
| Betrayal stronger models | Adaption `gemma-4-31b`, `gemma-4-26b-a4b`, `qwen3.8-27b`, `deepseek-v4-flash` |
| Single-scene test | Adaption `glm-53-flash` (low, high), `deepseek-v4-flash` (high), `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `qwen3.8-27b` (low) |
| Mechanism (local) | `google/gemma-2-2b-it`, `Qwen/Qwen2.5-1.5B-Instruct@989aa79`, `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2` |

Hosted models can change behind a fixed ID, and provider routing varies. Expect small numeric drift on a rerun, with
the same direction of effect.

## Reproduce each result

Plain names follow the paper. Internal codes, used in file names, are given in parentheses (paper Appendix N).
Costs and runtimes are what the paper's runs used. The runtimes are wall-clock times from the run logs, at the
concurrency shown in each launcher. Every betrayal command below runs from `betrayal/`, with
`PYTHONHASHSEED=0 MPLBACKEND=Agg` and the key file as the first argument of `harness/launch.py`. The generic form is:

```bash
OUT=out/<run> ARMS=<code> SEEDS_<code>=<n> COST_CAP=<usd> PAR_CELLS=10 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  python -s -P -u harness/launch.py "$OPENROUTER_KEY_FILE" runs/v2run harness/run_v2.py
python -I harness/analyze_<code>.py out/<code>_v2_2 out/<run>          # writes results_*.json + figs/*.csv
```

### Betrayal study (`betrayal/`)

| Paper result | Run (launcher / settings) | Seeds | Analysis → expected output | Cost | Runtime |
|---|---|---|---|---|---|
| Pre-registered Qwen study (A) and mixed swarm (D), with deterrence, scaling and validity arms (B, C, E); App. F | `run_v2.py`, `ARMS=A,B,C,D,E SEEDS_A=20 SEEDS_B=20 SEEDS_C=10 SEEDS_D=10 SEEDS_E=10`; extension `SEED_START_*`; merge with `harness/merge_runs.py` | A, B: 0–19; C, D, E: 0–9 | `harness/analyze_v2.py out/final_v2_1 out/final_v2_1/figs` → `out/final_v2_1/results.json`, `figs/fig1–5*.csv` | $5.66 | ~10 min per part (3 parts) |
| Stronger models (F); App. C | `runs/launch_armF_main.sh` (Adaption; `SEED_START_F=1 SEEDS_F=10`) | 8 seeds analysed (paper App. C) | `harness/analyze_arm_f.py` → `out/arm_F_v2_2/results.json`; `harness/analyze_by_model.py` → `out/by_model/figs/fig_by_model_F.csv` | Adaption (not in the OpenRouter tallies) | — |
| Who gets turned in the mixed swarm; App. C | (A–E logs) | — | `harness/analyze_by_model.py` → `out/by_model/figs/fig_by_model_D.csv` | $0 | seconds |
| Qwen outside briber (G), growth condition (GR) | `ARMS=G` (and `GR`) | 0–9 | `harness/analyze_g.py`, `analyze_gr.py` → `out/arm_G_v2_2/results_arm_G.json`, `results_growth.json` | $0.93 | ~5 min |
| Outside briber on stronger models (G on Adaption) | `runs/adaption_queue.sh` | 0–2 | `analyze_g.py` → `out/arm_G_ada_v2_2/results_arm_G.json` | Adaption only | — |
| Model mix (H) | `ARMS=H` | 0–4 | `harness/analyze_h.py` → `out/arm_H_v2_2/results_arm_H.json` | $0.41 | ~5 min |
| Swarm size (I), insider damage by size (Fig. 1c; App. D figure) | `ARMS=I` | 0–9 (8–40 per point with H, A–E) | `harness/analyze_i.py`; `analysis/size_keydamage.py` → `out/size_keydamage_honest.csv`, `out/arm_I_v2_2/figs/figI1_*.csv`, `figI2_*.csv` | $0.62 | ~10 min |
| Where insider harm comes from (withholding vs defection); App. D | (A–E, I logs) | — | `harness/analyze_p1.py out/p1_review` → `out/p1_review/p1_damage_decomposition_and_ledger.csv` | $0 | seconds |
| Payment schedule (J), ledger rule; App. H | `ARMS=J` | 0–19 | `harness/analyze_j.py` → `out/arm_J_v2_2/results_arm_J.json`; `analysis/recompute_armJ_ledger.py` → `out/p1_review/p1_ledger_by_payment_schedule_armJ.csv` | $1.05 | 11 min |
| Outsider entry stage (K) | `runs/launch_K.sh` (seed count set from the remaining budget: 4) | 0–3 | `harness/analyze_k.py` → `out/arm_K_v2_2/results_arm_K.json` | $0.37 | 7 min |
| **Gemma head-to-head (U)**: bribe vs forgery (Fig. 1b, Sec. 4.1–4.3), defenses (App. K, Fig. "Defenses on Gemma-3"), message content (App. L) | `ARMS=U SEEDS_U=36` | 0–35 | `harness/analyze_u.py` → `out/arm_U_v2_2/results_arm_U.json`, `headtohead.json`, `figs/figU_cells.csv`; `analysis/recompute_armU.py` → `out/recompute_armU.txt` | $6.97 (+$0.22 pilot) | 80 min |
| Factorial replication (UF); App. G | `ARMS=UF SEEDS_UF=10` | 0–9 | `harness/analyze_uf.py` → `out/arm_UF_v2_2/results_arm_UF.json`, `figs/figUF_effects.csv` | $1.61 | 20 min |
| Payment timing (T), tie-split ledger (Figs. 3a/3b); App. H | `runs/launch_T_C16.sh` (first half) | 0–39 | `harness/analyze_t.py` → `out/arm_T_v2_2/results_arm_T.json`; `analysis/armT_ledger_by_round_tiesplit.py` → `figs/figT_detection_over_rounds_tiesplit.csv` | $0.57 | 8 min |
| Contagion and bribe size at 8 agents (C); App. H | `runs/launch_C.sh` | 0–39 | `harness/analyze_c.py` → `out/arm_C_v2_2/results_arm_C.json`, `figs/figC_contagion.csv` | $0.54 | 7 min |
| Contagion at 16 agents (C16), N8 vs N16 (Fig. "Contagion at 8 vs 16") | `runs/launch_T_C16.sh` (second half) | 0–39 | `analyze_c.py` → `out/arm_C16_v2_2/`; `harness/compare_c_n.py` → `figs/figC_N8_vs_N16.csv` | $1.18 | 18 min |
| Bribe-size dose at 16 agents (C2K + C16; Fig. "Bribe-size dose-response") | `runs/launch_C2K.sh` | 0–39 | `harness/analyze_dose.py` → `out/dose_N16_v2_2/results_dose_N16.json`, `figs/figDose_N16.csv` | $0.73 | 9 min |
| Disclosure bounty (DB); App. H | `runs/launch_DB.sh` | 0–19 | `harness/analyze_db.py` → `out/arm_DB_v2_2/results_arm_DB.json` | $0.22 | 4 min |
| Social-engineering levers (SE); App. I | `runs/launch_SE.sh` | 0–19 | `harness/analyze_se.py` → `out/arm_SE_v2_2/results_arm_SE.json`, `figs/figSE_levers.csv` | $0.33 | 5 min |
| **Dual control (DC), pre-registered**; App. J | `runs/launch_DC.sh` (driver checks `prespec_dual_control.md` against `PREREG_DC_SHA256.txt`) | 0–39 | `harness/analyze_dc.py` → `out/arm_DC_v2_2/results_arm_DC.json`, `figs/figDC_cells.csv` | $0.70 | 10 min |
| **Pre-registered follow-ups (JF123, JF4, JF5, JF6)**; App. O | `runs/launch_jenfu.sh` → `harness/run_jenfu.py` | 300–319 | `harness/analyze_jenfu.py out/jenfu_results out/jenfu_JF*` → `out/jenfu_results/results_jenfu.json`, `jenfu_cells.csv`, `analysis_stdout.txt` | $1.91 | 30 min |
| **Pre-registered multi-model replication (prespec_noor)**; App. P | `runs/launch_noor.sh` (Gemma slot probe `harness/noor_gemma_probe.py`, then 4 models in parallel) | 700–723 | `harness/analyze_noor.py` → `out/noor_analysis/results_noor.json`, `noor_primary.csv`, `noor_cells.csv`; post hoc `harness/explore_noor_scripted.py` → `noor_exploratory_scripted.csv` | $2.18 | 40 min |
| Provider-timeout confirmation rerun (TOFIX, pre-registered); App. B | `runs/launch_tofix.sh` → `harness/run_tofix.py` | 800–819 | `harness/analyze_tofix.py` → `out/tofix/analysis/results_tofix.json`, `out/tofix/RESULTS.md` | $0.30 | 10 min |
| Family-wide correction; App. M | — | — | `analysis/familywide_correction.py` → `out/familywide_correction.csv`, `.md` | $0 | seconds |
| Empty replies and timeout sensitivity; App. B | — | — | `analysis/empty_reply_audit.py`, `analysis/timeout_sensitivity.py` → `out/empty_reply_audit.md`, `out/timeout_sensitivity.md` | $0 | seconds |

Every analysis script reads the raw run logs (`games.jsonl`, `actions.jsonl`, ...) of a fresh run. The raw logs are
not included (see Notes). The aggregates they produced are included, so every figure and table can be rebuilt without
the raw logs (next section).

### Keys & God (`keys_and_god/`)

Run from `keys_and_god/`. A single game is `python -m swarm.run --config configs/<name>.yaml`. The paper's sweeps used
the grid driver, which runs worlds with matched God / no-God / noise branches and stops at a spend cap:
`python infra/grid_driver.py --cell <config>:<cap_usd>:<concurrency>:<opts> --seeds <n> --rounds <R> --ledger runs/<ledger>.jsonl --ledger-stop <usd>`.
The exact launch lines are in `infra/sweep30/launch*.sh`, `infra/sweep60/*.sh` and `infra/sweep100/run_queue.sh`.

| Paper result | Code / config | Worlds (seeds) | Analysis → output | Cost | Runtime |
|---|---|---|---|---|---|
| Forger dose curve K = 1, 2, 4, 6 and the every-round forger K = 18 (E1); board off; Fig. 1a, App. D "Forger dose curve" | **`legacy_e1/`** (commit cf2c46b): `configs/sweep/e1.yaml`, `k1.yaml`, `k4.yaml`, `board_off.yaml`, `configs/grid_oss_sim_k2.yaml`, `grid_oss_sim_k6.yaml`; launcher `legacy_e1/infra/sweep/launch.sh` | `run.seed 1`, game_idx 0–15 (8 or 16 worlds) | `python -m swarm.analysis.sweep_stats e1=runs/sweep_e1 k1=runs/sweep_k1,runs/b2_sweep_base ...` (full line in `results/E1_RESULTS.md`) → `results/e1_stats.json`; `legacy_e1/infra/e1/e1_extra.py` → `results/e1_extra.json` | E1 $0.79 (+$0.03 smoke) | E1: 62 min |
| 60 agents, LLM God, Realistic Env (booking portal), God dose k = 0/1/5; App. C | `configs/sweep60_or_glm.yaml`, `sweep60_or_glm_k5.yaml` | seed 1, worlds 0, 1, 4, 6, 16 | `swarm/analysis/sweep_metrics.py` → `results/SWEEP60_OR_LIVE.md`, `results/god_dose_k5_metrics.md` | $2.52 per game (k1 grid $25.19) | hours (80 rounds × 60 agents) |
| Game Env vs Realistic Env, ± God (2×2); render-fixed rerun; C2/C3 controls; App. C | `sweep60_or_oldframe.yaml`, `sweep60_or_oldframe_k1.yaml`, `sweep60_or_glm_i2fix.yaml`, `sweep60_or_c2_maxtok.yaml`, `sweep60_or_c3_minimal.yaml`; launchers `infra/sweep60/queue_*.sh`, `infra/sweep30/resume_after_403.sh` | worlds 0, 1, 4 | `infra/sweep30/final_report.py` → `results/SWEEP30_FINAL_RESULTS.md` | part of $148.82 total OpenRouter spend | hours |
| 30-agent framing sweep (sweep30 A1–A7, B0–B10) | `configs/sweep30_*.yaml`, `infra/sweep30/launch_all.sh` | 3 worlds | `infra/sweep30/final_report.py` → `results/SWEEP30_FINAL_RESULTS.md` | $20.38 | ~2 h |
| Refusal-broadcast tie-in (gemma-4-31b, 20 agents); App. C | `configs/noor_tiein_gemma.yaml` (+ `noor_tiein_v_*.yaml`), generated by `infra/noor_tiein/make_config.py` | worlds g00, g01 | `single_agent/data/tiein_final/tiein_final.py` → `tiein_final_by_game.csv`, `tiein_final_summary.json` | $6.02 (Adaption) | ~1 h |
| CoT suspicion (Jev-1.13 judge): Game vs Realistic, and God dose; App. C | `cot_suspicion/run_jev113*.sh`, `cot_suspicion/god_dose/run_god*.sh` (read the 60-agent run dirs) | as above | `cot_judge.py` → `cot_suspicion/summary.json`, `per_round.csv`; `god_dose/summary.json`, `per_round.csv` (`add_ta2.py` merges the test-awareness pass) | judge $0.89 and $2.41 (cap $6.50 per run) | ~20 min |
| 100-agent sweep (configs only; not reported in this paper) | `configs/sweep100_*.yaml`, `adaption_*_nogod.yaml`, `infra/sweep100/` | — | `swarm/analysis/sweep_metrics.py` | Adaption | — |

### Single agents and the model's inside (`single_agent/`)

| Paper result | Command | Output | Cost |
|---|---|---|---|
| Single-scene test, 300 scenes × 6 hosted setups (Sec. 3.3, Fig. 1/App. D panel a) | `python -I tools/tempt_states300.py states.json` (seed 2026), then `python -I tools/tempt300_api.py states.json results.jsonl "$ADAPTION_KEY_FILE"`, then `python -I tools/tempt300_report.py states.json results.jsonl data/tempt300/` | `data/tempt300/summary.json`, `scored.json`, `obey_breakdown.json`, `TEMPT300.md` | $1.12 |
| Act-vs-wait control for the obey/help overlap (Sec. 4.4) | `ACT_CACHE=<dir> python -I tools/act_control.py` (GPU), then `ACT_CACHE=<dir> python -I tools/act_control_save_dirs.py` for the directions | `data/act_control/RESULTS.md`, `act_control.json`, `act_control_layers.csv` | local GPU |
| Obey vs help geometry, cross-model and letter-axis control (App. E) | `tools/separate.py`, `SAVE_DIRECTIONS=<dir> python -I tools/cross_model.py`, `tools/cross_model_letterfix.py` (GPU) | `data/noor_steer/separate/cos_by_layer.csv`, `cos_cross_model.csv`, `cross_model.md`, `letterfix.csv`, `mcnemar_obey_only_f0.1.json` | local GPU |
| Probe-switch trade-off sweep, neuron push, held-out finals (Sec. 4.4, App. E) | `tools/noor_steer.py` with `vectors/gemma-2-2b-it/neuron_push/*` (C9 probe = `obey_11`) | `data/noor_steer/sweep.jsonl`, `tempt300.jsonl` | local GPU |
| OBEY / MATCHED / OBEY_ONLY / OBEY2 steering (App. E) | `tools/separate.py --cfgs ...` with `vectors/gemma-2-2b-it/directions.npz` | `data/noor_steer/separate/separate.csv`, `separate.json`, `mcnemar_obey_only_f0.1.json` | local GPU |
| Team steering, SAE/dim units (App. E background) | `tools/steer_team.py`, `tools/steer_team_holdout.py` with `vectors/gemma-2-2b-it/team/*` + public Gemma Scope SAEs | team-steering tables | local GPU |

**Released steering vectors:** `single_agent/vectors/` holds unit directions for all layers of the three small open
models studied. Gemma-2-2B-it has OBEY, HELP, MATCHED, OBEY_ONLY and OBEY2 (`directions.npz`), plus the neuron-push
and SAE/residual "team" push specs. Gemma-2-2B-it, Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct each have
OBEY, HELP, MATCHED and OBEY_ONLY (`cross_model.npz`; Gemma's are in `directions.npz`), and the act-vs-wait control's
letter-balanced OBEY, HELP, ACT, ACT_board and residual directions (`act_control.npz`). Total size is about 7 MB. `single_agent/vectors/README.md` documents
each file: model, layers, how it was built, and the result it reproduces. The `cross_model` vectors
were regenerated on 9 Nov 2026 by rerunning the unchanged script, which reproduced its published outputs
byte-for-byte. The `act_control` vectors were saved from the act-control pipeline's pair differences
(`act_control_save_dirs.py`). OBEY2 exists for Gemma only.

To **recompute** the directions, the mechanism scripts need inputs that are not in this release: local model weights
under `$LFM_ROOT/models/`, the A/B contrast pairs, the benign scene sets, the agent-name list from a Keys & God run, and
the cached activation arrays. To **apply** the released vectors, you need only the model weights and the scene
prompts. See Notes.

## Reproduce the figures from the included aggregates (no API)

All of these were run on a clean copy of this folder before the first commit. Each one wrote its figure.

```bash
# Paper Figure 1 and the betrayal-study appendix figures (writes PDF/PNG next to the scripts)
cd betrayal/out/viz_figs
viz run fig_P1_figure1.py                 # Fig. 1 (a: e1_stats.json, b: arm U, c: size_keydamage_honest.csv)
viz run make_figs.py                      # fig1_bribe_vs_forgery, fig2_dose_N16, fig3a/3b paytiming, fig4_dual_control,
                                          # fig5_se_levers, fig6_defenses_gemma, fig7_contagion_N8_N16
viz run fig_P1_appD_damage_by_model.py    # App. D insider damage by swarm size
viz run fig_probe_switch_tradeoff.py      # App. E probe-switch trade-off (single_agent/data/noor_steer/sweep.jsonl)
viz run figC_contagion_clean.py figC_contagion.png
python tiein_final_clean.py tiein_final.png        # App. C tie-in (matplotlib only)
python schematic_clean.py .                         # App. D Keys & God schematic (needs keys_and_god deps)

# Keys & God
cd keys_and_god/cot_suspicion/god_dose && viz run make_fig_mean.py per_round.csv cot_suspicion_god_dose_mean
cd keys_and_god/figs && viz run god_dose_k0_k1_k5_viz.py && viz run framing_2x2_god_viz.py \
  && viz run cot_suspicion_summary_viz.py && viz run cot_suspicion_game_vs_realistic_viz.py
python keys_and_god/figs/fig_forgery_dose.py fig_forgery_dose.png   # App. D forger dose curve (matplotlib only)
python keys_and_god/figs/framing_game_vs_realistic.py          # App. C Game Env vs Realistic Env (matplotlib only)

# Single agents
cd single_agent/figs && viz run fig_bridge_reward.py && viz run make_noor_figs.py
```

Some scripts (`fig_forgery_dose.py`, `fig1_headline.py`, `god_dose_k0_k1_k5_viz.py`, `framing_2x2_god_viz.py`)
contain their numbers inline. A comment next to each number names the results file it came from. The rest read the
aggregate files listed above. `framing_game_vs_realistic.png` (App. C): its original script was not saved, so
`python keys_and_god/figs/framing_game_vs_realistic.py` rebuilds it (plain matplotlib). It reads the per-world rows of
`keys_and_god/results/SWEEP30_FINAL_RESULTS.md`, asserts the paper's round-40, final-welfare and winner numbers, and
draws the same values as the paper figure (means r20–r80: Realistic 0.598/0.804/0.926/0.965, Game
0.498/0.635/0.726/0.878; winners 59/60/43 vs 37/43/41).
`keys_and_god/figs/agent_god_*` and `god_*_compact_viz.py` are supporting figures not in the paper, and they need
the raw game logs. `cot_eval_awareness_viz.py` also needs raw judge output and will not run from this release.

## Pre-registrations

Each pre-registration hashes the body of its prespec (line 2 to the end; line 1 holds the hash) and the frozen code.
The drivers `run_jenfu.py`, `run_noor.py`, `run_closers.py` and `run_tofix.py`, and arm DC in `run_v2.py`, **refuse to
start** unless every listed hash matches. The A–E study recorded `prereg_sha256` in its `results.json`. Run
`bash scripts/check_prereg.sh` to recheck all 41 entries plus the package. All of them match in this release.

| Study | Hash file | Prespec (body sha256) | Frozen code |
|---|---|---|---|
| Pre-registered Qwen study (A–E), H1–H8 | `PREREG_SHA256.txt` | `HYPOTHESES.md` cd480e82… | `harness/battery.py` |
| Dual control (DC) | `PREREG_DC_SHA256.txt`, `PREREG_DC_CODE_SHA256.txt` | `prespec_dual_control.md` 71a62d7a… (whole file) | `dual_control.py`, `analyze_dc.py`, `run_v2.py`, tests, event schema |
| Follow-ups (JF123–JF6) | `PREREG_JENFU_SHA256.txt` | `prespec_jenfu.md` 766ebd0c… | `run_jenfu.py`, `jenfu_patch.py`, `outsider_rumour.py`, `analyze_jenfu.py` |
| Multi-model replication | `PREREG_NOOR_SHA256.txt` | `prespec_noor.md` 404f9f25… | `run_noor.py`, `short_prompt.py`, `noor_decode.py`, `noor_gemma_probe.py`, `outsider_bribe.py`, `outsider_rumour.py`, `u_factors.py`, `analyze_noor.py` |
| Closing experiments | `PREREG_CLOSERS_SHA256.txt` | `prespec_closers.md` cf5dbf00… | `run_closers.py`, `outsider_rumour.py`, `bribe_paraphrase.py`, `analyze_closers.py` |
| Timeout confirmation (TOFIX) | `PREREG_TOFIX_SHA256.txt` | `prespec_tofix.md` 47ab4a2e… | `run_tofix.py`, `tofix_decode.py`, `analyze_tofix.py`, `identity_check_tofix.py` and 7 reused modules |
| Game package | `pkg_module_sha256.txt` | — | 17 modules of `pkg/swarm_llm/` |

The frozen files are byte-identical to the registered versions, so the release does not edit them. One of them,
`harness/test_dual_control.sh`, still contains two absolute paths from the original machine. Set the paths at its top
before you run it.

## Notes and caveats

- **Provider timeouts were played as idle moves.** OpenRouter sometimes returns HTTP 200 with a "Provider timed out"
  error body. The drivers up to the Gemma head-to-head and the follow-ups treated these as empty replies, so the agent
  idled that turn: 7.6–8.1% of Gemma replies (App. B, O). The later drivers (`run_tofix.py`, `harness/tofix_decode.py`)
  retry them and count failures. The pre-registered TOFIX rerun (20 new seeds; all first-try timeouts recovered)
  replicated bribe-only vs lie-only (+0.65 [0.20, 1.10], Holm p = 0.034). Even with timeouts fixed, 14–21% of turns
  remained forced idles from replies that did not parse.
- **Tie-split ledger.** The ledger detector ("name the agent with the most unpaid promises") can tie. Arm T is reported
  with ties split evenly (`analysis/armT_ledger_by_round_tiesplit.py`, `figT_detection_over_rounds_tiesplit.csv`). The
  unsplit version (`figT_detection_over_rounds.csv`) is kept for comparison.
- **Gemma-3-12B → 27B.** In the multi-model replication, the pre-registered Gemma slot used Gemma-3-12B only if its one
  top_k-honouring provider passed a 16-call probe. It passed 1 of 16, so the run used Gemma-3-27B (16 of 16), as the
  prespec required (`out/noor_prereg/gemma_slot.json`).
- **Two engine versions.** The 10-agent dose runs (E1, K = 1–6, board off) ran at commit `cf2c46b`, which is shipped
  as `keys_and_god/legacy_e1/` (plus its `infra/e1/` analysis from the next commit). The 60-agent and later runs used the
  framing branch in `keys_and_god/`. Config hashes in run manifests cover game semantics only. Spend caps, paths and
  concurrency are not hashed, so the release's path edits leave them unchanged.
- **Exploratory vs confirmatory.** Only the A–E study, dual control, the follow-ups, the multi-model replication and
  TOFIX were pre-registered. Everything else is exploratory, marked † in the paper.
- **Release edits to non-frozen code:** absolute paths became `SWARM_ROOT` / `PLANC_ROOT` / `LFM_ROOT` / `PY`
  variables or file-relative paths. Key loaders also accept environment variables. One Adaption unit test now sets a
  fake key instead of reading a local key file. `cross_model.py` gained an optional `SAVE_DIRECTIONS` output, and
  `act_control_save_dirs.py` writes into `single_agent/vectors/`. Neither changes any computed result.

### What is withheld, and why

- **Steering vectors.** As the paper states, we release the steering vectors and directions for the small open
  models studied (Gemma-2-2B-it, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct; `single_agent/vectors/`). They are
  direction files and push specs only. Our own results show these pushes trade obedience for helpfulness rather than
  giving a clean control (Section 4.4, App. E). No vectors are released for any other model.
- **No cached activations or trained weights.** Whole-dataset activation arrays (`acts_*.npy`, `nat_*.npy`,
  `traces.npz`), LoRA weights (`lora_*.pt`) and DPO artifacts are not included. Vectors for LFM2.5-2.6B, which is
  not part of this paper, are also left out.
- **No DPO training pairs and no A/B contrast pairs** (`pairs_*.jsonl`, `states_v1pairs.json`, preference data).
  Without them `separate.py`, `cross_model*.py`, `act_control.py` and `extract_acts.py` cannot recompute the
  directions from this folder. Their aggregate outputs (cosines, counts, McNemar tests) and the resulting direction
  vectors are included.
- **No raw logs.** Per-call logs (`games.jsonl`, `actions.jsonl`, `messages*.csv`, LLM call caches, chains of thought,
  raw judge outputs) are not included. They are large, they hold full model outputs including generated attack
  messages, and some came through third-party APIs. The analysis scripts regenerate everything from a fresh run.
- **No keys or credentials.** Every key comes from an environment variable or a key-file path you supply.
  `python -I scripts/secret_scan.py` checks the tree. It prints nothing when clean. The allowlist in the script
  explains every pattern match it accepts: key prefixes, fake test canaries, file hashes, and the one frozen file with
  local paths.
- **Attack messages.** The attacks are one-line templates: bribes ("send me a key first and I'll give you info N"),
  rumours and peer-name forgery. The betrayal study's social-engineering arm left out the authority lever by design.
- **God-voice orders.** The "God" orders in `single_agent/tools/tempt_states*.py`, `separate.py` and
  `cross_model*.py` are fictional in-game orders from the game's scripted God character (scriptural wording, including
  the unused `--escalate` variant). They are used only for the single-agent obedience study and are needed to
  reproduce the fake-boss result. They are kept unchanged.

## Authorship and credit

| Component | Author |
|---|---|
| Keys & God engine, framing sweeps, 60/100-agent runs, CoT suspicion judge (`keys_and_god/`) | Iraj Moradi |
| Betrayal study: game package `betrayal/pkg/swarm_llm/` (Jenifer's key-trading game, used unmodified, hash-pinned) and all of its arms, pre-registered experiments and readouts | Jenifer Nguyen |
| Single-agent tests (tempt300), in-model mechanism and steering work, multi-model replication design (`single_agent/`, `prespec_noor.md`) | Noor |

Harness and analysis code for the betrayal study was written during the sprint (with LLM coding assistants, as stated
in the paper's LLM Usage Statement). `betrayal/pkg/swarm_llm/` is a hash-pinned copy of Jenifer's package (an earlier
snapshot lived in a separate project folder). The harness wraps it through monkey-patching modules (`fastpaths.py`,
`v2patch.py`, `jwin_fixes.py`, ...) and never edits it.

**License:** Code is released under the MIT License (see LICENSE).

## Citation

```
Noor, Lance Santiago, Iraj Moradi, Jenifer Nguyen, Mario Sepriano (equal contribution; author order randomized). Offers, Not Rumours: What Turns Agents in an LLM Swarm.
Swarm Alignment Hackathon (Apart Research), Track 2: The Science of Agent Networks, November 2026.
```
