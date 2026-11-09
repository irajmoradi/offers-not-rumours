# Steering vectors and directions (small open models)

Directions and steering specs for the three small open models studied in the paper. All were run locally with plain
`transformers` (bf16, no `trust_remote_code`):

| Model | HF id (revision) | Hidden size | Layers |
|---|---|---|---|
| Gemma-2-2B-it | `google/gemma-2-2b-it` | 2304 | 26 |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct@989aa79` | 1536 | 28 |
| SmolLM2-1.7B-Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2` | 2048 | 24 |

Every vector is float32 and unit length, stored under the key `<name>_<L>`. Layer `L` is the residual stream
**output of decoder layer L** (`hidden_states[L+1]`), 0-indexed, read at the answer-letter token of an A/B prompt that
ends in `Answer: (`. Nothing here is a cached activation of a dataset, a LoRA/DPO weight, or a training pair.

| File | Size | Contents |
|---|---|---|
| `gemma-2-2b-it/directions.npz` | 1.2 MB | OBEY, HELP, MATCHED, OBEY_ONLY, OBEY2 × 26 layers (`separate.py`) |
| `gemma-2-2b-it/act_control.npz` | 1.7 MB | act-vs-wait control: `OBEY`, `HELP`, `ACT`, `ACT_board`, `OBEY_resid`, `HELP_resid`, `letter_axis`, each `[26, 2304]` (`act_control_save_dirs.py`) |
| `gemma-2-2b-it/neuron_push/` | 35 KB | neuron-level push specs (selection, calibration, C9 probe, sweet-spot setting) |
| `gemma-2-2b-it/team/` | 25 KB | SAE/residual-dim "team" push specs and causal unit ranking |
| `qwen2.5-1.5b-it/cross_model.npz` | 0.7 MB | OBEY, HELP, MATCHED, OBEY_ONLY × 28 layers (`cross_model.py`) |
| `qwen2.5-1.5b-it/act_control.npz` | 1.2 MB | as for Gemma, `[28, 1536]` |
| `smollm2-1.7b-it/cross_model.npz` | 0.8 MB | OBEY, HELP, MATCHED, OBEY_ONLY × 24 layers |
| `smollm2-1.7b-it/act_control.npz` | 1.4 MB | as for Gemma, `[24, 2048]` |

There is no Gemma `cross_model.npz`. `cross_model.py` builds Gemma's OBEY/HELP/MATCHED/OBEY_ONLY with the same code
as `separate.py`, and the result matches `directions.npz` (cosine ≥ 0.9999999 at every layer).

**Provenance.** `directions.npz` and the push specs are the files saved by the original runs (8 Nov 2026). The
`act_control.npz` files were written by `tools/act_control_save_dirs.py` (Noor's act-control pipeline, 9 Nov 2026) from
the pair differences `act_control.py` collected. Their documentation is `ACT_CONTROL.md` and their mid-depth cosines
are in `act_control_manifest.json`. The original `cross_model.py` run saved only cosines and statistics. On 9 Nov 2026
we reran the unchanged script on the same machine (RTX 5090, same env) with the new optional `SAVE_DIRECTIONS` output.
The rerun reproduced `cos_cross_model.csv`, `cross_model_causal.json` and `cross_model.md` byte-for-byte. An
independent rerun of `act_control.py` reproduced `act_control.json` and `act_control_layers.csv` byte-for-byte too,
and gave the same vectors (cosine 1.0 at every layer). So these vectors are the ones behind the paper's numbers.

## `cross_model.npz` (Qwen2.5-1.5B, SmolLM2-1.7B)

Same construction as Gemma's `directions.npz`, keys `obey_<L>`, `help_<L>`, `matched_<L>`, `obey_only_<L>`. OBEY uses
the 30 trust pairs, HELP the 100 benign training scenes (seed 6161), and MATCHED the same scenes with the same "give
key K to P" option under a fake-authority DM vs the fair partner. OBEY_ONLY is OBEY with HELP projected out. Build:
`SAVE_DIRECTIONS=<dir> python -I tools/cross_model.py`. They reproduce App. E "Direction geometry": obey-help
mid-depth cosine 0.35 (Qwen) and 0.32 (SmolLM), split-half 0.86 / 0.89, in `data/noor_steer/separate/cos_cross_model.csv`.
They also reproduce the mid-layer push test (`cross_model_causal.json`): `+/- 0.25 · median‖h‖ · v` at the "Answer: ("
position, at layer 14 (Qwen) and 12 (SmolLM). The letter-axis control is computed by `cross_model_letterfix.py`.

## `act_control.npz` (all three models)

These are from the act-vs-wait control (Sec. 4.4, App. E "Act-vs-wait control"). Unlike the other files, each array
is `[n_layers, d_model]` with unit rows, plus index arrays `layers`, `mid_layers` (the 0.4–0.7 depth window) and
`depth`. Per layer, the A/B `letter_axis` (unit mean act(A) − act(B) over the OBEY and HELP pairs) is projected out of
every pair difference first. Each direction is then the unit mean over pairs:

- `OBEY`: the 30 trust pairs, letter-balanced (15 with obey as A).
- `HELP`: the 100 benign scenes, fair trade vs post-and-wait.
- `ACT`: 100 ambiguous-trade scenes (`data/act_control/act_scenes.json`, seed 7171), "Do the trade" vs "Hold off and
  wait". No authority or fairness cue.
- `ACT_board`: the same scenes vs HELP's "Post on the board and wait" wording (sensitivity check).
- `OBEY_resid`, `HELP_resid`: OBEY and HELP with `ACT` removed (one Gram-Schmidt step), renormalised.

Build: `ACT_CACHE=<dir> python -I tools/act_control.py` (GPU), then `ACT_CACHE=<dir> python -I
tools/act_control_save_dirs.py` (CPU). They reproduce the residual obey-help cosine falling from 0.465 to 0.115 (Gemma),
0.351 to 0.239 (Qwen) and 0.317 to 0.062 (SmolLM), with ACT split-half 1.00 (`data/act_control/act_control.json`,
`act_control_layers.csv`, `RESULTS.md`). Full notes: `ACT_CONTROL.md`.

## `neuron_push/`

The "neuron push" interventions in Section 4.4 and App. E (α 0.1 cuts obeying 80 → 3 of 300; the 37-setting sweep;
the held-out finals 49/300 and 55/300) do not add a residual vector. They add a per-neuron delta to the MLP
`down_proj` input at selected neurons in layers 12–16. The files here specify those pushes:

- `select.json`: neuron selection (`top10_attr`, `top20_by_layer`, `sae_matched`) and per-layer residual-norm
  calibration (`calibration`, `gemma_calibration`, median ‖h_L‖ for L12–16).
- `fraction_calib.json`: for each neuron `layer:index` and strength α, the activation delta that achieves the target
  write (`delta_a`, `achieved`, `target`).
- `fphase_selection.json`, `sweep_selection.json`: the settings picked on training scenes, and the baseline.
- `c9_probe.json`: the C9 probe (layer 11, threshold τ, AUC 0.53).
- `sweetspot_SW_a0.06_C9wnone_tau-.json`: the full spec of one swept setting (neurons + `delta_a`), and
  `sweetspot_t2.json`, its check on 100 more scenes.

Rebuild and run: `python -I single_agent/tools/noor_steer.py --help` (GPU). `noor_steer.py` also reads the cached trust
activations to recompute `obey_L`. Those activations are not released, but `directions.npz` gives the same unit
vectors.

## `team/`

"Team" steering pushes a set of causally ranked units. Each unit is either a Gemma Scope SAE latent (`sae|L|f`) or a
raw residual dimension (`dim|L|i`), in layers 12–16. The push is added on generated tokens only.

- `causal.json`: per-unit causal effects on the obey-vs-ignore logit over the 30 tempting scenes. The top 10 or 20
  units toward obey or toward ignore form "team10" and "team20".
- `teamsteer_meta.json` (30-scene run) and `teamsteer_holdout_meta.json` (300-scene holdout): `diff[unit]` is the
  unit's obey-minus-ignore difference. The push for that unit is `k · diff · W_dec[f]` for an SAE latent, or `k · diff`
  on the dimension itself, at k = 2, 4 or 8.
- The SAE decoder rows `W_dec[f]` come from the public Gemma Scope release `google/gemma-scope-2b-pt-res`,
  `layer_L/width_16k/average_l0_{82,84,84,78,78}` for L = 12..16. They are not redistributed here.

Rebuild and run: `steer_team.py <model_dir> <tempt_dir> <sae_root> team/causal.json <gemma_out> <out_dir>` (GPU).
This reproduces the team-steering table (team10 and team20 toward obey 27 of 30 at k = 2, vs 13 unsteered) and the
300-scene holdout.

## Not included

- **OBEY2 for Qwen and SmolLM.** OBEY2 ("whom to trust") was built and steered for Gemma only (`separate.py`).
- **LFM2.5-2.6B vectors.** That model is not part of this paper.
- Cached activations of whole datasets (`acts_*.npy`, `nat_*.npy`, `traces.npz`, the `ACT_CACHE` per-pair arrays,
  57–79 MB each), LoRA weights (`lora_*.pt`), DPO and contrast training pairs, and the Gemma Scope SAE weights (public,
  third-party).
