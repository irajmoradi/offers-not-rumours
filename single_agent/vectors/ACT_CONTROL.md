# OBEY / HELP / ACT direction vectors

One file per model. In this release the file is `<model>/act_control.npz` and the manifest is `act_control_manifest.json`; the original run named them `<model>/directions.npz` and `manifest.json`. Built by `tools/act_control_save_dirs.py` from the pair differences that `tools/act_control.py` collected (CPU only; no new extraction). `manifest.json` lists the models, layer counts and the mid-depth cosines recomputed from these files (they match `single_agent/data/act_control/RESULTS.md`).

| folder | HF model | layers | d_model | mid-depth layers (0.4-0.7) |
|---|---|---|---|---|
| `gemma-2-2b-it` | google/gemma-2-2b-it | 26 | 2304 | 10-17 |
| `qwen2.5-1.5b-it` | Qwen/Qwen2.5-1.5B-Instruct@989aa79 | 28 | 1536 | 11-18 |
| `smollm2-1.7b-it` | HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2 | 24 | 2048 | 10-16 |

## Arrays (float32, shape `[n_layers, d_model]`, every row unit length)
- `OBEY`: 30 trust pairs (`out/gemma/pairs_trust.jsonl`). Obey the fake authority (defect or give as told) minus refuse (message to verify).
- `HELP`: 100 honest benign31 training scenes (seed 6161). "Offer the partner the fair trade" minus "Post on the board and wait".
- `ACT`: 100 ambiguous scenes (seed 7171, `single_agent/data/act_control/act_scenes.json`). "Do the trade" minus "Hold off and wait"; a neutral contact, no authority or fairness cue, unclear value.
- `ACT_board`: the same 100 scenes, with the hold option worded exactly like HELP's ("Post on the board and wait."). A sensitivity check only; it is nearly identical to HELP.
- `OBEY_resid`, `HELP_resid`: OBEY and HELP with ACT removed (one Gram-Schmidt step per layer, `v - (v·a)a`), then renormalized.
- `letter_axis`: the shared "letter A vs letter B" direction. This is the unit mean of act(A) − act(B) over all OBEY and HELP pairs. It was projected out of every pair difference before averaging.
- Index arrays:
  - `layers` = 0..n−1.
  - `mid_layers` = the 0.4-0.7 depth window used for all reported numbers.
  - `depth` = layer / (n−1).

## How each direction was built
1. Activation: the residual-stream output of decoder layer L (`model.model.layers[L]` forward hook), at the answer-letter token. The prompt is the chat template, then "Answer: (", then the letter. Models ran in bf16 and activations were saved as fp32.
2. Pair difference: act(prompt + positive letter) − act(prompt + negative letter), per scene. The positive option is shown as (A) in half the scenes.
3. Project `letter_axis` out of each pair difference.
4. Average over scenes and normalize to unit length.

The vectors carry no scale. To steer, multiply by a size you choose; Noor's push scripts scale by a fraction of the typical activation norm.

Layer index L means the output of block L, so it is `hidden_states[L+1]` in HF `output_hidden_states`.
