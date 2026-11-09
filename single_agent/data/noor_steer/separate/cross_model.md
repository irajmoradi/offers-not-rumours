# Cross-model check: are 'obey' and 'help' entangled in other models too?

This checks only the geometry of the directions; no steering sweep was run. All directions are A/B letter-choice CAA at the letter token, built from the same TRAINING scenes for every model:
- **OBEY:** the 30 trust pairs.
- **HELP:** 100 benign seed-6161 scenes.
- **MATCHED:** the same action under fake-authority vs fair framing.

All models are ungated, downloaded from the HF hub at pinned revisions, and run with plain transformers (no trust_remote_code).

## Cosine similarity, averaged over mid depth (0.4–0.7 of layers), with [min, max]

| model | cos(OBEY, HELP) | cos(MATCHED, OBEY) | cos(MATCHED, HELP) |
|---|---|---|---|
| google/gemma-2-2b-it (reference) | +0.46 [+0.37, +0.51] | +0.06 [-0.02, +0.18] | -0.07 [-0.21, +0.01] |
| Qwen/Qwen2.5-1.5B-Instruct@989aa79 | +0.35 [+0.06, +0.57] | +0.04 [-0.30, +0.22] | -0.07 [-0.23, +0.07] |
| HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2 | +0.32 [+0.21, +0.40] | -0.01 [-0.05, +0.01] | +0.05 [-0.00, +0.14] |

## Null and reliability at mid depth (0.4–0.7)

- **Null:** the 2.5–97.5% range of cos(OBEY, HELP) over 50 label shuffles (each pair's difference sign-flipped at random), averaged over mid-depth layers.
- **Split-half:** the mean cos between a direction built on one random half of the scenes and the same direction built on the other half (20 splits).

| model | cos(OBEY, HELP) | null 95% | cos(MATCHED, OBEY) | null 95% | split-half OBEY / HELP / MATCHED |
|---|---|---|---|---|---|
| google/gemma-2-2b-it (reference) | +0.46 | [-0.76, +0.76] | +0.06 | [-0.14, +0.12] | 0.78 / 0.97 / 1.00 |
| Qwen/Qwen2.5-1.5B-Instruct@989aa79 | +0.35 | [-0.93, +0.91] | +0.04 | [-0.14, +0.15] | 0.08 / 0.71 / 0.99 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2 | +0.32 | [-0.90, +0.87] | -0.01 | [-0.10, +0.10] | 0.44 / 0.87 / 0.99 |

## Causal check at the mid layer (push ±0.25 × median‖resid‖ at the last prompt token)

Each cell is the mean A/B log-odds of the obey letter (20 trust pairs) / the help letter (20 held-out help pairs).

| model (layer) | base | +obey | −obey | +obey_only | −obey_only | +help | −help | +matched | −matched |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it (reference) (L13) | -2.29 / +5.70 | -1.54 / +5.48 | -2.91 / +5.14 | -1.48 / +5.49 | -2.89 / +5.20 | -1.97 / +5.78 | -2.46 / +5.52 | -1.99 / +5.82 | -2.28 / +4.83 |
| Qwen/Qwen2.5-1.5B-Instruct@989aa79 (L14) | +0.22 / -0.84 | +0.46 / -0.56 | +0.08 / -0.52 | +0.46 / -0.89 | +0.08 / -0.19 | +0.20 / +0.10 | +0.03 / -2.59 | +0.17 / -0.61 | +0.23 / -0.59 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2 (L12) | +0.03 / +0.39 | +0.11 / +0.46 | -0.04 / +0.33 | +0.12 / +0.41 | -0.07 / +0.36 | +0.00 / +0.47 | +0.09 / +0.29 | +0.06 / +0.43 | +0.03 / +0.12 |

Files:
- `cos_cross_model.csv` (per layer, all models)
- `cross_model.png`
- `cross_model_causal.json`
