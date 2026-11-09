# Letter-axis control for the obey/help tangle

**What could fake the cosine.** Every A/B pair difference carries the shared "letter A vs letter B" axis, with a sign that depends on which letter shows the positive option. A plain sign-flip null picks that axis up, which made the earlier null as wide as ±0.8. That says nothing about content.

**The fix:**
- Project the letter axis (the mean of act(A) − act(B) over all OBEY and HELP pairs, per layer) out of every pair difference.
- Recompute the cosines from the projected differences.
- Null: a letter-balanced sign flip (a random half flipped within each letter group), 200 draws.
- Split-half reliability with halves stratified by letter.

## Results at mid depth (0.4–0.7 of layers)

| model | cos(OBEY, HELP) raw | **after letter fix** | letter-balanced null 95% | cos(MATCHED, OBEY) / (MATCHED, HELP) fixed | split-half OBEY / HELP / MATCHED (fixed) | letter share of a pair diff, obey / help |
|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | +0.46 | **+0.46** (outside null) | [-0.18, +0.18] | +0.06 / -0.07 | 0.94 / 1.00 / 1.00 | 0.70 / 0.70 |
| Qwen/Qwen2.5-1.5B-Instruct@989aa79 | +0.35 | **+0.35** (outside null) | [-0.19, +0.19] | +0.04 / -0.06 | 0.86 / 1.00 / 0.99 | 0.89 / 0.91 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2 | +0.32 | **+0.32** (outside null) | [-0.18, +0.18] | -0.01 / +0.05 | 0.89 / 1.00 / 0.99 | 0.86 / 0.87 |

Per-layer values are in `letterfix.csv`. A claim of a "tangle" needs the fixed cosine outside the null AND a reliable OBEY direction (split-half well above 0).
