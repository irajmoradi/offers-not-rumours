# ACT-vs-WAIT control for the OBEY/HELP overlap

**Question.** Mid-depth cos(OBEY, HELP) is +0.465 / +0.351 / +0.317 (Gemma-2-2B / Qwen2.5-1.5B / SmolLM2-1.7B). Both contrasts are "take the transactional action" vs "hold off". Is the shared part just a generic act-vs-wait axis?

**Verdict.** Mostly yes. In Gemma and SmolLM2, removing the ACT direction drops the overlap into the chance range (+0.12 and +0.06, null ±0.17); act-vs-wait explains 88% of the overlap. Qwen keeps a clearly positive leftover (+0.24, null ±0.18; 5 of 8 mid layers outside, all from layer 15 on), and ACT explains only 52% there. "Obey = help" beyond acting is not supported in 2 of 3 models; in Qwen it is a possible extra shared part, open to the caveats below.

## Method
- **ACT scenes** (`act_scenes.json`): 100 new scenes, seed 7171, from the `tools/benign_eval.build` template (ids `ambig7171_*`; no overlap with the benign31 training/test scenes). A contact proposes a swap with no authority cue, no fairness cue and unclear value:
  - the offered key is not one of my needs;
  - the contact's board post lists only holdings ("Holds …. Open to trades."), so I can't tell whether they need my key;
  - 4 neutral message wordings ("Up to you", "No pressure either way", …).
- A/B choice: "Do the trade with P: your key Y for their key X." vs "Hold off and wait.", letter-balanced 50/50.
- **Models, token and layers** as in `cross_model_letterfix.py`: same local weights, the residual stream at the answer-letter token for every layer, pair difference act(pos letter) − act(neg letter).
- **OBEY and HELP** were rebuilt exactly as before. The letter axis is estimated from OBEY+HELP and projected out of every pair difference, ACT included.
- **Key test:** per layer, Gram-Schmidt the unit ACT direction out of the unit OBEY and HELP directions, then compute the cosine of the residuals. This uses the identity cos(O,H) = cos(O,A)·cos(H,A) + O_r·H_r, so the "via ACT" column is the part of the overlap that ACT carries.
- **Nulls:** 500 letter-balanced sign-flip draws, 2.5–97.5%.
  - OBEY has 15 pairs per letter group (an odd number), so each draw drops one random pair per group and then flips exactly half of the rest. The old "flip 7 of 15" leaves one unflipped pair, which leaks real OBEY signal; that had skewed the cos(OBEY, ACT) null to [−0.16, +0.45].
  - Residual null: OBEY and HELP flipped, real ACT projected out. Flipping ACT too gives the same band.
- **Split-half:** 50 letter-stratified random halves.
- All numbers are means over the mid-depth layers (0.4–0.7 of depth: Gemma L10–17, Qwen L11–18, SmolLM2 L10–16).

## Results (mid depth)

| model | cos(OBEY,HELP) raw [null] | cos(OBEY,ACT) [null] | cos(HELP,ACT) [null] | via ACT (share) | **residual cos(OBEY,HELP) after removing ACT** [null] | mid layers outside null | split-half ACT / OBEY / OBEY-residual |
|---|---|---|---|---|---|---|---|
| Gemma-2-2B | +0.465 [±0.17] | +0.494 [±0.32] | +0.823 [±0.19] | +0.407 (88%) | **+0.115** [−0.17, +0.17] | 1/8 | 1.00 / 0.94 / 0.93 |
| Qwen2.5-1.5B | +0.351 [±0.19] | +0.299 [±0.31] | +0.646 [±0.23] | +0.184 (52%) | **+0.239** [−0.18, +0.18] | 5/8 (L13, L15–18) | 1.00 / 0.85 / 0.84 |
| SmolLM2-1.7B | +0.317 [±0.17] | +0.357 [±0.41] | +0.785 [±0.28] | +0.278 (88%) | **+0.062** [−0.16, +0.17] | 1/7 (L10) | 1.00 / 0.89 / 0.89 |

The raw cosines reproduce `letterfix.md` exactly (0.465 / 0.351 / 0.317), which checks the pipeline.

![Raw vs ACT-removed cosine by layer, with null bands](act_control.png)

## Reading
- **HELP is mostly act-vs-wait.** cos(HELP, ACT) = 0.65–0.82, even though the ACT scenes have no fairness cue at all.
- **OBEY is partly act-vs-wait.** cos(OBEY, ACT) = 0.30–0.49. It is outside its null only for Gemma. The null is wide because OBEY has just 30 pairs and the cosine is taken against a fixed, strong ACT vector, so OBEY's alignment with ACT is noisy in Qwen and SmolLM2.
- **After removing ACT:**
  - Gemma: +0.465 → +0.115 (inside the null).
  - SmolLM2: +0.317 → +0.062 (inside the null).
  - Qwen: +0.351 → +0.239. Still outside the null, carried by its late-mid layers 15–18, where the raw overlap also peaks at about 0.57.
- **The leftover OBEY direction is still reliable** (residual split-half 0.84–0.93). So OBEY carries a real component beyond acting; it just does not line up with HELP's leftover in Gemma or SmolLM2.

## Sensitivity: hold option worded like HELP's
`ACT_board` uses HELP's exact hold wording, "Post on the board and wait.":
- cos(HELP, ACT_board) = 0.95 / 0.86 / 0.92 and cos(ACT, ACT_board) = 0.86 / 0.74 / 0.85.
- Projecting ACT_board out removes nearly all of HELP. Residual cos(OBEY, HELP) becomes −0.35 / −0.10 / −0.11, an over-correction that is not interpretable as a test.
- What this does show is that the HELP direction is almost entirely "do the trade vs post and wait", whatever the scene's fairness. The primary test above uses the neutral "Hold off and wait." wording.

## Caveats
- One ACT direction (rank-1) is removed. Act-vs-wait could take more than one dimension, so the residual is an upper bound on what is "beyond acting".
- OBEY's positive option is sometimes "defect" (smite), not a trade, and its negative option is "message to verify", which is itself an action, not waiting. ACT only matches the trade/hold-off part.
- **Behaviour on the ACT scenes:** mean P(act letter) is 0.49 for Gemma, so the scenes really are ambiguous for it. Qwen and SmolLM2 lean toward acting (0.69 and 0.74; SmolLM2 prefers act in 100/100 scenes). The direction is a contrast of forced answers, so this does not invalidate it, but the scenes are less "ambiguous" for those two models.
- Geometry only; no steering test of the residual directions.

## Files
- `tools/act_control.py`: scene builder + collection + analysis. Rerun the analysis with `--analyse` and `ACT_CACHE=<dir>`; the activations are cached in the session scratchpad, not here.
- `tools/act_control_fig.py`: the figure.
- `act_control.json`: per-model mid-depth summary, including the ACT_board variant and all nulls.
- `act_control_layers.csv`: per-layer values.
- `act_scenes.json`: the 100 ACT scenes (without the system prompt).
- `act_control.log`: the run log.
- `act_control.png`, `act_control_alt.txt`: the figure and its alt text.
- Run: local RTX 5090, HF transformers, bf16, one model at a time, `set_per_process_memory_fraction(0.6)`, RAM guard at 15 GB. Env `~/miniconda3/envs/swarm`. Activation collection took about 3 min. Cost $0.
