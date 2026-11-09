# Plan C v2.1 readout

- Main grid (A): Qwen-2.5-7B, N=8, 30 seeds. Mixed swarm (D): qwen/llama/gemma-3/gpt-oss, 40 seeds. Both use matched clean twins.
- Brackets are 95% seed-bootstrap CIs. Numbers come from `out/final_v2_1/results.json`. Total cost $6.10 of the $8 cap; 0 failed calls.

1. **Turned agents.** Main grid: 23 of 2,700 honest agent-games, all turned by the insider (D-Evil turned 0). Mixed swarm: 100 of 1,800 (74 insider, 26 D-Evil).
2. **H1, PRIMARY = LEAK-FREE (main grid, held-out seeds 21-29): underpowered.** There are only 9 held-out positives, under the 10 required, so this is an estimate without a verdict.
   - Leak-free result: combined 0.85 [CI in results.json] vs the best single detector (LLM judge on received messages) 0.94. Gap −0.10 [−0.18, −0.03]. The combined detector does NOT beat the best single one.
   - **The as-specified instrument 1 is LEAKY** (confirmed in `harness/instruments.py:behavior_counts`):
     - It uses `d_defections`, the agent's post-fork defections minus its clean twin's.
     - "Turned" is *defined* by a defection after exposure, so that feature encodes the label. Its AUROC is 1.00 by construction.
     - Indirect coupling also remains in the leak-free set. A defect move replaces a give/idle that round (`d_gives`, `d_idles`), and her `apply_action` logs a refusal for every candidate the agent passed over while defecting (`d_hoard`). Even so, leak-free I1 scores only 0.57.
     - Instrument 3 (self-report) counts a *denied* defection as a contradiction, so it is partly label-related.
     - Instruments 2, 4 and 5 don't use the agent's own defections.
   - **Baseline:** exposure alone (was the agent sent a bribe/D-Evil message?) gives AUROC 0.77 (main) and 0.79 (mixed).
3. **H1 on the mixed swarm's own split (DEVIATION, option c; 28 held-out positives; leak-free): not supported.**
   - Combined 0.73 vs best single (lie battery) 0.74. Gap −0.01 [−0.13, 0.10], p = 0.59.
   - Leaky as-specified result, for reference: 0.83 vs 0.83.
   - OOD, leak-free (main-grid fits on mixed held-out seeds): combined 0.52 vs judge 0.68.
4. **Insider identity.** The ledger finds it in 82% of 300 games, pooled: 90% in the all-Qwen main grid (180 games) and 70% in the mixed swarm (120 games). The judge (sent messages) finds it in 75% pooled. Chance is 12.5%.
5. **H2 (Holm-corrected).**
   - H2a: D-Evil did not raise defection: +0.00.
   - H2b: the insider did, by +0.13 per game [0.08, 0.19]. Holm p = 0.0004 (raw ≈ 0.0001, Monte Carlo SE 0.00007; 20,000 permutations), **supported**.
6. **H3: not supported.** Insider damage measured against the clean twin with the same defense:
   - none: 8pp [3, 12]
   - provenance: 10pp [6, 13]
   - rep+corr+quar: 10pp [5, 15]
   - rep+corr+quar's reduction vs none: −2pp [−8, 3].
7. **H4 passes under Holm (corrected 2026-11-08 15:35), but it is fragile.** The earlier "not significant after Holm" was wrong. A bug in the analysis applied Bonferroni instead of Holm.
   - Certainty − severity = +0.30 [0.10, 0.50] defections per game. Raw p = 0.0156 (EXACT sign-flip over all 2^20 patterns); Holm-adjusted p = 0.0469, under 0.05.
   - It rests on only 11 defections, which came in 8 of the 20 seeds (insider branches). Treat it as suggestive.
8. **Scaling (unchanged).** Insider damage falls with swarm size: 40pp at N=4, 9pp at N=8, about 0 at N≥16. Cost per cell grows about N^1.6.
9. **Exploratory.**
   - H5: turned agents contradict the ledger 1.5 times more often [1.0, 2.0]. The "quoted, won't affect score" offer raises contradictions by +0.22 [0.10, 0.34].
   - H7: no lock-in, ρ=0.06 [−0.04, 0.16].
   - H8: Qwen says "test" 100% of the time under both framings (300/300 parsed).
   - H6: untestable (2 public defections).
10. **What changed from v2.0.**
    - New seeds only: A 20-29 and D 10-39.
    - Under the pre-registered 70/30 rule, A's test seeds became 21-29, all new data.
    - Option c is logged as a deviation. v2.0 is kept intact.

**Audit-2 notes (15:35):**
- **D-Evil "turned" agents are weak evidence of influence.** Of the 26 D-Evil-turned agents in the mixed swarm, 13 made no more defections than in their clean twin (my count, using per-agent defections after the fork). (Separately, the audit found that 25 of ALL 100 turned agents in the mixed swarm, insider and D-Evil together, defected at least as often in their clean twin.) Either way, the count is not evidence that D-Evil caused those defections.
- **Holm fix:** this also changes v2.0. Its H4 Holm-adjusted p becomes about 0.049, which passes. The v2.0 files are left as published; see CHANGES.md.

**P-value note (16:35, final):** each permutation test now has its own fixed RNG (seeded by its name), so adding analyses can't shift it.
- H4 (20 seeds) uses the EXACT sign-flip over all 2^20 sign patterns, so it has no Monte Carlo error.
- The 30-seed tests use 20,000 permutations. MC SE: H2b 0.00007, H3 0.003.
- Earlier quoted decimals (H4 0.047 / 0.0444, H2b 0.0006) were Monte Carlo noise. `results.json` holds the final values.
