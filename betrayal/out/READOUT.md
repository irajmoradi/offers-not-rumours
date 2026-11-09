# Plan C v2 readout

Qwen-2.5-7B, N=8, matched clean twins. Brackets are 95% seed-bootstrap CIs. All numbers come from `out/final/results.json`.
Total cost $4.22, 0 failed calls.

1. **Turned agents are rare.** 14 of 1,800 honest agent-games in the main grid were turned, all by the insider (D-Evil: 0). The mixed swarm had 18 of 450, 5 of them from D-Evil.
2. **H1 is underpowered** (4 held-out positives): there is an estimate but no verdict.
   - Combined detector AUROC 0.96 [0.90, 1.00] vs best single (revealed preference) 1.00, a gap of −0.04 [−0.10, 0.00].
   - On the mixed swarm (18 positives) H1 is not supported: combined 0.72 vs revealed preference 0.95.
3. **Who is the insider?** The ledger finds it (who never gives) in 88% of 150 games, and the LLM judge on sent messages in 85%, vs 12.5% chance.
4. **H2.** D-Evil did not raise defection (+0.00 per game). The insider did: +0.13 per game [0.06, 0.20], Holm p=0.015 (supported).
5. **H3 is not supported.** Insider damage vs the same-defense clean twin: none 10pp [5, 14], provenance 10pp [6, 14], rep+corr+quar 13pp [8, 18]. Defense cost with no attacker is about 0.
6. **H4 is not significant after Holm.** Public detection cut defections (certainty effect +0.18 [0.03, 0.33]); the 2-key penalty did not (−0.13 [−0.25, 0.00]). Contrast p=0.016, Holm p=0.066. Only 11 defections in 80 insider games.
7. **Scaling.** Insider damage falls with swarm size: 40pp at N=4, 9pp at N=8, 1pp at N=16, −2pp at N=32. Cost per cell grows from 0.3c to 5.7c (about N^1.6).
8. **H5 (exploratory).** Turned agents' self-reports contradict the ledger 1.7 more times [1.1, 2.4]. The "quoted, won't affect your score" offer *raised* contradictions by 0.25 [0.07, 0.42].
9. **H7/H8 (exploratory).**
   - No lock-in: early gives vs final PageRank ρ=0.06 [−0.05, 0.16].
   - Qwen says "this is a test" 100% of the time under both framings (300/300 parsed), and test framing changed no behavior.
   - H6 is untestable (2 public defections).
10. **Caveats.**
    - Unparsed self-reports were treated as missing (audit fix; parse rates 98-100%).
    - The leak-free detector results are in `results.json`.
    - The Jennifer-question defaults are listed in CHANGES.md.
