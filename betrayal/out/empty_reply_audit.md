# Empty-reply audit: Gemma-3-12B Plan C arms

**Scope:** read-only, $0. Produced by p1-critic-plan with `analysis/empty_reply_audit.py`. Per-branch numbers are in `out/empty_reply_audit.json`.

**The bug.** A provider timeout comes back as HTTP 200 with an `{"error":…, "code":504}` body and no `choices`. `costpatch._one` returns `""` for it and counts it as `empty`. It never counts it as `failed`. The game then parses the empty string as action `""`, which it plays as **idle**. Nothing raises an error.

## 1. Call-level counts (from each run's `cost_tally.json`)

"No usage" is the `missing_cost` count. A timeout body carries no `usage`, so if almost every empty also has no usage, the empties are timeouts.

| Arm (run dir) | Gemma calls | Empty | Empty rate | Empties with no usage (timeout signature) | Failed |
|---|---|---|---|---|---|
| I (main + pilot) | 5,480 | 96 | 1.8% | ≈0 | 0 |
| J | 18,933 | 266 | 1.4% | 12 | 0 |
| K (rerun) | 5,847 | 82 | 1.4% | 0 | 0 |
| U pilot (seed 0) | 3,458 | 31 | 0.9% | 4 | 0 |
| **U main** | 115,174 | 10,577 | **9.2%** | 9,178 of 10,581 (87%) | 0 |
| **UF** | 15,729 | 1,351 | **8.6%** | 1,218 of 1,367 (89%) | 0 |
| **T** | 11,239 | 2,237 | **19.9%** | 2,135 (95%) | 0 |
| **C** | 12,391 | 1,983 | **16.0%** | 1,875 (95%) | 0 |
| **C16** | 24,919 | 5,155 | **20.7%** | 4,734 (92%) | 0 |
| **C2K** | 15,129 | 2,706 | **17.9%** | 2,425 (90%) | 0 |
| **DB** | 4,751 | 1,135 | **23.9%** | 1,110 (98%) | 0 |
| **SE** | 7,640 | 1,122 | **14.7%** | 1,068 (95%) | 0 |
| **DC (pre-registered)** | 13,978 | 1,293 | **9.3%** | 1,180 (91%) | 0 |

Notes on the table:
- The "no usage" column is counted over all calls in the run, including Qwen judge calls. Qwen had 0 empties, so the column is effectively Gemma's.
- "Failed" was 0 in every run, because a timeout is never counted as a failure.

**Reading:**
- **No timeouts in the early arms.** Arms I, J, K and the U pilot have about 1–2% empties, and those empties carry usage, so they are genuine empty completions.
- **Timeouts from arm U main on.** From arm U main onward (Nov 8 evening), 9–24% of Gemma calls are timeouts that were silently played as idle.
- **The "~14% empty" figure.** No "14%" figure appears in P1 or the readouts.
  - The nearest is the addendum's arm DB line: "Empty API completions are 22% (C16 20%, T 18%, C 15%)". Those figures **are** these timeouts.
  - P1's "115 empty Gemma replies" (arm I, Appendix B.1) is **not** timeouts: those empties carry usage, and the rate is 1.8%.

## 2. Is the forced-idle rate balanced between attacked and clean-twin branches?

**Measure:** the share of agent turns with action `""` (an empty or unparseable reply), from `actions.jsonl`. The paired difference is attacked minus its own clean twin (same cell and seed, same defense).

This measure also includes ordinary parse failures. Those run at about 5% in arms without timeouts (U pilot, K).

| Arm | Clean twins | Attacked | Paired diff (attacked − twin) [95% CI] | Pairs |
|---|---|---|---|---|
| U main | 0.195 | 0.186 | −0.034 [−0.038, −0.029] | 1,225 |
| UF | 0.076 | 0.081 | −0.011 [−0.025, +0.003] | 144 |
| T | 0.334 | 0.295 | −0.039 [−0.053, −0.026] | 160 |
| C | 0.292 | 0.285 | −0.008 [−0.024, +0.009] | 160 |
| C16 | 0.251 | 0.259 | +0.008 [−0.003, +0.019] | 160 |
| C2K | 0.218 | 0.224 | +0.006 [−0.005, +0.018] | 80 |
| DB | 0.317 | 0.272 | −0.035 [−0.069, −0.001] | 40 |
| SE | 0.280 | 0.293 | +0.014 [−0.003, +0.029] | 100 |
| DC | 0.204 | 0.187 | −0.015 [−0.028, −0.001] | 160 |
| K (control, no timeouts) | 0.052 | 0.053 | +0.012 [−0.001, +0.025] | 60 |

**Arm U headline branches** (N = 8, all-Gemma, closed records; forced-idle share):

| Branch | Share |
|---|---|
| clean\|none | 0.238 |
| BASE | 0.201 |
| outsider bribe | **0.225** |
| outsider forged | 0.211 |
| insider bribe-only | **0.226** |
| insider lie-only | 0.200 |
| both, bribe | 0.194 |
| both, forged | 0.189 |
| defense = resilience | 0.139 |
| defense = ALL | 0.131 |
| clean\|resilience | 0.210 |

## 3. Verdict

**Roughly balanced; any distortion is small.**
- The forced-idle rate differs between attacked branches and their twins by −4 to +1.4 pp.
- Attacked branches idle slightly *less* than their twins in U, T, DB and DC.

**Defection effects are biased toward the null (conservative):**
- Gemma clean twins never defect, so the twins' extra idling cannot lower their defection counts. The extra-defection estimates are therefore not inflated.
- In every headline contrast, the bribe branch idled as often as its comparator or more:
  - outsider bribe vs. forged: 0.225 vs. 0.211;
  - insider bribe-only vs. lie-only: 0.226 vs. 0.200;
  - both attackers: 0.194 vs. 0.189.
- Those contrasts are, if anything, understated.

**Resilience:** resilience branches idled *less* (0.139 vs. BASE 0.201), yet produced 0 bought defections. Fewer forced idles cannot explain the defense effect.

**One real caveat: key damage.** In arm U the clean twin idled 3–4 pp more than the attacked branches. That handicaps the twin's key progress, so it biases Gemma **key damage toward zero, or toward negative**. Arm U's "key damage near zero in every cell", and its BASE key damage of −0.06, may partly reflect the twins' extra timeouts. The pre-registered DC arm has the same direction, but smaller (−1.5 pp).

**Arms without timeouts:**
- Arm I (the size-dilution result) and arm J predate the timeouts.
- The Qwen arms (v2.1, G) had 0 empties.

## 4. Suggested Limitations sentence (sent to p1-paper-writer)

"In the later Gemma-3-12B arms (U, UF, T, C, C16, C2K, DB, SE and DC), 9–24% of agent calls hit provider timeouts that our harness logged as empty replies and played as idle moves. Forced-idle rates were similar in attacked branches and their clean twins (paired differences −4 to +1 pp), and in each headline contrast the bribe branch idled at least as often as its comparator, so this biases our defection contrasts toward the null. The clean twins idled slightly more, which may understate key damage in these arms."
