# TOFIX results: arm U core on Gemma-3-12B with the provider-timeout fix (PRE-REGISTERED CONFIRMATION)

- **Pre-registration:** `prespec_tofix.md`, body sha256 `47ab4a2e…`, frozen with the code in `PREREG_TOFIX_SHA256.txt` before any data. The driver checked it at launch, and every frozen hash re-verified after the run.
- **Analysis:** `harness/analyze_tofix.py`, run unchanged. The full output is in `out/tofix/analysis/results_tofix.json`.
- **Run:** seeds 800-819 (n=20), 5 branches each, 100 games, complete. Main run 01:56-02:06 ET, exit 0, no deviations. The smoke on seed 899 (`out/tofix_smoke`) is excluded from the analysis.
- **Design:** the original arm U settings. Temperature 0.7, long prompt, N=8, complete graph, closed records, 12 rounds, act 0.7, framing "assistant", no defense, default OpenRouter routing.
- **Identity check:** with flags off, her code and the harness with all modules plus `tofix_decode` both give `f3a70ed6…` (`out/tofix_prereg/`).

## 1. Timeouts before vs after the fix

| | Old path (arm U main) | This run, first attempt (old path would idle these) | This run, after the fix |
|---|---|---|---|
| Gemma calls hitting a 504 "Provider timed out" body | 8.0% (empty with no usage), 9.2% empty in total | **7.05%** (413 of 5,861 calls) | **0 failed (0.0%)**: all 413 recovered on retry |
| Genuine empty completions (with usage) | – | – | 1.4% (80), played as idle as in her code |

- **Stop rule:** never tripped (0 failed calls in any 200-call window).
- **First-attempt timeout rate varied by branch:** clean 3.8%, A/A 4.7%, bribe-only 6.0%, BASE 9.6%, lie-only 12.3%. Under the old path, the attacked branches here would have lost more moves than the twins. That is the reverse of arm U.

## 2. Primary outcomes (paired by seed, n=20, exact sign-flip, Holm over 2)

| | New (fixed) | Arm U (old) | New − old [95% CI] | Holm p | Verdict |
|---|---|---|---|---|---|
| **P1 key damage, BASE vs twin** (honest agents) | **−0.083 [−0.138, −0.021]** | −0.058 [−0.086, −0.030] | −0.025 [−0.088, +0.043] | 0.034 | Non-zero, but **negative** |
| **P2 bribe-only − lie-only defections** | **+0.65 [0.20, 1.10]** | +0.89 [0.56, 1.22] | −0.24 [−0.81, +0.36] | 0.034 | **Replicated** |
| ↳ bribe-only extra defections vs twin | +1.00 [0.65, 1.40] | +1.03 | | | |
| ↳ lie-only extra defections vs twin | +0.35 [0.15, 0.60] | +0.14 | | | |
| BASE extra defections (exploratory) | +2.65 [2.20, 3.05] | +2.22 | | | |

**Reading P1, using the pre-registered rules:**
- The "arm U understated key damage" rule fails: the new mean is not above the old one, and the new − old CI includes 0.
- The "≈0 confirmed" rule also fails, because the CI excludes 0.
- The fix does not reveal any hidden damage. As in arm U, honest agents end with slightly *more* keys when the BASE insider is present (completion as well: −0.35 [−0.46, −0.24]).
- Against the A/A twin the result is the same: −0.079 [−0.133, −0.026].

**Reading P2:**
- Bribes still drive betrayal: 17 turned agents in bribe-only vs 0 in lie-only.
- The new effect is not shown to be smaller than arm U's (the new − old CI includes 0).

## 3. A/A null (clean_AA − clean, same prefix and branch seed)

| Metric | Value |
|---|---|
| Defections | 0.00 (0 of 20 seeds non-zero; clean twins never defect) |
| Keys, insider agent excluded (matched to P1) | −0.005 [−0.050, +0.045], p=0.93 |
| Keys, all agents | −0.019 [−0.065, +0.027], p=0.50 |

## 4. Caveat found (exploratory, not pre-registered): forced idles are mostly parse failures, not timeouts

The fix removed every timeout idle. Even so, the forced-idle share (turns with action "") is still high:

| Branch | Forced-idle share, new | Arm U, old |
|---|---|---|
| Clean twin | 0.21 | 0.24 |
| A/A twin | 0.20 | – |
| BASE | 0.16 | 0.20 |
| Bribe-only | 0.21 | 0.23 |
| Lie-only | 0.14 | 0.20 |

- Nearly all of these idles are **non-empty replies that don't parse** (no JSON with an "action"): 12-21% of post-fork calls. Empty replies are only 0.5-2.7%.
- The gap between the twin and BASE (about 5 pp more idling in the twin) is still there, so it is not a timeout artifact.
- So the audit's attribution of arm U's ~20% forced-idle rate to timeouts was only partly right: timeouts were about 8 pp of it.
- A likely cause, not yet checked, is Gemma replies being cut off at 160 tokens before the JSON closes. Raw replies were not logged in this run.

## 5. Cost

| Item | Cost |
|---|---|
| Main run | $0.2997 (5,861 calls; costpatch usage.cost) |
| Smoke | $0.0145 |
| **Total** | **$0.314 of the $0.75 cap** |

Retries of error bodies carried no usage cost.

## Files
- Code: `harness/run_tofix.py`, `harness/tofix_decode.py`, `harness/analyze_tofix.py`, `harness/identity_check_tofix.py`
- Launcher: `runs/launch_tofix.sh`
- Prespec and hashes: `prespec_tofix.md`, `PREREG_TOFIX_SHA256.txt`
- Raw outputs: `out/tofix/`, including `games.jsonl`, `actions.jsonl`, `call_audit.json` (per-branch call outcomes, error codes and providers) and `cost_tally.json`
- Results: `out/tofix/analysis/results_tofix.json`
- Log: `out/tofix_log.txt`
