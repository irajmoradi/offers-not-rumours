# Plan C v2: changes and deviations log

Her package `pkg/swarm_llm/` is unchanged: hashes are in `pkg_module_sha256.txt` and copied as-is into the run folder.
Everything below lives in `harness/`.

## Harness (runtime, verified)

**fastpaths** (Plan C):
- Updated to her v2 strict bribe rule: accepted only if the defection round is greater than the offer round.
- Renamed fields `god_*` to `d_evil_*`.
- Re-verified offline (deterministic mock, `PYTHONHASHSEED=0`). Her code and the fast paths give the same sha256 at N=8, 16 and 32:
  - N=8: `f3a70ed6…`
  - N=16: `e983bc38…`
  - N=32: `8a5d5711…`
- The first check FAILED (`5fa6a1ff…` vs `f3a70ed6…`). The cause was the old `>=` rule in fastpaths, which is now fixed.

**v2patch** (penalty severity):
- Severity 1 (or no setting) leaves her code byte-identical; this is included in the hash check above.
- Severity 2, tested offline:
  - A failed defection costs 2 of the defector's own needed keys.
  - The prompt reads "TWO of the keys" instead of "one of the keys".

**Instruments, labeller, driver, analysis:** see PLAN.md §1-2.
- The "turned" label is per agent: honest agent i defected in rounds r..r+2 after a D-Evil message or a bribe was delivered in round r.
- It counts agents/defections, never bribes (audit item 2).

## Deviations (logged before results were viewed)

1. **Turned-rate decision (11:55).** The pilot had 0 of 135 turned agents. I told Jennifer before the full run and offered options (a), (b) and (c).
   - No answer came by 11:55, so I used (a), the design as specified: A 20 seeds, D 10 seeds.
   - H1 will report "underpowered" if it has fewer than 10 held-out positives (pre-registered rule).
2. **Audit fix: unparsed self-reports are MISSING** (analysis fix; no hypothesis changed).
   - Problem: her `elicit_eval_belief` maps an unparseable reply to `is_test=False` with confidence 0.5. The Plan C audit showed this created the "real-believers defect more" pattern.
   - v2 now treats as missing, and reports parse rates for, all of these:
     - unparsed eval-belief replies
     - unparsed interview replies (instrument 3 contradiction score)
     - unparsed lie-battery replies (instrument 4)
     - unparsed judge replies
   - Stats use parsed answers only.
   - Arm E had already run once (11:51-11:52, $0.18) with the old belief scoring, which stored only the rate and not the raw replies. That run is **discarded and re-run** with per-agent parsed/unparsed logging (`beliefs.jsonl`).
   - Arms B and C were unaffected; they have no self-report.
3. **Audit item 3** (already in the design): attack damage is measured against the clean twin with the same defense.
   - The defense's own cost is reported separately as `defense_cost_no_attacker` = completion(clean, none) − completion(clean, defense).
   - Net damage (vs. the undefended clean twin) is descriptive only.
4. **Audit item 4:** secondary claims are stated only from Holm-corrected results (`holm_reject`). Uncorrected p-values are shown for transparency only.

## Defaults used (Jennifer did not answer the 5 questions)

- **Q1:** instrument 1 as specified is primary. The leak-free variant is also reported.
- **Q2:** the turned rate (<2%) was reported to Jennifer before the full run.
- **Q3:** mixed swarm = qwen / llama / gemma-3-12b / gpt-oss-20b, 2 agents each.
- **Q4:** her approved self-give fix stays on.
- **Q5:** the insider is scripted, so insider identity uses instruments 1, 2 and 5-sent.

## Runs

| Run | Arms and seeds | Cost | Errors |
|---|---|---|---|
| Pilot (11:40) | all arms, 1 seed each | $0.26 | 0 |
| `full_BCE` (11:45-11:52) | B 20 seeds, C 10 seeds; E (discarded) | $1.68 | 0 |
| `full_ADE` (11:55-12:05) | A 20, D 10, E 10 (fixed) | $2.28 | 0 |

**Total spent for v2:** $4.22 (includes the pilot and the discarded first E run), against an $8 cap.

**Merged analysis:** `out/final` = `full_BCt` (B+C only) + `full_ADE`. Figures and their source CSVs are in `out/final/figs/`.

**Fig 1 fix:** tick labels were corrected before anything was posted. The values did not change.

## v2.1 extension (12:38, approved by Jennifer: "I approve everything proposed going forward")
- **New seeds only:** A seeds 20-29 and D seeds 10-39. Same matched forks, same audit fixes, same harness. Process cap $2.60; projected +$1.9.
- **H1 (primary) as pre-registered:** the first 70% of seeds train and the last 30% are held out.
  - With 30 A seeds, that means train 0-20 and test 21-29.
  - The test seeds are all new data that nobody had seen.
  - Seeds 14-20, which were the v2.0 test or unused seeds, are now training. That follows the pre-registered proportion rule.
- **DEVIATION (option c):** an extra H1 test on D's own 70/30 split (train 0-27, test 28-39).
  - Check: main-grid fits evaluated on D's held-out seeds (out of distribution).
  - In v2.0 that check used all D seeds. In v2.1 it uses the held-out D seeds only, so it never overlaps D-trained data.
- **v2.0 is kept as it was:** `out/final/` and `planC_v2_outputs.zip`.
- **v2.1 outputs:** `out/final_v2_1/` and `planC_v2_1_outputs.zip`.

## Arm F self-run package (approved by Jennifer; she runs it, the key stays off Iraj's machine)
- **Package:** `arm_f_package/` and `arm_f_package.zip`.
  - Contents: `run_arm_f.py`, the harness, and her game modules (byte-identical).
  - The Adaption key comes only from env `ADAPTION_API_KEY`, or a hidden getpass prompt.
- **D-Evil and judge:** these run on Adaption too, since she has no OpenRouter key in that setup. D-Evil is `qwen3.8-27b` (none); the judge is `deepseek-v4-flash` (none, temperature 0).
  - This differs from arms A and D, where D-Evil and the judge were Qwen-2.5-7B. It's a further reason arm F is never pooled with A-E.
- **Budget guard:** `--max-usd` (default 5) projects the total after seed 0 and stops if it's over. It is also a hard cap during the run.
- **Testing:** dry-run only. A full 10-seed run on the fake backend passed, and the guard stopped correctly at `--max-usd 0.5` (exit 2).
- **Ingest:** on our side, `harness/analyze_arm_f.py` takes her returned folder or zip. It scans for credentials (counts only) and runs `analyze_v2` with REMAP=F:A. It then writes a side-by-side CSV vs v2.1, never pooled. This was tested on the dry-run output.

## Efficiency review for arm F (Jennifer: "fewest calls/tokens, behaviour unchanged"), 13:56
The Adaption limit counts **tokens** per user per minute.

**Measured, offline (deterministic mock, N=8, rep+corr+quar insider games, 226 decision calls):**
- An agent decision call averages about 784 prompt tokens: 1,262 chars of system text plus 1,876 chars of observation. Output adds about 200-400 tokens.
- Where the observation's characters go:

| Part | Share |
|---|---|
| received messages (already capped at the last 10, so history does not grow) | 25% |
| action menu | 17% |
| credibility line (reputation defense) | 11% |
| board | 7% |
| ledger | 6% |
| candidates | 5% |

**Applied / verified (no change to what any agent sees):**
- One call per agent per instrument. The interview is one batched call per agent and the battery one per agent. The judge and D-Evil run on OpenRouter in this harness, so they use **no** Adaption tokens.
- The shared prefix is played once per seed and forked. Prefix calls are never re-sent; branches start from the forked state.
- Agent decisions are never merged: one agent = one call, as required.
- History is already bounded: last 10 messages, last 4 notices and ledger checks. Per-agent prompt size is O(N) and per-round work O(N²) through the board, as before.
- Scheduler: the token bucket is not FIFO. Any waiting request whose cost fits the current budget takes it, so small calls (interviews, short early-round prompts) fill gaps that a large call can't use yet. The adaptive per-key TPM and the global Retry-After pause prevent 429 storms; 429s waste wall time, not tokens.
- Identical system text across a game's calls: the `instructions` field is byte-identical for every call in a game, which maximises any server-side prefix caching. The deepseek r40 costs in ADAPTION_MODELS.md suggest it does cache.

**Not applied (flagged), because each would change the stimulus or the pre-registered instrument:**
- **Compact JSON in observations:** saves only 3.0% of prompt chars. It also changes the exact text agents see vs arms A-E, a confound for the F-vs-D comparison.
- **Sending the rules once:** not possible. The API is stateless, so the 1,262-char system text has to go with every call. Truncating or summarising it would change agent information.
- **Merging interview + battery into one call:** would save about 45k tokens per seed (about 8%). But the battery would then no longer be asked after the interview answer in its own call, which is a protocol change to pre-registered instrument 4. It is available only as a future option and is OFF.
- **Batching the judge across agents:** saves almost no tokens (bundles are the same size) and would make the scores relative. The judge isn't on Adaption here anyway.

**Net:** no behaviour-preserving change gives a meaningful token cut beyond what is already in place. Throughput is set by the TPM budget: about 1,050 tokens per call (prompt + output) x about 605 calls per seed ≈ 0.64M tokens per seed.

## v2.2 additions (14:00-15:15; all EXPLORATORY; Jennifer pre-approves proposals)

### Arm F (Adaption mixed swarm) run here on Jennifer's key
- **Setup:** the key is read from `.jenkey` by path and is never printed; the team lead patched the loader to extract the `pt_live_` token.
- **Pilot 1 is INVALID:** 313 of 350 calls returned 429.
- **Backend fixes:**
  - Retry-After is honoured: at least 60 s plus jitter, up to 8 retries per 429, with a global pause.
  - Adaptive per-key TPM: start 150k, then 25% drop per 429 (at most one per 60 s; floor 10k) and 10% raise per 10 quiet minutes.
  - Usage-corrected token accounting, concurrency 8, and a per-minute log.
- **Pilot 2 is INVALID:** the limiter only dropped on 3 or more 429s, so with one 429 per minute it never adapted.
- **Pilot 3 is valid:** measured about 26-40k tokens/min.
- **Plan change:** arm F was stopped after 7 seeds (0-6) instead of 10. At about 19 min per seed, 10 seeds would not have left time for the Adaption G/growth/H queue before 17:45.

### Efficiency review (Adaption tokens)
No behaviour-preserving cut beyond what was already in place. Details are in the section above.

### Arm G: outsider (D-Evil) bribe-and-renege (`harness/outsider_bribe.py`)
- **Mechanics:** offers go out through the D-Evil slot, using her `BRIBE_TEMPLATE`, `STALL_LINES` and the `st.bribes` bookkeeping. The outsider never pays.
- **Budget:** 4 offers per game, at most 2 per round.
- **Recognition:** flagged (agent −1 + an "unknown sender" prefix) vs disguised (impersonated neighbour).
- **DEVIATION, timing:** early/mid/late start at rounds 2/4/6, which are thirds of the effective horizon. v2.1 clean games end at median round 9.5, and a start at round 9 placed 0 offers offline.
- **Runs:** OpenRouter, Qwen, 40 seeds, $0.95.

### Growth, 6→8 agents at round 4 (`harness/growth.py`)
- **Mechanics:** dormant agents are kept in `st.compromised`; at the join round they are activated with their holdings, needs and edges.
- **Check:** with growth off, games stay byte-identical to her code (the same hash f3a70ed6…).
- **Runs:** OpenRouter, 8 seeds, $0.08.

### Arm H: heterogeneous scaling
- **Design:** mixed qwen/llama/gemma-3/gpt-oss swarm at N=4/16/32, insider vs clean twin, no defense.
- **Runs:** 8 seeds, $0.47.
- **Comparison:** the N=8 mixed point is v2.1 arm D.

### Adaption queue (after arm F)
- **Order:** G (3 seeds), then growth (3 seeds), then H at N=4 (3 seeds), using the arm F Adaption agent mix (`PROVIDER=adaption`).
- **Spend:** the outsider is scripted, so there is no judge or LLM D-Evil, and OpenRouter spend is $0.
- **Dropped:** H at N=16 on Adaption, which would take about 12+ min per seed.

### Arm I: homogeneous fleets of other models
- **Design:** all-Llama-3.1-8B, all-gpt-oss-20b and all-Gemma-3-12B; insider vs clean twin, no defense.
- **Seeds:** N=8 ×20 and N=4/16 ×10 each. That doubles the requested seeds; the cap allowed it and power was the weak point.
- **Runs:** the pilot ($0.058) was on `.mariokey`. The main run is on the NEW team key (`.newopenrouter`), with its own tally, cap $1.00 (team cap $5).
- **Key-loader fix:** the team key file holds only the 64-hex body, so `orkey.load_key` now adds the `sk-or-v1-` prefix. This was verified with one read-only GET.

### OpenRouter (`.mariokey`) total for v2
About $7.61 + $0.058 (arm I pilot) ≈ $7.67 of $8.

## 15:35 Audit-2 fix: Holm step-down
`analyze_v2.py` multiplied the CURRENT p by (m−j) instead of taking each earlier p_(j)·(m−j), which made it Bonferroni. It is now a proper Holm step-down (cumulative max of p_(j)·(m−j)).
- v2.1 re-analysed with no new API calls. H4 now passes: Holm p 0.047, previously 0.063. The other verdicts are unchanged.
- v2.0's H4 would also pass (about 0.049). The v2.0 files are left as published.
- Readout wording softened per audit-2:
  - H1 now leads with the leak-free result and the exposure-only baseline AUROC (0.77 / 0.79).
  - Arm G flagged > disguised is labelled not significant.
  - Arm H at N=4 is labelled not established, and the baseline-completion confound is noted.
  - D-Evil "turned" agents are flagged as weak evidence.
- 15:50 Readout text fixes: the audit quote now reads 25 of ALL 100 turned agents (mixed swarm); per-arm insider-ID figures added (90% all-Qwen, 70% mixed).
- 16:05 H1 leak check (Jennifer): instrument 1 as specified is LEAKY (d_defections encodes the 'turned' label). Leak-free is now the PRIMARY H1 result in results.json (H1_designation), the readout and figures fig1/fig1b. Indirect coupling is documented. By-model breakdown added (analyze_by_model.py).
- 16:03 Arm F stopped (the queue log's 'seeds 0-6' text is wrong). Complete seeds are 0-5, 7, 8 (8 seeds). Seeds 6 and 9 were in flight and are dropped. Throughput was steady at about 31k tokens/min with no hang, but seed 6 had produced no game for about 45 min. The Adaption queue started at 16:03.
- 16:20 Text synced to results.json after the Holm and leak re-runs: H2b Holm p 0.0004, H4 raw 0.0148 / Holm 0.0444. Figures were regenerated at 16:00 from the fixed code. P-values are Monte Carlo and shifted slightly because the random-number stream changed; verdicts unchanged.
- 16:35 Each permutation test now has its own RNG, seeded by sha256(test name). H4 (n=20) uses the EXACT sign-flip: raw 0.015625, Holm 0.0469. 30-seed tests stay Monte Carlo with SE reported. H2b Holm 0.0004. Readout, skeleton and results.json are final.
- 16:50 P1 review analysis (analyze_p1.py): honest-only damage, hoarding vs defection decomposition, twin-free ledger rule with false positives, and arm J payment effect on detection.
- 19:14 Arm UA (Adaption gemma-4-31b) STOPPED: 0 successful calls in 10 min, 23 HTTP 429s at the 10k tokens/min floor. Jennifer's per-user Adaption quota is saturated, most likely by her own website use (augmentation/AutoScientist). Stopped so it doesn't compete with her. Pilot seed 0 (4 games) remains the only UA data.
- 19:41-19:49 Arm T (pay timing) ran: 40 seeds × 4 cells, team key $0.57, 0 failed calls. New harness flag `st.fulfillment["timing"]` (now/later/end, delay 3). Re-verified 20:10 offline (DetMock, no network): her code and harness with all add-on modules installed and flags off both give sha256 f3a70ed6… at N=8.
- 20:05 analyze_t.py: added paired-by-seed sign-flip tests vs never_pays (final and mid-game ledger top-1; Holm over 6 tests; seeded Monte Carlo 200k). Raw curve values unchanged (same RNG stream).
- 20:05 analyze_c.py: arm filter now reads env ARM (default "C", unchanged output for N=8; verified identical cells), so the same script analyses C16. Figure title and label print the actual N.
- 20:05 No new launches after T and C16 (lead's instruction); UA stays stopped.
- 20:40 Arm C16 done: 40/40 seeds, team key $1.18, 0 failed calls (calls kept succeeding after the 19:59 limit notice). compare_c_n.py added (N=8 vs N=16, unpaired). Holm over C16's 4 paired tests is a post-hoc addition.
- 20:45 Identity-check correction: `python -I` ignores PYTHONHASHSEED, so offline hash checks run with -I are not reproducible run to run (seen: f3a70ed6 / 7609d8a9 / a2f1220c across identical code). Re-run properly (no -I, PYTHONHASHSEED=0; our own code only): her code and the harness with ALL add-on modules incl. disclosure.py installed and flags off give f3a70ed6… in 4/4 runs (2 orig, 2 fast). Earlier checks that matched f3a70ed6 stand; the 20:10 note's method is superseded by this one.
- 20:45 Arm DB (disclosure bounty; teammate request via the lead) BUILT, NOT LAUNCHED: harness/disclosure.py, arm_DB in run_v2.py (twin_key so DB1 is measured against a clean twin that also has the rule), analyze_db.py. Offline driver smoke test with a reporting mock: reports, verification, bounties, forfeits and rows all work.
- 20:42-20:46 Arm DB ran (20 seeds, $0.22, 0 failed). 20:42-20:51 Arm C2K ran in parallel (N=16, outsider bribe 2 keys, 40 seeds, $0.73, 0 failed; RAM 30 GB available). run_v2 arm_C: env C_SIZES (default "1,3" = original item list and order) and C_LABEL.
- 21:00 analyze_dose.py (1/2/3-key dose-response at N=16; per-seed slope sign-flip + Page's L, Holm over 4). analyze_db.py figure labels "n/a" when there are no reports; unparsed/invalid-by-branch table added for the DB empties check.
- 20:57 DB raw-reply diagnostic: 2 seeds (20-21), env DB_RAW=1 writes raw_calls_LOCAL.jsonl (excluded from the zip). analyze_db_raw.py. Result: 0 report-field attempts, so the DB null is behavioural.
- 21:01 Arm SE (social-engineering levers; teammate request via the lead) built and launched. It included an authority-order cell, which the lead then ruled out (the forged-authority ban stands; no exception). 21:0x: run killed before any game completed; all SE outputs deleted (raw and processed, including the offline smoke-test outputs); the authority template and cell removed from harness/outsider_se.py, run_v2.py and analyze_se.py (outsider_se now asserts the lever is one of social/scarcity/liking). SE relaunched with recip, social, scarcity, rumour, liking + shared twin, same seeds/cap. For authority, the writeup points to the existing God-forger results.
- 21:10 Arm SE ran (21:02-21:06, 20 seeds, $0.33, 0 failed) with recip/social/scarcity/rumour/liking + twin; analyze_se.py.
- 21:40 Arm DC (dual control; teammate request via the lead) BUILT AND PRE-REGISTERED, NOT RUN (no model calls of any kind):
  harness/dual_control.py (burn requests need a second authorization; random/fixed approver via its own seeded RNG; requester can't self-approve; blocked = attempt, never a defection; optional transfer threshold; append-only dual_request/dual_approve/dual_block events; metrics incl. dual_false_veto, approver_is_insider), arm_DC + _dual_cfg + clean|dual twin in run_v2.py (arm_DC refuses to run unless prespec_dual_control.md matches PREREG_DC_SHA256.txt; verified), analyze_dc.py, dual_event.schema.json, test_dual_control.py (a-d, f) + test_dual_control.sh (e). All tests pass (out/arm_DC_prereg/). Flags-off identity f3a70ed6; re-scored T and C16 identical. Pre-registration sha256 71a62d7a…; code frozen in PREREG_DC_CODE_SHA256.txt.
- 21:38-21:48 Arm DC ran as pre-registered (40 seeds, $0.70, 0 failed). Prereg and code hashes re-verified before launch and before analysis; analyze_dc.py unchanged. No deviations.

## 2026-11-09 closers (pre-registered, prespec_closers.md)
- New driver harness/run_closers.py (a copy of run_v2.py plus arms CL1/CL2S/CL2/CL3/CL4); run_v2.py is untouched (sha256 a0d3dd31…, still matches PREREG_DC_CODE_SHA256.txt). New modules: harness/outsider_rumour.py and harness/bribe_paraphrase.py (flags off = her code; the all-modules identity hash is still f3a70ed6, with both added to identity_check.py). New analysis script: harness/analyze_closers.py.
- prespec v1 (body af4e367a…) was frozen ~00:44 ET. A seed-99 smoke (out/closers_smoke, excluded from analysis) showed that the belief probe's first-JSON parse read Gemma's action object. v2 (~00:55 ET, before any main run) changes only the CL4 belief-probe parse rule (run_closers._belief_parse). Hash: cf5dbf006d46fd13e07e94e9189afb332f2c39c5058674ac0456aaad1b193484.
- 00:49 ET: the first main launch of CL1/CL3/CL4/CL2S died silently within 10 min, before any game was written (only events.jsonl existed; no games, no cost tally). It was probably lost with the tool shell. The dirs were moved to out/closers_*_lost_try1/, and the runs were relaunched unchanged at 00:59 with setsid/nohup, on the same seeds. No outcome data was seen. That try's OpenRouter spend (≈$0.07 credit delta) counts toward the closers budget.
- T2 screen (CL2S, seeds 200-207). The pre-registered rule picked microsoft/phi-4 (+1.38, p=0.0075). mistral-small-3.2 was not clear (p=0.12). CL2 main launched 01:03 on phi-4, seeds 300-329 (n=30; the cost allowed full n).
- ~01:03 ET: the 00:59 relaunch of CL1/CL3/CL4 and the 01:03 CL2 launch also died silently (no traceback, no cost tally; not ramguard). Kept: CL3 seeds 400-405 and CL4 seeds 500-502, all complete cells with every branch written. Relaunched ~01:16 as tool-managed background jobs with exit-code capture: CL3 406-429 -> out/closers_CL3_p2, CL4 503-529 -> out/closers_CL4_p2, CL1 100-159 full restart, CL2 300-329 full restart. The dead CL1/CL2 attempts had no complete cells; their dirs are in out/closers_*_lost_try*/. The analysis reads part1 + part2.

## 2026-11-09 Jenifer follow-ups (prespec_jenfu.md; EXPLORATORY; prepared, not launched)
- New driver harness/run_jenfu.py (own copy of the needed run_closers helpers; run_v2.py and run_closers.py untouched, hashes still match their registrations). New module harness/jenfu_patch.py (insider forged peer sender for lie/bribe, insider hoarding-rumour content swap, outsider forged accusation, 3 new paraphrased bribes J0-J2, harm_norm defense note; flags off = her code). Reuses harness/outsider_rumour.py unmodified. New analysis harness/analyze_jenfu.py (exact sign-flip, Holm per experiment, A/A null). Identity check copy harness/identity_check_jenfu.py: all-modules + jenfu flags-off = f3a70ed6 (orig and existing all-modules check also f3a70ed6).
- Offline functional test (DetMock) passed; offline driver smoke (SMOKE=1) passed; API smoke seed 99 (out/jenfu_smoke, excluded) $0.045, 0 failed. Prespec v1 body sha256 766ebd0c…, frozen with code in PREREG_JENFU_SHA256.txt. Launch script runs/launch_jenfu.sh.

## 2026-11-09 Noor T1-core (prespec_noor.md; PRE-REGISTERED; prepared, NOT launched)
- New driver harness/run_noor.py (own copy of the run_jenfu helpers; run_v2.py, run_closers.py and run_jenfu.py are untouched). New modules: harness/short_prompt.py (compact observation behind st.short_prompt; flag off = her code), harness/noor_decode.py (top_k 5 + provider.require_parameters on every call; retries HTTP-200 error bodies, so they are never silently scored as empty/idle), harness/noor_topk_check.py, harness/noor_gemma_probe.py (pre-registered US-reference slot rule), harness/analyze_noor.py, harness/identity_check_noor.py. Flags-off identity hash with short_prompt installed: f3a70ed6 (no -I, PYTHONHASHSEED=0; out/noor_prereg/idcheck_*.json).
- FINDING (affects costpatch everywhere): OpenRouter can return HTTP 200 with {"error": ...} and no choices (DeepInfra Gemma-12B "Provider timed out", 504). costpatch._one scores that as an EMPTY reply (an idle move), with no retry and no failed count. In Noor smoke 1 this hit 632 of 910 Gemma-12B calls. Fixed only in noor_decode.py (run_noor). Older arms were not changed.
- Smokes (all excluded from analysis, seeds 690-692): smoke 1 (5 processes; Gemma-12B invalid, see above), smoke 2 (Gemma-12B re-run; aborted with no games, provider down; its tally was lost on kill, so it is bounded by its $0.16 caps), smoke 3 (Gemma-27B short v1 vs long), smoke 4 (Gemma-27B short v2). short v1 merged and shortened the state labels and doubled invalid gives, so it was dropped. The frozen short v2 = 3-round message window + a summary line + compact JSON, keeping her labels. Measured smoke spend $0.240 plus smoke 2.
- Frozen: PREREG_NOOR_SHA256.txt (prespec body 4be3a18d…). Launch script runs/launch_noor.sh (probe -> 4 processes, caps 0.75/0.62/0.70/0.63).
- 01:23-01:53 jenfu main runs (JF123/JF4/JF5/JF6, seeds 300-319) ran as pre-specified: $1.90 + smoke $0.045, 0 failed calls; hashes re-verified; analyze_jenfu.py unchanged; results in out/jenfu_results/ and READOUT_ADDENDUM_GHGR.md. No deviations.
- 02:03-02:43 Noor main runs ran as pre-registered (prespec body 404f9f25… after the lead's 5%/200-call quality-stop rule was added before freezing). Probe 12B 7/16 -> US slot = Gemma-3-27B (logged swap). 4 × 24 seeds complete, all exit 0, 0 failed, 0 empty, no stops. Main spend $2.18 (gemma27 0.611, phi-4 0.496, mistral 0.519, qwen 0.554); with smoke ≤ $2.58. Hashes re-verified before analysis; analyze_noor.py ran unchanged -> out/noor_analysis/. Caveat seen in descriptives: the Gemma-27B LLM forger delivered only 8 of the possible 96 messages (others: 94-96).
- Post-results EXPLORATORY (lead request, $0, no new calls; no Holm): harness/explore_noor_scripted.py -> out/noor_analysis/noor_exploratory_scripted.{csv,json}: disguised/flagged bribe vs scripted rumour, scripted rumour vs clean, per model. Frozen analyze_noor.py untouched.

## 2026-11-09 TOFIX (pre-registered, prespec_tofix.md; Noor approved ≤ $0.75 via the lead)
- New driver harness/run_tofix.py and module harness/tofix_decode.py: OpenRouter HTTP-200 error bodies (504 "Provider timed out") are retried, then counted FAILED; per-branch call accounting; stop if > 5% of the last 200 calls fail. Default routing (no top_k/require_parameters). costpatch/run_v2/run_closers/run_jenfu/run_noor untouched. Flags-off identity f3a70ed6 (identity_check_tofix.py, no -I, PYTHONHASHSEED=0). Prespec body 47ab4a2e…, frozen before data.
- Ran 01:56-02:06 ET, seeds 800-819, arm U core (clean, A/A, BASE, bribe-only, lie-only), 0 failed calls; 7.05% of calls hit a 504 on the first try and all recovered. $0.30 + $0.0145 smoke (seed 899). Results: out/tofix/RESULTS.md. Exploratory caveat: forced idles of 14-21% remain, mostly non-empty unparseable replies, not timeouts.
