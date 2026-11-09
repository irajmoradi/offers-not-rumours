# Captions: Noor figures

Made by `make_noor_figs.py` (`viz run make_noor_figs.py`, run in this folder). It reads only Noor's frozen results files in `~/projects/lfm_lab/NOOR-Research/data/`. Each figure comes in three files: `<name>.png` (screen size, with the title), `<name>.pdf` (paper width, no title; use this in LaTeX) and `<name>_paper.png` (a preview of the PDF). `captions.json` holds the auto captions and the alt text. In LaTeX, write each `%` as `\%`.

---

## fig_hosted: how often hosted models obey a fake boss

**Caption.** Hosted models obeyed a fake boss in 7.7% to 52.7% of scenes. Error bars are 95% Wilson intervals. Each model saw the same 300 held-out tempt300 scenes (150 attack orders, 150 hand-over-a-key orders, 100 at each of three pressure levels) and replied once per scene. "Obeyed" counts only doing what the fake boss ordered: handing over the key outright, or carrying out the ordered attack. It does not count hedged counter-trades ("I'll give K if you give N"). Counting those too, the rates are 38% to 76%. The scenes are new games from the same template, not new kinds of prompt.

Source: `data/tempt300/obey_breakdown.json` (give + defect). Audited: the ranking pairs were checked with McNemar tests (gemma-31b vs deepseek p = 0.00075; glm low vs high p = 0.0015).

## fig_tradeoff: no fix cut obeying and kept the model helpful

**Caption.** Fixes that cut obeying below 10% cut honest engagement to 0-3%. Model: Gemma-2-2B. Each point is one intervention, scored on the 300 tempt300 scenes (y axis) and on 100 honest scenes with a fair trade offer and no fake boss (x axis). The neuron push uses the fraction version of Noor's formula at α 0.1. Adding a fade gate gives 3/300 and 2/100, on top of that point, so it is not drawn. The DPO point is two runs (layers 12-16 and all layers), both 0/300 and 0/100. Noor's original formula is left out because 0 of 300 replies could be read. The switch-off point is the 10 "obey team" units switched off. Its counts are audited, but the "obey team" reading is retracted, so it is labeled only "Switch-off (n.s. vs random)". It keeps engagement at 48/100, but 20 of its 100 honest replies could not be read, and it was no better than switching off 10 random units (55 vs 66 of 300, p = 0.28). The honest scenes used to choose the gated settings overlap the ones reported here, so treat those points as exploratory.

Sources: `noor_steer/tempt300.jsonl`, `neuron_dpo/table.json`, `neuron_dpo/benign.log`, `benign3.log`, `gentle.log`, `active.log`, `benign_abl.log`, `gemma_ablation/summary.json`. Audited (AUDIT2_noor rows 1-5, 13-19, 23).

## fig_obey_only: removing the help part raised helping, not resistance

**Caption.** Fair trades rose from 32% to 52%, but obeying did not fall. Error bars are 95% Wilson intervals. Model: Gemma-2-2B, pushed at layer 15 (f = 0.1) away from the obey direction after the help direction was projected out. The no-push run comes from the same script and scorer. Left: tempt300 obeying, 71 vs 68 of 300 (paired exact McNemar p = 0.81). Right: completed fair trades on the 50 honest scenes not used for selection, 16 vs 26 of 50 (p = 0.013). Not corrected for the repeated looks at tempt300. Not independently audited; recomputed by the paper writer.

Source: `noor_steer/separate/mcnemar_obey_only_f0.1.json`.

## fig_cosine: obey and help directions overlap in three small models

**Caption.** Obey and help directions had cosine 0.32 to 0.46 at mid depth. The red line is the cosine between the OBEY direction (obey vs refuse a fake authority) and the HELP direction (offer a fair trade vs post and wait), with the shared answer-letter axis projected out. The gray band is a letter-balanced sign-flip null (95%, 200 draws). Mid depth (0.4-0.7 of the layers, between the dashed lines) averages +0.46 in Gemma-2-2B-it, +0.35 in Qwen2.5-1.5B-Instruct and +0.32 in SmolLM2-1.7B-Instruct, all outside the null of about ±0.18. Split-half reliability of OBEY is 0.94 / 0.86 / 0.89. Caveats: both directions contrast "take the transaction" with "hold off", so the shared part may be "act vs wait" rather than "obeying is helping". The letter fix moves the cosine by at most 0.007; what it fixes is the null. Outside mid depth the picture changes: in Qwen and SmolLM the cosine keeps rising to about 0.5-0.67 in late layers, and in Gemma it is negative in the first quarter of the layers.

Source: `noor_steer/separate/letterfix.csv`. Audited (AUDIT_obey2 §5: verified, with the confound caveat).
