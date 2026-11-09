"""Compare arm C (N=8) vs C16 (N=16) from their results_arm_C.json files. EXPLORATORY, analysis only.
Usage: compare_c_n.py OUT_DIR C8_RESULTS_JSON C16_RESULTS_JSON
Per-agent normalisation: extra defections / N (the outsider uses the D-Evil slot, so all N agents start honest).
Unpaired across N (different worlds), so only CIs are shown; no test."""
import sys, os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
OUT, A, B = sys.argv[1:4]; os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
ORDER = ["outsider_only", "outsider_only_3key", "outsider+contagion", "outsider+contagion_3key"]
rows = []
for N, f in ((8, A), (16, B)):
    for c in json.load(open(f))["cells"]:
        hops = {int(k): v for k, v in (c.get("hop_counts") or {}).items()}
        rows.append(dict(N=N, cell=c["cell"], seeds=c["seeds"], extra_defections=c["extra_defections"], ed_lo=c["ed_lo"], ed_hi=c["ed_hi"],
                         extra_def_per_agent=round(c["extra_defections"] / N, 4), key_damage=c["key_damage"],
                         offers=c["bribe_offers"], accepted=c["bribes_accepted"], accept_rate=round(c["bribes_accepted"] / max(1, c["bribe_offers"]), 3),
                         recruited_total=c["recruited_total"], games_with_recruit=c["games_with_recruit"], max_hop=c["max_hop"],
                         recruits_per_recruiter=c["mean_recruits_per_recruiter"],
                         hops=" / ".join(f"g{k}:{hops[k]}" for k in sorted(hops))))
T = pd.DataFrame(rows); T["o"] = T.cell.map(ORDER.index); T = T.sort_values(["o", "N"]).drop(columns="o")
T.to_csv(os.path.join(OUT, "figs", "figC_N8_vs_N16.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
w = 0.38; col = {8: "#2a78d6", 16: "#eb6834"}
for j, N in enumerate((8, 16)):
    d = T[T.N == N].set_index("cell").reindex(ORDER)
    xs = [i + (j - 0.5) * w for i in range(len(ORDER))]
    ax[0].bar(xs, d.extra_defections, w, color=col[N], label=f"N={N}",
              yerr=[d.extra_defections - d.ed_lo, d.ed_hi - d.extra_defections], capsize=3, ecolor="#52514e")
    ax[1].bar(xs, d.recruited_total / d.seeds, w, color=col[N], label=f"N={N}")
for a in ax:
    a.set_xticks(range(len(ORDER))); a.set_xticklabels([c.replace("outsider+", "+").replace("_3key", "\n3-key") for c in ORDER], fontsize=8)
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e5e0"); a.set_axisbelow(True); a.legend(frameon=False)
ax[0].set_ylabel("extra defections per game vs clean twin"); ax[0].set_title("Betrayal: N=8 vs N=16 (95% CI)")
ax[1].set_ylabel("recruits per game"); ax[1].set_title("Recruits per game (recruits use the 1-key template)")
fig.suptitle("Arm C vs C16 (exploratory): same design, all-Gemma-3, 40 seeds each; unpaired across N", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "figC_N8_vs_N16.png"), dpi=200); plt.close(fig)
pd.set_option("display.width", 250); print(T.to_string(index=False))
