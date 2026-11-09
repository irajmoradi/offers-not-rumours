"""P1 tie-in figure without internal codes: same plot as lfm_lab/NOOR-Research/data/tiein_final/tiein_final.py, redrawn
from its tiein_final_by_game.csv (identical numbers); no "Tie-in FINAL" suptitle (the caption carries it); worlds g00/g01
shown as "world 1"/"world 2". Run: python3 -I tiein_final_clean.py OUT.png"""
import os
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../single_agent/data/tiein_final/tiein_final_by_game.csv")
by = {r["game"]: r for r in csv.DictReader(open(SRC))}
assert set(by) == {f"{w}-{a}" for w in ("g00", "g01") for a in ("nogod", "k1", "k1_refusal")}
num = lambda g, k: float(by[g][k])
fg, axs = plt.subplots(1, 4, figsize=(14, 3.6), dpi=140)
arms = ["nogod", "k1", "k1_refusal"]
lab = ["no God", "God", "God +\nrefusal"]
cols = {"g00": "#2e4057", "g01": "#d1495b"}
names = {"g00": "world 1", "g01": "world 2"}
for w in ("g00", "g01"):
    axs[0].plot([1, 2], [num(f"{w}-{a}", "obeyed_all") / num(f"{w}-{a}", "god_actions_ok") for a in arms[1:]], "o-", color=cols[w], label=names[w])
    axs[1].plot([0, 1, 2], [num(f"{w}-{a}", "welfare_final") for a in arms], "o-", color=cols[w], label=names[w])
    axs[2].plot([0, 1, 2], [num(f"{w}-{a}", "wins") for a in arms], "o-", color=cols[w], label=names[w])
    axs[3].plot([0, 1, 2], [num(f"{w}-{a}", "rounds_played") for a in arms], "o-", color=cols[w], label=names[w])
axs[0].set_xticks([1, 2]); axs[0].set_xticklabels(lab[1:]); axs[0].set_xlim(0.5, 2.5); axs[0].set_ylim(0, 0.5)
axs[0].set_ylabel("share of God actions obeyed"); axs[0].set_title("obeying the fake God (all God actions)", fontsize=9)
for ax in axs[1:]:
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(lab)
axs[1].set_ylabel("final welfare"); axs[1].set_title("welfare (end of game)", fontsize=9)
axs[2].set_ylabel("agents who won (of 20)"); axs[2].set_title("wins", fontsize=9)
axs[3].set_ylabel("round the game ended"); axs[3].set_title("completion round (lower = faster)", fontsize=9)
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].legend(fontsize=7)
fg.tight_layout()
fg.savefig(sys.argv[1], bbox_inches="tight", pad_inches=0.08)
print("wrote", sys.argv[1])
