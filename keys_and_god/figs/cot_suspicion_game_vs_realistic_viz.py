"""% of CoTs with Jev-1.13 suspicion > 5 by round (5-round bins), Game Env vs Realistic Env, no God.
Thin lines = each world; thick = mean over worlds. Data: ../cot_suspicion/per_round.csv"""
import csv, os
from collections import defaultdict
import viz
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "../cot_suspicion/per_round.csv")
B = 5
CONDS = ["Game Env", "Realistic Env"]

def binned(csv_path, conds):
    cnt = defaultdict(lambda: [0.0, 0])
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            n = int(r["n_cots"]); b = (int(r["round"]) - 1) // B
            cnt[(r["env"], r["world"], b)][0] += float(r["frac_suspicion_gt5"]) * n
            cnt[(r["env"], r["world"], b)][1] += n
    bins = sorted({b for (_, _, b) in cnt})
    x = [b * B + (B + 1) / 2 for b in bins]
    runs, mean = {}, {}
    for c in conds:
        worlds = sorted({w for (e, w, _) in cnt if e == c})
        per = [[100 * cnt[(c, w, b)][0] / cnt[(c, w, b)][1] if (c, w, b) in cnt else None for b in bins] for w in worlds]
        runs[c] = per
        mean[c] = [(lambda v: sum(v) / len(v) if v else None)([p[i] for p in per if p[i] is not None]) for i in range(len(bins))]
    return x, mean, runs

if __name__ == "__main__":
    x, mean, runs = binned(CSV, CONDS)
    viz.set_defaults(size="paper", titles=False)
    roles = {"control": "Realistic Env", "intervention": "Game Env"}
    fig = viz.line(mean, x=x, runs=runs, roles=roles, markers=False, end_values=False, xlabel="Round",
                   ylabel="CoTs with suspicion > 5 (%)", decimals=1,
                   title="2.6% of Game Env CoTs were suspicious vs 0.4% in Realistic Env")
    out = os.path.join(HERE, "cot_suspicion_game_vs_realistic_viz")
    fig.save(out + ".png"); fig.save(out + ".pdf")
    print(fig.caption())
