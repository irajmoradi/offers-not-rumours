"""viz figure: fraction of CoTs with suspicion > 5 by round, No God vs God k=1 vs God k=5 (5-round bins).
Usage: viz run make_fig.py <per_round.csv> <out_stem>"""
import csv, sys
from collections import defaultdict
import viz

csv_path, out = sys.argv[1], sys.argv[2]
B = 5
CONDS = ["No God", "God k=1", "God k=5"]
cnt = defaultdict(lambda: [0.0, 0])
with open(csv_path) as f:
    for row in csv.DictReader(f):
        n = int(row["n_cots"]); b = (int(row["round"]) - 1) // B
        k = (row["env"], row["world"], b)
        cnt[k][0] += float(row["frac_suspicion_gt5"]) * n; cnt[k][1] += n
bins = sorted({b for (_, _, b) in cnt})
x = [b * B + (B + 1) / 2 for b in bins]
runs, mean = {}, {}
for c in CONDS:
    worlds = sorted({w for (e, w, _) in cnt if e == c})
    per = {w: [100 * cnt[(c, w, b)][0] / cnt[(c, w, b)][1] if cnt.get((c, w, b), [0, 0])[1] else None for b in bins]
           for w in worlds}
    runs[c] = list(per.values())
    mean[c] = []
    for i in range(len(bins)):
        v = [p[i] for p in per.values() if p[i] is not None]
        mean[c].append(sum(v) / len(v) if v else None)
viz.set_defaults(size="paper", titles=False)
viz.set_roles(control="No God", ordered=["God k=1", "God k=5"])
fig = viz.line(mean, x=x, runs=runs, markers=False, xlabel="Round (5-round bins)",
               ylabel="CoTs with suspicion > 5", decimals=1,
               title="Suspicious CoTs rose from 0.39% without God to 3.71%")
fig.save(out + ".png"); fig.save(out + ".pdf")
print(fig.caption())
