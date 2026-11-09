"""% of CoTs with Jev-1.13 suspicion > 5 by round (5-round bins), Realistic Env, No God vs God k=1 vs God k=5.
Thin lines = each of 5 worlds; thick = mean over worlds. Data: ../cot_suspicion/god_dose/per_round.csv"""
import os, sys
import viz
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cot_suspicion_game_vs_realistic_viz import binned
CSV = os.path.join(HERE, "../cot_suspicion/god_dose/per_round.csv")
CONDS = ["No God", "God k=1", "God k=5"]
x, mean, runs = binned(CSV, CONDS)
viz.set_defaults(size="paper", titles=False)
roles = {"control": "No God", "intervention": "God k=1", "second": "God k=5"}
fig = viz.line(mean, x=x, runs=runs, roles=roles, markers=False, end_values=False, xlabel="Round",
               ylabel="CoTs with suspicion > 5 (%)", decimals=1,
               title="Suspicious CoTs rose from 0.39% without God to 3.71% with God k=5")
out = os.path.join(HERE, "cot_suspicion_god_dose_viz")
fig.save(out + ".png"); fig.save(out + ".pdf")
print(fig.caption())
