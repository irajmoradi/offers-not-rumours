import viz
viz.set_defaults(size="paper", titles=False)
W={"Game Env, no God":[(55.6,67.8,72.8,85.6),(45.6,56.1,69.4,90.0),(48.3,66.7,75.6,87.8)],
   "Game Env, God":   [(53.3,67.2,82.2,91.1),(43.9,56.7,60.6,66.1),(51.7,63.3,70.6,81.1)],
   "Realistic Env, no God":[(65.6,88.9,97.8,99.4),(56.7,82.2,98.3,100.0),(57.2,70.0,81.7,90.0)],
   "Realistic Env, God":[(63.9,82.8,95.6,98.9),(58.9,80.6,95.0,97.8),(59.4,76.1,86.7,96.7)]}
WINS={"Game Env, no God":[37,43,41],"Game Env, God":[47,23,35],"Realistic Env, no God":[59,60,43],"Realistic Env, God":[58,57,55]}
roles={"control":"Realistic Env, no God","third":"Realistic Env, God","second":"Game Env, no God","intervention":"Game Env, God"}
mean={k:[sum(c)/3 for c in zip(*v)] for k,v in W.items()}
runs={k:[list(r) for r in v] for k,v in W.items()}
a=viz.line(mean,x=[20,40,60,80],runs=runs,roles=roles,xlabel="Round",ylabel="Welfare",decimals=1,domain=(40,100),ticks=[40,60,80,100],end_labels="value",
           title="(a) Welfare by round")
b=viz.bars({k:sum(v)/3 for k,v in WINS.items()},runs=WINS,roles=roles,unit="",ylabel="Agents who won (of 60)",
           decimals=1,sort=False,title="(b) Winners at game end")
fig=viz.panels([a,b],cols=2,roles=roles,
     title="God lowered final welfare only in the Game Env")
fig.save("framing_2x2_god_viz.png"); fig.save("framing_2x2_god_viz.pdf")
print(fig.caption())
