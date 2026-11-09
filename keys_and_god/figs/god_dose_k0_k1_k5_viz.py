import viz
viz.set_defaults(size="paper", titles=False)
# 60 GLM agents, Realistic Env (original render), worlds g00,g01,g04,g06,g16; same code & seeds for all k
W={"No God":[(65.6,88.9,97.8,99.4),(56.7,82.2,98.3,100.0),(57.2,70.0,81.7,90.0),(58.3,83.3,97.8,99.4),(58.3,82.8,97.2,98.9)],
   "God k=1":[(63.9,82.8,95.6,98.9),(58.9,80.6,95.0,97.8),(59.4,76.1,86.7,96.7),(61.7,90.0,95.6,98.9),(59.4,86.1,93.9,99.4)],
   "God k=5":[(63.3,84.4,96.7,99.4),(56.1,77.8,91.1,98.3),(60.0,76.1,88.9,96.7),(58.9,81.7,89.4,97.2),(56.1,77.8,92.2,98.3)]}
DEF={"No God":[1,0,0,1,0],"God k=1":[1,4,3,2,0],"God k=5":[2,4,5,3,3]}
MAN={"No God":[0,0,0,0,0],"God k=1":[3,5,7,7,7],"God k=5":[12,14,5,9,15]}
roles={"control":"No God","ordered":["God k=1","God k=5"]}
mean=lambda d:{k:sum(v)/len(v) for k,v in d.items()}
wm={k:[sum(c)/len(c) for c in zip(*v)] for k,v in W.items()}
FW={k:[r[3] for r in v] for k,v in W.items()}
a=viz.bars(mean(FW),runs=FW,roles=roles,sort=False,decimals=1,ylabel="Final welfare",title="(c) Final welfare")
b=viz.bars(mean(DEF),runs=DEF,roles=roles,unit="",sort=False,decimals=1,ylabel="Defections per game",title="(a) Defections")
c=viz.bars(mean(MAN),runs=MAN,roles=roles,unit="",sort=False,decimals=1,ylabel="God manipulations per game",title="(b) God manipulations")
fig=viz.panels([b,c,a],cols=3,roles=roles,title="A stronger God caused more defections but no final welfare loss")
fig.save("god_dose_k0_k1_k5_viz.png"); fig.save("god_dose_k0_k1_k5_viz.pdf"); print(fig.caption())
