import os
import itertools, json, sys
sys.path.insert(0, f"{os.environ.get('PLANC_ROOT', '.')}/harness")
F = ["A","B","C","D","E","F","G"]
words = {"ABCE","BCDF","ACDG"}
gens = list(words)
rel = set()
for r in range(1, 4):
    for combo in itertools.combinations(gens, r):
        w = set()
        for g in combo: w ^= set(g)
        rel.add("".join(sorted(w)))
print("defining relation: I =", " = ".join(sorted(rel, key=lambda x:(len(x),x))))
def alias(eff):
    out = []
    for w in rel:
        a = "".join(sorted(set(eff) ^ set(w)))
        if len(a) <= 2: out.append(a)
    return out
print("main effects aliased with (<=2fi):", {f: alias(f) for f in F})
seen=set(); groups=[]
for a,b in itertools.combinations(F,2):
    e=a+b
    if e in seen: continue
    al=[x for x in alias(e) if len(x)==2]
    grp=sorted(set([e]+al)); seen|=set(grp); groups.append("=".join(grp))
print("2fi alias chains:", groups)
