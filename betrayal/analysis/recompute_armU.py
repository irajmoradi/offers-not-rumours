"""Independent recompute of arm U headline numbers from raw per-game rows (games.jsonl), not results json.
Extra defections per seed = attacked-branch defections minus the same seed's clean twin with the same defense
(clean|<defense>), within the N=8 / all-Gemma / closed-records cell. Seeds: arm_U_pilot (seed 0) + arm_U_main (1-35).
Tests: paired difference across seeds; two-sided sign-flip p (exact enumeration if n<=22, else 200k MC);
95% percentile bootstrap CI (10k). Written by p1-critic-plan; does not import the harness analysis code."""
import os
import json, sys, numpy as np
ROOT = f"{os.environ.get('PLANC_ROOT', '.')}/out"
rows = []
for f in (f"{ROOT}/arm_U_pilot/games.jsonl", f"{ROOT}/arm_U_main/games.jsonl"):
    rows += [json.loads(l) for l in open(f)]
rows = [g for g in rows if g["N"] == 8 and g["mix"] == "gemma" and g["records"] == "closed"]
by = {}
for g in rows:
    by.setdefault(g["seed"], {})[g["branch"]] = g
seeds = sorted(by)
MET = sys.argv[1] if len(sys.argv) > 1 else "defections"

ALL = "provenance+verify+monitor+reputation+corroboration+resilience+quarantine"
def defense_of(br):
    d = br.split("=", 1)[1] if br.startswith("defense=") else "none"
    return ALL if d == "ALL" else d

def extra(br, metric=MET):
    out = []
    for s in seeds:
        b = by[s]
        if br not in b or f"clean|{defense_of(br)}" not in b:
            continue
        out.append((s, b[br][metric] - b[f"clean|{defense_of(br)}"][metric]))
    return dict(out)

rng = np.random.default_rng(12345)
def signflip(x):
    x = np.asarray(x, float); obs = abs(x.mean()); x = x[x != 0]; n = len(x)   # zeros never change |mean| under sign flips
    if n == 0: return 1.0
    obs = abs(x.sum()); 
    if n <= 22:
        import itertools
        sg = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        sg = rng.choice([-1, 1], (200000, n))
    return float((np.abs((sg * x).sum(1)) >= obs - 1e-12).mean())
def boot(x, B=10000):
    x = np.asarray(x, float); m = rng.choice(x, (B, len(x))).mean(1); return np.percentile(m, [2.5, 97.5]).round(3).tolist()

def report(name, a, b=None):
    A = extra(a)
    if b is None:
        x = np.array(list(A.values())); print(f"{name}: n={len(x)} mean={x.mean():.3f} CI={boot(x)} p_vs0={signflip(x):.2g} sum_events={x.sum():.0f}")
        return
    B_ = extra(b); common = sorted(set(A) & set(B_))
    xa = np.array([A[s] for s in common]); xb = np.array([B_[s] for s in common]); d = xa - xb
    print(f"{name}: n={len(common)} {a}={xa.mean():.3f} ({xa.sum():.0f} events) {b}={xb.mean():.3f} ({xb.sum():.0f} events) "
          f"diff={d.mean():.3f} CI={boot(d)} p2={signflip(d):.2g}")

print("metric:", MET, "| seeds:", len(seeds), seeds[0], "-", seeds[-1])
report("outsider bribe vs forged", "pos=outsider", "posXmethod=outsider_forged")
report("insider bribe-only vs lie-only", "method=insider_bribe_only", "method=insider_lie_only")
report("both: bribe vs forged", "pos=both", "posXmethod=both_forged")
report("BASE", "BASE")
report("resilience", "defense=resilience")
report("BASE vs resilience", "BASE", "defense=resilience")
report("BASE vs ALL", "BASE", "defense=ALL")
# turned counts (as reported)
for br in ("pos=outsider", "posXmethod=outsider_forged", "method=insider_bribe_only", "method=insider_lie_only", "pos=both", "posXmethod=both_forged"):
    print("turned", br, sum(by[s][br]["turned"] for s in seeds if br in by[s]))
# clean-twin defections (floor check)
print("clean|none defections per game:", np.mean([by[s]["clean|none"]["defections"] for s in seeds]))
