"""EXPLORATORY (post hoc, requested by the team lead after results were seen; not in prespec_noor.md; no Holm).
Gemma-27B's LLM forger sent only 8 of 96 possible messages, so H3 (disguised > LLM rumour) is close to "offer vs almost no
attack". The scripted-rumour branch always sends (same RNG stream, targets, victims and forged senders as the disguised
bribe), so it is the content control. Per model: disguised bribe - scripted rumour, flagged bribe - scripted rumour,
scripted rumour - clean twin. Same test/CI code as analyze_noor.py (copied): two-sided paired sign-flip (exact n<=20,
else 200k MC seeded by sha256 of the test name), seed-cluster bootstrap 95% CI. No new model calls.
Usage: explore_noor_scripted.py OUT_DIR SRC_DIR [SRC_DIR ...]"""
import sys, os, json, hashlib
import numpy as np, pandas as pd
OUT, SRCS = sys.argv[1], sys.argv[2:]
G = pd.DataFrame([json.loads(l) for s in SRCS for l in open(os.path.join(s, "games.jsonl"))])
G = G[(G.arm == "NR") & (G.prompt == "short") & (G.seed >= 700) & (G.seed < 724)]
def _rng(name): return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
def signflip(d, name):
    d = np.asarray(d, float); n = len(d); obs = abs(d.sum())
    if obs == 0: return 1.0, 0.0, "exact" if n <= 20 else "mc"
    if n <= 20:
        idx = np.arange(2 ** n, dtype=np.int64); S = np.zeros(2 ** n)
        for j in range(n): S += np.where((idx >> j) & 1, -d[j], d[j])
        return float((np.abs(S) >= obs - 1e-9).mean()), 0.0, "exact"
    rng = _rng("flip|" + name); hits = 0
    for _ in range(10):
        sg = rng.choice((-1.0, 1.0), (20_000, n)); hits += int((np.abs(sg @ d) >= obs - 1e-9).sum())
    p = (1 + hits) / 200_001
    return p, float(np.sqrt(p * (1 - p) / 200_000)), "mc"
def ci(d, name):
    bs = _rng("boot|" + name).choice(np.asarray(d, float), (5000, len(d))).mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
rows = []
for m in sorted(G.model.unique()):
    ps = lambda b: dict(zip(G[(G.model == m) & (G.branch == b)].seed, G[(G.model == m) & (G.branch == b)].defections))
    for plus, minus in (("disguised_bribe", "scripted_rumour"), ("flagged_bribe", "scripted_rumour"), ("scripted_rumour", "clean|none")):
        a, b = ps(plus), ps(minus); sd = sorted(set(a) & set(b)); d = [a[s] - b[s] for s in sd]
        name = f"EXPL {plus} - {minus} [{m}]"; p, se, kind = signflip(d, name)
        rows.append(dict(test=name, model=m, n=len(d), diff=round(float(np.mean(d)), 3), ci95=ci(d, name), p=float(f"{p:.4g}"),
                         p_se=float(f"{se:.2g}"), p_method=kind, defections_plus=int(sum(a[s] for s in sd)),
                         defections_minus=int(sum(b[s] for s in sd)), label="EXPLORATORY (post hoc, no Holm)"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "noor_exploratory_scripted.csv"), index=False)
json.dump(rows, open(os.path.join(OUT, "noor_exploratory_scripted.json"), "w"), indent=1)
for r in rows: print(f"{r['test']}: {r['diff']} {r['ci95']} p={r['p']}({r['p_method']}) events {r['defections_plus']} vs {r['defections_minus']}")
