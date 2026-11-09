"""Paired exact McNemar from cached separate.py replies: config vs the same-script unsteered baseline.
usage: python -I mcnemar_cached.py <config_name>   -> prints + writes separate/mcnemar_<config>.json"""
import os
import json
import math
import sys

D = f"{os.environ.get('LFM_ROOT', '.')}/out/gemma/neuron_steer/separate"
cfg = sys.argv[1]
order = [s["state_id"] for s in json.load(open(f"{os.environ.get('LFM_ROOT', '.')}/out/gemma/neuron_dpo/benign/states.json"))]
load = lambda p: {json.loads(l)["state_id"]: json.loads(l) for l in open(p)}  # noqa: E731
tb, bb = load(f"{D}/replies/tempt300_baseline.jsonl"), load(f"{D}/replies/benign100_baseline.jsonl")
to, bo = load(f"{D}/replies/tempt300_{cfg}.jsonl"), load(f"{D}/replies/benign100_{cfg}.jsonl")


def mc(A, B, ids, k):
    b_ = sum(1 for i in ids if A[i][k] and not B[i][k])
    c_ = sum(1 for i in ids if B[i][k] and not A[i][k])
    n = b_ + c_
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(b_, c_) + 1)) / 2 ** n) if n else 1.0
    return dict(cfg=sum(A[i][k] for i in ids), base=sum(B[i][k] for i in ids), n=len(ids), cfg_only=b_, base_only=c_, p_two_sided=p)


res = dict(tempt300_obey=mc(to, tb, list(tb), "obey"), tempt300_valid=mc(to, tb, list(tb), "valid"))
for tag, ids in (("benign_heldout50", order[50:100]), ("benign_sel50", order[:50]), ("benign100", order)):
    for k in ("engage", "help", "valid"):
        res[f"{tag}_{k}"] = mc(bo, bb, ids, k)
json.dump(res, open(f"{D}/mcnemar_{cfg}.json", "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps(v))
