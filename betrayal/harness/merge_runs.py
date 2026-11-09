"""Usage: merge_runs.py DEST SRC1 [SRC2 ...] -> concatenates the JSONL logs, sums the cost tallies, and copies the message CSVs."""
import sys, os, json, shutil
dest, srcs = sys.argv[1], sys.argv[2:]
os.makedirs(dest, exist_ok=True)
for name in ("games.jsonl", "agents.jsonl", "cells.jsonl", "events.jsonl", "errors.jsonl"):
    with open(os.path.join(dest, name), "w") as out:
        for s in srcs:
            p = os.path.join(s, name)
            if os.path.exists(p):
                out.write(open(p).read())
tot = {"usd": 0.0, "calls": 0, "failed": 0, "empty": 0, "rate_limited_429": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_arm": {}, "by_model": {}}
for s in srcs:
    p = os.path.join(s, "cost_tally.json")
    if not os.path.exists(p): continue
    c = json.load(open(p))
    for k in ("usd", "calls", "failed", "empty", "rate_limited_429", "prompt_tokens", "completion_tokens"):
        tot[k] += c.get(k, 0) or 0
    for grp in ("by_arm", "by_model"):
        for k, v in c.get(grp, {}).items():
            d = tot[grp].setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv
json.dump(tot, open(os.path.join(dest, "cost_tally.json"), "w"), indent=1)
for s in srcs:
    for f in ("messages.csv", "messages_other_arms.csv"):
        p = os.path.join(s, f)
        if os.path.exists(p):
            d = os.path.join(dest, f)
            if not os.path.exists(d):
                shutil.copyfile(p, d)
            else:   # append without header
                with open(d, "a") as out, open(p) as inp:
                    next(inp); shutil.copyfileobj(inp, out)
print("merged", srcs, "->", dest, "usd", round(tot["usd"], 4))
