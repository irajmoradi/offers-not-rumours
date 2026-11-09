import json, re, statistics as stt, llm_game as lg, detmock, v2patch, fastpaths, jwin_fixes, os
fastpaths.install(lg); v2patch.install(lg); jwin_fixes.install("/tmp/claude-1000/-home-iraj-projects-swarm-data-/6da8a2b5-8033-4627-a30c-797c3dde5e45/scratchpad")
bk = detmock.DetMock()
rows = []
for seed in range(4):
    st = lg.new_game(8, "complete", seed, 12, 0.7); lg.play(st, bk, until=1, seed=seed)
    b = lg.fork(st, 31001 + seed); x = lg.pick_target(st, "random"); lg.compromise(b, x); b.defense = "reputation+corroboration+quarantine"
    lg.play(b, bk, seed=31001 + seed)
    rows += [(c["system"], c["prompt"]) for c in b.calls]
sysl = [len(s) for s, u in rows]; usr = [len(u) for s, u in rows]
def compact(u):   # same information, JSON without spaces after , and :
    return re.sub(r'(\{[^\n]*\}|\[[^\n]*\])', lambda m: m.group(0).replace(", ", ",").replace(": ", ":"), u)
cu = [len(compact(u)) for s, u in rows]
print(json.dumps(dict(calls=len(rows), system_chars_mean=round(stt.mean(sysl)), user_chars_mean=round(stt.mean(usr)),
      est_tokens_per_call=round((stt.mean(sysl) + stt.mean(usr)) / 4), compact_json_user_chars_mean=round(stt.mean(cu)),
      compact_saving_pct=round(100 * (1 - (stt.mean(sysl) + stt.mean(cu)) / (stt.mean(sysl) + stt.mean(usr))), 1))))
# share of the user prompt by line type
import collections
share = collections.Counter()
for s, u in rows:
    for ln in u.split("\n"):
        k = ln.split(":")[0][:40] if not ln.startswith("#") else "message line"
        share[k] += len(ln)
tot = sum(share.values())
print({k: round(100 * v / tot, 1) for k, v in share.most_common(10)})
