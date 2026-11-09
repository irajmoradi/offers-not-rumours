import llm_game as lg, detmock, growth, outsider_bribe, v2patch
v2patch.install(lg); growth.install(lg); outsider_bribe.install(lg); bk = detmock.DetMock()
st = lg.new_game(8, "complete", 5, 12, 0.7); lg.make_dormant(st, [6, 7], 4)
lg.play(st, bk, until=1, seed=5)
b = lg.fork(st, 31006); b.outsider_bribe = dict(budget=4, per_round=2, recognition="flagged")
lg.play(b, bk, seed=31006, d_evil=dict(backend=None, budget=4, per_round=2, start=4))
acts = {}
for c in b.calls: acts.setdefault(c["agent"], []).append(c["round"])
print("first round acted by 6/7:", min(acts.get(6, [99])), min(acts.get(7, [99])), "| others acted in round 1-3:", sorted({a for a, r in acts.items() if min(r) <= 3}))
print("deg 6:", b.G.degree(6), "joined:", b.growth["active"], "completion:", lg.game_metrics(b)["completion"], "rounds", b.round)
print("bribe targets:", [x["target"] for x in b.bribes], "joins in log events:", sum(1 for e in getattr(b, "log", []) if e.get("type") == "join"))
