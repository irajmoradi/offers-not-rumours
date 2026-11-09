import llm_game as lg, detmock, u_factors
u_factors.install(lg); bk = detmock.DetMock()
st = lg.new_game(8, "complete", 4, 12, 0.7); st.records = "closed"; lg.play(st, bk, until=1, seed=4)
b = lg.fork(st, 31005); b.compromise_plan = dict(x=3, at=6); lg.play(b, bk, seed=31005)
acts = {}
for c in b.calls: acts.setdefault(c["agent"], []).append(c["round"])
print("agent 3 last acted round:", max(acts.get(3, [0])), "(should be < 6)", "compromised:", sorted(b.compromised))
print("ledger line in prompts:", any("public ledger)" in c["prompt"] for c in b.calls), "(closed -> False)")
o = lg.fork(st, 1); o.records = "open_hides"; lg.play(o, bk, seed=1)
print("ledger line with open_hides:", any("public ledger)" in c["prompt"] for c in o.calls))
