import llm_game as lg, detmock, outsider_bribe, contagion
outsider_bribe.install(lg); contagion.install(lg); bk = detmock.DetMock()
for sd in range(6):
    st = lg.new_game(8, "complete", sd, 12, 0.7); lg.play(st, bk, until=1, seed=sd)
    b = lg.fork(st, 31001 + sd); b.outsider_bribe = dict(budget=4, per_round=2, recognition="flagged"); b.contagion = dict(gen={}, pending={}, log=[])
    lg.play(b, bk, seed=31001 + sd, d_evil=dict(backend=None, budget=4, per_round=2, start=2))
    print(sd, "compromised", sorted(b.compromised), "log", [(e["agent"], e["generation"], e["recruited_by"]) for e in b.contagion["log"]], "bribes", len(b.bribes))
