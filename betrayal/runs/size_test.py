import llm_game as lg, detmock, outsider_bribe, contagion
outsider_bribe.install(lg); contagion.install(lg); bk = detmock.DetMock()
for size in (1, 3):
    st = lg.new_game(8, "complete", 2, 12, 0.7); lg.play(st, bk, until=1, seed=2)
    b = lg.fork(st, 31003); b.outsider_bribe = dict(budget=4, per_round=2, recognition="flagged", size=size)
    lg.play(b, bk, seed=31003, d_evil=dict(backend=None, budget=4, per_round=2, start=2))
    ob = [q for q in b.bribes if q.get("outsider")]
    print("size", size, "offers keys:", [q["keys"] for q in ob])
    for i in range(8):
        for h in b.heard[i]:
            if h["kind"] == "bribe": print("   text:", h["text"][:170]); break
        else: continue
        break
