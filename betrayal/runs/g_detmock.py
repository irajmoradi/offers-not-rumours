import llm_game as lg, detmock, outsider_bribe
outsider_bribe.install(lg); bk = detmock.DetMock()
for rec in ("flagged", "disguised"):
    for start in (2, 5, 9):
        st = lg.new_game(8, "complete", 3, 12, 0.7); lg.play(st, bk, until=1, seed=3)
        b = lg.fork(st, 31004); b.outsider_bribe = dict(budget=4, per_round=2, recognition=rec)
        lg.play(b, bk, seed=31004, d_evil=dict(backend=None, budget=4, per_round=2, start=start))
        print(rec, start, "offers", len(b.bribes), "accepted(her strict)", sum(x["accepted"] for x in b.bribes), "stalls", sum(x["stalls"] for x in b.bribes), "defections", len(b.defections), "rounds", b.round)
