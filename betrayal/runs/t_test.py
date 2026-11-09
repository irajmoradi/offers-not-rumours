import llm_game as lg, detmock, fulfillment
fulfillment.install(lg); bk = detmock.DetMock()
for timing in ("now", "later", "end"):
    tot = []
    for sd in range(6):
        st = lg.new_game(8, "complete", sd, 12, 0.7); lg.play(st, bk, until=1, seed=sd)
        b = lg.fork(st, 31001 + sd); lg.compromise(b, lg.pick_target(st, "random"))
        b.fulfillment = dict(schedule="always", p=0.3, timing=timing, delay=3); b._ff_seed = 31001 + sd
        lg.play(b, bk, seed=31001 + sd)
        tot += [(q["accepted"], q.get("paid_round"), q.get("pay_due"), q["round"]) for q in b.bribes if q["accepted"]]
    print(timing, [(t[3], t[2], t[1]) for t in tot if t[0]][:8])
