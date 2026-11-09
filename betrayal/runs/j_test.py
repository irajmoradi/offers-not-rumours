import llm_game as lg, detmock, outsider_bribe, fulfillment
outsider_bribe.install(lg); fulfillment.install(lg); bk = detmock.DetMock()
for sched in ("never", "initial_only", "always"):
    for kind in ("insider", "outsider"):
        st = lg.new_game(8, "complete", 3, 12, 0.7); lg.play(st, bk, until=1, seed=3)
        b = lg.fork(st, 31004); b.fulfillment = dict(schedule=sched, p=0.3); b._ff_seed = 31004
        dv = None
        if kind == "insider": lg.compromise(b, lg.pick_target(st, "random"))
        else: b.outsider_bribe = dict(budget=4, per_round=2, recognition="flagged"); dv = dict(backend=None, budget=4, per_round=2, start=2)
        lg.play(b, bk, seed=31004, d_evil=dv)
        x = sorted(b.compromised)
        print(sched, kind, "accepted", sum(q["accepted"] for q in b.bribes), "paid", sum(q["paid"] for q in b.bribes), "failed", sum(bool(q.get("pay_failed")) for q in b.bribes),
              "insider gives", int(b.gifts[x[0]].sum()) if x else None, "stalls", sum(q["stalls"] for q in b.bribes))
