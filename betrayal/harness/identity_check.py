"""Offline profiling + identity check (DetMock backend, no network).
Usage (via child.py so RUN_DIR modules + tools/ are importable):
  python -I child.py RUN_DIR profile_game.py MODE N SEEDS OUT_JSON [--profile]
MODE: orig (her code as shipped) | fast (tools/fastpaths.py installed)
Plays, per seed: the insider sweep cell used in Plan C (base -> clean + every defense variant) and one
betrayal_game with God. Writes a canonical dump hash of every game (prompts, replies, results, messages
heard, bribes, defections, gifts, holdings, metrics) so orig vs fast can be compared byte for byte."""
import sys, os, json, time, hashlib, cProfile, pstats, io
mode, N, seeds, out_json = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
do_prof = "--profile" in sys.argv
import llm_game as lg
if mode == "fast":
    import fastpaths; fastpaths.install(lg)
    import v2patch
    if os.environ.get("V2P","1")=="1": v2patch.install(lg)
    if os.environ.get("GROWTH_MOD")=="1":
        import growth, outsider_bribe, fulfillment, u_factors, contagion, disclosure, outsider_se, dual_control, outsider_rumour, bribe_paraphrase; growth.install(lg); outsider_bribe.install(lg); fulfillment.install(lg); u_factors.install(lg); contagion.install(lg); disclosure.install(lg); outsider_se.install(lg); dual_control.install(lg); outsider_rumour.install(lg); bribe_paraphrase.install(lg)
import jwin_fixes; jwin_fixes.install(os.path.dirname(out_json))
from detmock import DetMock
from dataclasses import replace as _r
bk = DetMock()
VARIANTS = [("none", True), ("provenance", True), ("verify", True), ("reputation", True), ("corroboration", True),
            ("resilience", True), ("quarantine", True), ("reputation+corroboration+quarantine", True), ("none", False)]

def canon(st):
    return dict(calls=[(c["round"], c["agent"], c["prompt"], c["reply"], c["result"]) for c in st.calls],
                heard=st.heard, bribes=st.bribes, defections=st.defections, d_evil=st.d_evil_log,
                gifts=st.gifts.tolist(), hold=[sorted(h) for h in st.holdings], need=[sorted(n) for n in st.needs_left],
                burned=[sorted(b) for b in st.burned], refusals=st.refusals, board=st.board, notices=st.notices,
                m=lg.game_metrics(st), bm=lg.betrayal_metrics(st, 2), q=sorted(lg.quarantined_set(st)))

def run_all():
    dumps = []
    for seed in range(seeds):
        base = lg.new_game(N, "complete", seed, 12, 0.7, "private", "posted")
        lg.play(base, bk, until=2, seed=seed)
        x = lg.pick_target(base, "random"); bseed = 19_001 + seed
        clean = lg.fork(base, bseed); lg.play(clean, bk, seed=bseed); dumps.append(canon(clean))
        for d, lies in VARIANTS:
            b = lg.fork(base, bseed); b.defense = d
            if not lies:
                b.cfg = _r(b.cfg, insider_lie_prob=0.0)
            lg.compromise(b, x); lg.play(b, bk, seed=bseed); dumps.append(canon(b))
        rows, games = lg.betrayal_game(bk, "complete", seed, d_evil_budgets=(4,), defenses=("none", "provenance"), N=N)
        dumps.append(rows); dumps += [canon(g) for g in games.values()]
    return dumps

t = time.perf_counter()
if do_prof:
    pr = cProfile.Profile(); pr.enable(); dumps = run_all(); pr.disable()
else:
    dumps = run_all()
secs = time.perf_counter() - t
blob = json.dumps(dumps, sort_keys=True, default=str).encode()
res = dict(mode=mode, N=N, seeds=seeds, secs=round(secs, 3), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
           games=len(dumps), bribes=sum(len(d.get("bribes", [])) for d in dumps if isinstance(d, dict)),
           defections=sum(len(d.get("defections", [])) for d in dumps if isinstance(d, dict)))
if do_prof:
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18); res["profile_top"] = s.getvalue()[-6000:]
json.dump(res, open(out_json, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "profile_top"}))
if do_prof:
    print(res["profile_top"])
