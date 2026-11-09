"""Per-game LLM/play health of a run dir (grid abort checks, plan/GRID_PLAN.md).

    python -m swarm.analysis.call_metrics runs/<run_id>

Prints, per game: calls / paid / cache hits / errors, finish reasons, cost per paid call and per
round, prompt/completion/reasoning tokens, latency p50/p90, parse-failure and retry rates, max_tokens
truncation rate, welfare, wins and action counts. Works on finished or still-running games (an
unfinished game has no outcome.json yet). Reads only the run dir and the OpenRouter ledger.
"""
import collections
import json
import statistics
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    run = Path((argv or sys.argv[1:])[0])
    led = [json.loads(x) for x in open(run.parent / "openrouter_ledger.jsonl") if x.strip()]
    led = [d for d in led if d.get("run_id") == run.name]
    print(f"ledger: {len(led)} paid calls, ${sum(d['cost_usd'] for d in led):.5f}")
    for g in sorted((run / "games").iterdir()):
        calls = [json.loads(x) for x in open(g / "llm_calls.jsonl")]
        ev = [json.loads(x) for x in open(g / "events.jsonl")]
        out = json.load(open(g / "outcome.json")) if (g / "outcome.json").exists() else None
        paid = [c for c in calls if c["response"] and not c["response"]["cache_hit"]]
        errs = [c for c in calls if c["error"]]
        if not paid:
            ek = collections.Counter((c["error"] or "")[:90] for c in errs)
            print(f"\n== {g.name}: NO paid calls; {len(calls)} calls, {len(errs)} errors {dict(ek)}")
            continue
        fin = collections.Counter(c["response"]["finish_reason"] for c in calls if c["response"])
        cost = sum(c["response"]["cost_usd"] for c in paid)
        lat = [c["response"]["latency_s"] for c in paid]
        pt = [c["response"]["usage"]["prompt_tokens"] for c in paid]
        ct = [c["response"]["usage"]["completion_tokens"] for c in paid]
        rt = [c["response"]["usage"]["reasoning_tokens"] for c in paid]
        turns = [e["data"] for e in ev if e["type"] == "turn"]
        pf = sum(1 for t in turns if not t["parse_ok"])
        rtry = sum(1 for t in turns if t["retries"])
        errk = collections.Counter(x for t in turns for x in t["errors"])
        rounds = out["rounds_played"] if out else max((e["round"] for e in ev), default=0)
        per_round = collections.defaultdict(float)
        for c in paid:
            per_round[c["request"]["meta"].get("round")] += c["response"]["cost_usd"]
        ar = [e["data"] for e in ev if e["type"] == "action_result"]
        kinds = collections.Counter(f'{a["action"]["kind"]}:{"ok" if a["ok"] else "fail"}' for a in ar)
        pa = out["per_agent"].values() if out else []
        scores = [v["score"] for v in pa] or [float("nan")]
        wins = sum(1 for v in pa if v["won_round"])
        print(f"\n== {g.name}: rounds {rounds} end {(out or {}).get('end_reason', 'running')}")
        print(f"  calls {len(calls)} paid {len(paid)} cache_hits {len(calls)-len(paid)-len(errs)} errors {len(errs)}; finish {dict(fin)}")
        print(f"  cost ${cost:.5f}  per paid call ${cost/max(1,len(paid)):.6f}  per round ${cost/max(1,rounds):.5f}")
        print(f"  per-round cost: {', '.join(f'r{r}:{v:.4f}' for r, v in sorted(per_round.items()))}")
        print(f"  prompt tok mean {statistics.mean(pt):.0f} max {max(pt)}; completion mean {statistics.mean(ct):.0f} max {max(ct)}; reasoning mean {statistics.mean(rt):.0f} max {max(rt)}")
        print(f"  latency p50 {statistics.median(lat):.2f}s p90 {sorted(lat)[int(.9*len(lat))]:.2f}s max {max(lat):.1f}s")
        print(f"  turns {len(turns)} parse_fail {pf} ({pf/max(1,len(turns)):.1%}) turns_with_retry {rtry} ({rtry/max(1,len(turns)):.1%}) errors {dict(errk)}")
        print(f"  truncated (finish=length) {fin.get('length',0)}/{sum(fin.values())} = {fin.get('length',0)/max(1,sum(fin.values())):.1%}")
        print(f"  welfare {statistics.mean(scores):.3f} wins {wins}/{len(pa)}  actions {dict(sorted(kinds.items()))}")


if __name__ == "__main__":
    main()
