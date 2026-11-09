"""After a cache-replay restart: does the new run reproduce the archived run, round for round?
    python -I infra/sweep100/verify_replay.py <archived run dir> <new run dir>
For every game in the archived run: compares round_end state_hash, welfare and the cumulative win count for each
round both runs have finished, and prints the first mismatch (or "identical through r<N>") plus the new run's
paid (non-cache) calls so far. Read-only."""
import json
import sys
from pathlib import Path


def rounds(gdir: Path) -> dict[int, tuple[str, float, int]]:
    out, wins = {}, 0
    p = gdir / "events.jsonl"
    if not p.exists():
        return out
    for ln in open(p, encoding="utf-8", errors="ignore"):
        if '"type":"win"' in ln or '"type": "win"' in ln:
            wins += 1
        elif '"round_end"' in ln:
            e = json.loads(ln)
            d = e["data"]
            out[e["round"]] = (d.get("state_hash"), round(float(d.get("welfare") or 0), 6), wins)
    return out


def paid_calls(gdir: Path) -> int:
    p = gdir / "llm_calls.jsonl"
    if not p.exists():
        return 0
    return sum(1 for ln in open(p, encoding="utf-8", errors="ignore") if ('"cache_hit":false' in ln or '"cache_hit": false' in ln))


def main(old: str, new: str) -> int:
    bad = 0
    for g in sorted((Path(old) / "games").iterdir()):
        if not g.is_dir() or "." in g.name:
            continue
        a, b = rounds(g), rounds(Path(new) / "games" / g.name)
        common = sorted(set(a) & set(b))
        mism = next((r for r in common if a[r] != b[r]), None)
        last_a = max(a) if a else 0
        if mism is not None:
            bad += 1
            print(f"{g.name}: MISMATCH at r{mism}: archived {a[mism]} vs new {b[mism]}")
        else:
            r = common[-1] if common else 0
            wins = b[r][2] if r else 0
            print(f"{g.name}: identical through r{r} (archived last r{last_a}; wins at r{r}: {wins}); "
                  f"new run paid calls so far: {paid_calls(Path(new) / 'games' / g.name)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
