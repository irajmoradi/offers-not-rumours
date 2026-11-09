"""Board main effect without God (plan/SWEEP_RESULTS.md): per world, board-off noGod minus G1 noGod.

    python infra/sweep/board_effect.py LABEL=OFF_RUN:WORLDS [LABEL=OFF_RUN:WORLDS ...] [--ref runs/grid_oss_sim_k2]

WORLDS like 0-15 or 0-5. Per world (same world seed; different agent seeds, since the game differs):
welfare, win rate, and per agent gives, DMs, board posts, defects (correct / wrong), plus rounds played.
Reports the mean of each difference with a Student-t 95% CI, an exact sign-flip p and n, and the
two level means. Health: a world where either game is not `done`, ended budget_exceeded, or has
more than 10% LLM-call errors is dropped (grid_stats.game_health). Reads run dirs only.
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

from swarm.analysis.grid_stats import game_health, signflip_p, t_ci
from swarm.analysis.outcomes import game_summary
from swarm.runlog.trace import load_game

METRICS = ("welfare", "win_rate", "gives", "dms", "board_posts", "defects", "defects_correct", "defects_wrong",
           "rounds")


def metrics(gdir: Path) -> dict[str, float] | None:
    if not (gdir / "outcome.json").exists() or game_health(gdir)["reasons"]:
        return None
    tr = load_game(gdir, with_llm_calls=False, with_snapshots=False)
    s = game_summary(tr)
    n = max(1, len(tr.agents))
    return {"welfare": s["welfare"], "win_rate": s["win_rate"], "gives": s["gives"] / n, "dms": s["messages"] / n,
            "board_posts": s["board_posts"] / n, "defects": s["defects"] / n,
            "defects_correct": s["defects_correct"] / n, "defects_wrong": s["defects_wrong"] / n,
            "rounds": float(s["rounds"])}


def worlds(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--ref", type=Path, default=Path("runs/grid_oss_sim_k2"))
    a = ap.parse_args()
    rng = random.Random(0)
    for spec in a.cells:
        label, _, rest = spec.partition("=")
        run, _, ws = rest.rpartition(":")
        rows, dropped = [], []
        for i in worlds(ws):
            off = metrics(Path(run) / "games" / f"g{i:02d}-nogod")
            ref = metrics(a.ref / "games" / f"g{i:02d}-nogod")
            if off is None or ref is None:
                dropped.append(i)
                continue
            rows.append((i, off, ref))
        print(f"== {label}: {run} noGod - {a.ref.name} noGod, worlds {ws} (n={len(rows)}; dropped {dropped})")
        for m in METRICS:
            d = [off[m] - ref[m] for _, off, ref in rows]
            if len(d) < 2:
                continue
            lo, hi = t_ci(d)
            neg = sum(x < 0 for x in d)
            print(f"  {m:16s} off {statistics.fmean(o[m] for _, o, _ in rows):7.3f}  on {statistics.fmean(r[m] for *_, r in rows):7.3f}"
                  f"  diff {statistics.fmean(d):+.3f} t95 [{lo:+.3f}, {hi:+.3f}] p={signflip_p(d, rng):.4f}"
                  f"  ({neg}/{len(d)} worlds < 0)")


if __name__ == "__main__":
    main()
