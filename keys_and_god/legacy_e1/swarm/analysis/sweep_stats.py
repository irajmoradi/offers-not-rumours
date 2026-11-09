"""Sweep statistics (plan/SWEEP_PLAN.md): God vs noGod vs fork-matched noise, per condition.

    python -m swarm.analysis.sweep_stats [--seeds 0-7] [--json out.json] [--boot 10000] \
        LABEL=RUN[,NOISE_RUN[@BRANCH]] ...

Each cell is LABEL=RUN: the run dir holding the `god` and `nogod` branches. The noise baseline is the
branch `noisefork` (B'', resampled from God's first round) of NOISE_RUN (default RUN), or BRANCH
(e.g. `noise` = G1's B', resampled from round 1); LABEL=RUN,none = no noise baseline (A - B only). Per world (game_idx) present and healthy in both:
  A - B      God - noGod (outcome delta)
  N - B      noise - noGod (what resampling alone does)
  A - N      = (A - B) - (N - B): the God effect against the fork-matched noise baseline (primary)
for welfare, win_rate, burned, lost, rounds (outcome.json) and defects, defects_wrong, gives,
board_posts, dms per agent (events), plus divergence from B (ripple, coarse actions, fork round =
God's first power in A): div_god = post-fork mean share of agents acting differently in A vs B,
div_noise = the same for N vs B, div_gap = div_god - div_noise.
Each statistic: mean, t95 (Student t 95% CI; read this and the sign-flip p at n <= 16, the percentile
bootstrap is too narrow), exact sign-flip p, bootstrap CI, n. Health (grid_stats.load_run): a world
with an unhealthy branch (>10% call errors, >30% error-noop turns, budget_exceeded) is dropped.
Reads only run dirs; no LLM, no network.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from swarm.analysis.grid_stats import boot_ci, load_run, signflip_p, t_ci
from swarm.analysis.outcomes import game_summary
from swarm.analysis.ripple import ripple
from swarm.runlog.trace import load_game

BASE = ("welfare", "win_rate", "burned", "lost", "rounds")
EXTRA = ("defects", "defects_wrong", "gives", "board_posts", "dms")
CONTRASTS = (("god", "nogod"), ("noise", "nogod"), ("god", "noise"))


def per_agent(s: dict[str, Any], n: int) -> dict[str, float]:
    return {"defects": s["defects"] / n, "defects_wrong": s["defects_wrong"] / n, "gives": s["gives"] / n,
            "board_posts": s["board_posts"] / n, "dms": s["messages"] / n}


def _dirs(run_dir: Path) -> dict[tuple[int, str], Path]:
    man = json.loads((run_dir / "manifest.json").read_text())
    return {(int(g["game_idx"]), g["branch"]): run_dir / g["dir"] for g in man["games"]}


def parse_cell(spec: str) -> tuple[str, Path, Path | None, str]:
    label, _, rest = spec.partition("=")
    if not rest:
        rest, label = label, Path(label).name
    run, _, noise = rest.partition(",")
    if noise == "none":
        return label, Path(run), None, "none"
    noise_run, _, branch = noise.partition("@")
    return label, Path(run), Path(noise_run or run), branch or "noisefork"


def cell_rows(run_dir: Path, noise_dir: Path | None, noise_branch: str, seeds: set[int] | None) -> dict[str, Any]:
    main = load_run(run_dir)
    if noise_dir is None:                                   # no noise baseline: A - B only
        noise_dir, noise_branch = run_dir, "nogod"
    noise = load_run(noise_dir) if noise_dir != run_dir else main
    dm, dn = _dirs(run_dir), _dirs(noise_dir)
    rows: dict[int, dict[str, float]] = {}
    for i in sorted(set(main["seeds"]) & set(noise["seeds"])):
        if seeds is not None and i not in seeds:
            continue
        if not {"god", "nogod"} <= set(main["seeds"][i]) or noise_branch not in noise["seeds"][i]:
            continue
        g = {"god": load_game(dm[(i, "god")], with_llm_calls=False),
             "nogod": load_game(dm[(i, "nogod")], with_llm_calls=False),
             "noise": load_game(dn[(i, noise_branch)], with_llm_calls=False)}
        lv = {"god": dict(main["seeds"][i]["god"]), "nogod": dict(main["seeds"][i]["nogod"]),
              "noise": dict(noise["seeds"][i][noise_branch])}
        for k, tr in g.items():
            lv[k].update(per_agent(game_summary(tr), max(1, len(tr.agents))))
        r: dict[str, float] = {}
        for a, b in CONTRASTS:
            for m in (*BASE, *EXTRA):
                r[f"{m}:{a}-{b}"] = lv[a][m] - lv[b][m]
        ra = ripple(g["god"], g["nogod"]).summary()
        rn = ripple(g["noise"], g["nogod"], fork_from=g["god"]).summary()
        r["div_god"] = float(ra["post_fork_mean_frac"] or 0.0)
        r["div_noise"] = float(rn["post_fork_mean_frac"] or 0.0)
        r["div_gap"] = r["div_god"] - r["div_noise"]
        r["fork_round"] = float(ra["fork_round"] or 0)
        for k in ("god", "nogod", "noise"):
            r[f"welfare:{k}"] = lv[k]["welfare"]
            r[f"win_rate:{k}"] = lv[k]["win_rate"]
        rows[i] = r
    return {"rows": rows, "unhealthy": main["unhealthy"] + (noise["unhealthy"] if noise is not main else [])}


def summarize(xs: list[float], n_boot: int, rng: random.Random) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    lo, hi = boot_ci(xs, n_boot, rng)
    tlo, thi = t_ci(xs)
    return {"mean": statistics.fmean(xs), "t_lo": tlo, "t_hi": thi, "p": signflip_p(xs, rng),
            "lo": lo, "hi": hi, "n": len(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def parse_seeds(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def fmt(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return "n=0"
    return (f"{s['mean']:+.3f} t95 [{s['t_lo']:+.3f}, {s['t_hi']:+.3f}] p={s['p']:.3f} "
            f"boot [{s['lo']:+.3f}, {s['hi']:+.3f}] n={s['n']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cells", nargs="+", help="LABEL=RUN[,NOISE_RUN[@BRANCH]]")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="restrict to game_idx, e.g. 0-7")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    seeds = parse_seeds(args.seeds)
    out: dict[str, Any] = {}
    for spec in args.cells:
        label, run, noise_run, branch = parse_cell(spec)
        c = cell_rows(run, noise_run, branch, seeds)
        nz = f"{noise_run.name}@{branch}" if noise_run is not None else "none"
        res: dict[str, Any] = {"run": str(run), "noise": nz, "unhealthy": c["unhealthy"], "seeds": sorted(c["rows"])}
        print(f"== {label}: {run.name} + noise {nz} (n={len(c['rows'])} worlds)")
        for u in c["unhealthy"]:
            print(f"  UNHEALTHY {u} -> world excluded")
        keys = sorted({k for r in c["rows"].values() for k in r
                       if noise_run is not None or ("noise" not in k and k != "div_gap")})
        for k in keys:
            res[k] = summarize([r[k] for r in c["rows"].values() if k in r], args.boot, rng)
        for k in ("welfare:god-nogod", "welfare:noise-nogod", "welfare:god-noise", "win_rate:god-noise",
                  "defects:god-noise", "board_posts:god-noise", "div_god", "div_noise", "div_gap"):
            if k in res:
                print(f"  {k:22s} {fmt(res[k])}")
        out[label] = res
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
