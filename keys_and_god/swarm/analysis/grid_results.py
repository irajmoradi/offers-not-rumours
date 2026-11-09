"""Final grid tables (plan/GRID_RESULTS.md): paired God effects per cell + dose / turn-mode contrasts.

    python -m swarm.analysis.grid_results --g1 runs/grid_oss_sim_k2 --g2 runs/r2_grid_oss_seq_k2 \
        --g3 runs/grid_oss_sim_k6 [--boot 10000] [--json out.json]

Per seed (game_idx) and branch: welfare, win_rate, lost (needs lost to wrong defects, per agent),
gives / defects_ok / board_posts / dms (successful actions per agent over the game), and fork
divergence vs the nogod branch (swarm.analysis.ripple, coarse action signatures): post-fork mean
fraction of agents whose round actions differ, and the final cumulative fraction ever diverged.
Delta per seed = god - nogod (and for G1 also noise B' - nogod). Reported: mean, Student-t 95% CI
and exact/MC sign-flip p (primary), percentile bootstrap 95% CI (secondary). Seeds with any
unhealthy branch (grid_stats.game_health) are dropped. Contrasts pair seeds by world (same run.seed
and game_idx): dose = G3 delta - G1 delta, mode = G2 delta - G1 delta (on G2's seeds).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from swarm.analysis.grid_stats import boot_ci, game_health, game_metrics, signflip_p, t_ci
from swarm.analysis.ripple import ripple
from swarm.runlog.trace import load_game

ACTION_METRICS = {"gives": "give_key", "defects_ok": "defect", "board_posts": "post_board", "dms": "send_message"}
METRICS = ("welfare", "win_rate", "lost", *ACTION_METRICS, "div_post_fork", "div_final_cum")


def action_counts(game_dir: Path) -> dict[str, float]:
    n_agents, counts = 0, dict.fromkeys(ACTION_METRICS, 0)
    with open(game_dir / "events.jsonl", encoding="utf-8") as f:
        for ln in f:
            if '"type":"action_result"' in ln:
                d = json.loads(ln)["data"]
                if d.get("ok"):
                    for m, kind in ACTION_METRICS.items():
                        counts[m] += d["action"]["kind"] == kind
            elif '"type":"game_start"' in ln:
                n_agents = int(json.loads(ln)["data"]["n_agents"])
    return {m: c / max(1, n_agents) for m, c in counts.items()}


def load_cell(run_dir: Path) -> dict[str, Any]:
    man = json.loads((run_dir / "manifest.json").read_text())
    by_seed: dict[int, dict[str, Path]] = {}
    unhealthy: list[str] = []
    for g in man["games"]:
        d = run_dir / g["dir"]
        by_seed.setdefault(int(g["game_idx"]), {})[g["branch"]] = d
        h = game_health(d) if g.get("status") == "done" else {"reasons": [f"status {g.get('status')}"]}
        if h["reasons"]:
            unhealthy.append(f"{g['dir']}: {', '.join(h['reasons'])}")
    bad = {int(u.split("/")[1][1:3]) for u in unhealthy}
    seeds: dict[int, dict[str, dict[str, float]]] = {}
    for i, br in sorted(by_seed.items()):
        if i in bad or "god" not in br or "nogod" not in br:
            continue
        traces = {b: load_game(p, with_llm_calls=False, with_snapshots=False) for b, p in br.items()}
        out: dict[str, dict[str, float]] = {}
        for b, p in br.items():
            m = {**game_metrics(p), **action_counts(p)}
            if b == "nogod":
                m["div_post_fork"] = m["div_final_cum"] = 0.0
            else:
                s = ripple(traces["nogod"], traces[b], fork_from=traces["god"]).summary()
                m["div_post_fork"] = float(s["post_fork_mean_frac"] or 0.0)
                m["div_final_cum"] = float(s["final_cum"] or 0.0)
            out[b] = m
        seeds[i] = out
    return {"name": run_dir.name, "seeds": seeds, "unhealthy": unhealthy, "n_total": len(by_seed)}


def summarize(xs: list[float], n_boot: int, rng: random.Random) -> dict[str, Any]:
    return {"n": len(xs), "mean": statistics.fmean(xs) if xs else float("nan"), "t95": t_ci(xs),
            "p": signflip_p(xs, rng), "boot95": boot_ci(xs, n_boot, rng),
            "sd": statistics.stdev(xs) if len(xs) > 1 else float("nan")}


def delta(cell: dict[str, Any], m: str, a: str = "god", b: str = "nogod") -> dict[int, float]:
    return {i: br[a][m] - br[b][m] for i, br in cell["seeds"].items() if a in br and b in br}


def row(label: str, s: dict[str, Any]) -> str:
    t, bt = s["t95"], s["boot95"]
    return (f"| {label} | {s['n']} | {s['mean']:+.3f} | [{t[0]:+.3f}, {t[1]:+.3f}] | {s['p']:.3f} | "
            f"[{bt[0]:+.3f}, {bt[1]:+.3f}] |")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=Path, required=True)
    ap.add_argument("--g2", type=Path, required=True)
    ap.add_argument("--g3", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rng = random.Random(0)
    cells = {"G1": load_cell(args.g1), "G2": load_cell(args.g2), "G3": load_cell(args.g3)}
    res: dict[str, Any] = {}
    hdr = "| metric | n | mean Δ | t95 CI | sign-flip p | bootstrap 95% |\n|---|---|---|---|---|---|"
    for k, c in cells.items():
        print(f"\n### {k} {c['name']}: {len(c['seeds'])}/{c['n_total']} seeds kept; unhealthy: {c['unhealthy'] or 'none'}")
        print(hdr)
        for m in METRICS:
            s = summarize(list(delta(c, m).values()), args.boot, rng)
            res[f"{k}.god-nogod.{m}"] = s
            print(row(m, s))
        if k == "G1":
            print(f"\n### G1 noise baseline B' - nogod\n{hdr}")
            for m in METRICS:
                s = summarize(list(delta(c, m, "noise").values()), args.boot, rng)
                res[f"G1.noise-nogod.{m}"] = s
                print(row(m, s))
    for name, other in (("dose G3-G1 (K=6 vs K=2, sim)", "G3"), ("mode G2-G1 (seq vs sim, K=2)", "G2")):
        print(f"\n### contrast {name}, paired by world\n{hdr}")
        for m in METRICS:
            d1, d2 = delta(cells["G1"], m), delta(cells[other], m)
            shared = sorted(set(d1) & set(d2))
            s = summarize([d2[i] - d1[i] for i in shared], args.boot, rng)
            res[f"contrast.{other}-G1.{m}"] = s
            print(row(m, s))
    print("\n### per-branch means (all kept seeds)")
    print("| cell | branch | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS))
    for k, c in cells.items():
        for b in ("god", "nogod", "noise"):
            vals = [br[b] for br in c["seeds"].values() if b in br]
            if vals:
                print(f"| {k} | {b} | " + " | ".join(f"{statistics.fmean(v[m] for v in vals):.3f}" for m in METRICS) + " |")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, default=list))


if __name__ == "__main__":
    main()
