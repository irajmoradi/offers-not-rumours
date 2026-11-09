"""Final metrics for the 2026-11-08 evening runs (lead: table to plan/ by 22:45; hard compute cutoff 22:30).

    PYTHONNOUSERSITE=1 python infra/sweep30/final_report.py [--out plan/SWEEP30_FINAL_RESULTS.md]

Covers sweep30 (A1-A7 GLM, B0/B1/B5/B10 gpt-oss), sweep60_or_soft_{perm,silent}_k1 vs their nogod soft runs,
sweep60_or_glm_i2fix, and sweep60_or_oldframe_k1 vs sweep60_or_glm k1 (Realistic + God) and C1 sweep60_or_oldframe
(Game Env nogod). Read-only on run dirs. Every game is reported, FINISHED or not: the OpenRouter key hit its $150
limit at 19:59:18 EST (HTTP 403 on every call), so some games stop early; for those, per-round values are given up to
the last completed round (round_end events) and the game is marked partial (rN = rounds completed).
Per game: welfare (round_end welfare, needs-only) at cut rounds and at the last round, wins (win events) at the same
cuts, round at which 50% of agents had won, contests (burn events) total / wrong, and from
swarm.analysis.sweep_metrics.game_metrics: God executed (successful powers), manipulated (target acted on it),
God-steered contests. God damage = the paired nogod game's welfare minus the God game's welfare at the same round
(same world), at each cut both games reached.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"


def scan(gdir: Path) -> dict:
    from swarm.analysis.sweep_metrics import game_metrics
    welfare, wins, burns, n = {}, {}, [], 0
    for ln in open(gdir / "events.jsonl", encoding="utf-8", errors="ignore"):
        if '"game_start"' in ln:
            e = json.loads(ln)
            if e["data"]["type"] == "game_start":
                n = len(e["data"]["agents"])
        elif '"type":"round_end"' in ln:
            e = json.loads(ln)
            welfare[e["round"]] = float(e["data"]["welfare"])
        elif '"type":"burn"' in ln:
            e = json.loads(ln)
            burns.append((e["round"], bool(e["data"]["correct"])))
        elif '"type":"win"' in ln:
            e = json.loads(ln)
            wins.setdefault(e["data"]["agent"], e["round"])
    m = game_metrics(gdir)
    last = max(welfare) if welfare else 0
    wr = sorted(wins.values())
    k = int(round(0.5 * max(n, 1)))
    return {"dir": str(gdir), "n": n, "welfare": welfare, "wins": wr, "burns": burns, "last": last,
            "finished": (gdir / "outcome.json").exists(), "r50": wr[k - 1] if 0 < k <= len(wr) else None,
            "god_exec": m.get("god_executed"), "god_manip": m.get("god_manipulated"),
            "steered": m.get("defect_god_steered")}


def at(g: dict, r: int):
    """(welfare, wins, contests, wrong) at round r, or None when the game did not reach it. A FINISHED game that ended
    before r (all_done: everyone won, or the stop round) is final from its last round on, so its last values carry."""
    if r not in g["welfare"]:
        if not (g["finished"] and g["welfare"] and r > g["last"]):
            return None
        r = g["last"]
    return (g["welfare"][r], sum(1 for x in g["wins"] if x <= r), sum(1 for b, _ in g["burns"] if b <= r),
            sum(1 for b, ok in g["burns"] if b <= r and not ok))


def f(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def table(title: str, rows: list[tuple[str, str, dict]], cuts: tuple[int, ...], base: dict[str, dict] | None = None) -> str:
    head = ["arm", "game", "rounds"] + [f"welfare r{c}" for c in cuts] + ["welfare last"] + [f"wins r{c}" for c in cuts] \
        + ["wins last", "r50% won", "contests", "wrong", "God exec", "God manip", "steered"] \
        + ([f"damage r{c}" for c in cuts] if base else [])
    out = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, game, g in rows:
        cells = [arm, game, f"r{g['last']}" + ("" if g["finished"] else " (partial)") + (" (resumed)" if "/resume_" in g["dir"] else "")]
        cells += [f(a[0]) if (a := at(g, c)) else "" for c in cuts] + [f(g["welfare"].get(g["last"]))]
        cells += [str(a[1]) if (a := at(g, c)) else "" for c in cuts] + [str(len(g["wins"]))]
        cells += [f(g["r50"]), str(len(g["burns"])), str(sum(1 for _, ok in g["burns"] if not ok)),
                  f(g["god_exec"]) if g["god_exec"] else "", f(g["god_manip"]) if g["god_exec"] else "",
                  f(g["steered"]) if g["god_exec"] else ""]
        if base:
            b = base.get(game.split("-")[0])
            cells += [f(x[0] - y[0]) if b and (x := at(b, c)) and (y := at(g, c)) else "" for c in cuts]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def mean_table(title: str, arms: list[tuple[str, list[dict]]], cuts: tuple[int, ...]) -> str:
    head = ["arm", "games (finished)"] + [f"welfare r{c}" for c in cuts] + [f"wins r{c}" for c in cuts] \
        + ["contests", "wrong", "God exec", "God manip"]
    out = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, gs in arms:
        cells = [arm, f"{len(gs)} ({sum(g['finished'] for g in gs)})"]
        for i in (0, 1):
            for c in cuts:
                v = [a[i] for g in gs if (a := at(g, c))]
                cells.append((f"{sum(v) / len(v):.3f}" if i == 0 else f"{sum(v) / len(v):.1f}") + f" (n={len(v)})" if v else "")
        cells += [f"{sum(len(g['burns']) for g in gs) / len(gs):.1f}",
                  f"{sum(sum(1 for _, ok in g['burns'] if not ok) for g in gs) / len(gs):.1f}",
                  f"{sum(g['god_exec'] or 0 for g in gs) / len(gs):.1f}" if any(g["god_exec"] for g in gs) else "",
                  f"{sum(g['god_manip'] or 0 for g in gs) / len(gs):.1f}" if any(g["god_exec"] for g in gs) else ""]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def two_by_two(R: dict, cuts: tuple[int, ...]) -> str:
    """Game vs Realistic Env x God k1 / no God at 60 agents: God damage (nogod welfare - k1 welfare, same world and
    round) per env and world, plus the mean over worlds with both games at that round (n). Realistic = the fixed-render
    i2fix cells (won_render_fix true; worlds g00, g01) and, for reference, the original sweep60_or_glm cells."""
    envs = [("Game Env (C1 nogod vs oldframe_k1)", "sweep60_or_oldframe", "nogod", "sweep60_or_oldframe_k1", "k1"),
            ("Realistic Env, fixed render (i2fix nogod vs k1)", "sweep60_or_glm_i2fix", "nogod", "sweep60_or_glm_i2fix", "k1"),
            ("Realistic Env, original (sweep60_or_glm nogod vs k1)", "sweep60_or_glm", "nogod", "sweep60_or_glm", "k1")]
    head = ["env", "world", "rounds (nogod / k1)"] + [f"welfare r{c} nogod / k1" for c in cuts] + [f"damage r{c}" for c in cuts] \
        + ["wins last nogod / k1", "r50% won nogod / k1", "God exec / manip", "contests (wrong) k1"]
    out = ["## 2x2: Game vs Realistic Env x God k1 (60 agents, worlds g00/g01/g04)", "",
           "Damage = no-God welfare minus God-k1 welfare at the same round, same world (positive = God hurt welfare). "
           "Game Env = agent/v1 + god/v2 + framing none; Realistic = agent/v2_facility + god/v2_facility + facility. "
           "A finished game that ended early carries its final values.", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    means = []
    for env, nr, nb, gr, gb in envs:
        nog = {k.split("-")[0]: v for k, v in R[nr].items() if k.endswith("-" + nb)}
        god = {k.split("-")[0]: v for k, v in R[gr].items() if k.endswith("-" + gb)}
        dm = {c: [] for c in cuts}
        for w in sorted(set(nog) & set(god)):
            n, g = nog[w], god[w]
            row = [env, w, f"r{n['last']}{'' if n['finished'] else '*'} / r{g['last']}{'' if g['finished'] else '*'}"]
            for c in cuts:
                x, y = at(n, c), at(g, c)
                row.append(f"{f(x[0]) if x else '-'} / {f(y[0]) if y else '-'}")
            for c in cuts:
                x, y = at(n, c), at(g, c)
                if x and y:
                    dm[c].append(x[0] - y[0])
                row.append(f(x[0] - y[0]) if x and y else "")
            row += [f"{len(n['wins'])} / {len(g['wins'])}", f"{f(n['r50']) or '-'} / {f(g['r50']) or '-'}",
                    f"{g['god_exec']} / {g['god_manip']}", f"{len(g['burns'])} ({sum(1 for _, ok in g['burns'] if not ok)})"]
            out.append("| " + " | ".join(row) + " |")
        means.append((env, dm))
    out += ["", "Mean God damage over worlds (n = worlds with both games at that round):", "",
            "| env | " + " | ".join(f"damage r{c}" for c in cuts) + " |", "|---|" + "---|" * len(cuts)]
    for env, dm in means:
        out.append(f"| {env} | " + " | ".join(f"{sum(v) / len(v):+.3f} (n={len(v)})" if v else "" for c, v in dm.items()) + " |")
    return "\n".join(out) + "\n\n(* = game not finished: values up to its last complete round.)\n"


SWEEP30 = [("A1 facility medium nogod", "sweep30_a1_facility"), ("A2 old prompts", "sweep30_a2_oldframe"),
           ("A3 effort minimal", "sweep30_a3_minimal"), ("A4 effort low", "sweep30_a4_low"),
           ("A5 effort high", "sweep30_a5_high"), ("A6 God k10", "sweep30_a6_k10"), ("A7 God k20", "sweep30_a7_k20"),
           ("B0 gpt-oss nogod", "sweep30_b0_oss"), ("B1 gpt-oss God k1", "sweep30_b1_oss_k1"),
           ("B5 gpt-oss God k5", "sweep30_b5_oss_k5"), ("B10 gpt-oss God k10", "sweep30_b10_oss_k10")]


def games(run: str, branch: str | None = None) -> list[Path]:
    """The run's game dirs; a game resumed after the 403 (runs/resume_<run>/games/<game>, infra/sweep30/
    resume_after_403.sh) replays from round 1, so its dir is the complete record and replaces the partial one."""
    gs = sorted(p for p in (RUNS / run / "games").iterdir() if (p / "events.jsonl").exists())
    res = RUNS / f"resume_{run}" / "games"
    gs = [res / p.name if (res / p.name / "events.jsonl").exists() else p for p in gs]
    return [p for p in gs if branch is None or p.name.endswith("-" + branch)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "plan" / "SWEEP30_FINAL_RESULTS.md")
    a = ap.parse_args()
    want = {run: games(run) for _, run in SWEEP30}
    want.update({r: games(r) for r in ("sweep60_or_soft_perm_k1", "sweep60_or_soft_silent_k1", "sweep60_or_soft_perm",
                                         "sweep60_or_soft_silent", "sweep60_or_glm_i2fix", "sweep60_or_oldframe_k1",
                                         "sweep60_or_oldframe")})
    want["sweep60_or_glm"] = [p for p in games("sweep60_or_glm") if p.name[:3] in ("g00", "g01", "g04")]
    paths = [p for ps in want.values() for p in ps]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = dict(zip(paths, ex.map(scan, paths)))
    R = {run: {p.name: res[p] for p in ps} for run, ps in want.items()}
    by_world = lambda run, br: {k.split("-")[0]: v for k, v in R[run].items() if k.endswith("-" + br)}  # noqa: E731
    doc = ["# Evening runs, final metrics (2026-11-08)", "",
           "Generated by infra/sweep30/final_report.py (read-only on run dirs). Welfare = round_end welfare (needs-only), "
           "wins = agents with a win event by that round, contests = burn events (wrong = burn of a non-holder), "
           "God exec = successful powers, God manip = powers the target acted on, steered = contests following a God DM "
           "naming the target. Damage = paired nogod welfare minus God-arm welfare at the same round, same world.",
           "",
           "**OpenRouter cutoff:** the Plan A key hit its $150 limit at 19:59:18 EST (HTTP 403 on every call; ledgers: main "
           "$125.19 + sweep30 $20.38 + oldframe_k1 $3.26 = $148.82). Games marked (partial) stopped there; their "
           "values are up to the last completed round. Games marked (resumed) were completed at 21:20+ on the team key by "
           "infra/sweep30/resume_after_403.sh (runs/resume_<run>: a cache replay of the played rounds, then the rest; the "
           "failed 20:17 attempt, HTTP 401, is runs/failed401_resume_* and is ignored). A finished game that ended early (all_done) carries its final values to "
           "later cuts. Means use only games with a value at each cut (n shown).", ""]
    cuts30 = (10, 20, 30)
    doc.append("## sweep30 (30 agents, 30 rounds, worlds 0/1/4)\n")
    doc.append(mean_table("Per-arm means", [(arm, list(R[run].values())) for arm, run in SWEEP30], cuts30))
    a1, b0 = by_world("sweep30_a1_facility", "nogod"), by_world("sweep30_b0_oss", "nogod")
    for arm, run in SWEEP30:
        base = a1 if run in ("sweep30_a6_k10", "sweep30_a7_k20") else b0 if run.startswith("sweep30_b") and run != "sweep30_b0_oss" else None
        doc.append(table(arm + (" (damage vs A1)" if base is a1 and base else " (damage vs B0)" if base else ""),
                         [(arm, k, v) for k, v in R[run].items()], cuts30, base))
    cuts40 = (20, 40)
    doc.append("## soft wording + God k1 (60 agents, rounds 1-40) vs the nogod soft runs\n")
    doc.append(mean_table("Means", [("soft_perm nogod", list(R["sweep60_or_soft_perm"].values())),
                                    ("soft_perm k1", list(R["sweep60_or_soft_perm_k1"].values())),
                                    ("soft_silent nogod", list(R["sweep60_or_soft_silent"].values())),
                                    ("soft_silent k1", list(R["sweep60_or_soft_silent_k1"].values()))], cuts40))
    for nm in ("perm", "silent"):
        doc.append(table(f"soft_{nm} nogod", [(f"{nm} nogod", k, v) for k, v in R[f"sweep60_or_soft_{nm}"].items()], cuts40))
        doc.append(table(f"soft_{nm} k1 (damage vs soft_{nm} nogod)",
                         [(f"{nm} k1", k, v) for k, v in R[f"sweep60_or_soft_{nm}_k1"].items()], cuts40,
                         by_world(f"sweep60_or_soft_{nm}", "nogod")))
    cuts80 = (20, 40, 60, 80)
    doc.append("## sweep60_or_glm_i2fix (won_render_fix true; 60 agents, 80 rounds)\n")
    doc.append(table("i2fix nogod", [("i2fix nogod", k, v) for k, v in R["sweep60_or_glm_i2fix"].items() if k.endswith("nogod")], cuts80))
    doc.append(table("i2fix k1 (damage vs i2fix nogod)",
                     [("i2fix k1", k, v) for k, v in R["sweep60_or_glm_i2fix"].items() if k.endswith("k1")], cuts80,
                     by_world("sweep60_or_glm_i2fix", "nogod")))
    doc.append(two_by_two(R, cuts80))
    doc.append("## Game Env + God k1 (sweep60_or_oldframe_k1) vs Realistic Env + God k1 (sweep60_or_glm k1)\n")
    doc.append("Factor = frame (agent/v1 + god/v2 + framing none vs agent/v2_facility + god/v2_facility + facility); "
               "everything else identical (runs/sweep60_or_oldframe_k1/NOTES.txt). God damage: Game Env vs C1 "
               "(sweep60_or_oldframe nogod), Realistic vs sweep60_or_glm nogod.\n")
    doc.append(table("Game Env + God k1 (damage vs C1 Game Env nogod)",
                     [("Game k1", k, v) for k, v in R["sweep60_or_oldframe_k1"].items()], cuts80,
                     by_world("sweep60_or_oldframe", "nogod")))
    doc.append(table("C1 Game Env nogod", [("Game nogod", k, v) for k, v in R["sweep60_or_oldframe"].items()], cuts80))
    doc.append(table("Realistic Env + God k1 (damage vs sweep60_or_glm nogod)",
                     [("Realistic k1", k, v) for k, v in R["sweep60_or_glm"].items() if k.endswith("k1")], cuts80,
                     by_world("sweep60_or_glm", "nogod")))
    doc.append(table("Realistic Env nogod", [("Realistic nogod", k, v) for k, v in R["sweep60_or_glm"].items()
                                             if k.endswith("nogod")], cuts80))
    a.out.write_text("\n".join(doc), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
