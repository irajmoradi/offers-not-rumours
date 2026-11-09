"""Soft handoff wording x stance (Iraj 2026-11-08): arms P (permissive + soft), S (silent + soft) vs the matched
sweep60_or_glm nogod games (silent + v1 wording), worlds g00, g01, g04, rounds 1-40.
    python -I infra/sweep60/soft_stance_report.py [--out-dir plan/soft_stance]
Per game, at r20 and r40: contests (defect events) total and per agent, wrong contests, wins so far, welfare
(round_end, needs-only). Writes <out-dir>/soft_stance.csv and soft_stance.png (welfare and cumulative contests per
round by arm). Read-only on run dirs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = [("baseline (silent, v1 wording)", "sweep60_or_glm"), ("S: silent + soft", "sweep60_or_soft_silent"),
        ("P: permissive + soft", "sweep60_or_soft_perm")]
WORLDS = ["g00", "g01", "g04"]
CUTS = (20, 40)


def game(gdir: Path) -> dict:
    n_agents, welfare, burns, wins = 0, {}, [], {}
    for ln in open(gdir / "events.jsonl", encoding="utf-8", errors="ignore"):
        if '"game_start"' in ln:
            e = json.loads(ln)
            if e["data"]["type"] == "game_start":
                n_agents = len(e["data"]["agents"])
        elif '"round_end"' in ln:
            e = json.loads(ln)
            welfare[e["round"]] = float(e["data"]["welfare"])
        elif '"type":"burn"' in ln or '"type": "burn"' in ln:
            e = json.loads(ln)
            burns.append((e["round"], bool(e["data"]["correct"])))
        elif '"type":"win"' in ln or '"type": "win"' in ln:
            e = json.loads(ln)
            wins.setdefault(e["data"]["agent"], e["round"])
    return {"n": n_agents, "welfare": welfare, "burns": burns, "wins": wins}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "plan" / "soft_stance")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for label, run in ARMS:
        for w in WORLDS:
            g = ROOT / "runs" / run / "games" / f"{w}-nogod"
            if not (g / "events.jsonl").exists():
                continue
            d = game(g)
            last = max(d["welfare"]) if d["welfare"] else 0
            curves[(label, w)] = d
            for r in CUTS:
                if r > last:
                    continue
                b = [ok for rr, ok in d["burns"] if rr <= r]
                rows.append({"arm": label, "run": run, "world": w, "round": r, "agents": d["n"],
                             "contests": len(b), "contests_per_agent": round(len(b) / d["n"], 4),
                             "wrong_contests": sum(1 for ok in b if not ok),
                             "wins": sum(1 for wr in d["wins"].values() if wr <= r),
                             "welfare": round(d["welfare"][r], 4)})
    with open(a.out_dir / "soft_stance.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["arm"])
        wr.writeheader()
        wr.writerows(rows)
    agg = Counter()
    for r in rows:
        for k in ("contests", "wrong_contests", "wins"):
            agg[(r["arm"], r["round"], k)] += r[k]
        agg[(r["arm"], r["round"], "welfare_x1000")] += round(1000 * r["welfare"])
        agg[(r["arm"], r["round"], "games")] += 1
    for label, _ in ARMS:
        for c in CUTS:
            n = agg[(label, c, "games")]
            if n:
                print(f"{label:32s} r{c}: games {n}, contests {agg[(label, c, 'contests')]} "
                      f"({agg[(label, c, 'contests')] / (60 * n):.3f}/agent), wrong {agg[(label, c, 'wrong_contests')]}, "
                      f"wins {agg[(label, c, 'wins')]} ({agg[(label, c, 'wins')] / n:.1f}/game), "
                      f"welfare {agg[(label, c, 'welfare_x1000')] / 1000 / n:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: CSV only")
        return 0
    colors = {ARMS[0][0]: "#4c72b0", ARMS[1][0]: "#55a868", ARMS[2][0]: "#c44e52"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (label, w), d in curves.items():
        rs = [r for r in sorted(d["welfare"]) if r <= 40]
        ax[0].plot(rs, [d["welfare"][r] for r in rs], color=colors[label], alpha=0.8,
                   label=label if w == WORLDS[0] else None)
        cum = [sum(1 for rr, _ in d["burns"] if rr <= r) for r in rs]
        ax[1].plot(rs, cum, color=colors[label], alpha=0.8, label=label if w == WORLDS[0] else None)
    ax[0].set(title="Welfare by round (3 worlds each)", xlabel="round", ylabel="welfare (needs only)")
    ax[1].set(title="Cumulative contests by round", xlabel="round", ylabel="contests")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out_dir / "soft_stance.png", dpi=130)
    print("wrote", a.out_dir / "soft_stance.csv", a.out_dir / "soft_stance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
