"""Markdown tables for plan/SWEEP_RESULTS.md from the sweep_stats JSON (real numbers only).

    python infra/sweep/results.py plan/figs/sweep_stats.json > /tmp/tables.md

One table per factor. Rows are conditions. Columns: n worlds; welfare levels (God, noGod, noise);
A-N (God - fork-matched noise; primary), A-B (God - noGod) and N-B (noise - noGod), each as mean,
t95 and sign-flip p; win-rate A-N; defects and board posts per agent A-N; divergence God and noise
(post-fork share of agents acting differently from noGod) and their gap. n < 8 is flagged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import FACTORS  # noqa: E402


def cell(s: dict | None, digits: int = 3) -> str:
    if not s or not s.get("n"):
        return "–"
    f = f"{{:+.{digits}f}}"
    return f"{f.format(s['mean'])} [{f.format(s['t_lo'])}, {f.format(s['t_hi'])}] p={s['p']:.2f}"


def lvl(s: dict | None) -> str:
    return "–" if not s or not s.get("n") else f"{s['mean']:.3f}"


def table(stats: dict, factor: str) -> str:
    xlabel, cells = FACTORS[factor]
    rows = [(lab, x) for lab, x in cells if lab in stats]
    if not any(lab != "base" for lab, _ in rows):
        return ""
    out = [f"### {xlabel}", "",
           "| cond | n | welfare God / noGod / noise | **A−N welfare** (t95, p) | A−B welfare | N−B welfare "
           "| A−N win rate | A−N defects/agent | A−N board posts/agent | div God / noise (gap) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, _ in rows:
        s = stats[lab]
        n = len(s.get("seeds", []))
        flag = " (low n)" if n < 8 else ""
        has_noise = s.get("noise") != "none"
        dv = (f"{s['div_god']['mean']:.2f} / {s['div_noise']['mean']:.2f} ({cell(s.get('div_gap'), 2)})"
              if has_noise and s.get("div_noise", {}).get("n") else
              (f"{s['div_god']['mean']:.2f} / – " if s.get("div_god", {}).get("n") else "–"))
        out.append(
            f"| {lab} | {n}{flag} | {lvl(s.get('welfare:god'))} / {lvl(s.get('welfare:nogod'))} / "
            f"{lvl(s.get('welfare:noise')) if has_noise else '–'} | "
            f"{cell(s.get('welfare:god-noise')) if has_noise else '–'} | {cell(s.get('welfare:god-nogod'))} | "
            f"{cell(s.get('welfare:noise-nogod')) if has_noise else '–'} | "
            f"{cell(s.get('win_rate:god-noise')) if has_noise else '–'} | "
            f"{cell(s.get('defects:god-noise')) if has_noise else cell(s.get('defects:god-nogod'))} | "
            f"{cell(s.get('board_posts:god-noise'), 2) if has_noise else cell(s.get('board_posts:god-nogod'), 2)} | "
            f"{dv} |")
    return "\n".join(out) + "\n"


def main() -> None:
    stats = json.loads(Path(sys.argv[1]).read_text())
    for f in FACTORS:
        t = table(stats, f)
        if t:
            print(t)


if __name__ == "__main__":
    main()
