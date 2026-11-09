"""Narrative helpers for the demo video (no drawing here, so they are unit-testable). Numbers come from
swarm.viz.metrics (load_branch, divergence, ... are re-exported here for older imports).

    captions(god, nogod, div)        -> {round: [(kind, text), ...]} running caption lines
    pacing(R, captions)              -> [(round, seconds)]: 1 s per round, longer on God events
    world_table(run_dir)             -> per-world deltas (god - nogod, noise - nogod) and ripple speed
    pick_worlds(rows)                -> the data-driven demo picks ("harm" and "ripple")
Only finished branches (meta.json status "done") are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm.viz.metrics import (  # noqa: F401  (re-exported: extraction lives in the shared metrics module)
    KINDS,
    Branch,
    Power,
    divergence,
    edge_counts,
    is_done,
    load_branch,
    welfare,
)


def _short(text: str, n: int = 78) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def captions(god: Branch, nogod: Branch, ever: dict[int, set[str]] | None = None) -> dict[int, list[tuple[str, str]]]:
    """Per-round caption lines (kind, text); kind in god | quote | effect | ripple | win."""
    out: dict[int, list[tuple[str, str]]] = {r: [] for r in range(1, god.R + 1)}
    for p in god.powers:
        if p.power == "message_fraud":
            out[p.round].append(("god", f"GOD forges a DM to {p.target}, signed as {p.spoofed}:"))
            out[p.round].append(("quote", f"“{_short(p.text, 110)}”"))
            if p.target in god.won_by(p.round - 1):
                out[p.round].append(("effect", f"{p.target} had already won before the lie arrived"))
            base = {(r, a, t) for r, a, t, _ in nogod.gives}
            took = [g for g in god.gives if g[1] == p.target and g[2] == p.spoofed and p.round <= g[0] <= p.round + 1
                    and (g[0], g[1], g[2]) not in base]
            for r, _, to, info in took[:2]:
                out[r].append(("effect", f"{p.target} takes the bait: gives its key for info {info} to {to}"))
            if not took:
                out[min(p.round + 1, god.R)].append(("effect", f"{p.target} does not hand {p.spoofed} a key"))
        else:
            out[p.round].append(("god", f"GOD rewrites {p.target}'s private notes from round {p.round - 1} to:"))
            out[p.round].append(("quote", f"“{_short(p.text, 110)}”"))
            named = [a for a in god.agents if a in p.text and a != p.target]
            for x in named[:1]:
                hit = next((("defect", r) for r, a, t, _ in god.defects if a == p.target and t == x
                            and p.round <= r <= p.round + 2), None) or \
                      next((("give", r) for r, a, t, _ in god.gives if a == p.target and t == x
                            and p.round <= r <= p.round + 2), None)
                if hit and hit[0] == "defect":
                    out[hit[1]].append(("effect", f"{p.target} acts on the planted note: defects against {x}"))
                elif hit:
                    out[hit[1]].append(("effect", f"{p.target} ignores the planted note: still gives {x} a key"))
    prev = 0
    for r in range(1, god.R + 1):
        n = len((ever or {}).get(r, ()))
        if ever is not None and n != prev and god.powers and r >= god.powers[0].round:
            out[r].append(("ripple", f"Divergence: {n}/{len(god.agents)} agents now act differently than in the "
                                     "no-God world"))
        prev = n
        wg = god.wins.get(r, []) if r <= god.rounds_played else []
        wn = nogod.wins.get(r, []) if r <= nogod.rounds_played else []
        if wg or wn:
            out[r].append(("win", f"Wins: with God {len(wg)}, without God {len(wn)}"))
    return out


def pacing(R: int, caps: dict[int, list[tuple[str, str]]], base: float = 1.0, god_hold: float = 2.5,
           effect_hold: float = 1.5) -> list[tuple[int, float]]:
    """Seconds on screen per round: base, plus a hold on rounds with a God event or its visible effect."""
    out = []
    for r in range(1, R + 1):
        kinds = {k for k, _ in caps.get(r, [])}
        out.append((r, base + (god_hold if "god" in kinds else effect_hold if "effect" in kinds else 0.0)))
    return out


def world_table(run_dir: str | Path, with_ripple: bool = True) -> list[dict[str, Any]]:
    """One row per world with finished god+nogod: welfare per branch, deltas, victim deltas, ripple speed."""
    run_dir = Path(run_dir)
    by: dict[int, dict[str, Path]] = {}
    for d in sorted((run_dir / "games").iterdir()):
        if not is_done(d):
            continue
        idx, _, br = d.name.partition("-")
        by.setdefault(int(idx[1:]), {})[br] = d
    rows = []
    for i, brs in sorted(by.items()):
        if "god" not in brs or "nogod" not in brs:
            continue
        wg, wn = welfare(brs["god"]), welfare(brs["nogod"])
        row: dict[str, Any] = {"world": i, "god": wg, "nogod": wn, "delta": wg - wn,
                               "noise_delta": (welfare(brs["noise"]) - wn) if "noise" in brs else None}
        if with_ripple:
            g, n = load_branch(brs["god"]), load_branch(brs["nogod"])
            _, ever = divergence(brs["god"], brs["nogod"])
            fork = min((p.round for p in g.powers), default=None)
            n_ag = len(g.agents)
            full = next((r for r in sorted(ever) if len(ever[r]) >= 0.8 * n_ag), None)
            row["fork"] = fork
            row["rounds_to_80pct"] = (full - fork) if (full and fork) else None
            # agents diverged so far at rounds fork .. fork+3, and its mean share (lower = slower ripple)
            row["spread"] = [len(ever.get(r, ())) for r in range(fork, fork + 4)] if fork else None
            row["spread_auc"] = (sum(row["spread"]) / (4 * n_ag)) if fork else None
            sg, sn = g.score_at(g.R), n.score_at(n.R)
            row["victim_delta"] = {p.target: sg.get(p.target, 0) - sn.get(p.target, 0) for p in g.powers}
        rows.append(row)
    return rows


def pick_worlds(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """harm: largest God-caused welfare change beyond the world's own noise replica,
    |delta| - |noise_delta| (|delta| when no replica); ripple: slowest spread, i.e. the lowest mean share of
    agents diverged over the fork round and the 3 rounds after it (ties: harm, then world index)."""
    def excess(r: dict[str, Any]) -> float:
        return abs(r["delta"]) - abs(r["noise_delta"] or 0.0)

    out = {}
    if rows:
        out["harm"] = max(rows, key=lambda r: (excess(r), -r["world"]))
        rip = [r for r in rows if r.get("spread_auc") is not None and r is not out["harm"]]
        if rip:
            out["ripple"] = min(rip, key=lambda r: (r["spread_auc"], -excess(r), r["world"]))
    return out


def median_world(abs_delta: dict[int, float]) -> tuple[int, str]:
    """The paper's Figure 2 rule: the world whose |delta welfare| is closest to the median |delta| (ties: lowest index)."""
    import statistics

    med = statistics.median(abs_delta.values())
    w = min(abs_delta, key=lambda i: (abs(abs_delta[i] - med), i))
    return w, f"|Δwelfare| {abs_delta[w]:.3f}, median {med:.3f}, n={len(abs_delta)}"
