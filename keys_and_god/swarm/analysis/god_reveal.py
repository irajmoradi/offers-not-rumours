"""God reveal_needs effects (game-fixes; no LLM calls). For every ok reveal_needs power in a God
branch, with the paired no-God branch (same game_idx) as the control at the same rounds and agents:

  defects_on_target      defects on the revealed target in rounds [t, t+window] (by receivers / others)
  correct_on_target      ... of them correct (the target really needed the id)
  recv_defects           defects by the receivers (any target) in the window; recv_wrong = wrong ones
  recv_on_items          receivers' defects that name one of the revealed items
  target_burned          needs the target lost to burns in the window
  recv_needs_lost        needs the receivers lost (lose_need / lose_held_need) in the window
  recv_penalty           score points the receivers paid (score_penalty) and keys they lost
                         (lose_held_key) in the window
  final score delta      target / receivers, God minus no-God (Outcome per_agent score)

    PYTHONNOUSERSITE=1 python -m swarm.analysis.god_reveal runs/<run_id> [--window 3] [--json out.json]

Defect accuracy over all defects is also reported per branch. swarm/analysis/social.py has the
message-text (rumour / accusation) side of God's powers; this module is the defect side.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def load(gdir: Path) -> dict[str, Any]:
    """One game's reveals, burns, penalties, transfers and final scores."""
    g: dict[str, Any] = {"reveals": [], "burns": [], "pen_keys": [], "final": {}, "needs": {}}
    for ln in (gdir / "events.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        t, d, r = e["type"], e["data"], e["round"]
        if t == "game_start":
            g["needs"] = {a: s["needs"] for a, s in d["agents"].items()}
            g["n_needs"] = {a: len(s["needs"]) for a, s in d["agents"].items()}
        elif t == "god_power" and d["ok"] and d["power"] == "reveal_needs":
            a = d["action"]
            g["reveals"].append({"round": r, "target": a["target"], "receivers": list(a["receivers"]),
                                 "items": list(a["items"]), "variant": a["variant"], "sender": a["sender"]})
        elif t == "burn":
            g["burns"].append({"round": r, "actor": d["actor"], "target": d["target"], "info": d["info_id"],
                               "correct": d["correct"], "penalty": d["penalty"], "lost": d.get("lost_info_id")})
        elif t == "key_transfer" and d["cause"] == "defect_penalty":
            g["pen_keys"].append({"round": r, "from": d["from_agent"], "to": d["to_agent"], "info": d["info_id"]})
        elif t == "game_end":
            g["final"] = {a: o["score"] for a, o in d["outcome"]["per_agent"].items()}
    return g


def _penalty_points(g: dict[str, Any], b: dict[str, Any], score_penalty: float | None) -> float:
    if b["penalty"] != "score_penalty":
        return 0.0
    return score_penalty if score_penalty is not None else 1.0 / max(g["n_needs"].get(b["actor"], 1), 1)


def effects(g: dict[str, Any], rv: dict[str, Any], window: int, score_penalty: float | None) -> dict[str, float]:
    """The counters of one reveal in one branch (g may be the control: same rounds and agents)."""
    t0, t1, tgt, rec = rv["round"], rv["round"] + window, rv["target"], set(rv["receivers"])
    win = [b for b in g["burns"] if t0 <= b["round"] <= t1]
    on_t = [b for b in win if b["target"] == tgt]
    by_r = [b for b in win if b["actor"] in rec]
    return {
        "defects_on_target": len(on_t),
        "defects_on_target_by_receivers": sum(b["actor"] in rec for b in on_t),
        "correct_on_target": sum(b["correct"] for b in on_t),
        "recv_defects": len(by_r),
        "recv_wrong": sum(not b["correct"] for b in by_r),
        "recv_on_items": sum(b["target"] == tgt and b["info"] in rv["items"] for b in by_r),
        "target_burned": sum(b["correct"] for b in on_t),
        "recv_needs_lost": sum(b["lost"] is not None for b in by_r),
        "recv_score_penalty": round(sum(_penalty_points(g, b, score_penalty) for b in by_r), 6),
        "recv_keys_lost": sum(1 for k in g["pen_keys"] if t0 <= k["round"] <= t1 and k["from"] in rec),
    }


def analyse(run: Path, window: int = 3) -> dict[str, Any]:
    games = collections.defaultdict(dict)
    for gdir in sorted((run / "games").iterdir()):
        if (gdir / "events.jsonl").exists() and "-" in gdir.name:
            idx, branch = gdir.name.split("-", 1)
            games[idx][branch] = load(gdir)
    sp = None
    resolved = run / "config.resolved.yaml"
    if resolved.exists():
        import yaml
        sp = ((yaml.safe_load(resolved.read_text()).get("game") or {}).get("defect") or {}).get("score_penalty")
    rows, tot = [], collections.defaultdict(lambda: collections.Counter())
    for idx, br in sorted(games.items()):
        god, ctrl = br.get("god"), br.get("nogod")
        if god is None:
            continue
        for rv in god["reveals"]:
            row: dict[str, Any] = {"game": idx, **rv, "god": effects(god, rv, window, sp)}
            if ctrl is not None:
                row["nogod"] = effects(ctrl, rv, window, sp)
                for who in ("target", "receivers"):
                    agents = [rv["target"]] if who == "target" else rv["receivers"]
                    row[f"d_final_{who}"] = round(sum(god["final"].get(a, 0) - ctrl["final"].get(a, 0)
                                                      for a in agents) / max(len(agents), 1), 6)
            rows.append(row)
            for lab in ("god", "nogod"):
                if lab in row:
                    tot[(rv["variant"], lab)].update(row[lab])
                    tot[(rv["variant"], lab)]["reveals"] += 1
    acc = {}
    for lab in ("god", "nogod", "noise"):
        bs = [b for br in games.values() if lab in br for b in br[lab]["burns"]]
        acc[lab] = {"defects": len(bs), "correct": sum(b["correct"] for b in bs),
                    "accuracy": round(sum(b["correct"] for b in bs) / len(bs), 4) if bs else None}
    return {"run": str(run), "window": window, "n_reveals": len(rows),
            "totals": {f"{v}/{lab}": dict(c) for (v, lab), c in sorted(tot.items())},
            "defect_accuracy": acc, "reveals": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    out = analyse(a.run, a.window)
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(json.dumps({k: v for k, v in out.items() if k != "reveals"}, indent=1, default=str))


if __name__ == "__main__":
    main()
