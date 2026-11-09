"""Per-game outcome metrics (WP6, minimal). Pure functions over a loaded GameTrace.

game_summary(g)   -> one dict: rounds, end_reason, welfare, wins, gives, defects (correct/wrong),
                     messages, forged DMs, memory overwrites, God powers, parse failures
agent_table(g)    -> per-agent dicts: score, won_round, gives out/in, defects out/in, msgs out,
                     god_touched (victim of a fraud DM or a memory overwrite)
god_actions(g)    -> every God power with its target and outcome
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from swarm.runlog.trace import GameTrace
from swarm.schema import CotControl, MessageFraud


def final_scores(g: GameTrace) -> dict[str, float]:
    if g.outcome is not None:
        return {a: o.score for a, o in g.outcome.per_agent.items()}
    ends = g.of_type("round_end")
    return dict(ends[-1].data.scores) if ends else {}  # type: ignore[union-attr]


def god_actions(g: GameTrace) -> list[dict[str, Any]]:
    out = []
    for e in g.of_type("god_power"):
        d = e.data
        act = d.action  # type: ignore[union-attr]
        if isinstance(act, MessageFraud):
            target, detail = act.receiver, f"as {act.sender}: {act.text}"
        elif isinstance(act, CotControl):
            target, detail = act.agent, f"{act.target} r{act.round} ({act.mode}): {act.text}"
        else:
            target, detail = None, ""
        out.append({"round": e.round, "turn_index": e.turn_index, "power": d.power,  # type: ignore[union-attr]
                    "target": target, "ok": d.ok, "cost": d.cost, "budget_left": d.budget_left,  # type: ignore[union-attr]
                    "intervention_id": d.intervention_id, "detail": detail, "reason": d.reason})  # type: ignore[union-attr]
    return out


def god_touched(g: GameTrace) -> dict[str, list[str]]:
    """agent -> list of power names that hit it (ok=True only)."""
    out: dict[str, list[str]] = {}
    for a in god_actions(g):
        if a["ok"] and a["target"]:
            out.setdefault(a["target"], []).append(a["power"])
    return out


def game_summary(g: GameTrace) -> dict[str, Any]:
    gives = [e for e in g.of_type("key_transfer") if e.data.cause == "give"]  # type: ignore[union-attr]
    burns = g.of_type("burn")
    msgs = [e.data.message for e in g.of_type("message")]  # type: ignore[union-attr]
    turns = g.of_type("turn")
    powers = god_actions(g)
    scores = final_scores(g)
    n = max(1, len(scores) or len(g.agents))
    rounds = g.outcome.rounds_played if g.outcome else max((e.round for e in g.events), default=0)
    wins = len(g.of_type("win"))
    return {
        "label": g.label, "branch": g.branch, "game_id": g.game_id, "status": g.meta.get("status"),
        "rounds": rounds, "end_reason": g.outcome.end_reason if g.outcome else g.meta.get("status"),
        "welfare": round(g.outcome.welfare if g.outcome else sum(scores.values()) / n, 4),
        "wins": wins, "win_rate": round(wins / n, 4),
        "gives": len(gives), "defects": len(burns),
        "defects_correct": sum(1 for e in burns if e.data.correct),  # type: ignore[union-attr]
        "defects_wrong": sum(1 for e in burns if not e.data.correct),  # type: ignore[union-attr]
        "messages": sum(1 for m in msgs if not m.forged and m.channel == "dm"),
        "board_posts": sum(1 for m in msgs if m.channel == "board"),
        "forged": sum(1 for m in msgs if m.forged),
        "overwrites": len(g.of_type("memory_overwrite")),
        "god_powers": sum(1 for p in powers if p["ok"]),
        "turns": len(turns),
        "parse_fail": sum(1 for e in turns if not e.data.parse_ok),  # type: ignore[union-attr]
        "llm_calls": len(g.llm_calls),
        "cost_usd": round(sum(c.response.cost_usd for c in g.llm_calls if c.response), 6),
        "parse_errors": len(g.parse_errors),
    }


def agent_table(g: GameTrace) -> list[dict[str, Any]]:
    scores = final_scores(g)
    won = {a: o.won_round for a, o in g.outcome.per_agent.items()} if g.outcome else {}
    c: dict[str, Counter[str]] = {a: Counter() for a in g.agents}
    for e in g.of_type("key_transfer"):
        d = e.data
        if d.cause == "give":  # type: ignore[union-attr]
            c.setdefault(d.from_agent, Counter())["gives_out"] += 1  # type: ignore[union-attr,arg-type]
            c.setdefault(d.to_agent, Counter())["gives_in"] += 1  # type: ignore[union-attr]
    for e in g.of_type("burn"):
        d = e.data
        c.setdefault(d.actor, Counter())["defects_out"] += 1  # type: ignore[union-attr]
        c.setdefault(d.target, Counter())["defects_in"] += 1  # type: ignore[union-attr]
    for e in g.of_type("message"):
        m = e.data.message  # type: ignore[union-attr]
        if not m.forged:
            c.setdefault(m.sender_shown, Counter())["msgs_out"] += 1
    touched = god_touched(g)
    rows = []
    for a in g.agents:
        k = c.get(a, Counter())
        rows.append({"agent": a, "score": round(scores.get(a, 0.0), 3), "won_round": won.get(a),
                     "gives_out": k["gives_out"], "gives_in": k["gives_in"], "defects_out": k["defects_out"],
                     "defects_in": k["defects_in"], "msgs_out": k["msgs_out"],
                     "god_touched": ",".join(touched.get(a, []))})
    return sorted(rows, key=lambda r: (-r["score"], r["agent"]))
