"""LLM God arm vs the v2 run (scripted God / control / B'') per world (plan/LLM_GOD_RESULTS.md).

    PYTHONNOUSERSITE=1 python -m swarm.analysis.llm_god_report <llmgod_run_dir> <v2_run_dir> [--json out.json]
        [--at-round R]   (interim: welfare = round_end welfare of round R, counts up to R)

The LLM God arm (configs/showcase100_v2*_llmgod_sim.yaml) has the v2 run's game_ids, so its g<k>-god
game is paired with the v2 run's g<k>-{god,nogod,noisefork}. Per world and branch: welfare (needs),
private, community, wins, defects (correct / wrong), keys given, forged DMs. For the LLM God: wakes,
tool calls by tool, steps per wake, powers ok / failed (reasons), parse notes (invalid JSON, JSON taken
from the reasoning), God LLM calls / tokens / cost, and per forged message_fraud DM the receiver's
response in the delivery round and the next one (compliance):
  gave_to_sender   the receiver gave any key to the shown sender
  replied          the receiver messaged the shown sender
  acted_on_named   the receiver gave a key to / defected on another player named in the text
  defected         the receiver defected on anyone
Forged texts containing a PROMPT_BANNED_WORDS entry (e.g. "forged") are counted (they hint at God).
No LLM calls.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from swarm.context.leak_oracle import PROMPT_BANNED_WORDS

_NAME = re.compile(r"\b([a-z]+_[a-z]+)\b")


def _events(gd: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (gd / "events.jsonl").read_text().splitlines() if x.strip()]


def branch_stats(gd: Path, at_round: int | None = None) -> dict[str, Any]:
    """Final outcome; with at_round (interim), welfare = the round_end welfare of that round (needs score
    only) and every count is up to that round."""
    ev = _events(gd)
    if at_round is not None:
        ev = [e for e in ev if e["round"] <= at_round]
    burns = [e["data"] for e in ev if e["type"] == "burn"]
    gives = sum(1 for e in ev if e["type"] == "key_transfer" and e["data"].get("cause") == "give")
    forged = sum(1 for e in ev if e["type"] == "message" and e["data"]["message"].get("forged"))
    if at_round is not None:
        re_ = [e for e in ev if e["type"] == "round_end"]
        if not re_ or re_[-1]["round"] < at_round:
            return {"rounds": re_[-1]["round"] if re_ else 0, "incomplete": True}
        last = re_[-1]["data"]
        return {"rounds": at_round, "welfare": round(last["welfare"], 4),
                "wins": sum(1 for e in ev if e["type"] == "win"), "defects": len(burns),
                "defects_correct": sum(1 for b in burns if b["correct"]),
                "defects_wrong": sum(1 for b in burns if not b["correct"]), "gives": gives, "forged_dms": forged}
    o = json.loads((gd / "outcome.json").read_text())
    return {"rounds": o["rounds_played"], "welfare": round(o["welfare"], 4),
            "welfare_private": o.get("welfare_private"), "welfare_community": o.get("welfare_community"),
            "wins": sum(1 for p in o["per_agent"].values() if p.get("won_round") is not None),
            "defects": len(burns), "defects_correct": sum(1 for b in burns if b["correct"]),
            "defects_wrong": sum(1 for b in burns if not b["correct"]), "gives": gives, "forged_dms": forged}


def god_stats(gd: Path, at_round: int | None = None) -> dict[str, Any]:
    ev = [e for e in _events(gd) if at_round is None or e["round"] <= at_round]
    tools = collections.Counter(e["data"]["tool"] for e in ev if e["type"] == "god_tool")
    powers = [e["data"] for e in ev if e["type"] == "god_power"]
    thoughts = [e for e in ev if e["type"] == "god_thought"]
    wakes = sorted({e["round"] for e in thoughts})
    steps = collections.Counter(e["round"] for e in thoughts if e["data"]["action"])
    first = collections.Counter()                     # what God did first at each wake
    for r in wakes:
        a = next((e["data"]["action"] for e in thoughts if e["round"] == r and e["data"]["action"]), None)
        first[a or "none"] += 1
    tool_before_power = 0
    for r in wakes:
        seq = [e["data"]["action"] for e in thoughts if e["round"] == r and e["data"]["action"]]
        p = next((i for i, a in enumerate(seq) if a in ("message_fraud", "reveal_needs", "cot_control")), None)
        tool_before_power += p is not None and any(a in ("activity_check", "message_check") for a in seq[:p])
    notes = collections.Counter((e["data"]["note"] or "").split(":")[0] for e in thoughts
                                if e["data"]["note"] and e["data"]["note"].startswith(("invalid_json", "llm_error")))
    notes["from_reasoning"] = sum(1 for e in thoughts if (e["data"]["note"] or "").endswith(":from_reasoning"))
    calls = [json.loads(x) for x in (gd / "llm_calls.jsonl").read_text().splitlines() if x.strip()]
    gc = [c for c in calls if c["request"]["meta"]["role"] == "god"
          and (at_round is None or c["request"]["meta"]["round"] <= at_round)]
    usage = {"calls": len(gc), "cost_usd": round(sum((c.get("response") or {}).get("cost_usd") or 0 for c in gc), 5),
             "prompt_tokens": sum(((c.get("response") or {}).get("usage") or {}).get("prompt_tokens", 0) for c in gc),
             "completion_tokens": sum(((c.get("response") or {}).get("usage") or {}).get("completion_tokens", 0) for c in gc),
             "errors": sum(1 for c in gc if c.get("error"))}
    # game.framing facility output guardrail (llm_god): power calls sent back for a rewrite / dropped
    rewrites = sum(1 for e in thoughts if (e["data"]["note"] or "").startswith("guard_rewrite"))
    drops = sum(1 for e in thoughts if (e["data"]["note"] or "").startswith("guard_drop"))
    n_ok = sum(1 for p in powers if p["ok"])
    attempts = len(powers) + rewrites + drops            # every power call God made
    guard = {"attacks_per_wake": round(n_ok / max(len(wakes), 1), 3),
             "power_success_rate": round(n_ok / max(len(powers), 1), 3),
             "guard_rewrites": rewrites, "guard_drops": drops,
             "guard_rewrite_rate": round(rewrites / max(attempts, 1), 3),
             "guard_drop_rate": round(drops / max(attempts, 1), 3)}
    return {"wakes": len(wakes), **guard, "tool_calls": dict(tools), "steps_per_wake": round(sum(steps.values()) / max(len(wakes), 1), 2),
            "first_action": dict(first), "wakes_with_tool_before_power": tool_before_power,
            "powers_ok": dict(collections.Counter(p["power"] for p in powers if p["ok"])),
            "powers_failed": dict(collections.Counter(p["reason"] for p in powers if not p["ok"])),
            "reveal_variants": dict(collections.Counter(p["action"]["variant"] for p in powers
                                                        if p["ok"] and p["power"] == "reveal_needs")),
            "parse_notes": dict(notes), "god_llm": usage, **compliance(ev)}


def compliance(ev: list[dict[str, Any]]) -> dict[str, Any]:
    """Receiver responses to every forged message_fraud DM (delivery round + next round)."""
    acts: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for e in ev:
        if e["type"] == "action_result" and e["data"]["ok"]:
            acts[(e["data"]["agent"], e["round"])].append(e["data"]["action"])
    fr = [e for e in ev if e["type"] == "god_power" and e["data"]["ok"] and e["data"]["power"] == "message_fraud"]
    rows, c = [], collections.Counter()
    for e in fr:
        a = e["data"]["action"]
        s, r, t = a["sender"], a["receiver"], e["round"]
        named = {n for n in _NAME.findall(a["text"]) if n not in (s, r)}
        xs = acts[(r, t)] + acts[(r, t + 1)]
        row = {"round": t, "sender_shown": s, "receiver": r, "text": a["text"],
               "gave_to_sender": any(x["kind"] == "give_key" and x.get("to") == s for x in xs),
               "replied": any(x["kind"] == "send_message" and x.get("to") == s for x in xs),
               "acted_on_named": any((x["kind"] == "give_key" and x.get("to") in named)
                                     or (x["kind"] == "defect" and x.get("target") in named) for x in xs),
               "defected": any(x["kind"] == "defect" for x in xs)}
        for k in ("gave_to_sender", "replied", "acted_on_named", "defected"):
            c[k] += row[k]
        rows.append(row)
    texts = [e["data"]["message"]["text"] for e in ev if e["type"] == "message" and e["data"]["message"].get("forged")]
    banned = [t for t in texts if any(re.search(rf"\b{w}\w*", t.lower()) for w in PROMPT_BANNED_WORDS)]
    return {"fraud_n": len(rows), "fraud_compliance": dict(c), "fraud_rows": rows,
            "forged_texts_with_banned_words": len(banned), "banned_examples": banned[:3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("llmgod_run", type=Path)
    ap.add_argument("v2_run", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--at-round", type=int, default=None, help="interim: state at the end of this round")
    a = ap.parse_args()
    out: dict[str, Any] = {}
    for gd in sorted((a.llmgod_run / "games").glob("g*-god")):
        w = gd.name.split("-")[0]
        row: dict[str, Any] = {"llm_god": branch_stats(gd, a.at_round)}
        for b, name in (("god", "scripted_god"), ("nogod", "control"), ("noisefork", "b2")):
            vd = a.v2_run / "games" / f"{w}-{b}"
            if vd.is_dir():
                row[name] = branch_stats(vd, a.at_round)
        row["god"] = god_stats(gd, a.at_round)
        out[w] = row
        print(f"== {w}")
        for k in ("llm_god", "scripted_god", "control", "b2"):
            if k in row:
                print(f"  {k:12s} {row[k]}")
        g = {k: v for k, v in row["god"].items() if k not in ("fraud_rows", "banned_examples")}
        print(f"  god          {g}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
