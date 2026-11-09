"""Run-level checks: all games of a run, god/nogod fork prefixes, control_god cache replay (--base)."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from .common import Violation, _jsonl, load_cfg
from .game import GameChecker


# ---------------------------------------------------------------------- run level
def _norm_event(e: dict[str, Any]) -> str:
    e = {k: v for k, v in e.items() if k != "seq"}
    d = dict(e["data"])
    if d.get("type") == "round_end":
        d.pop("state_hash", None)      # the God branch carries god_budget_left in its state
    if d.get("type") == "game_start":
        d.pop("config_hash", None)
    e["data"] = d
    return json.dumps(e, sort_keys=True)


def fork_checks(run_dir: Path, games: dict[str, Path]) -> list[Violation]:
    out: list[Violation] = []
    pairs: dict[str, dict[str, Path]] = collections.defaultdict(dict)
    for label, p in games.items():
        idx, _, br = label.partition("-")
        pairs[idx][br] = p
    for idx, br in sorted(pairs.items()):
        if "god" not in br or "nogod" not in br:
            continue
        ea = list(_jsonl(br["god"] / "events.jsonl"))
        eb = list(_jsonl(br["nogod"] / "events.jsonl"))
        k = next((i for i, e in enumerate(ea) if e["actor"] == "GOD"), len(ea))
        fork_round = ea[k]["round"] if k < len(ea) else None
        for i in range(min(k, len(eb))):
            if _norm_event(ea[i]) != _norm_event(eb[i]):
                out.append(Violation(f"{idx}-god/nogod", "fork",
                                     f"pre-fork event {i} differs ({ea[i]['type']} r{ea[i]['round']} vs "
                                     f"{eb[i]['type']} r{eb[i]['round']})"))
                break
        if fork_round is None:
            continue
        ca = {c["request"]["call_id"]: c for c in _jsonl(br["god"] / "llm_calls.jsonl")
              if (c["request"]["meta"].get("round") or 0) < fork_round}
        cb = {c["request"]["call_id"]: c for c in _jsonl(br["nogod"] / "llm_calls.jsonl")
              if (c["request"]["meta"].get("round") or 0) < fork_round}
        if set(ca) != set(cb):
            out.append(Violation(f"{idx}-god/nogod", "fork",
                                 f"pre-fork call ids differ: {len(set(ca) ^ set(cb))} not in both"))
        for cid in sorted(set(ca) & set(cb)):
            x, y = ca[cid], cb[cid]
            if x["request"]["messages"] != y["request"]["messages"]:
                out.append(Violation(f"{idx}-god/nogod", "fork", f"pre-fork prompt of {cid} differs"))
                break
            tx = (x.get("response") or {}).get("text")
            ty = (y.get("response") or {}).get("text")
            if tx != ty:
                out.append(Violation(f"{idx}-god/nogod", "fork",
                                     f"pre-fork RESPONSE of {cid} (r{x['request']['meta'].get('round')}) differs: "
                                     "common random numbers broken"))
                break
    return out


def check_run(run_dir: Path, *, prompts: bool = True, only: str | None = None) -> list[Violation]:
    cfg = load_cfg(run_dir)
    games = {p.name: p for p in sorted((run_dir / "games").iterdir()) if p.is_dir()}
    if only:
        games = {k: v for k, v in games.items() if k == only or k.startswith(only + "-")}
    out: list[Violation] = []
    for label, gdir in games.items():
        gc = GameChecker(gdir, cfg, prompts=prompts)
        out += gc.run()
        if prompts:
            iids = [e["data"]["intervention_id"] for e in gc.events
                    if e["type"] == "god_power" and e["data"].get("intervention_id")]
            gc.check_prompts(iids)
            out += [v for v in gc.out if v.check == "prompts"]
        for chk, c in gc._count.items():
            if c > gc.max_per_check:
                out.append(Violation(label, chk, f"... {c - gc.max_per_check} more not listed"))
    out += fork_checks(run_dir, games)
    return out


def _uncacheable_round(calls_path: Path) -> int | None:
    """First round with a call the LLM cache cannot replay (error, or empty response text: never cached)."""
    rounds = [c["request"]["meta"].get("round") or 0 for c in _jsonl(calls_path)
              if c.get("error") or not ((c.get("response") or {}).get("text") or "").strip()]
    return min(rounds) if rounds else None


def _norm_cross(e: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in e.items() if k not in ("seq", "run_id")}, sort_keys=True)


def control_replay_checks(run_dir: Path, base_dir: Path) -> tuple[list[Violation], list[str]]:
    """runner.fork.control_god cells: each control (nogod) game must equal the base run's nogod game of
    the same game_idx (same game_id, events, prompts, responses) up to the base game's first
    uncacheable call; from that round on it may re-sample. -> (violations, info lines)."""
    out: list[Violation] = []
    info: list[str] = []
    for g in sorted((run_dir / "games").glob("*-nogod")):
        b = base_dir / "games" / g.name
        label = f"{g.name} vs {base_dir.name}"
        if not b.is_dir():
            out.append(Violation(label, "replay", "no base game with this label"))
            continue
        ea, eb = list(_jsonl(g / "events.jsonl")), list(_jsonl(b / "events.jsonl"))
        if not ea or not eb or ea[0]["game_id"] != eb[0]["game_id"]:
            out.append(Violation(label, "replay", "game_id differs (control_god not matching the base)"))
            continue
        brk = _uncacheable_round(b / "llm_calls.jsonl")
        div = next((i for i, (x, y) in enumerate(zip(ea, eb, strict=False)) if _norm_cross(x) != _norm_cross(y)), None)
        finished = bool(ea) and ea[-1]["type"] == "game_end"
        if div is None and len(ea) != len(eb) and (finished or len(ea) > len(eb)):
            div = min(len(ea), len(eb))                   # (a still-running control is just shorter)
        div_round = None if div is None else (ea[div]["round"] if div < len(ea) else eb[div]["round"])
        ca = {c["request"]["call_id"]: c for c in _jsonl(g / "llm_calls.jsonl")}
        cb = {c["request"]["call_id"]: c for c in _jsonl(b / "llm_calls.jsonl")}
        bad_calls = sorted(((ca[k]["request"]["meta"].get("round") or 0), k) for k in set(ca) & set(cb)
                           if ca[k]["request"]["messages"] != cb[k]["request"]["messages"]
                           or (ca[k].get("response") or {}).get("text") != (cb[k].get("response") or {}).get("text"))
        call_round = bad_calls[0][0] if bad_calls else None
        found = [x for x in (div_round, call_round) if x is not None]
        first = min(found) if found else None
        info.append(f"{label}: base break r{brk}, control diverges at "
                    f"{'never' if first is None else f'r{first}'}")
        if first is not None and (brk is None or first < brk):
            out.append(Violation(label, "replay", f"control diverges from the base at r{first} "
                                 f"(base break {'none' if brk is None else f'r{brk}'}): event {div}, "
                                 f"call {bad_calls[0][1] if bad_calls else '-'}"))
    return out, info
