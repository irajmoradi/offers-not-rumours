"""Tolerant parser: raw model text -> ParsedReply(thought, note, actions, ok, errors). Pure.

Output contract (agent prompt sets):
    {"thought": str?, "note": str?, "actions": [{"kind": ..., ...}, ...]}

Steps:
  1. strip <think>...</think> blocks (and an unclosed leading <think>) and ``` fences;
  2. take the first balanced {...} (string-aware) that decodes as JSON, with light repairs
     (trailing commas, Python True/False/None, smart quotes); a bare top-level [...] is an
     action list;
  3. normalise each action: kind aliases (type/action/name keys; "message"->send_message, "burn"->
     defect, ...), field aliases (recipient->to, message->text, info->info_id, ...), "7"/"info 7"
     -> 7; then validate against schema.Action;
  4. keep at most `actions_per_turn` actions (plus at most one free grep_log when grep_free) and
     drop the rest with an error code.

Never raises. ok=False means the reply was unusable (no JSON, no action list, or every action
invalid): the caller retries once, then falls back to [Noop()].

Error codes (strings, logged in TurnData.errors): no_json, bad_json, not_object, no_actions_key,
actions_not_list, action_not_object:<i>, unknown_kind:<i>:<kind>, invalid_action:<i>:<kind>:<field>,
too_many_actions:<n>, duplicate_grep:<i>, all_actions_invalid, think_leak, salvaged_truncated.

salvage=True (the caller passes it only for a reply cut off by max_tokens, reasoning.salvage_truncated):
if no JSON object decodes, keep the complete {...} action objects after the "actions" key (plus a
complete "thought"/"note" string before it) and add "salvaged_truncated".

Imports: schema only. Pure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from swarm.schema import ACTION_ADAPTER, Action, GrepLog, Noop

KIND_ALIASES: dict[str, str] = {
    "send_message": "send_message", "message": "send_message", "send": "send_message",
    "dm": "send_message", "msg": "send_message", "sendmessage": "send_message", "private_message": "send_message",
    "post_board": "post_board", "post": "post_board", "board": "post_board", "postboard": "post_board",
    "broadcast": "post_board", "board_post": "post_board",
    "give_key": "give_key", "give": "give_key", "givekey": "give_key", "transfer": "give_key",
    "send_key": "give_key", "trade": "give_key",
    "defect": "defect", "burn": "defect", "betray": "defect",
    "grep_log": "grep_log", "grep": "grep_log", "search": "grep_log", "search_log": "grep_log",
    "noop": "noop", "no_op": "noop", "pass": "noop", "wait": "noop", "none": "noop", "nothing": "noop",
}

FIELD_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "send_message": {"to": ("to", "recipient", "target", "receiver", "player"),
                     "text": ("text", "message", "content", "body", "msg")},
    "post_board": {"text": ("text", "message", "content", "body", "msg", "post")},
    "give_key": {"to": ("to", "recipient", "target", "receiver", "player"),
                 "info_id": ("info_id", "info", "id", "infoid", "key_id", "info_ID")},
    "defect": {"target": ("target", "to", "player", "victim", "agent"),
               "info_id": ("info_id", "info", "id", "infoid", "key_id", "info_ID")},
    "grep_log": {"pattern": ("pattern", "query", "text", "search", "term"),
                 "rounds": ("rounds",)},
    "noop": {},
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_INT = re.compile(r"-?\d+")


@dataclass
class ParsedReply:
    """Result of parse_reply. actions is never empty ([Noop()] when unusable)."""
    thought: str | None = None
    note: str | None = None
    actions: list[Action] = field(default_factory=lambda: [Noop()])
    ok: bool = True
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ extraction
def strip_noise(text: str) -> tuple[str, bool]:
    """Remove <think> blocks and code fences. Returns (text, had_think_leak)."""
    leak = "<think>" in text.lower() or "</think>" in text.lower()
    t = _THINK_BLOCK.sub("", text)
    low = t.lower()
    if "</think>" in low:                       # opening tag lost: drop everything before the close
        t = t[low.rindex("</think>") + len("</think>"):]
    elif "<think>" in low:                      # unclosed: keep what follows, the JSON may be inside
        t = t[low.index("<think>") + len("<think>"):]
    t = _FENCE.sub("", t)
    return t, leak


def _balanced_spans(t: str, open_ch: str, close_ch: str) -> list[str]:
    """Every top-level balanced open..close span, string-aware, in order."""
    spans, depth, start, in_str, esc = [], 0, -1, False, False
    for i, c in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif c == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(t[start:i + 1])
    return spans


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = _TRAILING_COMMA.sub(r"\1", s)
    s = re.sub(r"(?<![\"\w])True(?![\"\w])", "true", s)
    s = re.sub(r"(?<![\"\w])False(?![\"\w])", "false", s)
    s = re.sub(r"(?<![\"\w])None(?![\"\w])", "null", s)
    return s


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except ValueError:
        pass
    try:
        return json.loads(_repair(s))
    except ValueError:
        return None


def extract_json(text: str) -> tuple[Any, list[str]]:
    """First decodable JSON object (preferring one with an 'actions' key), else a bare list."""
    t, leak = strip_noise(text)
    errs = ["think_leak"] if leak else []
    objs = []
    for span in _balanced_spans(t, "{", "}"):
        v = _loads(span)
        if isinstance(v, dict):
            objs.append(v)
    for v in objs:
        if "actions" in v:
            return v, errs
    if objs:
        return objs[0], errs
    for span in _balanced_spans(t, "[", "]"):
        v = _loads(span)
        if isinstance(v, list):
            return v, errs
    if "{" in t:
        return None, [*errs, "bad_json"]
    return None, [*errs, "no_json"]


_ACTIONS_KEY = re.compile(r'"actions"\s*:\s*\[')
_STR_FIELD = r'"{}"\s*:\s*"((?:[^"\\]|\\.)*)"'


def salvage_truncated(text: str) -> dict[str, Any] | None:
    """{"actions": [complete objects], thought?, note?} from a JSON reply cut off mid-way, else None."""
    t, _ = strip_noise(text)
    m = _ACTIONS_KEY.search(t)
    if m is None:
        return None
    acts = [v for v in (_loads(sp) for sp in _balanced_spans(t[m.end():], "{", "}")) if isinstance(v, dict)]
    if not acts:
        return None
    out: dict[str, Any] = {"actions": acts}
    head = t[:m.start()]
    for k in ("thought", "note"):
        fm = re.search(_STR_FIELD.format(k), head)
        if fm:
            v = _loads('"' + fm.group(1) + '"')
            if isinstance(v, str):
                out[k] = v
    return out


# ------------------------------------------------------------------ normalisation
def _to_int(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        m = _INT.search(v)
        if m:
            return int(m.group())
    return v


def _to_text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    return json.dumps(v, ensure_ascii=False)


def _kind_of(a: dict[str, Any]) -> str | None:
    for k in ("kind", "type", "action", "name", "act"):
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower().replace("-", "_").replace(" ", "_")
    return None


def normalize_action(a: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """-> (dict ready for ACTION_ADAPTER, None) or (None, error_code_suffix)."""
    raw_kind = _kind_of(a)
    if raw_kind is None:
        return None, "unknown_kind:None"
    kind = KIND_ALIASES.get(raw_kind)
    if kind is None:
        return None, f"unknown_kind:{raw_kind[:30]}"
    args = a.get("args") if isinstance(a.get("args"), dict) else {}
    src = {**args, **{k: v for k, v in a.items() if k != "args"}}
    out: dict[str, Any] = {"kind": kind}
    for fname, aliases in FIELD_ALIASES[kind].items():
        for al in aliases:
            if al in src and src[al] is not None:
                out[fname] = src[al]
                break
    for f in ("info_id", "rounds"):
        if f in out:
            out[f] = _to_int(out[f])
    for f in ("to", "target", "text", "pattern"):
        if f in out and not isinstance(out[f], str):
            out[f] = _to_text(out[f]) if f in ("text", "pattern") else str(out[f])
        if f in ("to", "target") and isinstance(out.get(f), str):
            out[f] = out[f].strip().lstrip("@")
    return out, None


def parse_reply(text: str, *, actions_per_turn: int = 2, scratchpad: bool = True,
                want_thought: bool = True, grep_free: bool = False,
                max_text_chars: int | None = None, salvage: bool = False) -> ParsedReply:
    """Parse one model reply. Never raises.

    want_thought: keep the "thought" field (cot_mode=explicit). scratchpad: keep "note".
    grep_free: one grep_log does not consume an action slot (engine resolves it with a follow-up).
    max_text_chars: truncate send_message/post_board text.
    """
    res = ParsedReply()
    obj, errs = extract_json(text or "")
    if obj is None and salvage:
        obj = salvage_truncated(text or "")
        if obj is not None:
            errs = [e for e in errs if e not in ("bad_json", "no_json")] + ["salvaged_truncated"]
    res.errors.extend(errs)
    if obj is None:
        res.ok = False
        return res
    if isinstance(obj, list):
        obj = {"actions": obj}
    if want_thought:
        res.thought = _to_text(obj.get("thought") or obj.get("reasoning") or obj.get("thinking"))
    if scratchpad:
        res.note = _to_text(obj.get("note") or obj.get("notes") or obj.get("scratchpad"))
    raw = obj.get("actions")
    if raw is None:
        single = obj.get("action")
        if isinstance(single, dict):
            raw = [single]
        elif _kind_of(obj) in KIND_ALIASES and "actions" not in obj:
            raw = [obj]
        else:
            res.errors.append("no_actions_key")
            res.ok = False
            return res
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        res.errors.append("actions_not_list")
        res.ok = False
        return res

    valid: list[Action] = []
    attempted = 0
    for i, a in enumerate(raw):
        if isinstance(a, str) and KIND_ALIASES.get(a.strip().lower()) == "noop":
            a = {"kind": "noop"}
        if not isinstance(a, dict):
            res.errors.append(f"action_not_object:{i}")
            attempted += 1
            continue
        attempted += 1
        d, err = normalize_action(a)
        if d is None:
            res.errors.append(f"{err.split(':')[0]}:{i}:{err.split(':', 1)[1]}" if err else f"invalid:{i}")
            continue
        if max_text_chars and isinstance(d.get("text"), str):
            d["text"] = d["text"][:max_text_chars]
        if d["kind"] == "grep_log" and isinstance(d.get("pattern"), str):
            d["pattern"] = d["pattern"][:80]
        try:
            valid.append(ACTION_ADAPTER.validate_python(d))
        except ValidationError as e:
            loc = e.errors()[0].get("loc", ("?",))
            res.errors.append(f"invalid_action:{i}:{d['kind']}:{loc[-1] if loc else '?'}")

    # grep: at most one per turn
    seen_grep, kept = False, []
    for i, act in enumerate(valid):
        if isinstance(act, GrepLog):
            if seen_grep:
                res.errors.append(f"duplicate_grep:{i}")
                continue
            seen_grep = True
        kept.append(act)
    # drop noops if something real is present
    real = [a for a in kept if not isinstance(a, Noop)]
    kept = real or kept[:1]
    # slot limit
    out, used = [], 0
    for act in kept:
        free = grep_free and isinstance(act, GrepLog)
        if not free:
            if used >= actions_per_turn:
                continue
            used += 1
        out.append(act)
    n_dropped = len(kept) - len(out)
    if n_dropped:
        res.errors.append(f"too_many_actions:{n_dropped}")
    if attempted and not out:
        res.errors.append("all_actions_invalid")
        res.ok = False
    res.actions = out if out else [Noop()]
    return res
