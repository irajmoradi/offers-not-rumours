"""
Route 2: analyze PUBLIC datasets of real multi-agent LLM interactions with the
same emergence metrics. No system is touched; you read recorded conversations.

Good starting datasets on the Hugging Face hub (multi-agent LLM traces):
  * "mcemri/MAST"         - annotated multi-agent system failure traces (has
                            failure labels, so you can relate structure to failure)
  * any dataset whose rows are a conversation = a list of {role/name, content} turns

Because schemas vary, `conversation_turns` tries several common shapes and you
can pass a custom extractor. Each conversation becomes an interaction graph;
`analyze_dataset` returns one emergence row per conversation plus a summary.
"""

from __future__ import annotations

import json

from real_agents import analyze_run


def conversation_turns(row, speaker_keys=("name", "speaker", "role", "agent", "sender"),
                       text_keys=("content", "text", "message", "utterance", "value"),
                       convo_keys=("conversation", "messages", "dialogue", "turns", "trace", "history")):
    """Pull a list of {sender, text} turns out of one dataset row, trying common
    layouts: a list-of-dicts conversation, a chat-style 'messages' field, or a
    JSON string holding either."""
    convo = None
    for k in convo_keys:
        if k in row and row[k] is not None:
            convo = row[k]
            break
    if convo is None:
        # maybe the row itself is the list, or a JSON string
        convo = row
    if isinstance(convo, str):
        try:
            convo = json.loads(convo)
        except json.JSONDecodeError:
            return []
    if not isinstance(convo, (list, tuple)):
        return []
    turns = []
    for t in convo:
        if not isinstance(t, dict):
            continue
        sender = next((str(t[k]) for k in speaker_keys if k in t and t[k] is not None), None)
        text = next((str(t[k]) for k in text_keys if k in t and t[k] is not None), None)
        if sender and text is not None:
            turns.append(dict(sender=sender, text=text))
    return turns


def success_from_row(row, keys=("success", "label", "is_correct", "resolved", "passed")):
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, str):
                return 1 if v.lower() in ("success", "true", "correct", "pass", "resolved", "1") else 0
            return int(bool(v))
    return None


def analyze_dataset(dataset, mode="sequence", extractor=None, limit=None):
    """dataset: an iterable of rows (e.g. a Hugging Face `datasets` split).
    Returns (rows, summary). Each row is one conversation's emergence profile."""
    extractor = extractor or conversation_turns
    rows = []
    for k, row in enumerate(dataset):
        if limit and k >= limit:
            break
        turns = extractor(row)
        n_speakers = len({t["sender"] for t in turns})
        if n_speakers < 2:
            continue
        prof, _, _ = analyze_run(turns, mode=mode, success=success_from_row(row))
        prof["row"] = k
        rows.append(prof)
    return rows, summarize(rows)


def summarize(rows):
    import numpy as np
    if not rows:
        return {}
    by_size = {}
    for r in rows:
        bucket = "small (2-3)" if r["N"] <= 3 else "medium (4-7)" if r["N"] <= 7 else "large (8+)"
        by_size.setdefault(bucket, []).append(r)
    out = {}
    for bucket, rs in by_size.items():
        out[bucket] = {
            "n_conversations": len(rs),
            "mean_agents": round(float(np.mean([r["N"] for r in rs])), 2),
            "mean_centralization": round(float(np.mean([r["indegree_centralization"] for r in rs])), 3),
            "mean_leader_share": round(float(np.mean([r["leader_share"] for r in rs])), 3),
            "single_hub_rate": round(float(np.mean([r["single_hub"] for r in rs])), 3),
        }
        labeled = [r for r in rs if r["success"] is not None]
        if labeled:
            fail = [r for r in labeled if not r["success"]]
            ok = [r for r in labeled if r["success"]]
            if fail and ok:
                out[bucket]["centralization_failed"] = round(float(np.mean([r["indegree_centralization"] for r in fail])), 3)
                out[bucket]["centralization_succeeded"] = round(float(np.mean([r["indegree_centralization"] for r in ok])), 3)
    return out
