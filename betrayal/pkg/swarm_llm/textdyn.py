"""
Text-response dynamics: how each agent's language changes over time, and how
agents differ from one another.

Every agent's reply at every round is recorded in st.calls. This module turns
those into:

  * a response table  node = (agent, round), labelled "a<agent>.t<round>"
  * cosine similarity (TF-IDF) and Jaccard similarity (word sets) between any
    two responses
  * LONGITUDINAL analysis: each agent's drift over time (round-to-round and vs.
    its first response), and a 2-D trajectory (a<agent>.t1 -> ... -> a<agent>.tN)
  * CROSS-SECTIONAL analysis: at each round, how similar agents are to each
    other, split by group (leader vs. follower, aligned vs. misaligned)

Groups:
  * role     "influential" (leader / hub / broker / connector) vs "follower"
  * aligned  True unless the agent was compromised or defected at least once

Dependencies: scikit-learn (TfidfVectorizer, cosine_similarity, TruncatedSVD).
"""

from __future__ import annotations

import re
import numpy as np


def node_label(agent, rnd):
    return f"a{agent}.t{rnd}"


# --------------------------------------------------------------------------
# Pull responses and group labels
# --------------------------------------------------------------------------

def response_table(st, use_reason=True):
    """One row per agent reply: agent, round, text, result. `text` is the raw
    model reply, optionally with the parsed 'reason' appended for signal."""
    rows = []
    for c in getattr(st, "calls", []):
        text = str(c.get("reply", "") or "")
        if use_reason and isinstance(c.get("parsed"), dict):
            reason = c["parsed"].get("reason") or c["parsed"].get("message") or ""
            if reason:
                text = text + " || " + str(reason)
        rows.append(dict(agent=c["agent"], round=c["round"], text=text,
                         result=c.get("result"), label=node_label(c["agent"], c["round"])))
    rows.sort(key=lambda r: (r["agent"], r["round"]))
    return rows


def group_labels(st):
    """Return {agent: {"role": "influential"/"follower", "aligned": bool}}."""
    from netsci import structural_roles
    roles, _ = structural_roles(st)
    influential = {"leader", "hub", "broker", "connector"}
    defected = {d["actor"] for d in getattr(st, "defections", [])}
    out = {}
    for i in range(st.cfg.N):
        out[i] = dict(
            role="influential" if roles.get(i) in influential else "follower",
            structural_role=roles.get(i),
            aligned=(i not in st.compromised and i not in defected),
        )
    return out


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------

def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def jaccard(a, b):
    sa, sb = _tokens(a), _tokens(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def vectorize(texts):
    """TF-IDF matrix over all response texts (fit once so vectors are comparable)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=1, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform([t if t.strip() else "empty" for t in texts])
    return X, vec


def cosine_matrix(X):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(X)


# --------------------------------------------------------------------------
# LONGITUDINAL: each agent's evolution over time
# --------------------------------------------------------------------------

def self_drift(st):
    """Per agent: similarity of each response to the previous one and to the
    first one. Low similarity = the agent's language changed a lot."""
    rows = response_table(st)
    texts = [r["text"] for r in rows]
    X, _ = vectorize(texts)
    cos = cosine_matrix(X)
    idx = {(r["agent"], r["round"]): k for k, r in enumerate(rows)}
    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r["round"])
    out = {}
    for agent, rnds in by_agent.items():
        rnds = sorted(rnds)
        series = []
        for k, rnd in enumerate(rnds):
            i = idx[(agent, rnd)]
            prev_cos = cos[i, idx[(agent, rnds[k - 1])]] if k > 0 else 1.0
            first_cos = cos[i, idx[(agent, rnds[0])]]
            prev_jac = jaccard(rows[i]["text"], rows[idx[(agent, rnds[k - 1])]]["text"]) if k > 0 else 1.0
            series.append(dict(round=rnd, cos_to_prev=float(prev_cos), cos_to_first=float(first_cos),
                               jac_to_prev=float(prev_jac)))
        total = 1.0 - series[-1]["cos_to_first"] if series else 0.0
        out[agent] = dict(series=series, total_drift=float(total))
    return out


def trajectories(st, dims=2):
    """2-D projection (TruncatedSVD) of every response, so each agent is a path
    a<agent>.t1 -> ... -> a<agent>.tN. Returns points keyed by (agent, round)."""
    from sklearn.decomposition import TruncatedSVD
    rows = response_table(st)
    X, _ = vectorize([r["text"] for r in rows])
    k = min(dims, max(X.shape[1] - 1, 1))
    coords = TruncatedSVD(n_components=max(k, 1), random_state=0).fit_transform(X)
    if coords.shape[1] < 2:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    pts = {}
    for r, xy in zip(rows, coords):
        pts[(r["agent"], r["round"])] = (float(xy[0]), float(xy[1]))
    return pts, rows


# --------------------------------------------------------------------------
# CROSS-SECTIONAL: how agents differ at each round, by group
# --------------------------------------------------------------------------

def group_similarity_over_time(st, by="role"):
    """At each round, mean pairwise cosine within each group and between groups.
    by = "role" (influential vs follower) or "aligned" (aligned vs misaligned)."""
    rows = response_table(st)
    if not rows:
        return []
    X, _ = vectorize([r["text"] for r in rows])
    cos = cosine_matrix(X)
    groups = group_labels(st)
    key = (lambda a: groups[a]["role"]) if by == "role" else (lambda a: "aligned" if groups[a]["aligned"] else "misaligned")
    idx_by_round = {}
    for k, r in enumerate(rows):
        idx_by_round.setdefault(r["round"], []).append((k, r["agent"]))
    out = []
    for rnd in sorted(idx_by_round):
        members = idx_by_round[rnd]
        within, between, cross = {}, [], []
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                (ka, aa), (kb, ab) = members[a_i], members[b_i]
                s = cos[ka, kb]
                if key(aa) == key(ab):
                    within.setdefault(key(aa), []).append(s)
                else:
                    cross.append(s)
        row = dict(round=rnd)
        for g, vals in within.items():
            row[f"within_{g}"] = float(np.mean(vals)) if vals else None
        row["between_groups"] = float(np.mean(cross)) if cross else None
        out.append(row)
    return out


def pairwise_at_round(st, rnd):
    """Agent-by-agent cosine similarity matrix for one round (for a heatmap)."""
    rows = [r for r in response_table(st) if r["round"] == rnd]
    if len(rows) < 2:
        return [], np.zeros((0, 0))
    X, _ = vectorize([r["text"] for r in rows])
    return [r["agent"] for r in rows], cosine_matrix(X)


def summary(st):
    """One-call overview for a quick look or for the Discord agent to print."""
    drift = self_drift(st)
    groups = group_labels(st)
    gsim = group_similarity_over_time(st, by="role")
    return dict(
        n_responses=len(response_table(st)),
        mean_total_drift=round(float(np.mean([d["total_drift"] for d in drift.values()])), 3) if drift else 0.0,
        most_evolved_agent=max(drift, key=lambda a: drift[a]["total_drift"]) if drift else None,
        influential_agents=[a for a, g in groups.items() if g["role"] == "influential"],
        misaligned_agents=[a for a, g in groups.items() if not g["aligned"]],
        group_similarity_last_round=gsim[-1] if gsim else None,
    )
