"""
Route 1: run a REAL open-source multi-agent framework that YOU control, at
varying sizes, and measure leader emergence and failure with the same metrics
as the simulator.

The core is framework-agnostic. Give it a list of messages, each with a sender
and text (and optional recipient), and it builds the interaction graph and runs
the emergence profile. Thin adapters turn a framework's own log into that list,
so the framework-specific surface stays small.

Point any framework at your local open model by running an OpenAI-compatible
server with vLLM:

    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

then give the framework base_url="http://localhost:8000/v1", api_key="x".
This keeps everything local and free: real framework, real protocol, your
instances, full ground truth (you see every message and set N).
"""

from __future__ import annotations

import re

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------
# Framework-agnostic core
# --------------------------------------------------------------------------

def interaction_graph(messages, names=None, mode="sequence"):
    """messages: list of dicts with "sender" and "text" (optional "to").
    mode:
      "sequence"  edge sender -> next distinct speaker (who answers whom)
      "mention"   edge sender -> every agent named in the text
      "directed"  use each message's explicit "to"
    Returns a weighted DiGraph over agent names."""
    names = list(names) if names else sorted({m["sender"] for m in messages})
    idx = {n: k for k, n in enumerate(names)}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))

    def add(a, b):
        if a in idx and b in idx and idx[a] != idx[b]:
            e = G.get_edge_data(idx[a], idx[b], {})
            G.add_edge(idx[a], idx[b], weight=e.get("weight", 0) + 1)

    prev = None
    for m in messages:
        s = m["sender"]
        if mode == "sequence":
            if prev is not None and prev != s:
                add(prev, s)
            prev = s
        elif mode == "directed" and m.get("to"):
            tos = m["to"] if isinstance(m["to"], (list, tuple)) else [m["to"]]
            for t in tos:
                add(s, t)
        elif mode == "mention":
            for other in names:
                if other != s and re.search(rf"\b{re.escape(str(other))}\b", str(m.get("text", ""))):
                    add(s, other)
    return G, names


def gini(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) == 0 or xs.sum() == 0:
        return 0.0
    n = len(xs)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))


def graph_emergence(G, names, success=None):
    """Leader/role metrics on any interaction graph. `success` optional 0/1 for
    the task outcome, so you can relate structure to failure."""
    N = G.number_of_nodes()
    indeg = np.array([G.in_degree(i, weight="weight") for i in range(N)], dtype=float)
    outdeg = np.array([G.out_degree(i, weight="weight") for i in range(N)], dtype=float)
    deg = np.array([G.in_degree(i) for i in range(N)], dtype=float)
    cmax = deg.max() if N else 0
    centralization = float((cmax - deg).sum() / ((N - 1) * (N - 2))) if N > 2 else 0.0
    und = G.to_undirected()
    btw = nx.betweenness_centrality(und) if und.number_of_edges() else {i: 0 for i in range(N)}
    talk = indeg + outdeg
    top = int(np.argmax(talk)) if talk.sum() else None
    return dict(
        N=N,
        messages=int(G.size(weight="weight")),
        leader=names[top] if top is not None else None,
        leader_share=float(talk.max() / talk.sum()) if talk.sum() else 0.0,
        indegree_centralization=centralization,
        attention_gini=gini(indeg),
        single_hub=bool(talk.sum() and talk.max() >= 0.4 * talk.sum()),
        top_broker=names[int(max(btw, key=btw.get))] if und.number_of_edges() else None,
        reciprocity=float(nx.reciprocity(G) or 0.0) if G.number_of_edges() else 0.0,
        density=float(nx.density(G)),
        success=None if success is None else int(success),
    )


def analyze_run(messages, names=None, mode="sequence", success=None):
    G, names = interaction_graph(messages, names, mode)
    return graph_emergence(G, names, success), G, names


# --------------------------------------------------------------------------
# Adapter: AutoGen / AG2 group chat
# --------------------------------------------------------------------------

def autogen_messages(groupchat):
    """Turn an AutoGen GroupChat's `.messages` into the generic message list.
    Works with ag2 / pyautogen GroupChat objects (message dicts have 'name' and
    'content')."""
    out = []
    for m in getattr(groupchat, "messages", groupchat):
        sender = m.get("name") or m.get("role") or "unknown"
        out.append(dict(sender=str(sender), text=str(m.get("content", ""))))
    return out


def run_autogen_swarm(n_agents, task, base_url="http://localhost:8000/v1", model="local",
                      api_key="x", max_round=None, temperature=0.7):
    """Spin up N AutoGen assistants in a group chat on one shared task and return
    the generic message list. Requires `pip install pyautogen` (or ag2) and a
    running OpenAI-compatible server (vLLM). Agents are plain assistants; the
    structure that emerges is what you measure."""
    import autogen  # pyautogen or ag2

    llm_config = {"config_list": [{"model": model, "base_url": base_url, "api_key": api_key}],
                  "temperature": temperature, "cache_seed": None}
    agents = [autogen.AssistantAgent(
        name=f"agent_{i}",
        system_message=(f"You are agent_{i}, one of {n_agents} assistants collaborating to finish a shared "
                        f"task. Coordinate with the others, divide the work, and help whoever asks. "
                        f"Be concise."),
        llm_config=llm_config) for i in range(n_agents)]
    user = autogen.UserProxyAgent(name="coordinator", human_input_mode="NEVER",
                                  code_execution_config=False, max_consecutive_auto_reply=0,
                                  default_auto_reply="")
    groupchat = autogen.GroupChat(agents=agents + [user], messages=[],
                                  max_round=max_round or (3 * n_agents))
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    user.initiate_chat(manager, message=task)
    return autogen_messages(groupchat)


# --------------------------------------------------------------------------
# Adapter: CrewAI
# --------------------------------------------------------------------------

def crewai_messages(crew_output):
    """Best-effort extraction from a CrewAI result. CrewAI exposes per-task output
    and agent roles; we build a sender/text list from task outputs. For richer
    graphs, enable CrewAI's step callbacks and pass the collected steps instead."""
    out = []
    tasks = getattr(crew_output, "tasks_output", None) or []
    for t in tasks:
        sender = getattr(getattr(t, "agent", None), "role", None) or getattr(t, "agent", "agent")
        out.append(dict(sender=str(sender), text=str(getattr(t, "raw", t))))
    return out


def steps_to_messages(steps):
    """Generic: a list of (agent_name, text) tuples (e.g. from a framework's
    step/callback hook) into the message list."""
    return [dict(sender=str(a), text=str(x)) for a, x in steps]


# --------------------------------------------------------------------------
# Scaling over a real framework
# --------------------------------------------------------------------------

def scaling_over_framework(run_swarm, sizes, seeds, task, mode="sequence", success_fn=None, **kw):
    """run_swarm(n_agents, task, seed, **kw) -> message list. Returns one
    emergence row per (size, seed). success_fn(messages) -> 0/1 is optional."""
    rows = []
    for N in sizes:
        for seed in seeds:
            msgs = run_swarm(N, task, seed, **kw)
            succ = success_fn(msgs) if success_fn else None
            prof, _, _ = analyze_run(msgs, mode=mode, success=succ)
            rows.append(dict(seed=seed, requested_N=N, **prof))
    return rows
