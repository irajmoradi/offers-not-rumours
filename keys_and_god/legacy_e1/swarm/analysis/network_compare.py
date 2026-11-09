"""Network experiment comparison (plan/NETWORK_PLAN.md, NETWORK_RESULTS.md): conditions compared on the
same worlds. No LLM calls, read-only; builds on swarm.analysis.network (graph, diffusion, brokering)
and swarm.analysis.social (claim rules).

    PYTHONNOUSERSITE=1 python -m swarm.analysis.network_compare runs/net20_complete_sim runs/net20_er3_contacts_sim \
        runs/net20_er3_known_sim [runs/net20_ba_known_sim] [--ref-graph runs/net20_er3_contacts_sim] \
        [--json plan/figs/network_n20.json] [--figs plan/figs] [--examples 6]

Per condition (one run dir = one condition; games matched by game_idx):
  outcome     win rate, welfare, mean score, needs filled (held at the end, of 3), final round; mean and
              t95 CI over worlds; paired differences against the first run (by game_idx), t95 CI, sign test
  winners     cumulative share of agents that have won, by round (the figure)
  relays      key copies traced through give transfers: a RELAY = an agent receives a copy whose info it
              does not need (not an open need, or it already holds that info) and later gives the same
              copy on to a THIRD agent (handing it back to the giver is counted as a return). A relay chain = a copy delivered (to an agent that needed it) after >= 2 give hops
              with >= 1 relaying intermediary; also the graph distance origin -> final holder.
  distance    per (agent, need): d = graph distance (on the condition's graph, or on --ref-graph for the
              complete-graph run: a placebo) to the nearest initial holder of that info; fill rate by d;
              win rate by the agent's max d over its needs
  gossip      social.py claims per 1000 messages by type (third-party mentions, accusations, burned-me,
              needs / holds claims), hearsay markers, rumour chains (social.chains), introductions
              (INTRO rule: "ask / talk to / contact X", "X has / holds key N"), forwarding requests
              (FORWARD rule) and refusal notices (not_connected / unknown_contact)
  diffusion   (known_ids) name-coverage S-curve; channels; introductions = names learned from a DM or
              board MENTION (the mentioning sender introduced them); introducers' betweenness vs
              introductions (Spearman); hop distance of learned names
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

import networkx as nx

from swarm.analysis import network as netan
from swarm.analysis import social

INTRO_VERB = r"(?:ask|asking|talk to|contact|reach out to|message|dm|ping|try|check with|go to|speak (?:to|with))"
FORWARD = re.compile(r"(?<!looking )\bforward(?!ing to| to (?:receiv|get|hear|work|trad))|\bpass (?:it|them|this|that|the key)? ?(?:on|along|to)\b|\brelay|on (?:my|your|their) behalf|"
                     r"\bmiddle ?man|\bintermediar|\bthrough you\b|\bvia\b|\bbroker", re.I)
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20,
       12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13}


def mean_ci(xs: list[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": round(m, 4), "lo": None, "hi": None}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T95.get(n - 1, 1.96) * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4)}


def sign_p(diffs: list[float]) -> float | None:
    """Two-sided exact sign test (zeros dropped)."""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if not n:
        return None
    k = min(pos, neg)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


# ------------------------------------------------------------------ per game
def game_outcome(g: dict[str, Any]) -> dict[str, Any]:
    ev = g["events"]
    start = ev[0]["data"]["agents"]
    out = g["outcome"] or {}
    per = out.get("per_agent", {})
    n = len(g["names"])
    ends = [e for e in ev if e["type"] == "round_end"]
    welfare = ends[-1]["data"]["welfare"] if ends else None
    held = held_needs(g)
    wins_by_round: collections.Counter[int] = collections.Counter(
        e["round"] for e in ev if e["type"] == "win")
    status = "done" if any(e["type"] == "game_end" for e in ev) else "partial"
    return {"game_idx": g["game_idx"], "label": g["label"], "status": status, "n_agents": n,
            "max_rounds": min(ev[0]["data"]["max_rounds"], g.get("at_round") or 10**6),
            "end_reason": out.get("end_reason"), "rounds": ends[-1]["round"] if ends else 0,
            "win_rate": sum(per.get(a, {}).get("won_round") is not None for a in g["names"]) / n if per
            else sum(wins_by_round.values()) / n,
            "welfare": welfare,
            "mean_score": sum(per[a]["score"] for a in g["names"]) / n if per else None,
            "needs_filled": sum(len(held[a] & set(start[a]["needs"])) for a in g["names"]) / n,
            "wins_by_round": dict(sorted(wins_by_round.items()))}


def replay_holdings(g: dict[str, Any]):
    """Yield (event, holdings-before) over key_transfer events; holdings = agent -> {copy_id: info}."""
    start = g["events"][0]["data"]["agents"]
    hold = {a: {k["copy_id"]: k["info_id"] for k in s["keys"]} for a, s in start.items()}
    for e in g["events"]:
        if e["type"] == "key_transfer":
            d = e["data"]
            yield e, hold
            f, t = d.get("from_agent"), d.get("to_agent")
            if f in hold:
                hold[f].pop(d["copy_id"], None)
            if t in hold:
                hold[t][d["copy_id"]] = d["info_id"]
    yield None, hold


def held_needs(g: dict[str, Any]) -> dict[str, set[int]]:
    hold: dict[str, dict[str, int]] = {}
    for _e, h in replay_holdings(g):
        hold = h
    return {a: set(v.values()) for a, v in hold.items()}


def relays(g: dict[str, Any], dist: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Key copies passed on by agents that did not need them."""
    start = g["events"][0]["data"]["agents"]
    needs = {a: set(s["needs"]) for a, s in start.items()}
    path: dict[str, list[tuple[str, int, bool]]] = {}     # copy -> [(holder, round, needed_on_receipt)]
    for a, s in start.items():
        for k in s["keys"]:
            path[k["copy_id"]] = [(a, 0, k["info_id"] in needs[a])]
    relay_events = []
    for e, hold in replay_holdings(g):
        if e is None:
            break
        d = e["data"]
        c, t, f, info = d["copy_id"], d.get("to_agent"), d.get("from_agent"), d["info_id"]
        if t is None:
            continue
        needed = info in needs[t] and info not in set(hold.get(t, {}).values())
        p = path.setdefault(c, [] if f is None else [(f, 0, False)])
        if d["cause"] == "give" and len(p) >= 2 and p[-1][0] == f and not p[-1][2]:
            relay_events.append({"round": e["round"], "relay": f, "from": p[-2][0], "to": t, "info": info,
                                 "copy": c, "delivered_need": needed, "returned": t == p[-2][0]})
        p.append((t, e["round"], needed))
        if d["cause"] != "give":
            path[c] = [(t, e["round"], needed)]                  # minted copy: new origin
    chains = []
    for c, p in path.items():
        hops = len(p) - 1
        if hops >= 2 and p[-1][2] and p[-1][0] != p[0][0] and any(not x[2] for x in p[1:-1]):
            o, fin = p[0][0], p[-1][0]
            chains.append({"copy": c, "hops": hops, "path": [f"{x[0]}@r{x[1]}" for x in p],
                           "origin_dist": dist.get(o, {}).get(fin)})
    fwd = [r for r in relay_events if not r["returned"]]      # passed on to a third agent (not handed back)
    return {"relay_events": relay_events, "n_relay_events": len(fwd), "n_returns": len(relay_events) - len(fwd),
            "n_relay_delivered": sum(r["delivered_need"] for r in fwd),
            "chains": chains, "n_chains": len(chains),
            "chains_beyond_nbhd": sum((c["origin_dist"] or 0) >= 2 for c in chains),
            "gives_hist": dict(collections.Counter(len(p) - 1 for p in path.values() if len(p) > 1))}


def distance_fill(g: dict[str, Any], dist: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """(agent, need) rows: distance to the nearest initial holder, filled at the end, agent won."""
    start = g["events"][0]["data"]["agents"]
    holders = collections.defaultdict(set)
    for a, s in start.items():
        for k in s["keys"]:
            holders[k["info_id"]].add(a)
    held = held_needs(g)
    per = (g["outcome"] or {}).get("per_agent", {})
    rows = []
    for a, s in start.items():
        for i in s["needs"]:
            hs = holders[i] - {a}
            d = min((dist[a].get(h, 99) for h in hs), default=None) if i not in {k["info_id"] for k in s["keys"]} else 0
            rows.append({"agent": a, "need": i, "d": d, "filled": i in held[a],
                         "won": per.get(a, {}).get("won_round") is not None})
    return rows


def intro_rx(names: list[str]) -> re.Pattern[str]:
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"\b{INTRO_VERB}\s+@?(?P<a>{alt})\b|\b(?P<b>{alt})(?:'s)?\s+(?:has|holds|owns|"
                      rf"might have|may have|should have|can give|is holding|got)\b", re.I)


def gossip(g: dict[str, Any]) -> dict[str, Any]:
    sg = g["social"]
    msgs = [m for m in sg["msgs"] if not m["forged"]]
    cl = [c for c in social.claims_of(sg) if not c["m"]["forged"]]
    rx = intro_rx(g["names"])
    intros, forwards = [], []
    for m in msgs:
        for mt in rx.finditer(m["text"]):
            z = (mt.group("a") or mt.group("b")).lower()
            if z not in (m["s"], m["r"]):
                intros.append({"round": m["round"], "sender": m["s"], "receiver": m["r"], "subject": z,
                               "ch": m["ch"], "text": m["text"]})
                break
        if FORWARD.search(m["text"]):
            forwards.append({"round": m["round"], "sender": m["s"], "receiver": m["r"], "ch": m["ch"],
                             "text": m["text"]})
    refusals = collections.Counter()
    for e in g["events"]:
        if e["type"] == "action_result" and not e["data"]["ok"]:
            refusals[e["data"]["reason"]] += 1
    lens, paths, n_claims = social.chains({g["label"]: sg})
    return {"n_msgs": len(msgs), "n_dm": sum(m["ch"] == "dm" for m in msgs),
            "n_board": sum(m["ch"] == "board" for m in msgs),
            "third_party_msgs": len({c["seq"] for c in cl}),
            "claims": dict(collections.Counter(c["cat"] for c in cl)),
            "hearsay": sum(c["hearsay"] for c in cl),
            "liar_words_any_msg": sum(bool(social.LIAR.search(m["text"])) for m in msgs),
            "chains": lens, "chain_paths": [[{"round": c["round"], "sender": c["sender"], "r": c["r"], "subj": c["subj"],
                                              "cat": c["cat"], "text": c["m"]["text"]} for c in p] for _k, p in paths],
            "intros": intros, "forwards": forwards, "refusals": dict(refusals),
            "claim_examples": [{"cat": c["cat"], "round": c["round"], "sender": c["sender"], "r": c["r"],
                                "subj": c["subj"], "text": c["m"]["text"]} for c in cl]}


def introductions(g: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any] | None:
    """known_ids: who introduced whom (names learned from a DM / board mention), introducer centrality."""
    d = netan.diffusion(g)
    if d is None:
        return None
    # attribute each mention-learning to the first sender in that observation naming the learned name
    by_turn = {}
    for e in g["events"]:
        if e["type"] == "turn":
            by_turn[(e["data"]["agent"], e["round"], e["turn_index"])] = e["data"]["obs"]
    intro_by: collections.Counter[str] = collections.Counter()
    pairs = []
    known0 = {a: set(k) for a, k in g["known"].get(0, {}).items()}
    known = {a: set(k) for a, k in known0.items()}
    for e in g["events"]:
        if e["type"] != "turn":
            continue
        a, obs = e["data"]["agent"], e["data"]["obs"]
        for box in ("inbox", "board"):
            for m in obs.get(box, []):
                for z in sorted(netan.names_in_text(m["text"], g["names"])):
                    if z != a and z not in known[a]:
                        intro_by[m["sender"]] += 1
                        pairs.append({"round": e["round"], "introducer": m["sender"], "to": a, "name": z,
                                      "box": box, "text": m["text"]})
                        known[a].add(z)
            for m in obs.get(box, []):
                if m["sender"] in g["names"] and m["sender"] != a:
                    known[a].add(m["sender"])
        for nt in obs.get("notices", []):
            if nt["kind"] in netan.LEARN_NOTICE_KINDS:
                known[a] |= set(netan.names_in_text(nt["text"], g["names"])) - {a}
    btw = stats["betweenness"]
    xs = [btw[a] for a in g["names"]]
    ys = [intro_by.get(a, 0) for a in g["names"]]
    return {"coverage": d["coverage_by_round"], "channel_hist": d["channel_hist"], "hops_hist": d["hops_hist"],
            "n_learned": len(d["learned"]), "n_intro": len(pairs), "intro_by": dict(intro_by),
            "intro_pairs": pairs, "rows": [{"agent": a, "betweenness": btw[a], "degree": stats["degree"][a],
                                             "intros": intro_by.get(a, 0)} for a in g["names"]],
            "spearman_btw_intros": netan._spearman(xs, ys)}


# ------------------------------------------------------------------ per condition
def truncate(g: dict[str, Any], R: int) -> dict[str, Any]:
    """The game as it stood at the end of round R (matched-rounds comparison): events, messages,
    transfers and known sets of rounds <= R; outcome rebuilt from the win / round_end events."""
    ev = [e for e in g["events"] if e["round"] <= R and e["type"] != "game_end"]
    ends = [e for e in ev if e["type"] == "round_end"]
    scores = ends[-1]["data"]["scores"] if ends else {a: 0.0 for a in g["names"]}
    won = {}
    for e in ev:
        if e["type"] == "win":
            won.setdefault(e["data"]["agent"], e["round"])
    sg = dict(g["social"])
    sg["msgs"] = [m for m in sg["msgs"] if m["round"] <= R]
    sg["transfers"] = [t for t in sg["transfers"] if t["round"] <= R]
    sg.pop("_tok", None)
    out = {**g, "events": ev, "social": sg, "known": {r: k for r, k in g["known"].items() if r <= R},
           "outcome": {"per_agent": {a: {"score": scores.get(a, 0.0), "won_round": won.get(a)} for a in g["names"]},
                       "end_reason": f"truncated_r{R}", "welfare": ends[-1]["data"]["welfare"] if ends else None}}
    out["at_round"] = R
    if ends and ends[-1]["round"] == R:
        out["events"] = ev + [{"type": "game_end", "round": R, "data": {"outcome": out["outcome"]}}]
    return out


def analyse(run_dir: Path, ref_graphs: dict[Any, nx.Graph] | None = None, at_round: int | None = None,
            worlds: set[int] | None = None) -> dict[str, Any]:
    games = [g for g in netan.load_run(run_dir) if worlds is None or int(g["game_idx"]) in worlds]
    if at_round is not None:
        games = [truncate(g, at_round) for g in games]
    rows, rel, dfill, gos, intro, centr = [], [], [], [], [], []
    for g in games:
        graph = g["graph"]
        if g["complete"] and ref_graphs and g["game_idx"] in ref_graphs:
            graph = ref_graphs[g["game_idx"]]                     # placebo distances on the ER graph
        dist = netan._dist(graph)
        stats = netan.graph_stats(g["graph"])
        o = game_outcome(g)
        r = relays(g, netan._dist(g["graph"]))
        o.update({k: r[k] for k in ("n_relay_events", "n_returns", "n_relay_delivered", "n_chains", "chains_beyond_nbhd")})
        rows.append(o)
        rel.append({"label": g["label"], **r})
        dfill += [{"game_idx": g["game_idx"], **x} for x in distance_fill(g, dist)]
        gs = gossip(g)
        gs["label"] = g["label"]
        gos.append(gs)
        it = introductions(g, stats)
        if it:
            intro.append({"label": g["label"], **it})
        b = netan.brokering(g)
        centr += netan.centrality_rows(g, stats, b)
    return {"run": run_dir.name, "games": rows, "relays": rel, "distance_rows": dfill, "gossip": gos,
            "intro": intro, "centrality": netan.centrality_summary(centr), "mode": games[0]["mode"] if games else None}


def summarise(res: list[dict[str, Any]]) -> dict[str, Any]:
    base = {g["game_idx"]: g for g in res[0]["games"]}
    out: dict[str, Any] = {}
    for c in res:
        gs = c["games"]
        s: dict[str, Any] = {"n_games": len(gs), "partial": sum(g["status"] != "done" for g in gs),
                             "mode": c["mode"]}
        for k in ("win_rate", "welfare", "mean_score", "needs_filled", "rounds", "n_relay_events", "n_returns",
                  "n_relay_delivered", "n_chains", "chains_beyond_nbhd"):
            s[k] = mean_ci([g[k] for g in gs])
        for k in ("win_rate", "welfare", "needs_filled"):
            diffs = [g[k] - base[g["game_idx"]][k] for g in gs if g["game_idx"] in base and g[k] is not None
                     and base[g["game_idx"]][k] is not None]
            s[f"d_{k}_vs_{res[0]['run']}"] = {**mean_ci(diffs), "sign_p": sign_p(diffs)}
        bydist = collections.defaultdict(list)
        for r in c["distance_rows"]:
            bydist[r["d"]].append(r["filled"])
        s["fill_by_distance"] = {str(d): {"n": len(v), "fill": round(sum(v) / len(v), 3)}
                                 for d, v in sorted(bydist.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}
        agent_max = collections.defaultdict(lambda: [0, False])
        for r in c["distance_rows"]:
            k = (r["game_idx"], r["agent"])
            agent_max[k][0] = max(agent_max[k][0], r["d"] or 0)
            agent_max[k][1] = r["won"]
        wd = collections.defaultdict(list)
        for (dmax, won) in agent_max.values():
            wd[dmax].append(won)
        s["win_by_max_distance"] = {str(d): {"n": len(v), "win": round(sum(v) / len(v), 3)} for d, v in sorted(wd.items())}
        tot = collections.Counter()
        claims = collections.Counter()
        for gsp in c["gossip"]:
            tot.update({k: gsp[k] for k in ("n_msgs", "n_dm", "n_board", "third_party_msgs", "hearsay",
                                             "liar_words_any_msg")})
            tot["intros"] += len(gsp["intros"])
            tot["forwards"] += len(gsp["forwards"])
            tot["chains"] += len(gsp["chains"])
            tot["max_chain"] = max(tot["max_chain"], max(gsp["chains"], default=0))
            claims.update(gsp["claims"])
            for k, v in gsp["refusals"].items():
                tot[f"refused_{k}"] += v
        s["gossip"] = dict(tot)
        s["claims"] = dict(claims)
        per1k = lambda x: round(1000 * x / max(tot["n_msgs"], 1), 2)  # noqa: E731
        s["per_1000_msgs"] = {"third_party": per1k(tot["third_party_msgs"]), "intros": per1k(tot["intros"]),
                              "forwards": per1k(tot["forwards"]),
                              **{k: per1k(v) for k, v in claims.items()}}
        if c["intro"]:
            cov = collections.defaultdict(list)
            for it in c["intro"]:
                for r_, v in it["coverage"].items():
                    cov[int(r_)].append(v)
            s["coverage_by_round"] = {r_: round(sum(v) / len(v), 4) for r_, v in sorted(cov.items())}
            ch, hops = collections.Counter(), collections.Counter()
            for it in c["intro"]:
                ch.update(it["channel_hist"])
                hops.update({str(k): v for k, v in it["hops_hist"].items()})
            s["learn_channels"] = dict(ch.most_common())
            s["learn_hops"] = dict(sorted(hops.items()))
            rows = [r for it in c["intro"] for r in it["rows"]]
            s["spearman_btw_intros_pooled"] = netan._spearman([r["betweenness"] for r in rows],
                                                              [r["intros"] for r in rows])
            s["spearman_deg_intros_pooled"] = netan._spearman([r["degree"] for r in rows], [r["intros"] for r in rows])
            s["spearman_btw_intros_per_game"] = [it["spearman_btw_intros"] for it in c["intro"]]
            s["n_intro"] = sum(it["n_intro"] for it in c["intro"])
            s["n_learned"] = sum(it["n_learned"] for it in c["intro"])
            top = sorted(rows, key=lambda r: -r["intros"])[:5]
            s["top_introducers"] = top
        s["centrality"] = c["centrality"]
        cum = collections.defaultdict(float)
        for g in gs:
            acc = 0
            byr = {int(k): v for k, v in g["wins_by_round"].items()}
            for r_ in range(1, g["max_rounds"] + 1):
                acc += byr.get(r_, 0)
                cum[r_] += acc / g["n_agents"] / len(gs)
        s["cum_win_share"] = {r_: round(v, 4) for r_, v in cum.items()}
        out[c["run"]] = s
    return out


def examples(res: list[dict[str, Any]], k: int) -> dict[str, Any]:
    ex: dict[str, Any] = {}
    for c in res:
        e: dict[str, list] = collections.defaultdict(list)
        for gsp in c["gossip"]:
            for x in gsp["claim_examples"]:
                e[x["cat"]].append(f"[{gsp['label']} r{x['round']} {x['sender']}->{x['r'] or 'board'} re {x['subj']}] {x['text'][:220]}")
            for x in gsp["intros"]:
                e["intro"].append(f"[{gsp['label']} r{x['round']} {x['sender']}->{x['receiver'] or 'board'} re {x['subject']}] {x['text'][:220]}")
            for x in gsp["forwards"]:
                e["forward"].append(f"[{gsp['label']} r{x['round']} {x['sender']}->{x['receiver'] or 'board'}] {x['text'][:220]}")
            for p in gsp["chain_paths"]:
                e["chain"].append(" => ".join(f"r{x['round']} {x['sender']}->{x['r'] or 'board'} ({x['cat']} {x['subj']}): {x['text'][:120]}" for x in p))
        for rl in c["relays"]:
            for ch in rl["chains"]:
                e["relay_chain"].append(f"[{rl['label']}] copy {ch['copy']} hops {ch['hops']} origin_dist {ch['origin_dist']}: " + " -> ".join(ch["path"]))
        ex[c["run"]] = {cat: v[:k] for cat, v in e.items()}
        ex[c["run"]]["_counts"] = {cat: len(v) for cat, v in e.items()}
    return ex


def figures(summ: dict[str, Any], figdir: Path, res: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab0 = {"complete_sim": "A complete", "er3_contacts_sim": "B ER3 contacts",
            "er3_known_sim": "C ER3 known_ids", "ba_known_sim": "D BA known_ids"}
    col0 = {"complete_sim": "#4C72B0", "er3_contacts_sim": "#DD8452", "er3_known_sim": "#55A868",
            "ba_known_sim": "#8172B3"}
    labels = {r: v for r in summ for k, v in lab0.items() if r.endswith(k)}
    cols = {r: v for r in summ for k, v in col0.items() if r.endswith(k)}
    tag = next(iter(summ)).split("_")[0]
    out = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for run, s in summ.items():
        lab, col = labels.get(run, run), cols.get(run, "gray")
        xs = sorted(s["cum_win_share"])
        ax[0].plot(xs, [s["cum_win_share"][x] for x in xs], label=lab, color=col, lw=2)
        if "coverage_by_round" in s:
            cx = sorted(s["coverage_by_round"])
            ax[2].plot(cx, [s["coverage_by_round"][x] for x in cx], label=lab, color=col, lw=2)
    ax[0].set(xlabel="round", ylabel="share of agents that have won", title=f"Winners over time ({next(iter(summ.values()))['n_games']} worlds)")
    ax[0].legend(fontsize=8)
    names = list(summ)
    for j, k in enumerate(("win_rate", "welfare", "needs_filled")):
        for i, run in enumerate(names):
            m = summ[run][k]
            scale = 3 if k == "needs_filled" else 1
            y = (m["mean"] or 0) / scale
            err = [[y - (m["lo"] or y * scale) / scale], [(m["hi"] or y * scale) / scale - y]]
            ax[1].bar(j * (len(names) + 1) + i, y, color=cols.get(run, "gray"), yerr=err, capsize=3,
                      label=labels.get(run, run) if j == 0 else None)
    ax[1].set_xticks([(len(names) - 1) / 2 + j * (len(names) + 1) for j in range(3)],
                     ["win rate", "welfare", "needs filled / 3"])
    ax[1].set(title="End of game (mean, t95 CI over worlds)", ylim=(0, 1))
    ax[2].set(xlabel="round", ylabel="mean share of others whose name is known",
              title="Name diffusion (known_ids)", ylim=(0, 1))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    p = figdir / f"network_{tag}_outcomes.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))
    # fill rate by graph distance
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (run, s) in enumerate(summ.items()):
        ds = [d for d in s["fill_by_distance"] if d not in ("None",)]
        xs = [int(d) for d in ds]
        ax.plot(xs, [s["fill_by_distance"][d]["fill"] for d in ds], marker="o", color=cols.get(run, "gray"),
                label=labels.get(run, run) + (" (ER-graph distance, placebo)" if run.endswith("complete_sim") else ""))
        for d in ds:
            ax.annotate(str(s["fill_by_distance"][d]["n"]), (int(d), s["fill_by_distance"][d]["fill"]),
                        fontsize=6, xytext=(2, 3), textcoords="offset points", color=cols.get(run, "gray"))
    ax.set(xlabel="graph distance to nearest initial holder of the needed key (0 = own key)",
           ylabel="share of needs filled at the end", title="Need fill rate by distance (n annotated)", ylim=(0, 1.05))
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / f"network_{tag}_distance.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref-graph", default=None, help="run whose graphs give the placebo distances for complete runs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--at-round", type=int, default=None, help="analyse every game as of the end of this round")
    ap.add_argument("--worlds", default=None, help="game_idx range to include, e.g. 0-3")
    a = ap.parse_args(argv)
    worlds = None
    if a.worlds:
        lo, _, hi = a.worlds.partition("-")
        worlds = set(range(int(lo), int(hi or lo) + 1))
    ref = None
    if a.ref_graph:
        ref = {g["game_idx"]: g["graph"] for g in netan.load_run(Path(a.ref_graph))}
    res = [analyse(Path(r), ref, a.at_round, worlds) for r in a.runs]
    summ = summarise(res)
    ex = examples(res, a.examples)
    for run, s in summ.items():
        print(f"== {run} ({s['n_games']} games, {s['partial']} partial, mode {s['mode']})")
        for k in ("win_rate", "welfare", "needs_filled", "rounds", "n_relay_events", "n_returns", "n_relay_delivered",
                  "n_chains", "chains_beyond_nbhd"):
            print(f"  {k}: {s[k]}")
        for k in s:
            if k.startswith("d_"):
                print(f"  {k}: {s[k]}")
        print(f"  fill_by_distance: {s['fill_by_distance']}")
        print(f"  win_by_max_distance: {s['win_by_max_distance']}")
        print(f"  gossip: {s['gossip']}")
        print(f"  claims: {s['claims']}  per1000: {s['per_1000_msgs']}")
        for k in ("learn_channels", "learn_hops", "spearman_btw_intros_pooled", "spearman_deg_intros_pooled",
                  "spearman_btw_intros_per_game", "n_intro", "n_learned"):
            if k in s:
                print(f"  {k}: {s[k]}")
        if "coverage_by_round" in s:
            cv = s["coverage_by_round"]
            print("  coverage: " + " ".join(f"r{r}={cv[r]:.2f}" for r in sorted(cv) if r in (0, 1, 2, 3, 5, 10, 20, 30, 40)))
        print(f"  centrality: {s['centrality']}")
        print(f"  example counts: {ex[run]['_counts']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summ, "examples": ex,
                                            "games": {c["run"]: c["games"] for c in res}}, indent=1, default=str))
    if a.figs:
        print(figures(summ, Path(a.figs), res))


if __name__ == "__main__":
    main()
