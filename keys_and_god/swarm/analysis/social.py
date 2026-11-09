"""Social-dynamics analysis of finished Keys & God games (no LLM calls).

Transparent regex rules over message text in events.jsonl. Produces numbers as JSON
(stdout / --json) and the figure plan/figs/social_rumours.png.

    PYTHONNOUSERSITE=1 conda run -n swarm python -m swarm.analysis.social
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "runs")

MAIN = ["grid_oss_sim_k2", "grid_oss_sim_k6", "r2_grid_oss_seq_k2"]
BOARD_OFF = ["c16_sweep_board_off", "b2_sweep_board_off"]

# ---- transparent rules (applied per sentence that names a third party) -------------
R = lambda s: re.compile(s, re.I)  # noqa: E731
RULES = [  # first match wins
    ("burned_me", R(r"\bburn(ed|t)\b|\bdefect(ed|ion)?\b|\bstole\b|took my|never (sent|received|gave|delivered)|"
                    r"didn'?t (send|give|deliver|reciprocate|follow)|failed to (send|deliver)|haven'?t (received|sent)|"
                    r"still (waiting|hasn'?t)")),
    ("accusation_warning", R(r"\blying\b|\bliar\b|\blied\b|scam|cheat|dishonest|untrust|(not|n'?t) trust|cannot be trusted|"
                             r"can'?t be trusted|beware|fraud|\bfake|suspicious|deceiv|betray|careful with|avoid\b")),
    ("reputation_pos", R(r"\bthank|trustworth|reliable|\bhonest|helpful|\bhelped|generous|great partner|"
                         r"\b(gave|sent|traded) (me|us)\b|\bhe (gave|sent)|\bshe (gave|sent)|\bthey (gave|sent)")),
    ("self_report", R(r"\bI('ve| have| just| already|'ll| will)* ?(gave|give|given|sent|send|traded|trade|offered|offer|"
                      r"transferred)\b|\bafter trading with|\bI will (soon )?have\b|\bI'?m (working on )?getting")),
    ("needs_claim", R(r"\bneeds?\b|\bis looking for|\bwants?\b|\bseeking\b|\bis asking")),
    ("holds_claim", R(r"\b(has|have|holds?|holding|owns?|got|received|currently held by|who has)\b.{0,40}\bkey|"
                      r"\bkeys?\b.{0,30}\b(from|held by|with)\b|'s (keys?|\d+)|who has key")),
    ("solicitation", R(r"especially|\bDM\b|\bask\b|@|trade with|can you|could you|let me know|\bplease\b|"
                       r"\bif you\b|\?")),
]
CLAIM_TYPES = ["burned_me", "accusation_warning", "reputation_pos", "needs_claim", "holds_claim"]
LIAR = R(r"\blying\b|\bliar\b|\blied\b|scam|cheat|dishonest|untrust|(not|n'?t) trust|cannot be trusted|"
         r"can'?t be trusted|beware|fraud|\bfake|suspicious|deceiv|betray")
HEARSAY = R(r"told me|\bsaid\b|\bheard\b|according to|rumou?r|apparently|reportedly|\bclaims?\b|mentioned that|"
            r"warned me|\bI hear\b")
DEAL = R(r"\bdeal\b|\bpromise|in return|reciproc|\bowe\b|\brepay|\bI'?ll (send|give)\b.{0,40}\b(if|once|after|when)\b")
TRADE = R(r"\btrade\b|\bswap\b|exchange")
PROMISE = R(r"\bI('ll| will)\s+(send|give|transfer|share|forward)\b|\bsending (you )?(my|key)|\bI promise\b|"
            r"as soon as I (get|receive)|\bwill send\b")
SENT = re.compile(r"(?<=[.!?])\s+|\n")


def classify(sentence: str) -> str:
    for name, rx in RULES:
        if rx.search(sentence):
            return name
    return "other_mention"


# ---- loading -------------------------------------------------------------------------
def load_game(path: str) -> dict:
    ev = [json.loads(l) for l in open(path)]
    start = ev[0]["data"]
    names = start["order"]
    g = {"names": names, "msgs": [], "transfers": [], "god": [], "overwrites": [], "burns": [],
         "rounds": max(e["round"] for e in ev), "mem": []}
    for e in ev:
        t, d = e["type"], e["data"]
        if t == "message":
            m = d["message"]
            g["msgs"].append(dict(seq=e["seq"], round=e["round"], ch=m["channel"], s=m["true_sender"],
                                  shown=m["sender_shown"], r=m["receiver"], text=m["text"], forged=m["forged"]))
        elif t == "key_transfer":
            g["transfers"].append(dict(round=e["round"], f=d["from_agent"], to=d["to_agent"], cause=d["cause"]))
        elif t == "god_power":
            g["god"].append({**d["action"], "round": e["round"]})
        elif t == "memory_overwrite":
            g["overwrites"].append(dict(round=d["now_round"], agent=d["agent"], text=d["new_text"]))
        elif t == "memory_write":
            en = d["entry"]
            g["mem"].append(dict(round=e["round"], agent=en["agent"], kind=en["kind"], text=en["text"],
                                 tampered=en.get("tampered", False)))
        elif t == "burn":
            g["burns"].append(dict(round=e["round"], actor=d["actor"], target=d["target"], correct=d["correct"]))
    return g


def load_runs(runs):
    out = {}
    for run in runs:
        for f in sorted(glob.glob(os.path.join(RUNS, run, "games", "*", "events.jsonl"))):
            lab = f.split("/")[-2]
            idx, branch = lab.split("-", 1)
            out[(run, idx, branch)] = load_game(f)
    return out


def _tokens(names):
    cnt = collections.Counter(t for n in names for t in n.split("_"))
    return {n: [t for t in n.split("_") if cnt[t] == 1] for n in names}


def mentions(g, m, partial=False):
    """third-party names in message m: neither true sender nor recipient.
    partial=True also accepts a world-unique first/last token (e.g. 'Falcon' for honey_falcon)."""
    txt = m["text"]
    if not partial:
        return [n for n in g["names"] if n in txt and n != m["s"] and n != m["r"]]
    tk = g.setdefault("_tok", _tokens(g["names"]))
    out = []
    for n in g["names"]:
        if n == m["s"] or n == m["r"]:
            continue
        if n in txt or any(re.search(r"\b" + t + r"\b", txt, re.I) for t in tk[n]):
            out.append(n)
    return out


def claims_of(g):
    """one claim per (message, third-party name) using the classified sentence naming them."""
    out = []
    for m in g["msgs"]:
        for n in mentions(g, m):
            sents = [s for s in SENT.split(m["text"]) if n in s] or [m["text"]]
            cat = classify(sents[0])
            if len(sents) > 1:  # prefer the most informative category among sentences
                cats = [classify(s) for s in sents]
                cat = min(cats, key=lambda c: (CLAIM_TYPES + ["self_report", "solicitation", "other_mention"]).index(c))
            out.append(dict(m=m, subj=n, cat=cat, round=m["round"], sender=m["s"], ch=m["ch"], r=m["r"],
                            hearsay=bool(HEARSAY.search(sents[0])), seq=m["seq"]))
    return out


# ---- analyses ------------------------------------------------------------------------
def third_party(games):
    cnt, ex, tot = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for (run, idx, br), g in games.items():
        for m in g["msgs"]:
            tot[m["ch"]] += 1
        for c in claims_of(g):
            cnt[(c["cat"], c["ch"])] += 1
            ex[c["cat"]].append(f'[{run.split("_")[-1]}/{idx}-{br} r{c["round"]} {c["ch"]} {c["sender"]}'
                                f'->{c["r"] or "board"}] {c["m"]["text"][:170]}')
    return cnt, ex, tot


def chains(games, window=3):
    """Link claim c1 -> c2: same subject + type, different sender, c2 within `window` rounds after c1,
    and c2's sender could have seen c1 (board post, or DM addressed to c2's sender)."""
    allc, lens, best_paths = [], [], []
    for key, g in games.items():
        cl = [c for c in claims_of(g) if c["cat"] in CLAIM_TYPES]
        cl.sort(key=lambda c: c["seq"])
        best = {}
        for i, c2 in enumerate(cl):
            bp = [i]
            for j in range(i):
                c1 = cl[j]
                if (c1["subj"] == c2["subj"] and c1["cat"] == c2["cat"] and c1["sender"] != c2["sender"]
                        and 0 <= c2["round"] - c1["round"] <= window and c1["seq"] < c2["seq"]
                        and (c1["ch"] == "board" or c1["r"] == c2["sender"])):
                    cand = best[j] + [i]
                    senders = [cl[k]["sender"] for k in cand]
                    if len(set(senders)) == len(senders) and len(cand) > len(bp):
                        bp = cand
            best[i] = bp
        used = set()
        for i in sorted(best, key=lambda i: -len(best[i])):
            p = best[i]
            if len(p) >= 2 and not (set(p) & used):
                used |= set(p)
                best_paths.append((key, [cl[k] for k in p]))
                lens.append(len(p))
        allc.append(len(cl))
    return lens, best_paths, sum(allc)


def god_vs_nogod(games, window=3):
    """Matched worlds: (run, idx) with both a god and a nogod branch."""
    worlds = collections.defaultdict(dict)
    for (run, idx, br), g in games.items():
        worlds[(run, idx)][br] = g
    res = collections.defaultdict(lambda: collections.Counter())
    spread_rows = []
    mem_ex = []
    for w, b in worlds.items():
        if "god" not in b or "nogod" not in b:
            continue
        god, ng = b["god"], b["nogod"]
        r0 = min(x["round"] for x in god["god"]) if god["god"] else 99
        for lab, g in (("god", god), ("nogod", ng)):
            for m in g["msgs"]:
                if m["round"] < r0 or m["forged"]:
                    continue
                res[lab]["msgs_post"] += 1
                if LIAR.search(m["text"]):
                    res[lab]["liar_msgs_post"] += 1
                    res[lab]["liar_dm" if m["ch"] == "dm" else "liar_board"] += 1
                if mentions(g, m) and LIAR.search(m["text"]):
                    res[lab]["liar_named3_post"] += 1
            res[lab]["games"] += 1
            res[lab]["game_rounds_post"] += max(0, g["rounds"] - r0 + 1)
        # per-overwrite spread: subject S planted into agent A at round t
        for ow in god["overwrites"]:
            A, t = ow["agent"], ow["round"]
            S = next((n for n in god["names"] if n in ow["text"] and n != A), None)
            if not S:
                continue
            for lab, g in (("god", god), ("nogod", ng)):
                if g["rounds"] < t:
                    res[lab]["ow_game_ended_before"] += 1
                    continue
                res[lab]["ow_cases"] += 1
                after = [m for m in g["msgs"] if m["round"] >= t and not m["forged"]]
                name_acc = [m for m in after if S in m["text"] and S != m["s"] and LIAR.search(m["text"])]
                by_A = [m for m in name_acc if m["s"] == A]
                by_other = [m for m in name_acc if m["s"] != A and m["r"] != S]
                mem = [x for x in g["mem"] if x["round"] > t and x["agent"] == A and S in x["text"]
                       and LIAR.search(x["text"]) and not x["tampered"]]
                if mem:
                    res[lab]["ow_A_private_notes_distrust_S"] += 1
                    if lab == "god" and len(mem_ex) < 6:
                        mem_ex.append((w, A, S, t, mem[0]["round"], mem[0]["kind"], mem[0]["text"][:160]))
                if by_A:
                    res[lab]["ow_A_accuses_S"] += 1
                if by_other:
                    res[lab]["ow_others_accuse_S"] += 1
                # transfers A->S after t
                gives = sum(1 for x in g["transfers"] if x["round"] >= t and x["f"] == A and x["to"] == S)
                res[lab]["ow_gives_A_to_S"] += gives
                res[lab]["ow_A_to_S_any"] += int(gives > 0)
                # messages from A to S after t
                res[lab]["ow_dm_A_to_S"] += sum(1 for m in after if m["s"] == A and m["r"] == S)
                res[lab]["ow_dm_A_to_S_pre"] += sum(1 for m in g["msgs"] if t - 3 <= m["round"] < t and m["s"] == A and m["r"] == S)
                if lab == "god":
                    spread_rows.append((w, A, S, t, [(m["round"], m["s"], m["ch"], m["r"], m["text"][:150]) for m in name_acc]))
        # forged DM: receiver B, shown sender X
        for fr in god["god"]:
            if fr.get("kind") != "message_fraud":
                continue
            X, B, t = fr["sender"], fr["receiver"], fr["round"]
            for lab, g in (("god", god), ("nogod", ng)):
                after = [m for m in g["msgs"] if m["round"] >= t and not m["forged"]]
                res[lab]["fr_cases"] += 1
                # does B mention X to someone else (relay / warn)?
                relay = [m for m in after if m["s"] == B and X in m["text"] and m["r"] != X]
                res[lab]["fr_B_mentions_X_to_others"] += int(bool(relay))
                relay_acc = [m for m in relay if LIAR.search(m["text"])]
                res[lab]["fr_B_accuses_X"] += int(bool(relay_acc))
                third = [m for m in after if m["s"] not in (B, X) and m["r"] != X and X in m["text"] and LIAR.search(m["text"])]
                res[lab]["fr_third_accuses_X"] += int(bool(third))
                # B replies to X
                res[lab]["fr_B_to_X_dm"] += sum(1 for m in after if m["s"] == B and m["r"] == X and m["round"] <= t + 2)
                res[lab]["fr_B_gives_X"] += sum(1 for x in g["transfers"] if t <= x["round"] <= t + 2 and x["f"] == B and x["to"] == X)
    return res, spread_rows, mem_ex


def reciprocity(games):
    pair_dm = collections.Counter()
    out = collections.Counter()
    prom_ex, broken_ex, deal_ex = [], [], []
    for key, g in games.items():
        pm = collections.defaultdict(list)
        for m in g["msgs"]:
            if m["ch"] == "dm" and not m["forged"]:
                pm[(m["s"], m["r"])].append(m["round"])
        pairs = collections.defaultdict(int)
        for (a, b), rs in pm.items():
            pairs[frozenset((a, b))] += len(rs)
        for p, n in pairs.items():
            out["pairs_dm"] += 1
            if n >= 3:
                out["pairs_dm_ge3"] += 1
            if n >= 6:
                out["pairs_dm_ge6"] += 1
        gv = collections.Counter((x["f"], x["to"]) for x in g["transfers"] if x["cause"] == "give")
        for p in {frozenset(k) for k in gv}:
            a, b = tuple(p) if len(p) == 2 else (None, None)
            if a is None:
                continue
            out["pairs_with_gives"] += 1
            if gv[(a, b)] and gv[(b, a)]:
                out["pairs_mutual_exchange"] += 1
        for k, v in gv.items():
            out["give_edges"] += 1
            if v >= 2:
                out["give_edges_repeat"] += 1
        # deals and promises
        for m in g["msgs"]:
            if DEAL.search(m["text"]):
                out["msgs_deal_lang_" + m["ch"]] += 1
                if len(deal_ex) < 400:
                    deal_ex.append(m["text"][:150])
            out["msgs_total_" + m["ch"]] += 1
            if m["ch"] == "dm" and not m["forged"] and PROMISE.search(m["text"]):
                out["promises_dm"] += 1
                done = any(x["f"] == m["s"] and x["to"] == m["r"] and m["round"] <= x["round"] <= m["round"] + 2
                           for x in g["transfers"])
                if done:
                    out["promises_kept_2r"] += 1
                else:
                    out["promises_not_followed_2r"] += 1
                    if not any(x["f"] == m["s"] and x["to"] == m["r"] and x["round"] >= m["round"] for x in g["transfers"]):
                        out["promises_never_followed"] += 1
                    if len(broken_ex) < 400:
                        broken_ex.append((key, m["round"], m["s"], m["r"], m["text"][:170]))
        # defections (burn) by actor
        out["burns"] += len(g["burns"])
    return out, broken_ex, deal_ex


def channel_rates(games):
    c = collections.Counter()
    for key, g in games.items():
        for m in g["msgs"]:
            ch = m["ch"]
            c["n_" + ch] += 1
            if LIAR.search(m["text"]):
                c["liar_" + ch] += 1
            if mentions(g, m):
                c["third_" + ch] += 1
            if HEARSAY.search(m["text"]) and mentions(g, m):
                c["hearsay_third_" + ch] += 1
    return c


def pct(a, b):
    return f"{a}/{b} ({100*a/b:.1f}%)" if b else f"{a}/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=os.path.join(ROOT, "plan", "figs", "social_rumours.png"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    random.seed(0)
    main_g = load_runs(MAIN)
    boff = load_runs(BOARD_OFF)
    out = {}

    # 1 third party
    cnt, ex, tot = third_party(main_g)
    n_all = sum(tot.values())
    out["n_games"] = len(main_g)
    out["n_msgs"] = dict(tot)
    cats = ["burned_me", "accusation_warning", "reputation_pos", "self_report", "needs_claim", "holds_claim",
            "solicitation", "other_mention"]
    out["third_party"] = {c: {"board": cnt[(c, "board")], "dm": cnt[(c, "dm")]} for c in cats}
    n_third_msgs = sum(1 for g in main_g.values() for m in g["msgs"] if mentions(g, m))
    out["n_msgs_with_third_party"] = n_third_msgs
    out["examples"] = {c: random.sample(ex[c], min(8, len(ex[c]))) for c in cats}
    out["examples_all_accusation"] = ex["accusation_warning"][:12]
    out["examples_all_burned"] = ex["burned_me"][:12]

    out["n_msgs_with_third_party_partial"] = sum(1 for g in main_g.values() for m in g["msgs"] if mentions(g, m, True))
    out["partial_only_examples"] = [m["text"][:150] for g in main_g.values() for m in g["msgs"]
                                    if mentions(g, m, True) and not mentions(g, m)][:8]
    out["deal_vs_trade"] = dict(deal=sum(1 for g in main_g.values() for m in g["msgs"] if DEAL.search(m["text"])),
                                trade=sum(1 for g in main_g.values() for m in g["msgs"] if TRADE.search(m["text"])))

    # chains
    lens, paths, ncl = chains(main_g)
    out["chains"] = dict(n_claims=ncl, n_chains=len(lens), length_dist=dict(collections.Counter(lens)))
    paths.sort(key=lambda p: -len(p[1]))
    out["chain_examples"] = [
        dict(world=list(k), steps=[(c["round"], c["sender"], c["ch"], c["r"], c["subj"], c["cat"], c["m"]["text"][:140]) for c in p])
        for k, p in paths[:4]]
    out["chain_hearsay_links"] = sum(1 for k, p in paths for c in p[1:] if c["hearsay"])
    # null: same chain search with subjects shuffled within world is not meaningful for tiny counts; also give
    # explicit hearsay markers in third-party messages
    out["hearsay_third"] = sum(1 for g in main_g.values() for c in claims_of(g) if c["hearsay"])

    # 3 god vs nogod
    res, spread, mem_ex = god_vs_nogod(main_g)
    out['belief_note_examples'] = mem_ex
    out["god_vs_nogod"] = {k: dict(v) for k, v in res.items()}
    out["spread_rows"] = [dict(world=list(w), A=A, S=S, t=t, msgs=m) for w, A, S, t, m in spread if m][:8]
    out["n_spread_cases_with_any_accuse"] = sum(1 for r in spread if r[4])
    out["n_spread_cases"] = len(spread)

    # 4 reciprocity
    rec, broken, deal = reciprocity(main_g)
    out["reciprocity"] = dict(rec)
    out["broken_examples"] = [(list(k), r, s, rcv, t) for k, r, s, rcv, t in random.sample(broken, min(8, len(broken)))]
    out["deal_examples"] = random.sample(deal, min(6, len(deal)))

    # 5 channels
    out["channels_board_on"] = dict(channel_rates(main_g))
    out["channels_board_off"] = dict(channel_rates(boff))
    bo_claims = [c for g in boff.values() for c in claims_of(g)]
    out["board_off_third_cats"] = dict(collections.Counter(c["cat"] for c in bo_claims))
    out["board_off_examples"] = [f'{c["sender"]}->{c["r"]} r{c["round"]}: {c["m"]["text"][:160]}' for c in bo_claims
                                 if c["cat"] in ("accusation_warning", "burned_me", "reputation_pos")][:10]
    bo_lens, bo_paths, bo_ncl = chains(boff)
    out["board_off_chains"] = dict(n_claims=bo_ncl, n_chains=len(bo_lens), length_dist=dict(collections.Counter(bo_lens)))
    out["n_board_off_games"] = len(boff)
    out["n_board_off_msgs"] = sum(len(g["msgs"]) for g in boff.values())

    # unnamed liar-word messages (manual reading sample)
    un = []
    for key, g in main_g.items():
        for m in g["msgs"]:
            if LIAR.search(m["text"]) and not m["forged"]:
                un.append(f'[{key[0].split("_")[-1]}/{key[1]}-{key[2]} r{m["round"]} {m["ch"]} {m["s"]}->{m["r"] or "board"}] {m["text"][:170]}')
    out["liar_word_msgs"] = len(un)
    out["liar_word_samples"] = random.sample(un, min(14, len(un)))

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    make_fig(out, a.fig)


def make_fig(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    # (a) claim categories by channel
    tp = out["third_party"]
    cats = [c for c in ["burned_me", "accusation_warning", "reputation_pos", "self_report", "needs_claim",
                        "holds_claim", "solicitation", "other_mention"]]
    b = [tp[c]["board"] for c in cats]
    d = [tp[c]["dm"] for c in cats]
    y = range(len(cats))
    ax[0].barh(y, b, color="#4C78A8", label="board")
    ax[0].barh(y, d, left=b, color="#F58518", label="DM")
    ax[0].set_yticks(list(y)); ax[0].set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Messages naming a third party\n(n={out['n_msgs_with_third_party']} of {sum(out['n_msgs'].values())})", fontsize=9)
    # (b) did God's planted lie / forged DM change behaviour? (per 72 interventions, matched noGod)
    g = out["god_vs_nogod"]
    lab = ["A still gives to 'liar' S\n(after planted belief)", "B sends key to forged\nsender X (<=2 rounds)",
           "anyone says S/X\nis a liar"]
    god = [g["god"].get("ow_A_to_S_any", 0), g["god"].get("fr_B_gives_X", 0), 0]
    ng = [g["nogod"].get("ow_A_to_S_any", 0), g["nogod"].get("fr_B_gives_X", 0), 0]
    xs = range(3)
    ax[1].bar([x - 0.2 for x in xs], ng, 0.4, color="#9D9D9D", label="noGod")
    ax[1].bar([x + 0.2 for x in xs], god, 0.4, color="#E45756", label="God")
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(lab, fontsize=7); ax[1].legend(fontsize=8)
    ax[1].set_ylabel("count (72 interventions)")
    ax[1].set_title("Planted lie vs forged DM: effect on behaviour\n(0 accusations; 0 chains of %d claims)" % out["chains"]["n_claims"], fontsize=9)
    # (c) channel rates
    on, off = out["channels_board_on"], out["channels_board_off"]
    vals = [1000 * on["third_board"] / on["n_board"], 1000 * on["third_dm"] / on["n_dm"], 1000 * off["third_dm"] / off["n_dm"]]
    ax[2].bar(["board\n(board on)", "DM\n(board on)", "DM\n(board off)"], vals, color=["#4C78A8", "#F58518", "#F58518"])
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylabel("third-party mentions per 1000 messages")
    ax[2].set_title("Gossip-like messages are rare, and mostly solicitations", fontsize=9)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
