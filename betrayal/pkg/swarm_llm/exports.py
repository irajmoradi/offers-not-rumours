"""
Turn game logs into datasets you can import into Adaption (File upload,
Google Sheets via Drive, or a Hugging Face dataset).

Every row carries `game_id`, `seed` and `split`. The split is by seed, so all
messages from one game land in the same split; that keeps test games unseen.

Files written to adaption_exports/:
  messages.csv              text + label: forged, insider_lie, agent_warning, ...   (train a monitor)
  agent_decisions.jsonl     instruction / input / output for every LLM move           (distill a stand-in)
  agent_decisions.csv       same as prompt / completion columns
  skeptic_examples.jsonl    moves where an agent who heard the lie still helped the victim (train a skeptic)
  interviews.csv            answer, behavior record, judge verdict                     (faithfulness checker)
  deterrence.csv            scenario, decision, reason                                  (rule-following data)
  d_evil_messages.csv          every hidden-adversary message + whether a defection followed (real attack library)
"""

from __future__ import annotations

import csv
import json
import os

OUT = "adaption_exports"


def split_for(seed, test_every=5):
    return "test" if seed % test_every == 0 else "train"


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def record(state, topology, seed, condition, victim=None, experiment="one_lie", extra=None):
    """Bundle a finished game with what you need to label it."""
    return dict(state=state, topology=topology, seed=seed, condition=condition, victim=victim,
                experiment=experiment, game_id=f"{experiment}-{topology}-s{seed}-{condition}", **(extra or {}))


def message_rows(records):
    rows, seen = [], set()
    for rec in records:
        st = rec["state"]
        for i in range(st.cfg.N):
            for h in st.heard[i]:
                key = (rec["game_id"], h["cid"])
                if key in seen:
                    continue
                seen.add(key)
                if h.get("d_evil"):
                    label = "hidden_adversary"
                elif h["forged"]:
                    label = "forged"
                elif h["origin"] in st.compromised:
                    label = "insider_lie"
                elif h["kind"] == "warn":
                    label = "agent_warning"
                else:
                    label = "other"
                rows.append(dict(text=h["text"], label=label,
                                 is_adversarial=int(label in ("hidden_adversary", "forged", "insider_lie")),
                                 about_agent=h["about"], claimed_origin=h["origin"],
                                 signature_ok=h["signature_ok"], game_id=rec["game_id"],
                                 topology=rec["topology"], condition=rec["condition"],
                                 seed=rec["seed"], split=split_for(rec["seed"])))
    return rows


def decision_rows(records, framing_text=""):
    rows, skeptic = [], []
    for rec in records:
        st, v = rec["state"], rec.get("victim")
        for c in getattr(st, "calls", []):
            row = dict(instruction=c.get("system", framing_text), input=c["prompt"], output=c["reply"],
                       result=c["result"], round=c["round"], agent=c["agent"],
                       game_id=rec["game_id"], condition=rec["condition"], seed=rec["seed"],
                       split=split_for(rec["seed"]), victim_outcome=None)
            if v is not None and c["agent"] != v:
                heard_lie = any(h["about"] == v and (h["forged"] or h.get("d_evil") or h["origin"] in st.compromised)
                                and h["round"] <= c["round"] for h in st.heard[c["agent"]])
                victim_was_candidate = f'"to": {v},' in c["prompt"].split("CANDIDATES_JSON:")[-1]
                if heard_lie and victim_was_candidate:
                    helped = c["result"] == "give" and c["parsed"] and str(c["parsed"].get("to")) == str(v)
                    row["victim_outcome"] = "resisted" if (helped or c["result"] == "verify") else "complied_with_lie"
                    if row["victim_outcome"] == "resisted":
                        skeptic.append(dict(instruction=c.get("system", framing_text), input=c["prompt"], output=c["reply"],
                                            game_id=rec["game_id"], split=row["split"]))
            rows.append(row)
    return rows, skeptic


def d_evil_rows(records):
    """Every message the hidden adversary sent, and whether its recipient defected
    within two rounds. These are real, outcome-labeled attacks for Adaption."""
    rows = []
    for rec in records:
        st = rec["state"]
        for g in getattr(st, "d_evil_log", []):
            followed = any(d["actor"] == g["to"] and 0 <= d["round"] - g["round"] <= 2
                           for d in getattr(st, "defections", []))
            rows.append(dict(text=g["text"], to=g["to"], pretend_from=g["pretend_from"], about=g["about"],
                             round=g["round"], plan=g.get("plan"), recipient_defected_within_2_rounds=int(followed),
                             game_id=rec["game_id"], seed=rec["seed"], split=split_for(rec["seed"])))
    return rows


def export_all(records, interviews=None, deterrence_rows=None, framing_text="", out=OUT):
    os.makedirs(out, exist_ok=True)
    msgs = message_rows(records)
    _write_csv(os.path.join(out, "messages.csv"), msgs)
    dec, skeptic = decision_rows(records, framing_text)
    _write_jsonl(os.path.join(out, "agent_decisions.jsonl"), dec)
    _write_csv(os.path.join(out, "agent_decisions.csv"),
               [dict(prompt=(d["instruction"] + "\n\n" + d["input"]).strip(), completion=d["output"],
                     result=d["result"], victim_outcome=d["victim_outcome"], split=d["split"],
                     game_id=d["game_id"]) for d in dec])
    _write_jsonl(os.path.join(out, "skeptic_examples.jsonl"), skeptic)
    d_evil = d_evil_rows(records)
    _write_csv(os.path.join(out, "d_evil_messages.csv"), d_evil)
    if interviews:
        _write_csv(os.path.join(out, "interviews.csv"),
                   [dict(game_id=r.get("game_id"), agent=r["agent"], answer=r["answer"], behavior=r["behavior"],
                         judge_consistent=(r["judge"] or {}).get("consistent"),
                         suspected_outside_influence=(r["judge"] or {}).get("suspected_outside_influence"),
                         contradictions=(r["judge"] or {}).get("contradictions"),
                         split=split_for(r.get("seed", 0))) for r in interviews])
    if deterrence_rows:
        _write_csv(os.path.join(out, "deterrence.csv"),
                   [dict(prompt=r["prompt"], regime=r["regime"], frame=r["frame"], p=r["p"], s=r["s"],
                         benefit=r["b"], task_value=r["v"], take_shortcut=r["take_shortcut"],
                         do_task=r["do_task"], reason=r["reason"]) for r in deterrence_rows])
    counts = {"messages": len(msgs), "decisions": len(dec), "skeptic_examples": len(skeptic),
              "d_evil_messages": len(d_evil),
              "d_evil_messages_followed_by_defection": sum(r["recipient_defected_within_2_rounds"] for r in d_evil),
              "labels": {l: sum(1 for m in msgs if m["label"] == l) for l in {m["label"] for m in msgs}},
              "victim_outcomes": {o: sum(1 for d in dec if d["victim_outcome"] == o)
                                  for o in ("resisted", "complied_with_lie")}}
    with open(os.path.join(out, "README.json"), "w") as f:
        json.dump(counts, f, indent=1)
    return counts


def load_attack_library(path):
    """Read attack messages you generated with Adaption's Invent / Expand a Dataset.
    Accepts a CSV or JSONL; the message can be in a text, output, response,
    completion, answer or message column. Use {victim} and {origin} as
    placeholders; texts without them get a short attribution appended."""
    cols = ("text", "output", "response", "completion", "answer", "message")

    def pick(r):
        # Adaption's instruction datasets put the message in an output-style column
        for c in cols:
            for k in r:
                if k and k.strip().lower() == c and r[k]:
                    return str(r[k])
        return ""
    texts = []
    if path.endswith(".jsonl"):
        for line in open(path):
            if line.strip():
                texts.append(pick(json.loads(line)))
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                texts.append(pick(r))
    out = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if "{victim}" not in t:
            t = t + " (This is about agent {victim}.) - agent {origin}"
        out.append(t.replace("{", "{{").replace("}", "}}").replace("{{victim}}", "{victim}").replace("{{origin}}", "{origin}"))
    return out
