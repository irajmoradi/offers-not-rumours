"""Arm DB raw-reply diagnostic (2 seeds). Usage: analyze_db_raw.py OUT_DIR RAW_JSONL
For DB1 (rule on) vs DB0: among calls whose prompt shows a private offer delivered THIS round or last round
(a "Private offer" message from round >= current-1), does the raw reply mention it (offer/bribe/report/overseer/
private/defect against), does it contain the "report" field at all, and does any reply contain "report" but fail to
parse (parsed is None or lacks a dict "report")? Writes counts + a few short excerpts (agent replies only)."""
import sys, os, json, re, collections
OUT, RAW = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
MENT = re.compile(r"offer|bribe|report|overseer|private|defect against|reward", re.I)
OFF = re.compile(r"#\d+ round (\d+), from agent (\d+): \"Private offer")
C = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for l in open(RAW):
    g = json.loads(l); br = g["branch"]
    for c in g["calls"]:
        if c["round"] < 2: continue
        rounds = [int(m.group(1)) for m in OFF.finditer(c["prompt"])]
        recent = any(r >= c["round"] - 1 for r in rounds)
        rep = c["reply"] or ""; P = c["parsed"]
        k = C[br]; k["calls"] += 1; k["empty_reply"] += (not rep.strip()); k["unparsed"] += (P is None)
        k["menu_shown"] += ('"report"' in c["prompt"])
        k["reply_has_report_word"] += bool(re.search(r"report", rep, re.I))
        k["report_field_parsed"] += isinstance(P, dict) and isinstance(P.get("report"), dict)
        k["report_word_but_no_parsed_field"] += bool(re.search(r'"report"', rep)) and not (isinstance(P, dict) and isinstance(P.get("report"), dict))
        if recent:
            k["calls_with_recent_offer"] += 1; k["recent_offer_empty_reply"] += (not rep.strip())
            k["recent_offer_mentions"] += bool(MENT.search(rep))
            k["recent_offer_defect"] += isinstance(P, dict) and P.get("action") == "defect"
            if MENT.search(rep) and len(ex[br]) < 6: ex[br].append(dict(round=c["round"], agent=c["agent"], reply=rep[:300]))
        if re.search(r'"report"', rep) and len(ex[br + "_report"]) < 6:
            ex[br + "_report"].append(dict(round=c["round"], agent=c["agent"], parsed_ok=isinstance(P, dict), reply=rep[:400]))
res = {b: dict(v) for b, v in C.items()}
json.dump(dict(counts=res, excerpts=ex), open(os.path.join(OUT, "db_raw_diagnostic.json"), "w"), indent=1)
for b, v in sorted(res.items()): print(b, json.dumps(v))
for b, v in ex.items():
    print("==", b)
    for e in v[:4]: print("  ", json.dumps(e)[:420])
