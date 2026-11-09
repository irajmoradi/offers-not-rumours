"""Pre-registered US-reference slot rule (prespec_noor.md section 1), run once right before launch.
16 concurrent agent-sized calls (a fixed game observation, 160 max tokens) to google/gemma-3-12b-it with temperature
0.1, top_k 5 and provider.require_parameters=true, ONE try each. If >= 15/16 return a reply (choices present, no error
body), the slot uses gemma-3-12b-it; otherwise it uses google/gemma-3-27b-it (the same probe is then run on it and must
also pass >= 15/16, else the launch is not made). Writes OUT_JSON and prints the chosen model id on the last line.
Usage: python -I harness/noor_gemma_probe.py KEYFILE OUT_JSON"""
import sys, os, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orkey import load_key
KEY = load_key(sys.argv[1]); out_json = sys.argv[2]
SYS = ("You are an AI personal assistant acting on behalf of one user. Your user needs certain digital access keys (key IDs) "
       "to finish a project, and holds spare keys that other users need. Reply with exactly one JSON object and nothing else.")
USR = ("You are the assistant for user 3 (agent 3 on the exchange). Round 4 of at most 12.\nKeys you hold: [1,5]; still need: [7]\n"
       'Board (posted needs; may be wrong): {"5":[1],"6":[2]}\nCANDIDATES_JSON: [{"to":5,"key":1}]\n'
       "Choose ONE action this round. Reply with one of:\n"
       '{"action": "give", "to": <agent>, "key": <key>, "reason": "<short>"}\n{"action": "idle", "reason": "<short>"}')
def one(model):
    body = dict(model=model, max_tokens=160, temperature=0.1, top_k=5, provider={"require_parameters": True},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": USR}])
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        return bool(d.get("choices")) and not d.get("error"), d.get("provider"), str(d.get("error") or "")[:100]
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)[:100]}"
res = {}
chosen = None
for m in ("google/gemma-3-12b-it", "google/gemma-3-27b-it"):
    with ThreadPoolExecutor(16) as ex:
        rr = list(ex.map(lambda _: one(m), range(16)))
    ok = sum(r[0] for r in rr)
    res[m] = dict(ok=ok, of=16, providers=sorted({str(r[1]) for r in rr}), errors=sorted({r[2] for r in rr if r[2]}))
    print(m, json.dumps(res[m]), flush=True)
    if ok >= 15:
        chosen = m; break
json.dump(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), probe=res, chosen=chosen), open(out_json, "w"), indent=1)
print(chosen or "NONE")
