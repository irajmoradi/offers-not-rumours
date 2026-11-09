"""Noor spec item 2: per-model top_k verification. One tiny call per model with temperature 0.1, top_k 5 and
provider.require_parameters=true (so only providers that honour top_k may serve it). Prints HTTP status, served
provider, tokens and cost. The key is read by path via orkey.load_key and never printed.
Usage: python -I harness/noor_topk_check.py KEYFILE OUT_JSON MODEL [MODEL ...]"""
import sys, os, json, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orkey import load_key
keyfile, out_json, models = sys.argv[1], sys.argv[2], sys.argv[3:]
KEY = load_key(keyfile)
res = []
for m in models:
    body = dict(model=m, max_tokens=8, temperature=0.1, top_k=5, provider={"require_parameters": True},
                usage={"include": True},
                messages=[{"role": "system", "content": "Reply with exactly one JSON object and nothing else."},
                          {"role": "user", "content": 'Reply with {"action": "idle"}'}])
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "HTTP-Referer": "https://swarm-sprint.local", "X-Title": "swarm-sprint"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code, d = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code, d = e.code, {"error": e.read().decode(errors="replace")[:300]}
    except Exception as e:
        code, d = None, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    u = d.get("usage") or {}
    row = dict(model=m, status=code, provider=d.get("provider"), secs=round(time.time() - t0, 2),
               reply=((d.get("choices") or [{}])[0].get("message") or {}).get("content"),
               prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"), cost=u.get("cost"),
               error=d.get("error"))
    res.append(row); print(json.dumps(row), flush=True)
json.dump(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), params=dict(temperature=0.1, top_k=5, require_parameters=True), results=res),
          open(out_json, "w"), indent=1)
