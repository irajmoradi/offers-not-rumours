import os
from backends import make_backend
b = make_backend("meta-llama/llama-3.1-8b-instruct", engine="openrouter", temperature=0.0, max_new_tokens=5, max_workers=1)
k = os.environ.get("OPENROUTER_API_KEY", "")
print("env key prefixed:", k.startswith("sk-or-v1-"), "len", len(k))
try:
    r = b.client.chat.completions.create(model=b.model, max_tokens=5, messages=[{"role": "user", "content": "Say ok"}])
    print("OK", bool(r.choices))
except Exception as e:
    print(type(e).__name__, str(e)[:120])
