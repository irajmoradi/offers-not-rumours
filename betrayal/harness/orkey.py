"""Load the OpenRouter key from a file path; never print it."""
import os, re
def load_key(path):
    """path "-" or "env" (or empty): use OPENROUTER_API_KEY from the environment instead of a file."""
    if path in ("", "-", "env"):
        raw = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not raw:
            raise SystemExit("set OPENROUTER_API_KEY or pass a key file path")
        return raw
    raw = open(path).read().strip()
    m = re.search(r"sk-or-[A-Za-z0-9_\-]+", raw)
    if m:
        return m.group(0)
    if "=" in raw:
        raw = raw.split("=", 1)[1]
    raw = raw.strip().strip('"').strip("'")
    if re.fullmatch(r"[0-9a-f]{64}", raw):   # file holds only the key body (e.g. the team key): add the OpenRouter prefix
        raw = "sk-or-v1-" + raw
    return raw
