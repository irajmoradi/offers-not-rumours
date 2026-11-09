"""Secret scan for this release. Usage (from the repo root):  python -I scripts/secret_scan.py [--show-allowed]

Prints NOTHING and exits 0 when clean. Any match that is not on the explicit allowlist below is printed as
"path:line: <pattern name>" (never the matched text) and the exit code is 1.

Patterns: OpenRouter/OpenAI-style keys (sk-or-..., sk-<20+>), Google (AIza...), Hugging Face (hf_...), GitHub
(ghp_...), Slack (xox?-...), Adaption tokens (pt_live_...), PEM private keys, any run of 32+ hex digits, and
absolute home paths (/home/<user>, /Users/<user>).

Explicit allowlist (every item was checked by hand before the first commit):
  1. Bare key PREFIXES in key-loading / redaction code and docs: "sk-or-", "sk-or-v1-", "sk-or-env", "sk-".
  2. Fake test canaries in keys_and_god/**/tests (they are asserted to be redacted):
     sk-or-v1-zqTESTCANARYKEY0123456789abcdef, sk-or-v1-zqcanary0123456789abcdef0123456789abcdef,
     sk-or-v1-THISISNOTAREALKEY1234567890, sk-or-v1-test0000000000000000000000000000, sk-or-env-key-0000000000,
     sk-or-envkey-123456789, sk-or-x0123456789, ad-zqcanary-0123456789abcdef0123456789abcdef, and Adaption
     test tokens that start with pt_live_FAKE (pt_live_FAKEabc_123-xyz, pt_live_FAKEbare, pt_live_FAKEenv, ...).
  3. 32+ hex runs that are (a) the sha256 of a file shipped here (pre-registration and package hash lists, the
     prespec header lines, results.json prereg fields), (b) pinned config / prompt / event hashes in
     keys_and_god/**/tests (git-tracked regression fixtures), (c) game-identity hashes ("sha256") in
     betrayal/out/*_prereg/idcheck*.json, or (d) the public viz git commit (9cda155...) in requirements/figures.txt and README.md.
  4. betrayal/harness/test_dual_control.sh keeps two absolute paths: it is a pre-registered, hash-frozen file
     (PREREG_DC_CODE_SHA256.txt) and editing it would break the registered hash. It is a local test script only.
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOW = "--show-allowed" in sys.argv
PATS = {
    "openrouter-key": re.compile(r"sk-or-[A-Za-z0-9_\-]*"),
    "sk-key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}"),
    "google-key": re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    "hf-token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{10,}"),
    "github-token": re.compile(r"ghp_[A-Za-z0-9]{10,}"),
    "slack-token": re.compile(r"xox[abprs]-[A-Za-z0-9\-]+"),
    "adaption-token": re.compile(r"pt_live_[A-Za-z0-9_\-]{6,}"),
    "private-key": re.compile(r"BEGIN [A-Z ]*PRIVATE"),
    "hex32": re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])"),
    "home-path": re.compile(r"/home/[A-Za-z0-9_.\-]+|/Users/[A-Za-z0-9_.\-]+"),
}
PREFIXES = {"sk-or-", "sk-or-v1-", "sk-or-env"}
CANARIES = {
    "sk-or-v1-zqTESTCANARYKEY0123456789abcdef", "sk-or-v1-zqcanary0123456789abcdef0123456789abcdef",
    "sk-or-v1-THISISNOTAREALKEY1234567890", "sk-or-v1-test0000000000000000000000000000", "sk-or-env-key-0000000000",
    "sk-or-envkey-123456789", "sk-or-x0123456789",
}
FROZEN_HOME = {"betrayal/harness/test_dual_control.sh"}
VIZ_COMMIT = "9cda155aabf95125c996bda51eacb259c659c78d"


def files():
    for d, dirs, fs in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in fs:
            yield os.path.join(d, f)


hashes = set()
for p in files():
    if os.path.getsize(p) < 20_000_000:
        b = open(p, "rb").read()
        hashes.add(hashlib.sha256(b).hexdigest())
        if os.path.basename(p).startswith("prespec_"):
            hashes.add(hashlib.sha256(b.split(b"\n", 1)[-1]).hexdigest())


def allowed(rel, name, m):
    s = m.group(0)
    if name == "openrouter-key":
        return s in PREFIXES or s in CANARIES
    if name == "adaption-token":
        return s.startswith("pt_live_FAKE") and "/tests/" in "/" + rel
    if name == "hex32":
        h = s.lower()
        if h in hashes:
            return True
        if "/tests/" in "/" + rel:
            return True
        if re.match(r"betrayal/out/[a-z]+_prereg/idcheck[^/]*\.json$", rel):
            return True
        return rel in ("requirements/figures.txt", "README.md") and h == VIZ_COMMIT
    if name == "home-path":
        return rel in FROZEN_HOME
    return False


bad, ok = [], {}
for p in files():
    rel = os.path.relpath(p, ROOT)
    try:
        lines = open(p, encoding="utf-8").read().split("\n")
    except (UnicodeDecodeError, OSError):
        continue                                   # binary (png/pdf): not scanned as text
    if rel == "scripts/secret_scan.py":
        continue                                   # this file holds the patterns and the allowlist
    for i, ln in enumerate(lines, 1):
        for name, rx in PATS.items():
            for m in rx.finditer(ln):
                if allowed(rel, name, m):
                    ok[name] = ok.get(name, 0) + 1
                else:
                    bad.append(f"{rel}:{i}: {name}")
for b in bad:
    print(b)
if SHOW:
    print("allowed matches by pattern:", ok)
sys.exit(1 if bad else 0)
