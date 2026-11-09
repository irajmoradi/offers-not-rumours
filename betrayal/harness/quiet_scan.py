"""Usage: quiet_scan.py KEYFILE SRC_DIR OUT_ZIP. Writes MANIFEST.txt, then zips SRC_DIR and scans every file
(and every zip entry) for the real key and for sk-or-/sk-/hf_/Bearer token patterns. It prints NOTHING
when clean (exit 0). On a hit it prints the file paths only, never the match, and exits 1."""
import sys, os, re, zipfile, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orkey import load_key
key = load_key(sys.argv[1]).encode(); src, out = sys.argv[2], sys.argv[3]
pats = [re.compile(rb"pt_live_[A-Za-z0-9_\-]{8,}"), re.compile(rb"sk-or-[A-Za-z0-9_\-]{10,}"), re.compile(rb"\bsk-[A-Za-z0-9]{20,}"),
        re.compile(rb"\bhf_[A-Za-z0-9]{20,}"), re.compile(rb"Bearer [A-Za-z0-9_\-\.]{20,}")]
files = sorted(os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs if f != "MANIFEST.txt")
with open(os.path.join(src, "MANIFEST.txt"), "w") as m:
    m.write("sha256[:16]  bytes  path\n")
    for p in files:
        m.write(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}  {os.path.getsize(p):>10}  {os.path.relpath(p, src)}\n")
hits = []
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files + [os.path.join(src, "MANIFEST.txt")]:
        b = open(p, "rb").read()
        if key in b or any(rx.search(b) for rx in pats): hits.append(os.path.relpath(p, src))
        z.write(p, os.path.relpath(p, os.path.dirname(src)))
with zipfile.ZipFile(out) as z:
    for n in z.namelist():
        if key in z.read(n): hits.append("zip:" + n)
if hits:
    print("SECRET-SCAN HITS:", sorted(set(hits))); sys.exit(1)
