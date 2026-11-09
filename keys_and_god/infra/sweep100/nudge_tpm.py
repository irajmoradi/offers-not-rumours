"""Live-nudge the shared Adaption limiter's TPM (no restart): every sweep process re-reads the state file on each
request, under the same flock, so a new "tpm" takes effect on the next call.
    python infra/sweep100/nudge_tpm.py [--state runs/adaption_rpm.json] [--log ...] TPM
Sets tpm (clamped to [300k, 1.5M]), empties the token bucket (no burst on top of the jump), and restarts the
adaptive window (n429 = 0, last_adjust = now), so the normal +10% per clean 5 min / -25% per 429 burst logic
continues from the new value. Logs a "nudge_tpm" event to the limiter log. Reads/writes no secrets."""
import argparse
import fcntl
import json
import os
import time

H = f"{os.environ.get('SWARM_ROOT', '.')}/runs"

ap = argparse.ArgumentParser()
ap.add_argument("tpm", type=float)
ap.add_argument("--state", default=f"{H}/adaption_rpm.json")
ap.add_argument("--log", default=f"{H}/adaption_rpm_log.jsonl")
a = ap.parse_args()
new = max(300_000.0, min(1_500_000.0, a.tpm))
fd = os.open(a.state, os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    st = json.loads(os.read(fd, 4096) or b"{}")
    old = st.get("tpm")
    now = time.time()
    st.update(tpm=new, ttokens=0.0, n429=0, first429=None, burst=0, last_adjust=now)
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(st).encode())
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
with open(a.log, "a", encoding="utf-8") as f:
    f.write(json.dumps({"ts": now, "rpm": st.get("rpm"), "tpm": round(new), "event": "nudge_tpm",
                        "from": round(old or 0)}) + "\n")
print(f"tpm {old} -> {new}")
