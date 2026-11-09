import os, time
os.environ["OUT"] = "/tmp/claude-1000/-home-iraj-projects-swarm-data-/6da8a2b5-8033-4627-a30c-797c3dde5e45/scratchpad"
os.environ["ADAPTION_TPM"] = "60000"
import adaption_backend as adb
T = adb.TPM
t = time.monotonic(); T.acquire(9000); T.acquire(1000); print("burst ok", round(time.monotonic() - t, 2))
t = time.monotonic(); T.acquire(1000); print("waited for refill", round(time.monotonic() - t, 2), "s (expect ~1s)")
for _ in range(3): T.on_429(0.5)
print("after 3x429 tpm", T.tpm, "pause set", T.pause_until > time.monotonic())
t = time.monotonic(); T.acquire(10); print("waited through pause", round(time.monotonic() - t, 2))
