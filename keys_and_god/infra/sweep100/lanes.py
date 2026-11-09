"""sweep100 lane scheduler: 2 blocks in flight, picked in strict priority order (lead's decision A, 2026-11-08).

    cd keys_and_god && PYTHONNOUSERSITE=1 setsid nohup python infra/sweep100/lanes.py \
        [cell=<driver pid> ...] > runs/sweep100_lanes.out 2>&1 < /dev/null &
(cell=pid adopts an already-running block driver into a lane: the lane waits for it, then continues the queue).
RSS guard: the total RSS of the sweep game processes is logged every minute; above RSS_LIMIT_GB (18) no new
block starts (the lane waits). The runner's world gate (runner.world_gate.rss_gb) holds new worlds inside a block.

Each lane takes the next unclaimed block from QUEUE and runs ONE grid_driver for it (all guards), with the block's
lane settings (games in flight, a per-process RPM cap for FAST blocks so they cannot starve a slow block's
replies, the per-process in-flight cap). Both lanes share the cross-process adaptive RPM bucket file and the sweep
ledger. After a block's driver exits:
  - rc 0, no world-gate skip      -> the lane takes the next block;
  - abort by ledger / projection  -> STOP every lane (cost guard), log it;
  - world gate skipped a world    -> STOP every lane (the $1,300 sweep cap is reached);
  - any other abort / failure     -> log it (resume command below) and CONTINUE with the next block
    (a stall / truncation / error abort in one block must not kill the rest).
Resume a block that aborted: python -m swarm.run --config configs/<block>.yaml "runner.game_indices=[...]" --run-id
resume_<block> (everything played replays from the cache). State: runs/sweep100_lanes.json (claims, results).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", "python")
LEDGER = f"{os.environ.get('SWARM_ROOT', '.')}/runs/adaption_sweep_ledger.jsonl"
LOG = REPO / "runs" / "sweep100_driver.log"
STATE = REPO / "runs" / "sweep100_lanes.json"
SLOW = {"max_concurrency_games": None, "rpm_local": None, "inflight": 220}     # GLM high / DeepSeek / Gemma
FAST = {"max_concurrency_games": 3, "rpm_local": 50, "inflight": 120}          # agents GLM low: 3 s calls; ~0.5M
# prompt tokens/min (50 x ~10k): the shared TPM bucket (~1.5M/min, Adaption's binding limit) stays mostly block 1's
# (cell, grid_driver spec, lane settings) in strict priority order
QUEUE = [
    ("sweep100_glm_high", "sweep100_glm_high:340:2:n=100,g=0,seeds=2,shared=2", SLOW),          # block 1: W1+W2
    ("sweep100_glm_high_w3", "sweep100_glm_high_w3:170:2:n=100,g=0,seeds=1,shared=2", SLOW),    # then W3
    ("sweep100_glm_high_w45", "sweep100_glm_high_w45:340:2:n=100,g=0,seeds=2,shared=2", SLOW),  # then W4+W5
    ("sweep100_glm_low", "sweep100_glm_low:160:2:n=100,g=0,seeds=5,shared=2", FAST),
    ("sweep100_mixed_effort", "sweep100_mixed_effort:90:1:n=100,g=0,seeds=5,shared=0", FAST),
    ("sweep100_scripted", "sweep100_scripted:340:2:n=100,g=0,seeds=5,shared=2", SLOW),
    ("sweep50_glm_high", "sweep50_glm_high:170:2:n=50,g=0,seeds=5,shared=2", SLOW),
    ("sweep100_glm_high_k14", "sweep100_glm_high_k14:340:2:n=100,g=0,seeds=5,shared=2", SLOW),
    ("sweep100_deepseek", "sweep100_deepseek:140:2:n=100,g=0,seeds=5,shared=2,trunc=10", SLOW),
    ("sweep100_gemma", "sweep100_gemma:400:2:n=100,g=0,seeds=5,shared=2,trunc=10", SLOW),
]
LANES = int(os.environ.get("SWEEP_LANES", "1"))   # lead 2026-11-08 03:40: ONE block at a time (the TPM budget is
# fixed, overlap only splits it); in single-lane mode every block runs with SLOW settings (all games in flight)
RSS_LIMIT_GB = 18.0     # lead: above this total sweep RSS, start no new block (logged every minute)
_tlock = threading.Lock()
stop = threading.Event()
LOCKFILE = REPO / "runs" / "sweep100_lanes.lock"
# Key profiles: several lanes processes (one per Adaption key) share the claim file under an exclusive flock, so
# two keys never run the same block/game. "main" = the default key; "lance" = Lance's key (Iraj-approved): its own
# key file (by path only), ledger ($150 cap), RPM/TPM bucket and log; every game setting identical.
PROFILE = os.environ.get("SWEEP_PROFILE", "main")
# SWEEP_ADOPT_ONLY=1 (lead 2026-11-08 14:00, option b): a lane that adopted a running driver exits when it ends instead
# of claiming the next block (the main key is throttled to ~10 calls/min; Lance's lane takes W3 then W4+W5)
ADOPT_ONLY = os.environ.get("SWEEP_ADOPT_ONLY", "0") == "1"
HACK = f"{os.environ.get('SWARM_ROOT', '.')}/runs"
PROFILES = {
    "main": {"ledger": LEDGER, "ledger_stop": 1300, "sets": []},
    "lance": {"ledger": f"{HACK}/adaption_lance_ledger.jsonl", "ledger_stop": 150, "sets": [
        f"llm.adaption.key_file={os.environ.get('ADAPTION_KEY_FILE', '')}",
        f"llm.adaption.ledger_path={HACK}/adaption_lance_ledger.jsonl", "llm.adaption.global_cap_usd=150",
        f"llm.adaption.rpm_state_path={HACK}/adaption_rpm_lance.json",
        f"llm.adaption.rpm_log_path={HACK}/adaption_rpm_lance_log.jsonl",
        # lead 2026-11-08 14:00: per-user token limits seen today (main key ~10 calls/min since 12:22): start Lance's
        # bucket conservatively at 350k prompt tokens/min and let the adaptive ramp (+10% per clean 5 min) find it
        "llm.adaption.tpm=350000", "llm.adaption.tpm_burst=60000"]},
    # lead 2026-11-08 15:40: BOTH keys for one block (W3, then W4+W5): the main key + its bucket as primary, Lance's
    # key + Lance's bucket as an extra key lane (llm.adaption.extra_keys; every POST to the first bucket that admits
    # it). Spend goes to the main ledger (the $1300 stop and the world gate, which also sums Lance's ledger); Lance's
    # share is the calls with provider "adaption:k1".
    "dual": {"ledger": LEDGER, "ledger_stop": 1300, "sets": [
        f"llm.adaption.extra_keys=[{{key_file: \"{os.environ.get('ADAPTION_KEY_FILE', '')}\", "
        f"rpm_state_path: {HACK}/adaption_rpm_lance.json, rpm_log_path: {HACK}/adaption_rpm_lance_log.jsonl}}]"]},
}


@contextlib.contextmanager
def lock():
    """Thread lock + an exclusive flock shared by every lanes process (main key, Lance's key)."""
    with _tlock:
        with open(LOCKFILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def sweep_rss() -> tuple[float, dict[str, float]]:
    """Total RSS (GB) of every sweep game process (python -m swarm.run --config configs/sweep*), by run id."""
    per: dict[str, float] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
            if b"swarm.run" not in argv or not any(a.startswith(b"configs/sweep") for a in argv):
                continue
            rid = argv[argv.index(b"--run-id") + 1].decode() if b"--run-id" in argv else d.name
            kb = next(int(ln.split()[1]) for ln in (d / "status").read_text().splitlines() if ln.startswith("VmRSS"))
        except (OSError, ValueError, StopIteration, IndexError):
            continue
        per[rid] = per.get(rid, 0.0) + kb / 1024 / 1024
    return sum(per.values()), per


def rss_watch() -> None:
    while not stop.is_set():
        tot, per = sweep_rss()
        log(f"rss total {tot:.2f} GB ({', '.join(f'{k} {v:.2f}' for k, v in sorted(per.items())) or 'none'})"
            + (f" > {RSS_LIMIT_GB:.0f} GB: no new block starts" if tot > RSS_LIMIT_GB else ""))
        stop.wait(60)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} LANES {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"claimed": {}, "results": {}}


def save(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def done(name: str) -> bool:
    m = REPO / "runs" / name / "manifest.json"
    try:
        t = m.read_text()
    except OSError:
        return False
    return '"ended_utc": "' in t and '"status": "done"' in t


def claim() -> tuple[str, str, dict] | None:
    while not stop.is_set():                        # RSS guard: wait (do not start a block) while above the limit
        tot, _ = sweep_rss()
        if tot <= RSS_LIMIT_GB:
            break
        time.sleep(60)
    with lock():
        st = load()
        for name, spec, opts in QUEUE:
            if name in st["claimed"] or done(name):
                continue
            st["claimed"][name] = time.time()
            save(st)
            return name, spec, opts
    return None


def last_abort(name: str) -> str:
    try:
        lines = [ln for ln in LOG.read_text().splitlines() if f"ABORT {name}" in ln]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def run_block(lane: int, name: str, spec: str, o: dict) -> None:
    if LANES == 1:
        o = SLOW
    sets = [f"llm.adaption.global_max_concurrency={o['inflight']}"]
    if o["max_concurrency_games"]:
        sets.append(f"run.max_concurrency_games={o['max_concurrency_games']}")
    if o["rpm_local"]:
        sets.append(f"llm.adaption.rpm_local={o['rpm_local']}")
    prof = PROFILES[PROFILE]
    sets += prof["sets"]
    cmd = [PY, "infra/grid_driver.py", "--cell", spec, "--seeds", "5", "--rounds", "80", "--ledger", prof["ledger"],
           "--ledger-stop", str(prof["ledger_stop"]), "--stall-min", "20", "--log", str(LOG)]
    for s in sets:
        cmd += ["--set", s]
    shown = [x if "key_file" not in x else x.split("=", 1)[0] + "=<key path>" for x in sets]
    log(f"[{PROFILE}] lane {lane} START {name} ({', '.join(shown)})")
    rc = subprocess.call(cmd, cwd=REPO, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    ab = last_abort(name)
    with lock():
        st = load()
        st["results"][name] = {"rc": rc, "abort": ab, "ended": time.time()}
        save(st)
    man = REPO / "runs" / name / "manifest.json"
    skipped = man.exists() and '"status": "skipped"' in man.read_text()
    if rc == 0 and not skipped:
        log(f"lane {lane} END {name} ok")
        return
    if skipped or "ledger:" in ab or "projection:" in ab:
        log(f"lane {lane} STOP ALL: {name} rc={rc} {'world gate skipped worlds' if skipped else ab}")
        stop.set()
        return
    log(f"lane {lane} {name} FAILED rc={rc} ({ab or 'no abort line'}); continuing with the next block. Resume: "
        f"python -m swarm.run --config configs/{name}.yaml \"runner.game_indices=[<unfinished>]\" --run-id resume_{name}")


def lane(i: int, wait_pid: int | None = None, wait_name: str | None = None) -> None:
    if wait_pid:
        log(f"lane {i} waits for pid {wait_pid} (block 1 already running)")
        while Path(f"/proc/{wait_pid}").exists() and not stop.is_set():
            time.sleep(30)
        if wait_name:                                # the adopted block's outcome, as run_block would record it
            ab = last_abort(wait_name)
            man = REPO / "runs" / wait_name / "manifest.json"
            skipped = man.exists() and '"status": "skipped"' in man.read_text()
            with lock():
                st = load()
                st["results"][wait_name] = {"rc": None, "abort": ab, "ended": time.time(), "adopted": True}
                save(st)
            if skipped or "ledger:" in ab or "projection:" in ab:
                log(f"lane {i} STOP ALL: adopted {wait_name} {'world gate skipped worlds' if skipped else ab}")
                stop.set()
                return
            log(f"lane {i} adopted {wait_name} ended ({ab or 'no abort'})")
            if ADOPT_ONLY:
                log(f"[{PROFILE}] lane {i} exits after the adopted block (SWEEP_ADOPT_ONLY=1): no new block claimed")
                return
    while not stop.is_set():
        nxt = claim()
        if nxt is None:
            log(f"lane {i} DONE (queue empty)")
            return
        run_block(i, *nxt)


def main() -> int:
    # adopt already-running block drivers: lanes.py <cell>=<driver pid> [<cell>=<driver pid>]
    adopt = [a.split("=", 1) for a in sys.argv[1:]]
    with lock():
        st = load()
        for name, _pid in adopt:
            st["claimed"].setdefault(name, time.time())
        save(st)
    waits = [(int(pid), name) for name, pid in adopt] + [(None, None)] * LANES
    threading.Thread(target=rss_watch, daemon=True).start()
    ts = [threading.Thread(target=lane, args=(i + 1, *waits[i])) for i in range(LANES)]
    for t in ts:
        t.start()
        time.sleep(5)
    for t in ts:
        t.join()
    log("ALL LANES FINISHED" + (" (stopped by a cost guard)" if stop.is_set() else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
