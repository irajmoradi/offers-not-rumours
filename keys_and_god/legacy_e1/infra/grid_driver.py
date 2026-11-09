"""Grid driver + watchdog (plan/GRID_PLAN.md): runs grid cells one after another, each in its own
process group, and kills the group when an abort condition fires.

    OPENROUTER_API_KEY=... PYTHONNOUSERSITE=1 python infra/grid_driver.py \
        --cell grid_oss_sim_k2:1.35:3 --cell grid_oss_sim_k6:0.95:2 --seeds 16 --rounds 20

--cell name:cap_usd:branches_per_seed[:k=v,...] (config configs/<name>.yaml, run id <name> with "/" -> "_"; cap = the
cell's llm.budget_usd). Optional k=v (only change the projection's cost model): n=<agents> (default 10;
calls per round, prompt growth g=<tok/round> (default 55*n) capped at pcap=<tokens> (default 14000)), rounds=<max_rounds> (default --rounds), seeds=<the
config's run.n_games> (default --seeds), shared=<rounds
the forks share before God's first act> (default 2), free=1 (the control branch replays from the cache
via runner.fork.control_god and the God branch's shared prefix is cached too: only God's rounds > shared
are paid), paid=<branches paid from round shared+1 on> with ctl=<branches paid in full> (overrides free;
e.g. god + B'' on a cached control: paid=2,ctl=0; a game cell with B'': paid=2,ctl=1). Abort conditions, checked every 10 s (incremental reads of the run's files):
  ram        MemAvailable < 15 GB
  rg         more than 20 processes named rg
  ledger     global ledger total >= --ledger-stop (default 5.70)
  projection live projected cell cost > cap (after >= 150 paid calls): spent * W_total / W_done, with
             W(call in round r) = the smoke's per-call cost model (prompt grows ~550 tok/round, capped)
  errors     > 5% of LLM calls with an error (after >= 100 calls); any HTTP 400 at once
  parse      > 10% of turns parse_ok=false (after >= 100 turns)
  trunc      > 5% of responses finish_reason=length (after >= 200 responses)
  fallback   (only with --max-fallback F) > F of the fresh (non-cached) responses served by a provider
             other than --primary-provider (default PrimeIntellect), after >= 200 fresh responses
  branch     any single branch (game dir) with > 10% of its LLM calls errored (after >= 30 calls), or
             > 30% of its turns noop because of an error (llm_error or failed parse; after >= 20 turns):
             the "all calls error -> noop turns -> status done" failure
--attach: do not launch; watch already-running cells (found by `--run-id <name>` in /proc cmdlines,
waiting for each to appear) with the same checks, and kill their process group on abort.
On abort: SIGTERM to the cell's process group, SIGKILL after 20 s, no further cells; exit code 3.
Writes runs/grid_driver.log (status every 60 s). Never prints the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
LEDGER = RUNS / "openrouter_ledger.jsonl"
LOG = RUNS / "grid_driver.log"
FALLBACK: tuple[str, float] | None = None       # (primary provider, max fallback share): --max-fallback


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def call_weight(r: int, n: int = 10, g: int | None = None, pcap: int = 14000) -> float:
    """Smoke cost model per call in round r (micro-$): prompt 686 + g (r-1) tokens, capped at pcap;
    g defaults to 55 n tokens/round (550 at N=10, the smoke's measurement)."""
    return min(686 + (55 * n if g is None else g) * (r - 1), pcap) * 0.0142 + 350 * 0.0445


class Tail:
    """Incremental JSONL reader over a growing file."""

    def __init__(self, path: Path) -> None:
        self.path, self.pos, self.buf = path, 0, ""

    def new(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        self.buf += chunk
        lines = self.buf.split("\n")
        self.buf = lines.pop()
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out


def mem_available_gb() -> float:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def count_rg() -> int:
    n = 0
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if (p / "comm").read_text().strip() == "rg":
                    n += 1
            except OSError:
                pass
    return n


def ledger_total() -> float:
    t = 0.0
    for ln in open(LEDGER, encoding="utf-8"):
        try:
            t += float(json.loads(ln).get("cost_usd") or 0.0)
        except ValueError:
            pass
    return t


class Cell:
    def __init__(self, name: str, cap: float, branches: int, seeds: int, rounds: int,
                 opts: dict[str, int] | None = None) -> None:
        o = opts or {}
        self.n_agents = n = int(o.get("n", 10))
        self.g, self.pcap = (int(o["g"]) if "g" in o else None), int(o.get("pcap", 14000))
        rounds = int(o.get("rounds", rounds))
        seeds = int(o.get("seeds", seeds))
        self.name, self.cap, self.branches, self.seeds, self.rounds = name, cap, branches, seeds, rounds
        self.run_dir = RUNS / name
        k = int(o.get("shared", 2))
        per_branch = sum(n * call_weight(r, n, self.g, self.pcap) for r in range(1, rounds + 1))
        shared = sum(n * call_weight(r, n, self.g, self.pcap) for r in range(1, k + 1))  # replayed by forks
        if "paid" in o:          # paid branches from round k+1 on, plus ctl full branches
            self.w_total = seeds * (int(o["paid"]) * (per_branch - shared) + int(o.get("ctl", 0)) * per_branch)
        elif int(o.get("free", 0)):
            self.w_total = seeds * (per_branch - shared)                  # only God's rounds > k are paid
        else:
            # Only the no-God control B replays A's shared prefix from the cache; a 3rd branch (the
            # noise replica B', own seeds) pays for it too (was ~1% low: CODE_REVIEW #24).
            self.w_total = seeds * (branches * per_branch - min(branches - 1, 1) * shared)
        self.tails: dict[Path, Tail] = {}
        self.ledger = Tail(LEDGER)
        self.ledger.pos = 0                      # run ids are unique: count the cell's whole spend
        self.t0 = 0.0                            # ...since t0 (set at launch: a relaunched run id's
                                                 # earlier spend replays from the cache, not counted)
        self.per_game: dict[str, list[int]] = {}  # game dir -> [calls, errors, turns, noop_by_error]
        self.spent = self.w_done = 0.0
        self.calls = self.errors = self.http400 = self.paid = 0
        self.resp = self.length = self.turns = self.parse_fail = 0
        self.providers: dict[str, int] = {}      # fresh (non-cached) responses by serving provider

    def update(self) -> None:
        for d in self.ledger.new():
            if d.get("run_id") == self.name and float(d.get("ts") or 0.0) >= self.t0:
                self.spent += float(d.get("cost_usd") or 0.0)
        games = self.run_dir / "games"
        if not games.is_dir():
            return
        for g in games.iterdir():
            for fn in ("llm_calls.jsonl", "events.jsonl"):
                p = g / fn
                if p not in self.tails:
                    self.tails[p] = Tail(p)
                pg = self.per_game.setdefault(g.name, [0, 0, 0, 0])
                for d in self.tails[p].new():
                    if fn == "llm_calls.jsonl":
                        self._call(d)
                        pg[0] += 1
                        pg[1] += bool(d.get("error"))
                    elif d.get("type") == "turn":
                        td = d["data"]
                        self.turns += 1
                        self.parse_fail += not td.get("parse_ok", True)
                        pg[2] += 1
                        errored = (not td.get("parse_ok", True)
                                   or any(str(e).startswith("llm_error") for e in td.get("errors") or []))
                        pg[3] += errored and all((a or {}).get("kind") == "noop" for a in td.get("actions") or [{}])

    def _call(self, d: dict) -> None:
        self.calls += 1
        if d.get("error"):
            self.errors += 1
            self.http400 += "HTTP 400" in d["error"]
        r = d.get("response")
        if r:
            self.resp += 1
            self.length += r.get("finish_reason") == "length"
            if not r.get("cache_hit"):
                self.paid += 1
                pv = str(r.get("provider"))
                self.providers[pv] = self.providers.get(pv, 0) + 1
                self.w_done += call_weight(int(d["request"]["meta"].get("round") or 1), self.n_agents,
                                           self.g, self.pcap)

    def projected(self) -> float:
        return self.spent * self.w_total / self.w_done if self.w_done else 0.0

    def status(self) -> str:
        return (f"{self.name}: spent ${self.spent:.4f} projected ${self.projected():.3f} (cap ${self.cap:.2f}) "
                f"done {self.w_done / self.w_total:.1%} calls {self.calls} paid {self.paid} errors {self.errors} "
                f"turns {self.turns} parse_fail {self.parse_fail} length {self.length}/{self.resp}"
                + (f" fallback {self.fallback_share():.1%} {self.providers}" if self.providers else ""))

    def fallback_share(self) -> float:
        n = sum(self.providers.values())
        primary = FALLBACK[0] if FALLBACK else "PrimeIntellect"
        return 1 - self.providers.get(primary, 0) / n if n else 0.0

    def abort_reason(self, ledger_stop: float) -> str | None:
        if (m := mem_available_gb()) < 15:
            return f"ram: MemAvailable {m:.1f} GB < 15"
        if (n := count_rg()) > 20:
            return f"rg: {n} rg processes > 20"
        if (t := ledger_total()) >= ledger_stop:
            return f"ledger: total ${t:.3f} >= ${ledger_stop:.2f}"
        if self.paid >= 150 and self.projected() > self.cap:
            return f"projection: ${self.projected():.3f} > cap ${self.cap:.2f}"
        for gname, (c, e, t, ne) in sorted(self.per_game.items()):
            if c >= 30 and e / c > 0.10:
                return f"branch: {gname} {e}/{c} LLM calls errored"
            if t >= 20 and ne / t > 0.30:
                return f"branch: {gname} {ne}/{t} turns noop because of errors"
        if FALLBACK and sum(self.providers.values()) >= 200 and self.fallback_share() > FALLBACK[1]:
            return f"fallback: {self.fallback_share():.1%} of fresh responses not from {FALLBACK[0]} > {FALLBACK[1]:.0%}"
        if self.http400:
            return f"errors: {self.http400} HTTP 400"
        if self.calls >= 100 and self.errors / self.calls > 0.05:
            return f"errors: {self.errors}/{self.calls} calls"
        if self.turns >= 100 and self.parse_fail / self.turns > 0.10:
            return f"parse: {self.parse_fail}/{self.turns} turns"
        if self.resp >= 200 and self.length / self.resp > 0.05:
            return f"trunc: {self.length}/{self.resp} responses"
        return None


def parse_cell(spec: str) -> tuple[str, str, str, dict[str, int]]:
    """name:cap:branches[:k=v,k=v] -> (name, cap, branches, {k: int(v)})."""
    parts = spec.split(":")
    name, cap, branches = parts[:3]
    opts = {k: int(v) for k, v in (kv.split("=") for kv in (parts[3].split(",") if len(parts) > 3 else []))}
    return name, cap, branches, opts


def kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()                                  # reap it
    except ProcessLookupError:
        pass


def kill_target(pid: int) -> tuple[str, int]:
    """("group", pgid) when the cell leads its own process group (start_new_session), else ("pid",
    pid): killing a shared group could hit other cells or this watchdog (CODE_REVIEW #25)."""
    pgid = os.getpgid(pid)
    if pgid == pid and pgid != os.getpgid(0):
        return "group", pgid
    return "pid", pid


def send(target: tuple[str, int], sig: int) -> None:
    kind, ident = target
    (os.killpg if kind == "group" else os.kill)(ident, sig)


def find_cell_pid(run_id: str) -> int | None:
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"swarm.run" in argv and b"--run-id" in argv and run_id.encode() in argv:
            return int(p.name)
    return None


def attach(cells: list[Cell], ledger_stop: float) -> int:
    for cell in cells:
        pid = None
        while pid is None:
            pid = find_cell_pid(cell.name)
            if pid is None:
                if (cell.run_dir / "manifest.json").exists() and "ended_utc\": \"" in (cell.run_dir / "manifest.json").read_text():
                    break                                # finished before we looked
                time.sleep(10)
        if pid is None:
            cell.update()
            log(f"ATTACH {cell.name}: already finished; {cell.status()}")
            continue
        target = kill_target(pid)
        log(f"ATTACH {cell.name} pid {pid} pgid {os.getpgid(pid)}; abort kills the {target[0]} {target[1]}")
        last = 0.0
        while Path(f"/proc/{pid}").exists():
            time.sleep(10)
            cell.update()
            reason = cell.abort_reason(ledger_stop)
            if reason:
                log(f"ABORT {cell.name} (attach): {reason}; {cell.status()}")
                try:
                    send(target, signal.SIGTERM)
                    time.sleep(20)
                    send(target, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return 3
            if time.time() - last >= 300:
                log(f"(attach) {cell.status()}")
                last = time.time()
        cell.update()
        log(f"DETACH {cell.name} ended; {cell.status()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--ledger-stop", type=float, default=5.70)
    ap.add_argument("--set", dest="sets", action="append", default=[], help="override passed to every cell")
    ap.add_argument("--run-prefix", default="", help="run id prefix (dry runs)")
    ap.add_argument("--attach", action="store_true", help="watch running cells instead of launching")
    ap.add_argument("--cap-override", action="store_true",
                    help="pass llm.budget_usd=<the cell spec's cap> to each cell (per-cell hard caps)")
    ap.add_argument("--log", type=Path, default=None, help="log file (default runs/grid_driver.log)")
    ap.add_argument("--max-fallback", type=float, default=None, help="abort when this share of fresh calls is "
                    "served by a provider other than --primary-provider (after 200 fresh calls)")
    ap.add_argument("--primary-provider", default="PrimeIntellect")
    args = ap.parse_args()
    global LOG, FALLBACK
    if args.max_fallback is not None:
        FALLBACK = (args.primary_provider, args.max_fallback)
    if args.log is not None:
        LOG = args.log if args.log.is_absolute() else REPO / args.log
    if args.attach:
        cells = []
        for spec in args.cell:
            name, cap, branches, opts = parse_cell(spec)
            cells.append(Cell(args.run_prefix + name.replace("/", "_"), float(cap), int(branches), args.seeds,
                              args.rounds, opts))
        return attach(cells, args.ledger_stop)
    fake = any(s.startswith("roles.agent.backend=fake") for s in args.sets)
    if not fake and not os.environ.get("OPENROUTER_API_KEY"):
        log("OPENROUTER_API_KEY not set")
        return 2
    for spec in args.cell:
        name, cap, branches, opts = parse_cell(spec)
        cell = Cell(args.run_prefix + name.replace("/", "_"), float(cap), int(branches), args.seeds,
                    args.rounds, opts)   # name may be a subpath (configs/sweep/x.yaml -> run id sweep_x)
        if (r := cell.abort_reason(args.ledger_stop)) is not None:
            log(f"NOT starting {name}: {r}")
            return 3
        cmd = [sys.executable, "-m", "swarm.run", "--config", f"configs/{name}.yaml", "--run-id", cell.name,
               *args.sets, *([f"llm.budget_usd={cell.cap}"] if args.cap_override else [])]
        out = open(RUNS / f"{cell.name}.log", "a")
        cell.t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"START {cell.name} pid/pgid {proc.pid} cap ${cell.cap:.2f} seeds {cell.seeds} rounds {cell.rounds}")
        last = 0.0
        while proc.poll() is None:
            time.sleep(10)
            cell.update()
            reason = cell.abort_reason(args.ledger_stop)
            if reason:
                log(f"ABORT {name}: {reason}; {cell.status()}")
                kill_group(proc)
                return 3
            if time.time() - last >= 60:
                log(cell.status())
                last = time.time()
        cell.update()
        log(f"END {name} exit {proc.returncode}; {cell.status()}")
        if proc.returncode != 0:
            return 4
    log("GRID DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
