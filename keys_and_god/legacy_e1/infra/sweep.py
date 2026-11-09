"""Sweep launcher (plan/SWEEP_PLAN.md): one-factor-at-a-time cells around G1, each a paired fork
God / no-God / fork-matched noise B'' on shared worlds (run.seed 1, game_idx 0..n-1 = G1's worlds).

    python infra/sweep.py list                       # conditions, priority, seeds, caps, projections
    python infra/sweep.py write                      # (re)write configs/sweep/<cond>.yaml
    python infra/sweep.py dry  [COND ...|all] [--rounds 16] [--seeds 2]
                                                     # FakeLLM dry runs (no network, $0) + checks that
                                                     # God timing/K, board, keys, N, rounds took effect
    python infra/sweep.py launch COND [COND ...] [--tag A] [--run-prefix P] [--seeds N]
                                  [--only-branches noisefork] [--conc 4] [--dry-run]
                                                     # paid: one grid_driver (watchdog) running the
                                                     # cells one after another, with per-cell hard caps
    python infra/sweep.py stats [COND ...]           # sweep_stats over the finished cells

Design per condition (Cond below):
- God-only conditions (start round, K, base) set runner.fork.control_god = G1's god section: the
  control is exactly G1's nogod game (same game_id/seeds; replays from the LLM cache) and the God
  branch replays G1's cached prefix up to its first intervention.
- Game conditions (board, keys, N, rounds) change the world/rules, so they have their own control.
- Every condition has runner.fork.noise_fork: B'' = the control resampled from God's first
  scheduled round on (run.resample_from_round), the fork-matched noise baseline.
`launch` never exports or prints the key: the driver child reads OPENROUTER_API_KEY, which this
script sets from the key file only in the child's environment. --dry-run prints the plan only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CONF_DIR = REPO / "configs" / "sweep"
KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", ""))
PY = [sys.executable]
LEDGER_STOP = 14.80                # hard: the ledger must never reach this (raised from 7.80 for E1, lead 2026-11-07)
# Sweep-total guard: the ledger was $5.0575 when the main grid ended (before any sweep spend), and the
# whole sweep may spend $2.30, so new drivers stop at 5.0575 + 2.30 (other agents' spend counts too).
SWEEP_LEDGER_STOP = 7.35
# RunBudget reserves a worst-case estimate for EVERY pending call (before the concurrency semaphore):
# (prompt x price_in + max_tokens x price_out) x (concurrent games x N). With the old 0.1/0.5 $/M
# estimates that was ~$0.15 at r18 with 8-12 games, so tight caps ended games "budget_exceeded" while
# real spend was far below the cap (04:2x, b2_sweep_start8/14, sweep_start14). The configs now use
# 0.03/0.15 $/M (2x the realized fitted price; operational, hashed as fixed) and caps get this margin.
RESERVE_MARGIN = 0.06

FRAUD = {"power": "message_fraud", "after_turns": 0,
         "args": {"sender": "recent_partner", "receiver": "closest_to_winning"}}
COT = {"power": "cot_control", "after_turns": 0,
       "args": {"agent": "closest_to_winning", "round": "prev", "target": "scratchpad", "mode": "replace"}}
G1_GOD = {"kind": "scripted", "start_round": 3, "budget": {"k": 2, "scope": "game"},
          "schedule": [{"round": 3, **FRAUD}, {"round": 5, **COT}]}


def sched(*items: tuple[int, dict]) -> list[dict]:
    return [{"round": r, **p} for r, p in items]


@dataclass(frozen=True)
class Cond:
    factor: str
    priority: int                      # 1 = first (lead's order: start, board, K, N, keys, rounds)
    desc: str
    god: dict[str, Any] = field(default_factory=dict)       # merged onto G1_GOD
    game: dict[str, Any] = field(default_factory=dict)      # merged onto the base game block
    seeds: int = 8
    cap_usd: float | None = None       # hard cap = llm.budget_usd; None = 1.25 x projection at price 1.35
    conc_games: int = 12

    @property
    def cap(self) -> float:
        if self.cap_usd is not None:
            return self.cap_usd
        return round(1.25 * projection(self, 1.35)["total"] + RESERVE_MARGIN, 2)

    @property
    def god_only(self) -> bool:
        return not self.game

    @property
    def god_cfg(self) -> dict[str, Any]:
        return {**G1_GOD, **self.god}

    @property
    def n(self) -> int:
        return int(self.game.get("n_agents", 10))

    @property
    def rounds(self) -> int:
        return int(self.game.get("max_rounds", 20))

    @property
    def fork_round(self) -> int:
        g = self.god_cfg
        return min(s["round"] for s in g["schedule"] if s["round"] >= g["start_round"])

    @property
    def god_rounds(self) -> list[tuple[int, str]]:
        g = self.god_cfg
        return sorted((s["round"], s["power"]) for s in g["schedule"] if s["round"] >= g["start_round"])


CONDS: dict[str, Cond] = {
    # factor 1: God's first-intervention round (same two powers and spacing: fraud r, cot r+2)
    "base": Cond("start", 1, "G1 design (fraud r3, cot r5); adds B''@r3 to G1's worlds", {}, {}),
    "start1": Cond("start", 1, "God first acts r1", {"start_round": 1, "schedule": sched((1, FRAUD), (3, COT))}),
    "start8": Cond("start", 1, "God first acts r8", {"start_round": 8, "schedule": sched((8, FRAUD), (10, COT))}),
    "start14": Cond("start", 1, "God first acts r14",
                    {"start_round": 14, "schedule": sched((14, FRAUD), (16, COT))}),
    # factor 6: message board variants (own control; God = G1's)
    "board_off": Cond("board", 2, "no board (post_board not offered)", {}, {"board": {"enabled": False}}),
    "board_delay2": Cond("board", 2, "board posts visible 2 rounds late", {}, {"board": {"visible_delay_rounds": 2}}),
    "board_anon": Cond("board", 2, "anonymous board", {}, {"board": {"anonymous": True}}),
    "board_skip3": Cond("board", 2, "posting costs the next 3 turns", {}, {"board": {"post_skip_rounds": 3}}),
    # factor 2: God budget K (K=2 = G1/base; K=6 = G3)
    "k1": Cond("k", 3, "K=1 (fraud r3)", {"budget": {"k": 1, "scope": "game"}, "schedule": sched((3, FRAUD))}),
    "k4": Cond("k", 3, "K=4 (fraud r3/r5, cot r4/r6)",
               {"budget": {"k": 4, "scope": "game"}, "schedule": sched((3, FRAUD), (4, COT), (5, FRAUD), (6, COT))}),
    # E1 (plan/COHERENCE_REVIEW.md section 5): the original PDF's per-round budget, K=1 per round, message
    # fraud every round r3-r20 (18 acts), G1's templates/targeting; all 16 G1 worlds
    "e1": Cond("k", 3, "E1: K=1 per round, message fraud every round r3-r20 (18 acts)",
               {"budget": {"k": 1, "scope": "round"}, "schedule": sched(*((r, FRAUD) for r in range(3, 21)))},
               seeds=16),
    # factor 4: number of agents
    "n5": Cond("n", 4, "N=5", {}, {"n_agents": 5}, conc_games=18),
    "n20": Cond("n", 4, "N=20", {}, {"n_agents": 20}, conc_games=9),
    # factor 3: keys per agent (needs 3, copies 2 unchanged; feasibility checked by world.py)
    "keys3": Cond("keys", 5, "3 keys/agent", {}, {"keys_per_agent": 3}),
    "keys8": Cond("keys", 5, "8 keys/agent", {}, {"keys_per_agent": 8}),
    # factor 5: rounds
    "r40": Cond("rounds", 6, "40 rounds", {}, {"max_rounds": 40}),
}


# ------------------------------------------------------------------ cost model
# Per-call cost in round r (USD): price * (prompt(r) * $0.0142/M + 350 * $0.0445/M), prompt(r) =
# min(686 + g(N) (r-1), cap). g(N) = 49 N tokens/round (fitted: N=10 G1 ~490, N=20 smoke ~820 => 41 N,
# N=5 smoke ~230 => 46 N; 49 N is the conservative envelope), cap 16500 (context budget 14000 Qwen
# tokens ~ 16k gpt-oss tokens). Calls per branch-round = N. price = realized / fitted list price:
# G1 1.06; the smokes and G3 on 2026-11-07 1.35-1.6. Calibration (price 1.06): N=10 20-round branch
# $0.0187 (measured G1 $0.0177 = $0.85 / 48 branches); N=20 $0.083 (smoke pair $0.147 => $0.074);
# N=5 $0.0072 (smoke pair $0.0146 => $0.0073).
def call_cost(r: int, n: int, price: float) -> float:
    prompt = min(686 + 49 * n * (r - 1), 16500)
    return price * (prompt * 0.0142 + 350 * 0.0445) / 1e6


def branch_cost(c: Cond, price: float, from_round: int = 1) -> float:
    return sum(c.n * call_cost(r, c.n, price) for r in range(from_round, c.rounds + 1))


def projection(c: Cond, price: float, seeds: int | None = None,
               only: list[str] | None = None) -> dict[str, float]:
    """Paid cost per seed by branch, and the cell total. God-only: control replays (about 15% of
    it re-pays where G1's cached control had an uncacheable empty response); God pays from its fork
    round (G1's cached prefix). Game cells: control pays everything, God and B'' from their fork."""
    s = c.seeds if seeds is None else seeds
    f = c.fork_round
    if c.god_only:
        per = {"god": branch_cost(c, price, f), "nogod": 0.15 * branch_cost(c, price),
               "noisefork": branch_cost(c, price, f)}
    else:
        per = {"nogod": branch_cost(c, price), "god": branch_cost(c, price, f),
               "noisefork": branch_cost(c, price, f)}
    if only:
        per = {k: (v if k in only else 0.0) for k, v in per.items()}
    seed_cost = sum(per.values())
    calls = sum(c.n * (c.rounds - (f - 1 if k != "nogod" or c.god_only else 0)) for k in per if per[k] > 0)
    return {**per, "per_seed": seed_cost, "total": seed_cost * s, "calls": calls * s}


# ------------------------------------------------------------------ configs
def config_dict(name: str, c: Cond, conc: int = 4) -> dict[str, Any]:
    fork: dict[str, Any] = {"enabled": True, "noise_replica": False, "serial_pair": False, "noise_fork": True}
    if c.god_only:
        fork["control_god"] = G1_GOD
    game = {"n_agents": 10, "max_rounds": 20, "turn_mode": "simultaneous", "turn_order": "reshuffle", **c.game}
    return {
        "extends": "../base.yaml",
        "run": {"name": f"sweep_{name}", "seed": 1, "n_games": c.seeds, "max_concurrency_games": c.conc_games},
        "reasoning": {"salvage_truncated": True},
        "game": game,
        "god": c.god_cfg,
        "runner": {"engine": "real", "report": True, "fork": fork},
        "roles": {r: {"backend": "openrouter", "model": "openai/gpt-oss-120b",
                      **({"max_tokens": 1000} if r == "agent" else {})}
                  for r in ("agent", "god", "judge", "summarizer")},
        "llm": {"budget_usd": c.cap, "cache": {"mode": "read_write"},
                "openrouter": {"enabled": True, "model": "openai/gpt-oss-120b", "max_concurrency": conc,
                               "global_cap_usd": 20.0, "price_in_per_mtok": 0.03, "price_out_per_mtok": 0.15,
                               "reasoning_effort": "low", "provider_sort": "price"}},
    }


def dump_yaml(o: dict, ind: int = 0) -> str:
    """Small YAML emitter (flow style for leaves) -- short, diffable files."""
    pad, out = "  " * ind, []
    for k, v in o.items():
        if isinstance(v, dict) and v and any(isinstance(x, (dict, list)) for x in v.values()):
            out.append(f"{pad}{k}:\n{dump_yaml(v, ind + 1)}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.append(f"{pad}{k}:")
            out += [f"{pad}- {json.dumps(x)}" for x in v]
        else:
            out.append(f"{pad}{k}: {json.dumps(v)}")
    return "\n".join(out)


def write_configs(names: list[str]) -> list[Path]:
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        c = CONDS[name]
        head = (f"# Sweep condition {name} (factor {c.factor}, priority {c.priority}): {c.desc}.\n"
                f"# Generated by infra/sweep.py write (plan/SWEEP_PLAN.md); base = configs/grid_oss_sim_k2.yaml (G1).\n"
                f"# Branches: god, nogod, noisefork (B'' resamples the control from God's first round, r{c.fork_round}).\n"
                + ("# control_god = G1's god: the control is G1's nogod game (cache replay).\n" if c.god_only
                   else "# Own paired control (the game itself changes).\n")
                + "# llm.budget_usd = this cell's hard cap.\n")
        p = CONF_DIR / f"{name}.yaml"
        p.write_text(head + dump_yaml(config_dict(name, c)) + "\n")
        paths.append(p)
    return paths


# ------------------------------------------------------------------ dry run + checks
def conda_py(args: list[str], env: dict[str, str] | None = None, log: Path | None = None) -> int:
    e = {**os.environ, "PYTHONNOUSERSITE": "1", **(env or {})}
    out = open(log, "w") if log else None
    try:
        return subprocess.call(PY + args, cwd=REPO, env=e, stdout=out, stderr=subprocess.STDOUT if out else None)
    finally:
        if out:
            out.close()


def dry(names: list[str], rounds: int, seeds: int) -> bool:
    write_configs(names)
    ok_all = True
    for name in names:
        c = CONDS[name]
        r = max(rounds, max(rr for rr, _ in c.god_rounds) + 2)
        run_id = f"fake_sweep_{name}"
        if (REPO / "runs" / run_id).is_dir():          # our own fake output; run dirs append, so start clean
            shutil.rmtree(REPO / "runs" / run_id)
        args = ["-m", "swarm.run", "--config", f"configs/sweep/{name}.yaml", "--run-id", run_id,
                "roles.agent.backend=fake", "roles.god.backend=fake", "roles.judge.backend=fake",
                "roles.summarizer.backend=fake", "llm.openrouter.enabled=false", "llm.cache.mode='off'",
                "runner.report=false", f"game.max_rounds={r}", f"run.n_games={seeds}"]
        rc = conda_py(args, log=REPO / "runs" / f"{run_id}.log")
        problems = [f"exit {rc}"] if rc else check_dry(name, REPO / "runs" / run_id, r, seeds)
        ok_all &= not problems
        print(f"{name:13s} rounds {r:2d} seeds {seeds}: " + ("OK" if not problems else "FAIL " + "; ".join(problems)))
    return ok_all


def _events(gdir: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(gdir / "events.jsonl", encoding="utf-8")]


def check_dry(name: str, run_dir: Path, rounds: int, seeds: int) -> list[str]:
    """Verify in the logs that the condition's knobs took effect. Returns a list of problems."""
    from swarm.agents.llm_agent import system_context
    from swarm.game.world import generate_world, is_feasible
    from swarm.prompts import load_prompt_set
    from swarm.runner.run import load_experiment
    from swarm.runner.spec import world_seed

    c = CONDS[name]
    probs: list[str] = []
    cfg = load_experiment(REPO / "configs" / "sweep" / f"{name}.yaml", [f"game.max_rounds={rounds}"])[0]
    for i in range(seeds):
        br = {b: run_dir / "games" / f"g{i:02d}-{b}" for b in ("god", "nogod", "noisefork")}
        missing = [b for b, d in br.items() if not (d / "outcome.json").exists()]
        if missing:
            probs.append(f"g{i:02d} missing {missing}")
            continue
        ev = {b: _events(d) for b, d in br.items()}
        metas = {b: json.loads((d / "meta.json").read_text()) for b, d in br.items()}
        if len({m["game_id"] for m in metas.values()}) != 1:
            probs.append(f"g{i:02d} branches do not share a game_id")
        # God: powers exactly at the scheduled rounds (K = number of entries), only in the god branch
        got = sorted((e["round"], e["data"]["power"]) for e in ev["god"] if e["type"] == "god_power"
                     and e["data"].get("ok", True))
        if got != c.god_rounds:
            probs.append(f"g{i:02d} god powers {got} != {c.god_rounds}")
        for b in ("nogod", "noisefork"):
            if any(e["type"] == "god_power" for e in ev[b]):
                probs.append(f"g{i:02d} {b} has god powers")
        # B'': agent seeds equal to the control's before the fork round, all different from it on
        seeds_by = {}
        for b in ("nogod", "noisefork"):
            seeds_by[b] = {}
            for ln in open(br[b] / "llm_calls.jsonl", encoding="utf-8"):
                d = json.loads(ln)
                m = d["request"]["meta"]
                if m.get("role") == "agent":
                    seeds_by[b][(m["agent"], m["round"], m["attempt"], m["purpose"])] = d["request"]["seed"]
        shared = set(seeds_by["nogod"]) & set(seeds_by["noisefork"])
        pre = [k for k in shared if k[1] < c.fork_round]
        post = [k for k in shared if k[1] >= c.fork_round]
        if any(seeds_by["nogod"][k] != seeds_by["noisefork"][k] for k in pre):
            probs.append(f"g{i:02d} B'' seeds differ before r{c.fork_round}")
        if not post or any(seeds_by["nogod"][k] == seeds_by["noisefork"][k] for k in post):
            probs.append(f"g{i:02d} B'' seeds not resampled from r{c.fork_round}")
        # game knobs
        start = next(e for e in ev["nogod"] if e["type"] == "game_start")["data"]
        if start["n_agents"] != c.n or start["max_rounds"] != rounds:
            probs.append(f"g{i:02d} n_agents/max_rounds {start['n_agents']}/{start['max_rounds']}")
        w = generate_world(cfg.game, world_seed(cfg, i), "w")
        kpa = {len(a.keys) for a in w.agents.values()}
        if kpa != {cfg.game.keys_per_agent} or not is_feasible(w) or len(w.agents) != c.n:
            probs.append(f"g{i:02d} world keys/agent {kpa}, feasible {is_feasible(w)}")
        rp = json.loads((br["nogod"] / "outcome.json").read_text())["rounds_played"]
        if rp > rounds:
            probs.append(f"g{i:02d} rounds_played {rp} > {rounds}")
        probs += _board_checks(name, cfg, ev["nogod"], i)
    sysp = load_prompt_set("agent/v1").render("system.j2", **system_context(cfg))
    if (not cfg.game.board.enabled) == ("post_board" in sysp):
        probs.append("system prompt post_board offer does not match game.board.enabled")
    return probs


def _board_checks(name: str, cfg: Any, ev: list[dict], i: int) -> list[str]:
    b = cfg.game.board
    probs = []
    msgs = {e["data"]["message"]["msg_id"]: e["data"]["message"] for e in ev if e["type"] == "message"}
    board_msgs = [m for m in msgs.values() if m["channel"] == "board"]
    attempts = [e for e in ev if e["type"] == "action_result" and e["data"]["action"]["kind"] == "post_board"]
    if not b.enabled:
        if board_msgs:
            probs.append(f"g{i:02d} board off but {len(board_msgs)} board posts")
        if any(e["data"]["ok"] or e["data"]["reason"] != "board_disabled" for e in attempts):
            probs.append(f"g{i:02d} board off but a post_board did not fail board_disabled")
        if any(t["data"]["obs"]["board"] for t in ev if t["type"] == "turn"):
            probs.append(f"g{i:02d} board off but an observation has board items")
        return probs
    if not board_msgs:
        return [f"g{i:02d} no board posts to check (fake policy)"] if name.startswith("board") else []
    # per post, the earliest round any viewer sees it minus the round it was posted: must be exactly
    # 1 + visible_delay_rounds in sim mode (skipped viewers may see it later, nobody earlier)
    first: dict[str, int] = {}
    for t in ev:
        if t["type"] != "turn":
            continue
        for m in t["data"]["obs"]["board"]:
            first[m["msg_id"]] = min(first.get(m["msg_id"], 10**9), t["round"])
            if b.anonymous and m["sender"] != "(anonymous)":
                probs.append(f"g{i:02d} anonymous board shows a sender")
                break
    lags = {r - msgs[mid]["round"] for mid, r in first.items()}
    want = 1 + b.visible_delay_rounds if cfg.game.turn_mode == "simultaneous" else None
    if want is not None and lags and lags != {want}:
        probs.append(f"g{i:02d} board lag {sorted(lags)} != {{{want}}}")
    if b.post_skip_rounds:
        posted = {(e["data"]["agent"], e["round"]) for e in ev if e["type"] == "action_result"
                  and e["data"]["action"]["kind"] == "post_board" and e["data"]["ok"]}
        turns = {(t["data"]["agent"], t["round"]) for t in ev if t["type"] == "turn"}
        for a, r in posted:
            if any((a, r + k) in turns for k in range(1, b.post_skip_rounds + 1)):
                probs.append(f"g{i:02d} {a} posted r{r} but took a turn within {b.post_skip_rounds} rounds")
                break
    return probs


# ------------------------------------------------------------------ launch
def launch(names: list[str], tag: str, run_prefix: str, seeds: int | None, only: list[str] | None,
           conc: int, price: float, dry_run: bool, cap_usd: float | None = None,
           paid_seeds: int | None = None, ledger_stop: float | None = None, max_fallback: float | None = None,
           driver_log: str = "runs/sweep_driver.log") -> int:
    write_configs(names)
    cells, caps, total = [], [], 0.0
    for name in names:
        c = CONDS[name]
        s = seeds or c.seeds
        ps = paid_seeds or s                 # seeds that are not already cached (projection / watchdog model)
        pr = projection(c, price, ps, only)
        total += pr["total"]
        branches = only or ["god", "nogod", "noisefork"]
        paid = sum(1 for b in branches if b != "nogod")
        ctl = int("nogod" in branches and not c.god_only)
        opts = [f"n={c.n}", f"g={49 * c.n}", "pcap=16500", f"rounds={c.rounds}", f"seeds={ps}",
                f"shared={c.fork_round - 1}", f"paid={paid}", f"ctl={ctl}"]
        cap = cap_usd if cap_usd is not None else round(1.25 * pr["total"] + RESERVE_MARGIN, 2)
        caps.append(cap)
        cells.append(f"sweep/{name}:{cap:.2f}:{len(branches)}:{','.join(opts)}")
        print(f"{name:13s} seeds {s} projected ${pr['total']:.3f} (cap ${cap:.2f}) calls ~{pr['calls']:.0f}")
    print(f"total projected ${total:.3f}; hard caps ${sum(caps):.2f}")
    sets = [f"llm.openrouter.max_concurrency={conc}"]
    if seeds:
        sets.append(f"run.n_games={seeds}")
    if only:
        sets += [f"runner.fork.only_branches={json.dumps(only)}", "runner.report=false"]
    cmd = PY + [str(REPO / "infra" / "grid_driver.py"), "--seeds", str(seeds or 8), "--rounds", "20",
                "--ledger-stop", str(min(LEDGER_STOP, ledger_stop or SWEEP_LEDGER_STOP)), "--log", driver_log, "--run-prefix", run_prefix,
                "--cap-override"] + (["--max-fallback", str(max_fallback)] if max_fallback is not None else [])
    for cell in cells:
        cmd += ["--cell", cell]
    for s_ in sets:
        cmd += ["--set", s_]
    print("command:", " ".join(cmd))
    if dry_run:
        return 0
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "OPENROUTER_API_KEY": KEY_FILE.read_text().strip()}
    out = open(REPO / "runs" / f"sweep_driver_{tag}.out", "a")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    print(f"driver {tag} pid/pgid {p.pid} (log {driver_log}, out runs/sweep_driver_{tag}.out)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--price", type=float, default=1.35)
    sub.add_parser("write").add_argument("conds", nargs="*")
    d = sub.add_parser("dry")
    d.add_argument("conds", nargs="*")
    d.add_argument("--rounds", type=int, default=16)
    d.add_argument("--seeds", type=int, default=2)
    la = sub.add_parser("launch")
    la.add_argument("conds", nargs="+")
    la.add_argument("--tag", default="L")
    la.add_argument("--run-prefix", default="")
    la.add_argument("--seeds", type=int, default=None)
    la.add_argument("--only-branches", nargs="*", default=None)
    la.add_argument("--conc", type=int, default=4)
    la.add_argument("--price", type=float, default=1.35)
    la.add_argument("--dry-run", action="store_true")
    la.add_argument("--cap", type=float, default=None, help="hard cap per cell (default 1.25 x projection)")
    la.add_argument("--paid-seeds", type=int, default=None, help="seeds not yet cached (projection only)")
    la.add_argument("--ledger-stop", type=float, default=None,
                    help=f"driver ledger stop (default {SWEEP_LEDGER_STOP}; never above {LEDGER_STOP})")
    la.add_argument("--max-fallback", type=float, default=None,
                    help="driver aborts when this share of fresh calls is not from the primary provider")
    la.add_argument("--driver-log", default="runs/sweep_driver.log")
    st = sub.add_parser("stats")
    st.add_argument("conds", nargs="*")
    a = ap.parse_args()
    names = [n for n in (getattr(a, "conds", None) or ["all"]) if n]
    names = list(CONDS) if names == ["all"] else names
    unknown = [n for n in names if n not in CONDS]
    if unknown:
        ap.error(f"unknown conditions {unknown}; known: {list(CONDS)}")
    if a.cmd == "list":
        print(f"{'cond':13s} {'factor':7s} pri seeds  fork  cap    $/seed  total   calls  (price x{a.price})")
        for n, c in sorted(CONDS.items(), key=lambda kv: kv[1].priority):
            pr = projection(c, a.price)
            print(f"{n:13s} {c.factor:7s} {c.priority:3d} {c.seeds:5d}  r{c.fork_round:<3d} ${c.cap:.2f}  "
                  f"${pr['per_seed']:.4f} ${pr['total']:.3f} {pr['calls']:6.0f}")
        return 0
    if a.cmd == "write":
        for p in write_configs(names):
            print(p.relative_to(REPO))
        return 0
    if a.cmd == "dry":
        return 0 if dry(names, a.rounds, a.seeds) else 1
    if a.cmd == "launch":
        return launch(names, a.tag, a.run_prefix, a.seeds, a.only_branches, a.conc, a.price, a.dry_run, a.cap,
                      a.paid_seeds, a.ledger_stop, a.max_fallback, a.driver_log)
    if a.cmd == "stats":
        runs = [str(REPO / "runs" / f"sweep_{n}") for n in names if (REPO / "runs" / f"sweep_{n}").exists()]
        return conda_py(["-m", "swarm.analysis.sweep_stats", *runs])
    return 2


if __name__ == "__main__":
    sys.exit(main())
