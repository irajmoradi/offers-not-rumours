"""Experiment runner (WP9): N games (and their paired forks) in one asyncio pool -> runs/<run_id>/.

    python -m swarm.run --config configs/demo.yaml [a.b.c=value ...] [--engine synth] [--no-report]

Side effects live here and in runlog: the run directory, manifest, per-game logs, report.
Each game is isolated: an exception marks that game `error` (traceback in games/<label>/error.txt)
and the pool continues. BudgetExceeded marks it `budget_exceeded`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from swarm.config import REPO_ROOT, RunConfig, apply_overrides, dump_yaml, from_dict, load_raw
from swarm.llm.secrets import install_log_redaction
from swarm.runlog.manifest import Manifest, copy_prompts, make_run_id
from swarm.runlog.writer import CURRENT_GAME, RunWriter, atomic_write_text
from swarm.runner.spec import GameSpec, RunnerOpts, build_specs
from swarm.runner.wiring import Clients, build_clients, close_clients, play_game

log = logging.getLogger("swarm.run")


# --------------------------------------------------------------------------- config
def load_experiment(path: str | Path, overrides: list[str] | None = None) -> tuple[RunConfig, RunnerOpts]:
    """Load an experiment YAML. The `runner:` block (engine, fork, report) is split off before
    RunConfig validation (RunConfig is extra=forbid). Overrides `runner.x=y` go to RunnerOpts."""
    raw = load_raw(path)
    runner_raw = raw.pop("runner", None) or {}
    cfg_ov, run_ov = [], []
    for o in overrides or []:
        (run_ov if o.startswith("runner.") else cfg_ov).append(o)
    if run_ov:
        runner_raw = apply_overrides({"runner": runner_raw}, run_ov)["runner"]
    return from_dict(raw, cfg_ov), RunnerOpts.from_dict(runner_raw)


# --------------------------------------------------------------------------- pool
async def _run_one(spec: GameSpec, run: RunWriter, clients: Clients, engine: str,
                   sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        gw = run.game(spec.label, spec.game_id, spec.meta())
        token = CURRENT_GAME.set(gw)
        t0 = time.monotonic()
        entry = dict(spec.meta())
        try:
            outcome = await play_game(spec, gw, clients, engine)
            gw.write_outcome(outcome)
            # the engine ends a game on BudgetExceeded itself: report that, not "done" (CODE_REVIEW #11)
            status = "budget_exceeded" if outcome.end_reason == "budget_exceeded" else "done"
            gw.close(status)
            entry.update(status=status, end_reason=outcome.end_reason, rounds=outcome.rounds_played,
                         welfare=round(outcome.welfare, 4), win_rate=round(outcome.win_rate, 4))
        except Exception as e:  # isolation: one game failing never stops the pool
            from swarm.llm.base import BudgetExceeded

            status = "budget_exceeded" if isinstance(e, BudgetExceeded) else "error"
            (gw.dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            gw.close(status, f"{type(e).__name__}: {e}")
            entry.update(status=status, error=f"{type(e).__name__}: {e}")
            log.error("game %s failed: %s", spec.label, e)
        finally:
            CURRENT_GAME.reset(token)
        entry.update(wall_s=round(time.monotonic() - t0, 2), dir=f"games/{spec.label}")
        log.info("game %s %s (%.1fs)", spec.label, entry["status"], entry["wall_s"])
        return entry


async def _run_group(group: list[GameSpec], run: RunWriter, clients: Clients, opts: RunnerOpts,
                     sem: asyncio.Semaphore) -> list[dict[str, Any]]:
    # Concurrent branches rely on the client's single-flight dedup, which needs the response cache.
    serial = opts.fork.serial_pair or group[0].cfg.llm.cache.mode == "off"
    if serial and len(group) > 1:
        first = await _run_one(group[0], run, clients, opts.engine, sem)
        rest = await asyncio.gather(*(_run_one(s, run, clients, opts.engine, sem) for s in group[1:]))
        return [first, *rest]
    return list(await asyncio.gather(*(_run_one(s, run, clients, opts.engine, sem) for s in group)))


async def run_experiment(cfg: RunConfig, opts: RunnerOpts, *, overrides: list[str] | None = None,
                         out_root: str | Path | None = None, run_id: str | None = None) -> Path:
    install_log_redaction()
    run_id = run_id or make_run_id(cfg)
    root = Path(out_root) if out_root is not None else Path(cfg.run.out_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_dir = root / run_id
    run = RunWriter(run_dir, run_id)
    resolved = dump_yaml(cfg) + yaml.safe_dump({"runner": opts.to_dict()}, sort_keys=False)
    atomic_write_text(run_dir / "config.resolved.yaml", resolved)
    copy_prompts(cfg, run_dir / "prompts")
    man = Manifest(run_dir, run_id, cfg, runner_opts=opts.to_dict(), overrides=overrides)
    man.data["budget_usd"] = cfg.llm.budget_usd
    man.write()

    status = "done"
    clients = Clients()
    try:
        groups = build_specs(cfg, opts)
        clients = build_clients(cfg, run, opts.engine)
        sem = asyncio.Semaphore(cfg.run.max_concurrency_games)
        results = await asyncio.gather(*(_run_group(g, run, clients, opts, sem) for g in groups))
        for grp, res in zip(groups, results, strict=True):
            for e in res:
                man.add_game(e)
            if len(grp) > 1:
                man.data["forks"].append({"game_idx": grp[0].game_idx,
                                          "branches": {s.branch: f"games/{s.label}" for s in grp}})
        sts = {e["status"] for r in results for e in r}
        if "budget_exceeded" in sts:
            status = "budget_exceeded"
        elif sts - {"done"}:
            status = "partial" if "done" in sts else "error"
    except Exception as e:
        status = "error"
        man.data["errors"].append(f"{type(e).__name__}: {e}")
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        await close_clients(clients)
        run.close()
        man.finalize(status, run.usage().to_dict(cfg.llm.budget_usd))

    if opts.report:
        try:
            from swarm.viz.report import build_report

            path = build_report(run_dir)
            log.info("report: %s", path)
        except Exception as e:  # a report failure never invalidates the run
            man.data["errors"].append(f"report: {type(e).__name__}: {e}")
            man.write()
            log.error("report failed: %s", e)
    return run_dir


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m swarm.run", description=__doc__.splitlines()[0])
    ap.add_argument("--config", "-c", required=True, help="experiment YAML (extends base.yaml)")
    ap.add_argument("overrides", nargs="*", help="a.b.c=value overrides (runner.* go to the runner block)")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="same as a positional override")
    ap.add_argument("--engine", choices=["real", "synth"], help="override runner.engine")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--out", help="output root (default: run.out_dir)")
    ap.add_argument("--run-id", help="explicit run id (default: <name>_<utc>_<cfghash6>)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ov = list(args.overrides) + list(args.sets)
    if args.engine:
        ov.append(f"runner.engine={args.engine}")
    if args.no_report:
        ov.append("runner.report=false")
    cfg, opts = load_experiment(args.config, ov)
    run_dir = asyncio.run(run_experiment(cfg, opts, overrides=ov, out_root=args.out, run_id=args.run_id))
    print(run_dir)
    man = (run_dir / "manifest.json").read_text()
    return 0 if '"status": "done"' in man else 1


if __name__ == "__main__":
    sys.exit(main())
