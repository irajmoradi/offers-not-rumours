"""Preflight checks (WP9): `python -m swarm.doctor [--config configs/demo.yaml]`.

Checks, each printed as OK / WARN / FAIL with one line of detail:
  config   the experiment YAML loads and validates (RunConfig + runner block)
  vllm     GET <base_url>/models answers and serves the configured model (only if a role uses vllm)
  disk     free space under run.out_dir and the LLM cache dir (WARN < 5 GB, FAIL < 0.5 GB)
  ledger   OpenRouter global spend so far vs global_cap_usd (WARN >= 80 %, FAIL >= 100 %)
  key      OpenRouter key presence (masked; only matters if openrouter is enabled)
Exit code 1 if any check FAILs. Never prints a key (mask() only). Network: one HTTP GET, 3 s timeout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from swarm.config import REPO_ROOT, RunConfig

GB = 1024**3


@dataclass
class Check:
    """One doctor line."""

    name: str
    status: str          # OK | WARN | FAIL
    detail: str

    def line(self) -> str:
        return f"[{self.status:4}] {self.name:7} {self.detail}"


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def check_vllm(cfg: RunConfig, timeout: float = 3.0) -> Check:
    """Is the vLLM OpenAI server up, and does it serve the configured model?"""
    roles = [r for r in ("agent", "god") if cfg.model_for(r)[0] == "vllm"
             and (r != "god" or cfg.god.kind == "llm")]
    if not roles:
        return Check("vllm", "OK", "no role uses vllm (skipped)")
    url = cfg.llm.vllm.base_url.rstrip("/") + "/models"
    want = {cfg.model_for(r)[1] for r in roles}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            served = {m.get("id") for m in json.loads(r.read()).get("data", [])}
    except Exception as e:
        return Check("vllm", "FAIL", f"{url} unreachable ({type(e).__name__}: {e})")
    missing = want - served
    if missing:
        return Check("vllm", "FAIL", f"server up but missing {sorted(missing)}; serves {sorted(served)}")
    return Check("vllm", "OK", f"{url} serves {sorted(want)}")


def check_disk(cfg: RunConfig) -> Check:
    """Free space where runs and the cache are written."""
    worst = None
    for p in (_abs(cfg.run.out_dir), _abs(cfg.llm.cache.path).parent):
        q = p
        while not q.exists():
            q = q.parent
        free = shutil.disk_usage(q).free
        worst = free if worst is None else min(worst, free)
    gb = (worst or 0) / GB
    status = "FAIL" if gb < 0.5 else "WARN" if gb < 5 else "OK"
    return Check("disk", status, f"{gb:.1f} GB free")


def check_ledger(cfg: RunConfig) -> Check:
    """OpenRouter global ledger spend vs cap."""
    o = cfg.llm.openrouter
    path = _abs(o.ledger_path)
    spent = 0.0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                spent += float(json.loads(line).get("cost_usd") or 0.0)
            except (ValueError, AttributeError):
                continue
    frac = spent / o.global_cap_usd if o.global_cap_usd else 0.0
    status = "FAIL" if frac >= 1 else "WARN" if frac >= 0.8 else "OK"
    return Check("ledger", status, f"${spent:.4f} of ${o.global_cap_usd:.2f} global cap "
                                   f"(openrouter {'enabled' if o.enabled else 'disabled'})")


def check_key(cfg: RunConfig) -> Check:
    """Key presence, masked. Only a FAIL when openrouter is enabled and the key is missing."""
    from swarm.llm.secrets import key_status

    st = key_status(cfg.llm.openrouter.api_key_env)
    if st["api_key_present"]:
        return Check("key", "OK", f"{st['api_key_env']} is set")
    status = "FAIL" if cfg.llm.openrouter.enabled else "OK"
    return Check("key", status, f"{st['api_key_env']} not set" + ("" if cfg.llm.openrouter.enabled
                                                                   else " (openrouter disabled)"))


def run_checks(config: str, overrides: list[str] | None = None, *, network: bool = True) -> list[Check]:
    """All checks; a config failure short-circuits the rest."""
    from swarm.runner.run import load_experiment

    try:
        cfg, opts = load_experiment(config, overrides)
    except Exception as e:
        return [Check("config", "FAIL", f"{config}: {type(e).__name__}: {str(e).splitlines()[0]}")]
    out = [Check("config", "OK", f"{config} -> {cfg.run.name} ({cfg.game.n_agents} agents, "
                                 f"{cfg.game.max_rounds} rounds, god={cfg.god.kind}, "
                                 f"fork={'on' if opts.fork.enabled else 'off'}, engine={opts.engine})")]
    if network and opts.engine != "synth":
        out.append(check_vllm(cfg))
    out += [check_disk(cfg), check_ledger(cfg), check_key(cfg)]
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m swarm.doctor", description=__doc__.splitlines()[0])
    ap.add_argument("--config", "-c", default="configs/demo.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--offline", action="store_true", help="skip the vLLM HTTP check")
    args = ap.parse_args(argv)
    checks = run_checks(args.config, args.overrides, network=not args.offline)
    for c in checks:
        print(c.line())
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
