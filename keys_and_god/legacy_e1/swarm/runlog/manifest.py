"""Run manifest (WP5): manifest.json at start (status=running) and atomically rewritten at the end.

Fields: run_id, schema_version, status, started_utc, ended_utc, wall_s, git{sha,dirty,diff_sha256},
host{python,platform,gpu,packages}, config_sha256, seeds{run,per_game}, prompts{role:{set,hash,files}},
models{role:{backend,model,base_url,api_key_env,api_key_present}}, usage, games[], forks[], errors[].

SECRETS: the manifest never contains a key. `scrub()` drops any key-like field (except the
whitelisted `api_key_env` / `api_key_present`) and redacts registered secrets plus anything that looks
like an OpenRouter/OpenAI key; it is applied to the whole manifest before every write.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from swarm import SCHEMA_VERSION
from swarm.config import REPO_ROOT, RunConfig
from swarm.llm.secrets import key_status, redact
from swarm.runlog.writer import atomic_write_json

PROMPTS_ROOT = REPO_ROOT / "swarm" / "prompts" / "templates"
_KEYLIKE = re.compile(r"(api[_-]?key|secret|password|token|authorization|bearer)", re.I)
_KEY_WHITELIST = {"api_key_env", "api_key_present", "api_key_source", "prompt_tokens",
                  "completion_tokens", "reasoning_tokens", "max_tokens", "budget_tokens",
                  "tokenizer", "max_tokens_total"}
_KEY_VALUE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|sk-or-[A-Za-z0-9_\-]{8,})")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(cfg: RunConfig, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.UTC)
    return f"{cfg.run.name}_{now.strftime('%Y%m%dT%H%M%S')}_{cfg.config_hash()[:6]}"


# ------------------------------------------------------------------ scrubbing
def scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if _KEYLIKE.search(ks) and ks not in _KEY_WHITELIST and not ks.endswith("_tokens"):
                continue
            out[ks] = scrub(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return _KEY_VALUE.sub("***", redact(obj))
    return obj


# ------------------------------------------------------------------ host / git
def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def git_info(repo: Path = REPO_ROOT, save_diff_to: Path | None = None) -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"], repo)
    if sha is None:
        return {"sha": None, "dirty": None, "diff_sha256": None}
    status = _run(["git", "status", "--porcelain"], repo) or ""
    diff = _run(["git", "diff", "HEAD"], repo) or ""
    dirty = bool(status.strip())
    if dirty and save_diff_to is not None:
        save_diff_to.write_text(redact(diff), encoding="utf-8")
    return {"sha": sha.strip(), "dirty": dirty,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None}


def host_info() -> dict[str, Any]:
    pk = {}
    for name in ("openai", "pydantic", "jinja2", "httpx", "matplotlib", "networkx", "vllm"):
        try:
            pk[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = None
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        gpu = out.strip() if out else None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "gpu": gpu, "packages": pk}


# ------------------------------------------------------------------ prompts
def prompt_set_hash(set_name: str, root: Path = PROMPTS_ROOT) -> dict[str, Any]:
    """sha256(sorted (relpath, sha256(file bytes))). Missing set -> hash None (recorded, not fatal)."""
    d = root / set_name
    if not d.is_dir():
        return {"set": set_name, "hash": None, "files": {}, "missing": True}
    files = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    h = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"set": set_name, "hash": h, "files": files}


def prompts_info(cfg: RunConfig, roles: tuple[str, ...] = ("agent", "god")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in roles:
        rc = getattr(cfg.roles, role)
        info = prompt_set_hash(rc.prompt_set)
        if rc.framing:
            info["framing"] = prompt_set_hash(rc.framing)
        out[role] = info
    return out


def copy_prompts(cfg: RunConfig, dest: Path, roles: tuple[str, ...] = ("agent", "god")) -> None:
    for role in roles:
        rc = getattr(cfg.roles, role)
        for name in filter(None, (rc.prompt_set, rc.framing)):
            src = PROMPTS_ROOT / name
            if src.is_dir():
                shutil.copytree(src, dest / name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__"))


# ------------------------------------------------------------------ models
def models_info(cfg: RunConfig, roles: tuple[str, ...] = ("agent", "god")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in roles:
        backend, model = cfg.model_for(role)
        rc = getattr(cfg.roles, role)
        m: dict[str, Any] = {"backend": backend, "model": model, "temperature": rc.temperature,
                             "top_p": rc.top_p, "max_tokens": rc.max_tokens}
        if backend == "vllm":
            m.update(base_url=cfg.llm.vllm.base_url, max_model_len=cfg.llm.vllm.max_model_len)
        elif backend == "openrouter":
            m.update(base_url=cfg.llm.openrouter.base_url, **key_status(cfg.llm.openrouter.api_key_env))
        out[role] = m
    return out


# ------------------------------------------------------------------ manifest
class Manifest:
    """Mutable manifest dict + atomic writer. All writes are scrubbed."""

    def __init__(self, run_dir: Path, run_id: str, cfg: RunConfig, *, runner_opts: dict[str, Any] | None = None,
                 overrides: list[str] | None = None) -> None:
        self.path = Path(run_dir) / "manifest.json"
        self._t0 = time.monotonic()
        self.data: dict[str, Any] = {
            "run_id": run_id, "schema_version": SCHEMA_VERSION, "status": "running",
            "started_utc": utc_now(), "ended_utc": None, "wall_s": None,
            "argv": sys.argv[1:], "overrides": list(overrides or []),
            "git": git_info(save_diff_to=Path(run_dir) / "git.diff"),
            "host": host_info(),
            "config_sha256": cfg.config_hash(), "config_name": cfg.run.name,
            "runner": dict(runner_opts or {}),
            "seeds": {"run": cfg.run.seed, "per_game": []},
            "prompts": prompts_info(cfg), "models": models_info(cfg),
            "usage": {}, "games": [], "forks": [], "errors": [],
        }

    def add_game(self, entry: dict[str, Any]) -> None:
        self.data["games"].append(entry)
        if "seed" in entry:
            self.data["seeds"]["per_game"].append({"label": entry.get("label"), "seed": entry["seed"]})

    def write(self) -> None:
        atomic_write_json(self.path, scrub(self.data))

    def finalize(self, status: str, usage: dict[str, Any] | None = None) -> None:
        self.data.update(status=status, ended_utc=utc_now(), wall_s=round(time.monotonic() - self._t0, 3))
        if usage is not None:
            self.data["usage"] = usage
        self.write()
