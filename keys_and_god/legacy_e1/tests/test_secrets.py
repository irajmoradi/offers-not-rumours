"""The OpenRouter key must never appear in logs, events, llm_calls, configs or manifests.

Assertions only report file paths, never the key (pytest would print compared values otherwise).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from swarm.config import REPO_ROOT, from_dict
from swarm.llm import secrets
from swarm.schema import CallMeta, ChatMessage, ChatRequest, ChatResponse, LLMCall, write_jsonl

FAKE_KEY = "sk-or-v1-zqTESTCANARYKEY0123456789abcdef"
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt", ".log", ".j2", ".toml", ".cfg", ""}


def _files_containing(root: Path, needle: str) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES or ".git" in p.parts or p.stat().st_size > 5e6:
            continue
        try:
            if needle in p.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(p))
        except OSError:
            pass
    return hits


def test_key_loaded_from_env_or_file_and_masked(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert secrets.load_openrouter_key(env={}) is None
    kf = tmp_path / "keyfile"
    kf.write_text(FAKE_KEY + "\n")
    k = secrets.load_openrouter_key(kf, env={})
    found = k is not None and k.get_secret_value() == FAKE_KEY
    assert found
    leaked = FAKE_KEY in repr(k) or FAKE_KEY in str(k) or FAKE_KEY in secrets.mask(k)
    assert not leaked
    k2 = secrets.load_openrouter_key(kf, env={"OPENROUTER_API_KEY": "sk-or-env-key-0000000000"})
    assert k2.get_secret_value().startswith("sk-or-env")  # env wins


def test_key_never_in_run_dir(tmp_run_dir, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    key = secrets.load_openrouter_key()
    assert key is not None
    # logging, with redaction installed on a file handler
    log_path = tmp_run_dir / "run.log"
    lg = logging.getLogger("swarm.test_secrets")
    h = logging.FileHandler(log_path)
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    secrets.install_log_redaction(lg)
    lg.info("building client with key %s", key.get_secret_value())
    lg.info("repr %r", key)
    h.close()
    lg.removeHandler(h)
    # config (resolved), LLM records, manifest
    cfg = from_dict({}, ["llm.openrouter.enabled=true", "roles.god.backend=openrouter"])
    (tmp_run_dir / "config.resolved.yaml").write_text(json.dumps(cfg.model_dump(mode="json")))
    req = ChatRequest(call_id="c", messages=[ChatMessage(role="user", content="hi")], max_tokens=8,
                      temperature=0.0, seed=1, meta=CallMeta(role="god"))
    write_jsonl(tmp_run_dir / "llm_calls.jsonl",
                [LLMCall(run_id="r", backend="openrouter", model=cfg.llm.openrouter.model, request=req,
                         response=ChatResponse(call_id="c", text="ok"))])
    manifest = {"models": {"god": {"backend": "openrouter", **secrets.key_status()}}, "key_repr": repr(key),
                "err": secrets.redact(f"401 for key {key.get_secret_value()}")}
    (tmp_run_dir / "manifest.json").write_text(json.dumps(manifest))
    assert manifest["models"]["god"]["api_key_present"] is True
    assert _files_containing(tmp_run_dir, FAKE_KEY) == []
    assert "***" in log_path.read_text()


def test_real_key_not_in_repo_or_runs():
    """If the lead's key file exists, its value must not be anywhere in the repo tree or runs/."""
    key = secrets.load_openrouter_key(secrets.DEFAULT_OPENROUTER_KEY_FILE, env={})
    if key is None:
        return
    assert _files_containing(REPO_ROOT, key.get_secret_value()) == []
