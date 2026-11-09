"""API-key handling. The key NEVER goes into config, logs, events, llm_calls, manifests or reprs.

- load_openrouter_key(): env OPENROUTER_API_KEY first, else an explicit key file holding the raw key
  (no KEY= prefix). Returns pydantic SecretStr (repr/str are masked) and registers the value with
  `redact()`.
- redact(text): replaces every registered secret with "***". RunWriter/log handlers should pass text
  through it; `install_log_redaction()` adds a logging filter that does this for every record.
- key_status(): {api_key_env, api_key_present} for manifests.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import SecretStr

OPENROUTER_ENV = "OPENROUTER_API_KEY"
# Raw-key file supplied by the lead (outside the repo). Pass explicitly; never copy it into the repo.
DEFAULT_OPENROUTER_KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", ""))

_SECRETS: set[str] = set()


def register_secret(value: str) -> None:
    """Remember a secret value so redact() masks it."""
    if value and len(value) >= 8:
        _SECRETS.add(value)


def redact(text: str) -> str:
    """Replace every registered secret in text with '***'."""
    for s in _SECRETS:
        if s in text:
            text = text.replace(s, "***")
    return text


def mask(secret: SecretStr | str | None) -> str:
    """For doctor output only: 'sk-or-...ab12'."""
    if secret is None:
        return "<missing>"
    v = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
    if len(v) < 12:
        return "***"
    return f"{v[:6]}...{v[-4:]}"


def load_openrouter_key(key_file: str | Path | None = None, *, env: Mapping[str, str] | None = None,
                        env_var: str = OPENROUTER_ENV) -> SecretStr | None:
    """Key from env (first) or the raw-key file; registered for redaction; None if absent."""
    env = os.environ if env is None else env
    value = (env.get(env_var) or "").strip()
    if not value and key_file is not None:
        p = Path(key_file)
        if p.is_file():
            value = p.read_text(encoding="utf-8").strip()
    if not value:
        return None
    register_secret(value)
    return SecretStr(value)


def key_status(env_var: str = OPENROUTER_ENV, *, env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Manifest-safe key info: env var name and whether it is set."""
    env = os.environ if env is None else env
    return {"api_key_env": env_var, "api_key_present": bool((env.get(env_var) or "").strip())}


class RedactingFilter(logging.Filter):
    """logging.Filter that masks registered secrets in every record."""
    def filter(self, record: logging.LogRecord) -> bool:
        """Mask secrets in the record's message; always keep the record."""
        if _SECRETS:
            msg = record.getMessage()
            red = redact(msg)
            if red != msg:
                record.msg, record.args = red, None
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> None:
    """Attach the filter to `logger` (root by default) and to each of its current handlers.
    Handlers added later need `h.addFilter(RedactingFilter())` themselves."""
    lg = logger or logging.getLogger()
    f = RedactingFilter()
    lg.addFilter(f)
    for h in lg.handlers:
        h.addFilter(f)
