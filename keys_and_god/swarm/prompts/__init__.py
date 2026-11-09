"""Prompt template sets (WP0 loader; WP3/WP4 own the template contents).

A set is a directory  swarm/prompts/templates/<role>/<name>/  of Jinja2 files (+ optional meta.yaml).
An optional FRAMING set (templates/framing/<name>/framing.j2) is rendered once and passed to every
template as the string variable `framing` ("" when no framing set).

Hashes:
  set hash      = sha256 over sorted (relpath, sha256(bytes)) of every file in the set dir
  combined hash = sha256(set hash + framing hash)    (PromptSet.hash; goes into manifest + LLMCall meta)
  prompt_hash(messages) = sha256(canonical_json(messages)) of the rendered request messages

Rendering: StrictUndefined (missing variable = error), autoescape off, trim/lstrip blocks, loader
restricted to the set dir. Context values must be primitives, dict/list of them, or AgentView
models: passing a truth-side record (GameState, Message, MemoryEntry, Event, ...) raises TypeError.
Never edit a set a logged run has used: copy it to a new name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jinja2
from pydantic import BaseModel

from swarm.rng import canonical_json, sha256_hex
from swarm.schema import AgentView, ChatMessage

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def hash_dir(d: Path) -> tuple[str, dict[str, str]]:
    """(combined hash, {relpath: file sha256}) of a directory."""
    files = {p.relative_to(d).as_posix(): sha256_hex(p.read_bytes())
             for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return sha256_hex(canonical_json(sorted(files.items()))), files


def prompt_hash(messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
    """sha256 of the canonical JSON of rendered messages."""
    return sha256_hex(canonical_json(list(messages)))


def _check_ctx(value: Any, path: str = "ctx") -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, AgentView):
        return
    if isinstance(value, BaseModel):
        raise TypeError(f"{path}: {type(value).__name__} is not an AgentView; templates get agent views only")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _check_ctx(v, f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_ctx(v, f"{path}[{i}]")
        return
    raise TypeError(f"{path}: unsupported template value type {type(value).__name__}")


def _env(d: Path) -> jinja2.Environment:
    return jinja2.Environment(loader=jinja2.FileSystemLoader(str(d)), undefined=jinja2.StrictUndefined,
                              autoescape=False, keep_trailing_newline=True, trim_blocks=True,
                              lstrip_blocks=True)


class PromptSet:
    """A loaded, hashed template set (+ optional framing set)."""
    def __init__(self, name: str, framing: str | None = None, root: Path | str = TEMPLATES_DIR) -> None:
        root = Path(root)
        self.name = name
        self.dir = root / name
        if not self.dir.is_dir():
            raise FileNotFoundError(f"prompt set {name!r} not found under {root}")
        self.set_hash, self.files = hash_dir(self.dir)
        self.framing_name = framing
        self.framing_text = ""
        self.framing_hash = ""
        self.framing_files: dict[str, str] = {}
        if framing:
            fdir = root / framing
            if not (fdir / "framing.j2").is_file():
                raise FileNotFoundError(f"framing set {framing!r} needs {fdir / 'framing.j2'}")
            self.framing_hash, self.framing_files = hash_dir(fdir)
            self.framing_text = _env(fdir).get_template("framing.j2").render().strip()
        self.hash = sha256_hex(self.set_hash + self.framing_hash)
        self._env = _env(self.dir)

    def has(self, template: str) -> bool:
        """Whether the set contains this template file."""
        return template in self.files

    def render(self, template: str, **ctx: Any) -> str:
        """Render a template with an AgentView-only context (StrictUndefined)."""
        _check_ctx(ctx)
        return self._env.get_template(template).render(framing=self.framing_text, **ctx)

    def manifest(self) -> dict[str, Any]:
        """Set name, hashes and per-file hashes for manifest.json."""
        return {"set": self.name, "hash": self.hash, "set_hash": self.set_hash, "files": self.files,
                "framing": self.framing_name, "framing_hash": self.framing_hash or None}


def load_prompt_set(name: str, framing: str | None = None, root: Path | str = TEMPLATES_DIR) -> PromptSet:
    """Load a PromptSet by name (e.g. 'agent/v1')."""
    return PromptSet(name, framing=framing, root=root)
