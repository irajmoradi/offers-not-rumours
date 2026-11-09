"""Game specs (WP9): what one game in a run is, including paired-fork branches.

Paired fork (ripple measurement). For each game_idx we run:
  god   : the config as given (God active)                         -> branch A
  nogod : identical config with god.kind=none                      -> control B
  noise : (optional) nogod with run.replica="1"                    -> noise replica B'
  noisefork: (optional) nogod with run.resample_from_round = F     -> fork-matched noise B''
A and B share run.branch / run.replica, so every agent seed (derive_seed(..., branch, replica)) is
identical: before God's first intervention both branches see the same prompts with the same seeds
(common random numbers; with the LLM cache on, B is served A's cached responses, so the prefix is
exactly identical). They branch at God's first successful power. B' changes only the replica, so its
seeds/cache keys differ from B everywhere: B vs B' is the sampling-noise floor from round 1.
B'' (fork.noise_fork) keeps B's game_id and seeds before round F = God's first scheduled power (the
cached prefix is identical) and resamples every agent from F on: B'' vs B is the noise floor of a fork
AT GOD'S ROUND, so (A - B) - (B'' - B) = A - B'' is the God effect net of fork-matched noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swarm.config import RunConfig
from swarm.rng import derive_seed, make_game_id

BRANCH_GOD, BRANCH_NOGOD, BRANCH_NOISE, BRANCH_MAIN = "god", "nogod", "noise", "main"
BRANCH_NOISE_FORK = "noisefork"


@dataclass(frozen=True)
class ForkOpts:
    enabled: bool = False
    noise_replica: bool = False
    # False (default): branches run concurrently; the LLM client's single-flight dedup gives B the
    # same pre-fork responses as A. True: A first, then B/B' (warm cache). Forced on if cache is off.
    serial_pair: bool = False
    # None (default): the control is cfg with god.kind=none. A dict: the control (and the pair's id_hash)
    # is cfg with god={**cfg.god, **control_god, kind: none}. Sweep cells that change only `god` pass the
    # base cell's god section here, so they reuse the base cell's exact control games (same game_ids,
    # seeds and cache keys: the control replays from the cache, and A shares B's cached prefix).
    control_god: dict[str, Any] | None = None
    # True: add B'' (BRANCH_NOISE_FORK), the control resampled from God's first round (fork_round()).
    noise_fork: bool = False
    # None: run every branch. A list: run only these branches (e.g. ["noisefork"] to add B'' to a
    # finished pair; the pair's prefix comes from the LLM cache, game_ids are unchanged).
    only_branches: list[str] | None = None


@dataclass(frozen=True)
class RunnerOpts:
    """The `runner:` block of an experiment YAML (stripped before RunConfig validation)."""

    engine: str = "real"         # real | synth
    fork: ForkOpts = ForkOpts()
    report: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RunnerOpts:
        d = dict(d or {})
        fork = ForkOpts(**(d.pop("fork", None) or {}))
        unknown = set(d) - {"engine", "report"}
        if unknown:
            raise ValueError(f"unknown runner keys: {sorted(unknown)}")
        return cls(fork=fork, **d)

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "report": self.report,
                "fork": {"enabled": self.fork.enabled, "noise_replica": self.fork.noise_replica,
                         "serial_pair": self.fork.serial_pair, "control_god": self.fork.control_god,
                         "noise_fork": self.fork.noise_fork, "only_branches": self.fork.only_branches}}


@dataclass(frozen=True)
class GameSpec:
    game_idx: int
    branch: str
    cfg: RunConfig
    game_id: str
    world_seed: int
    label: str = field(default="")

    @property
    def god_active(self) -> bool:
        return self.cfg.god.kind != "none"

    def meta(self) -> dict[str, Any]:
        return {"label": self.label, "game_id": self.game_id, "game_idx": self.game_idx,
                "branch": self.branch, "god": self.cfg.god.kind, "seed": self.world_seed,
                "run_seed": self.cfg.run.seed, "replica": self.cfg.run.replica,
                "config_sha256": self.cfg.config_hash()}


def _with(cfg: RunConfig, **updates: dict[str, Any]) -> RunConfig:
    raw = cfg.model_dump(mode="json")
    for section, vals in updates.items():
        raw[section] = {**raw[section], **vals}
    return RunConfig.model_validate(raw)


def world_seed(cfg: RunConfig, game_idx: int) -> int:
    """Independent of branch/replica on purpose: all branches of a pair play the same world."""
    return derive_seed(cfg.run.seed, game_idx, "world")


def make_spec(cfg: RunConfig, game_idx: int, branch: str, id_hash: str | None = None) -> GameSpec:
    """id_hash: config hash used for the game_id. Paired branches A and B pass the SAME hash (that of
    the God-free config), so they share game_id and therefore every agent seed and cache key
    (LLMAgent seeds include game_id). B' differs through run.replica."""
    gid = make_game_id(cfg.run.seed, game_idx, id_hash or cfg.config_hash(), branch=cfg.run.branch,
                       replica=cfg.run.replica)
    return GameSpec(game_idx, branch, cfg, gid, world_seed(cfg, game_idx), f"g{game_idx:02d}-{branch}")


def fork_round(cfg: RunConfig) -> int:
    """God's first scheduled round (scripted: the earliest schedule entry at or after start_round;
    otherwise start_round): where B'' starts resampling."""
    rounds = [sp.round for sp in cfg.god.schedule if sp.round >= cfg.god.start_round]
    return min(rounds) if cfg.god.kind == "scripted" and rounds else cfg.god.start_round


def build_specs(cfg: RunConfig, opts: RunnerOpts) -> list[list[GameSpec]]:
    """-> groups; each group is one game_idx (a single game, or the A/B/B' branches of a pair)."""
    groups: list[list[GameSpec]] = []
    for i in range(cfg.run.n_games):
        if not opts.fork.enabled:
            groups.append([make_spec(cfg, i, BRANCH_MAIN)])
            continue
        if cfg.god.kind == "none":
            raise ValueError("paired fork needs god.kind != none in the base config")
        nogod = _with(cfg, god={**(opts.fork.control_god or {}), "kind": "none"})
        h = nogod.config_hash()
        g = [make_spec(cfg, i, BRANCH_GOD, h), make_spec(nogod, i, BRANCH_NOGOD, h)]
        if opts.fork.noise_replica:
            g.append(make_spec(_with(nogod, run={"replica": "1"}), i, BRANCH_NOISE, h))
        if opts.fork.noise_fork:
            g.append(make_spec(_with(nogod, run={"resample_from_round": fork_round(cfg)}), i,
                               BRANCH_NOISE_FORK, h))
        if opts.fork.only_branches is not None:
            g = [s for s in g if s.branch in opts.fork.only_branches]
        groups.append(g)
    return groups
