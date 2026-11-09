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

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from swarm.config import GodCfg, RunConfig
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
    # Operational (not in RunConfig, so never hashed): stop every game after this round (smokes that
    # must keep the full run's config hash, game_ids and cache keys). None = game.max_rounds.
    stop_after_round: int | None = None
    # Arms (sweeps): a list of {"label": str, "god": {...}} deep-merged into the config's god section;
    # every world (game_idx) plays every arm as one group (concurrently, paired: the same world seed and
    # game_id = the God-free config's hash, so all arms share agent seeds and the cached pre-God prefix,
    # exactly like the god / nogod fork). Exclusive with fork.enabled. None = off.
    arms: tuple[dict[str, Any], ...] | None = None
    # The worlds to play, in this order (game_idx; e.g. [3] = world 4 alone; world seeds depend only on
    # run.seed and game_idx). None = range(run.n_games).
    game_indices: tuple[int, ...] | None = None
    # "raise" (default): a TruthLeak aborts that game (it is marked error; the pool continues). "warn":
    # log + count it in the game's meta (leak_hits / leak_examples) and send the request anyway, so a
    # false positive can never end an overnight game; real leaks stay visible in meta for analysis.
    leak_action: str = "raise"
    # World gate (sweeps; stop cleanly at a world boundary): {"ledger": path, "cap_usd": X, "world_usd": Y}.
    # Before a world (game_idx group) starts its FIRST game, the gate reads the ledger total; if total + Y > X
    # every game of that world is skipped (status "skipped"); a world that has started always finishes all its
    # arms, so finished worlds stay paired. None = off.
    world_gate: dict[str, Any] | None = None
    # Per-game stall watchdog (minutes without a new event -> cancel + restart that game from the cache; None =
    # off) and the number of restarts before the game fails.
    game_stall_min: float | None = None
    game_stall_restarts: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RunnerOpts:
        d = dict(d or {})
        fork = ForkOpts(**(d.pop("fork", None) or {}))
        unknown = set(d) - {"engine", "report", "stop_after_round", "arms", "game_indices", "leak_action",
                            "world_gate", "game_stall_min", "game_stall_restarts"}
        if unknown:
            raise ValueError(f"unknown runner keys: {sorted(unknown)}")
        if d.get("arms") is not None:
            arms = tuple(dict(a) for a in d["arms"])
            labels = [a.get("label") for a in arms]
            if not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels):
                raise ValueError("runner.arms: every arm needs a unique non-empty label")
            if any(set(a) - {"label", "god", "game"} for a in arms):
                raise ValueError("runner.arms: an arm has only 'label', 'god' and 'game'")
            for a in arms:   # arm game overrides: only the peer-refusal knob (it changes nothing before God acts)
                g = a.get("game")
                if g is not None and (set(g) != {"view_log"} or not set(g["view_log"]) <= {"refusal_broadcast", "norm_notice"}):
                    raise ValueError("runner.arms: an arm's 'game' may only set view_log.refusal_broadcast / norm_notice")
            if fork.enabled:
                raise ValueError("runner.arms and runner.fork.enabled are exclusive")
            d["arms"] = arms
        if d.get("game_indices") is not None:
            d["game_indices"] = tuple(int(i) for i in d["game_indices"])
        wg = d.get("world_gate")
        if wg is not None and (set(wg) - {"ledger", "cap_usd", "world_usd", "rss_gb", "extra_ledgers"}
                               or "ledger" not in wg
                               or "cap_usd" not in wg):
            raise ValueError("runner.world_gate needs ledger and cap_usd (world_usd optional)")
        if d.get("leak_action", "raise") not in ("raise", "warn"):
            raise ValueError("runner.leak_action must be raise or warn")
        return cls(fork=fork, **d)

    def to_dict(self) -> dict[str, Any]:
        out = {"engine": self.engine, "report": self.report,
               "fork": {"enabled": self.fork.enabled, "noise_replica": self.fork.noise_replica,
                        "serial_pair": self.fork.serial_pair, "control_god": self.fork.control_god,
                        "noise_fork": self.fork.noise_fork, "only_branches": self.fork.only_branches}}
        if self.stop_after_round is not None:          # listed only when set (resolved yaml unchanged)
            out["stop_after_round"] = self.stop_after_round
        if self.arms is not None:
            out["arms"] = [dict(a) for a in self.arms]
        if self.game_indices is not None:
            out["game_indices"] = list(self.game_indices)
        if self.leak_action != "raise":
            out["leak_action"] = self.leak_action
        if self.world_gate is not None:
            out["world_gate"] = dict(self.world_gate)
        if self.game_stall_min is not None:
            out["game_stall_min"] = self.game_stall_min
            out["game_stall_restarts"] = self.game_stall_restarts
        return out


@dataclass(frozen=True)
class GameSpec:
    game_idx: int
    branch: str
    cfg: RunConfig
    game_id: str
    world_seed: int
    label: str = field(default="")
    stop_after_round: int | None = None      # RunnerOpts.stop_after_round (operational; not in meta)
    leak_action: str = "raise"               # RunnerOpts.leak_action (operational)
    game_stall_min: float | None = None      # RunnerOpts.game_stall_min (operational)
    game_stall_restarts: int = 2

    @property
    def god_active(self) -> bool:
        return self.cfg.god.kind != "none"

    def meta(self) -> dict[str, Any]:
        out = {"label": self.label, "game_id": self.game_id, "game_idx": self.game_idx,
               "branch": self.branch, "god": self.cfg.god.kind, "seed": self.world_seed,
               "run_seed": self.cfg.run.seed, "replica": self.cfg.run.replica,
               "config_sha256": self.cfg.config_hash()}
        o = self.cfg.llm.openrouter
        if self.cfg.roles.agent.backend == "openrouter":     # per-game model / reasoning (manifest + meta)
            out.update(agent_model=self.cfg.model_for("agent")[1], reasoning_effort=o.reasoning_effort)
        elif self.cfg.roles.agent.backend == "adaption":
            out.update(agent_model=self.cfg.model_for("agent")[1], backend="adaption",
                       reasoning_effort=self.cfg.llm.adaption.reasoning_effort)
        if self.cfg.god.kind != "none":
            out["god_k"] = self.cfg.god.budget.k
            if self.cfg.god.kind == "llm":
                gl = self.cfg.god_llm()
                if self.cfg.roles.god.backend == "adaption":    # God's own effort, else the agents'
                    ad = self.cfg.llm.adaption
                    out.update(god_model=self.cfg.model_for("god")[1],
                               god_reasoning_effort=ad.god_reasoning_effort or ad.reasoning_effort)
                else:
                    out.update(god_model=gl["model"], god_reasoning_effort=gl["reasoning_effort"])
        if self.cfg.game.framing.mode != "none":
            out.update(framing=self.cfg.game.framing.mode, stance=self.cfg.game.framing.stance)
        return out


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


def _deep(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        out[k] = _deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def arm_cfg(cfg: RunConfig, arm: dict[str, Any]) -> RunConfig:
    """cfg with an arm's god overrides (and its game.view_log.refusal_broadcast, Noor tie-in) deep-merged
    (runner.arms). The arms' shared game id hash ignores both, so a refusal-broadcast arm stays paired."""
    raw = cfg.model_dump(mode="json")
    raw["god"] = _deep(raw["god"], arm.get("god") or {})
    if arm.get("game"):
        raw["game"] = _deep(raw["game"], arm["game"])
    return RunConfig.model_validate(raw)


def build_specs(cfg: RunConfig, opts: RunnerOpts) -> list[list[GameSpec]]:
    """-> groups; each group is one game_idx (a single game, the A/B/B' branches of a pair, or the arms)."""
    groups = _build_specs(cfg, opts)
    upd: dict[str, Any] = {}
    if opts.stop_after_round is not None:
        upd["stop_after_round"] = opts.stop_after_round
    if opts.leak_action != "raise":
        upd["leak_action"] = opts.leak_action
    if opts.game_stall_min is not None:
        upd["game_stall_min"] = opts.game_stall_min
        upd["game_stall_restarts"] = opts.game_stall_restarts
    if not upd:
        return groups
    return [[dataclasses.replace(s, **upd) for s in g] for g in groups]


def _build_specs(cfg: RunConfig, opts: RunnerOpts) -> list[list[GameSpec]]:
    groups: list[list[GameSpec]] = []
    indices = list(opts.game_indices) if opts.game_indices is not None else list(range(cfg.run.n_games))
    if opts.arms is not None:                      # sweeps: every world plays every arm, paired
        # id hash = the config with the WHOLE god section at its defaults (kind none): every arm and every block
        # with the same agent-side config (model, effort, game) shares game ids, seeds and cached prefixes,
        # whatever its God (LLM k1/k2/k4, scripted heuristic / random, God effort)
        raw = cfg.model_dump(mode="json")
        raw["god"] = {**GodCfg().model_dump(mode="json"), "kind": "none"}
        h = RunConfig.model_validate(raw).config_hash()
        for i in indices:
            groups.append([make_spec(arm_cfg(cfg, a), i, str(a["label"]), h) for a in opts.arms])
        return groups
    for i in indices:
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
