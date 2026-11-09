"""Config models + loading (WP0, FROZEN).

load(path, overrides) -> RunConfig:
  1. read YAML; if it has `extends: <relative path>`, deep-merge it onto that base (ONE level only;
     dicts merge, lists replace),
  2. apply CLI overrides "a.b.c=value" (value parsed with yaml.safe_load),
  3. validate (pydantic, extra="forbid", plus cross-field checks).
`configs/base.yaml` mirrors the defaults below exactly (tested).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-35b-a3b"
REPO_ROOT = Path(__file__).resolve().parent.parent


class Cfg(BaseModel):
    """Base for config sections: frozen, extra=forbid (typos are errors)."""
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- run
class RunCfg(Cfg):
    """Run identity: name, base seed, games, output dir, fork branch/replica labels."""
    name: str = "base"
    seed: int = 0
    n_games: int = Field(1, ge=1)
    out_dir: str = "runs"
    max_concurrency_games: int = Field(1, ge=1)   # >1 only with enough vLLM KV (REVIEW_v2 5.1)
    branch: str = "main"                          # fork branch label (folded into seeds)
    replica: str = "0"                            # noise replica id (folded into seeds)
    # Fork-matched noise B'' (runner.fork.noise_fork): from this round on, agent sampling seeds use
    # the replica tag "<replica>@r<round>" (fresh sampling stream; cache keys differ), before it they
    # are the control's (identical prefix). None = off; left out of config_hash while None.
    resample_from_round: int | None = Field(None, ge=1)
    # Log prompt provenance in real runs (CODE_REVIEW #5): turn.prompt_hash / prompt_set_hash and one
    # context_built event per turn. Logging only (never changes play), so it is never hashed; off by
    # default because it changes events.jsonl.
    log_prompt_trace: bool = False


# ---------------------------------------------------------------- game
class ViewLogCfg(Cfg):
    """What agents see of others' actions (view_log, defect visibility, grep_log)."""
    content: Literal["none", "actions_only", "actions_msg_meta", "full"] = "actions_only"
    actor_identity: Literal["shown", "hidden"] = "shown"
    reveal_provenance: bool = False                # ablation only: tag forged DMs / tampered entries
    window_turns: int | None = Field(None, ge=1)   # None = n_agents (the last round)
    defect_visibility: Literal["public", "victim_only", "delayed", "victim_then_delayed"] = \
        "victim_then_delayed"
    defect_delay_rounds: int = Field(3, ge=0)
    defect_actor_identity: Literal["shown", "hidden"] | None = None   # None = inherit actor_identity
    grep_enabled: bool = True
    grep_max_results: int = Field(20, ge=1)
    grep_max_pattern_chars: int = Field(80, ge=1, le=80)


class BoardCfg(Cfg):
    """Public message board options (added after the WP0 freeze; both default off = original rules).
    post_costs_turn: a post_board uses the agent's WHOLE action budget for that turn. The first
        post_board of a turn is applied and every other (non-grep) action of that turn fails with
        reason "board_post_uses_whole_turn" (the agent also gets a system notice). DMs cost one slot.
    anonymous: agents see board posts without the author (board view, view_log, grep_log and so
        the context built from them). God, events and analysis still see the true author.
    post_skip_rounds: k > 0 = an agent whose post_board succeeds in round r takes no turns in rounds
        r+1..r+k (AgentState.delayed_until_round, the wrong-defect delay mechanism; it still receives
        keys and messages). 0 = off."""
    post_costs_turn: bool = False
    anonymous: bool = False
    post_skip_rounds: int = Field(0, ge=0)
    # Added for the board sweep; defaults = original rules, left out of config_hash at their defaults.
    # enabled=False: no board at all (post_board is not offered and fails "board_disabled"; no board
    #   section or board lines in observations). visible_delay_rounds=k: others see a post (board view,
    #   view_log, grep_log) k rounds later than they otherwise would (sim: a round-r post from round r+1+k).
    enabled: bool = True
    visible_delay_rounds: int = Field(0, ge=0)
    # Added with game.network (default global = original rules; left out of config_hash at its default).
    # neighbors: an agent sees only the posts of its graph neighbours (board view, view_log, grep_log).
    scope: Literal["global", "neighbors"] = "global"

    @model_validator(mode="after")
    def _check(self) -> BoardCfg:
        if not self.enabled and (self.post_costs_turn or self.anonymous or self.post_skip_rounds
                                 or self.visible_delay_rounds or self.scope != "global"):
            raise ValueError("game.board.enabled=false cannot be combined with other board options")
        return self


class NetworkCfg(Cfg):
    """Social network over the agents (added after the freeze; topology "complete" = the original
    rules, and the whole section is left out of config_hash while it equals its defaults).
    The graph is generated from the WORLD seed (game.network module), so the God / no-God / noise
    branches of one game share it.
    topology: complete | erdos_renyi (mean_degree; resampled from the same seeded stream until
        connected) | watts_strogatz (k, p) | barabasi_albert (m) | ring.
    mode: contacts = only graph edges can DM / give_key / defect ("not_connected" otherwise);
        known_ids = every pair may interact, but an agent starts knowing only its neighbours' names
        and learns a name when it is named in a message or board post it sees, or gives to / defects
        on it ("unknown_contact" for an unknown name)."""
    topology: Literal["complete", "erdos_renyi", "watts_strogatz", "barabasi_albert", "ring"] = "complete"
    mode: Literal["contacts", "known_ids"] = "contacts"
    mean_degree: float = Field(3.0, gt=0)          # erdos_renyi: p = mean_degree / (n - 1)
    k: int = Field(4, ge=2)                        # watts_strogatz: ring-lattice degree (even)
    p: float = Field(0.1, ge=0.0, le=1.0)          # watts_strogatz: rewiring probability
    m: int = Field(2, ge=1)                        # barabasi_albert: edges per new node

    @property
    def on(self) -> bool:
        """Is any network restriction active? (complete graph = the original rules)."""
        return self.topology != "complete"


class GameCfg(Cfg):
    """Game rules knobs (PDF items mapped to config; ARCHITECTURE_v2 s1)."""
    n_agents: int = Field(20, ge=2)
    keys_per_agent: int = Field(5, ge=1)
    needs_per_agent: int = Field(3, ge=1)
    copies_per_info: int = Field(2, ge=1)
    actions_per_turn: int = Field(2, ge=1, le=2)
    max_rounds: int = Field(40, ge=1)
    turn_mode: Literal["sequential", "simultaneous"] = "sequential"
    turn_order: Literal["fixed", "reshuffle"] = "reshuffle"
    observe_mode: Literal["folded"] = "folded"     # "tools" cut (REVIEW_v2 s3)
    immediate_reply: bool = False                  # not implemented; True is rejected
    defect_gain: Literal["mint", "steal", "none"] = "mint"
    wrong_defect_penalty: Literal["lose_need", "delay"] = "lose_need"
    delay_n: int = Field(3, ge=1)
    winner_policy: Literal["sit_out", "continue"] = "continue"   # continue: winners act, cannot be burned
    max_message_chars: int = Field(500, ge=1)
    view_log: ViewLogCfg = ViewLogCfg()
    board: BoardCfg = BoardCfg()
    network: NetworkCfg = NetworkCfg()

    @model_validator(mode="after")
    def _check(self) -> GameCfg:
        if self.immediate_reply:
            raise ValueError("game.immediate_reply is not implemented (cut, REVIEW_v2 s3)")
        nw = self.network
        if self.board.scope == "neighbors" and not nw.on:
            raise ValueError("game.board.scope=neighbors needs a game.network.topology other than complete")
        if nw.topology == "watts_strogatz" and (nw.k % 2 or nw.k >= self.n_agents):
            raise ValueError("game.network.k must be even and < n_agents (watts_strogatz)")
        if nw.topology == "barabasi_albert" and nw.m >= self.n_agents:
            raise ValueError("game.network.m must be < n_agents (barabasi_albert)")
        if nw.topology == "erdos_renyi" and nw.mean_degree > self.n_agents - 1:
            raise ValueError("game.network.mean_degree must be <= n_agents - 1 (erdos_renyi)")
        if self.needs_per_agent > (self.n_agents - 1) * self.keys_per_agent:
            raise ValueError("needs_per_agent cannot be satisfied by the other agents' keys")
        return self

    @property
    def window_turns(self) -> int:
        """Effective view_log window in agent-turn slots (default n_agents)."""
        return self.view_log.window_turns or self.n_agents


# ---------------------------------------------------------------- reasoning
class ScratchpadCfg(Cfg):
    """Scratchpad (free `note` field): on/off, entries shown, char cap."""
    enabled: bool = True
    window: int | None = Field(None, ge=1)         # None = all entries (subject to the context budget)
    max_chars: int = Field(400, ge=1)


class ReasoningCfg(Cfg):
    """CoT mode, CoT replay window and caps; the scratchpad is the persistent store."""
    cot_mode: Literal["none", "native", "explicit"] = "explicit"
    cot_replay_window: int = Field(0, ge=0)        # 0: past CoT is logged, never re-shown
    max_cot_chars: int = Field(1200, ge=1)         # stored/replayed CoT only
    scratchpad: ScratchpadCfg = ScratchpadCfg()
    # Added after the freeze (default False = original behaviour, left out of config_hash): when a reply
    # was cut off by max_tokens (finish_reason=length) and has no decodable JSON object, keep every
    # complete action object after its "actions" key instead of retrying. A reply cut inside "thought"
    # (before "actions") has nothing to salvage and is retried as before.
    salvage_truncated: bool = False


# ---------------------------------------------------------------- roles / llm
Backend = Literal["vllm", "openrouter", "fake"]


class RoleCfg(Cfg):
    """Per-role LLM choice: backend, model override, prompt set, framing, sampling."""
    backend: Backend = "vllm"
    model: str | None = None                       # None = backend default model
    prompt_set: str = "agent/v1"
    framing: str | None = None                     # e.g. "framing/none"; None = no framing set
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float | None = None
    max_tokens: int = Field(700, ge=1)


class RolesCfg(Cfg):
    """LLM settings for each role (agent, god, judge, summarizer)."""
    agent: RoleCfg = RoleCfg()
    god: RoleCfg = RoleCfg(prompt_set="god/v1", temperature=0.7, max_tokens=700)
    judge: RoleCfg = RoleCfg(prompt_set="judge/v1", temperature=0.0, max_tokens=200)
    summarizer: RoleCfg = RoleCfg(prompt_set="summarizer/v1", temperature=0.0, max_tokens=300)


class VLLMCfg(Cfg):
    """Local vLLM server settings (infra/README.md)."""
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-8B"
    max_concurrency: int = Field(64, ge=1)
    max_model_len: int = 16384


class OpenRouterCfg(Cfg):
    """OpenRouter settings. OFF by default; key only via env/key file; $8 global ledger cap."""
    enabled: bool = False                          # OFF by default; any role on openrouter needs True
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = DEFAULT_OPENROUTER_MODEL
    api_key_env: str = "OPENROUTER_API_KEY"        # the key itself is NEVER in config
    max_concurrency: int = Field(8, ge=1)
    global_cap_usd: float = Field(20.0, ge=0.0)     # across all runs, enforced via the ledger
    ledger_path: str = "runs/openrouter_ledger.jsonl"
    price_in_per_mtok: float = 1.0                 # [E] conservative pre-call estimate; actual cost from provider
    price_out_per_mtok: float = 4.0
    # Added after the freeze; None = original behaviour (reasoning {enabled: req.thinking}, default
    # provider routing) and left out of config_hash. Both change play, so set values are hashed.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None   # e.g. gpt-oss: reasoning
    #   is mandatory (enabled=false is HTTP 400); send {effort: ...} instead
    provider_sort: Literal["price", "throughput", "latency"] | None = None        # OpenRouter provider.sort
    # Provider pin (OpenRouter provider.order / allow_fallbacks). None = default routing; left out of
    # config_hash while None (a pin changes who serves the model, so set values are hashed).
    provider_order: list[str] | None = None
    allow_fallbacks: bool | None = None


class FakeCfg(Cfg):
    """FakeLLM backend settings (tests)."""
    model: str = "fake"


class CacheCfg(Cfg):
    """sqlite response cache mode and path (WP2)."""
    mode: Literal["off", "read_write", "read_only", "write_only"] = "read_write"
    path: str = "cache/llm.sqlite"


class RetryCfg(Cfg):
    """Retry/backoff and per-request timeouts (WP2)."""
    max_retries: int = Field(5, ge=0)
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    timeout_s: float = 120.0
    timeout_thinking_s: float = 300.0


class LLMCfg(Cfg):
    """LLM layer: per-run budget, backends, cache, retry."""
    budget_usd: float = Field(2.0, ge=0.0)         # per-run cap on paid spend
    vllm: VLLMCfg = VLLMCfg()
    openrouter: OpenRouterCfg = OpenRouterCfg()
    fake: FakeCfg = FakeCfg()
    cache: CacheCfg = CacheCfg()
    retry: RetryCfg = RetryCfg()


# ---------------------------------------------------------------- god
class BudgetCfg(Cfg):
    """God power budget K, per game or per round."""
    k: int = Field(3, ge=0)
    scope: Literal["game", "round"] = "game"


class MessageFraudCfg(Cfg):
    """Message Fraud power: enabled and cost."""
    enabled: bool = True
    cost: int = Field(1, ge=0)


class CotControlCfg(Cfg):
    """CoT Control power: enabled, cost, allowed memory targets, edit mode."""
    enabled: bool = True
    cost: int = Field(1, ge=0)
    targets: list[Literal["scratchpad", "cot", "summary"]] = ["scratchpad"]
    mode: Literal["replace", "append"] = "replace"


class PowersCfg(Cfg):
    """God's powers."""
    message_fraud: MessageFraudCfg = MessageFraudCfg()
    cot_control: CotControlCfg = CotControlCfg()


class ScheduledPower(Cfg):
    """One ScriptedGod schedule entry (round, wake point, power, args)."""
    round: int = Field(ge=1)
    after_turns: int = Field(0, ge=0)              # sequential: wake after this many turns of the round
    power: Literal["message_fraud", "cot_control"]
    args: dict[str, Any] = {}                      # ScriptedGod resolves helpers (e.g. victim selectors)


class GodCfg(Cfg):
    """God: kind, timing, free-observation safety ceiling, budget, powers, schedule."""
    kind: Literal["none", "scripted", "llm"] = "scripted"
    start_round: int = Field(3, ge=1)
    wake_every_turns: int | None = Field(None, ge=1)   # None = once per round
    tool_call_safety_ceiling: int = Field(50, ge=1)    # observation is free; this stops runaway loops
    budget: BudgetCfg = BudgetCfg()
    powers: PowersCfg = PowersCfg()
    schedule: list[ScheduledPower] = []


# ---------------------------------------------------------------- context
IMPLEMENTED_POLICIES = ("full_truncate", "window")
STUB_POLICIES = ("rolling_summary", "hybrid")
SUMMARIZING_POLICIES = ("rolling_summary", "hybrid")


class ContextCfg(Cfg):
    """Agent context policy and token budget (WP13 implements policies)."""
    policy: Literal["full_truncate", "window", "rolling_summary", "hybrid"] = "full_truncate"
    budget_tokens: int = Field(14000, ge=256)      # whole prompt incl. system
    safety_margin: float = Field(0.05, ge=0.0, lt=0.5)
    tokenizer: str = "hf:Qwen/Qwen3-8B"            # | approx | words (tests)
    window_rounds: int = Field(3, ge=1)            # policy=window
    per_kind_caps: dict[str, int] = {}             # e.g. {scratchpad: 5, inbox_msg: 12} (window policy)

    @model_validator(mode="after")
    def _check(self) -> ContextCfg:
        if self.policy in STUB_POLICIES:
            raise ValueError(f"context.policy={self.policy!r} is a stub name only (not implemented)")
        return self


# ---------------------------------------------------------------- root
class RunConfig(Cfg):
    """Root config. Validates cross-field rules (see _cross)."""
    run: RunCfg = RunCfg()
    game: GameCfg = GameCfg()
    reasoning: ReasoningCfg = ReasoningCfg()
    roles: RolesCfg = RolesCfg()
    llm: LLMCfg = LLMCfg()
    god: GodCfg = GodCfg()
    context: ContextCfg = ContextCfg()
    sweep: dict[str, list[Any]] = {}

    @model_validator(mode="after")
    def _cross(self) -> RunConfig:
        errs: list[str] = []
        targets = self.god.powers.cot_control.targets
        if "cot" in targets and (self.reasoning.cot_mode == "none" or self.reasoning.cot_replay_window == 0):
            errs.append("cot_control target 'cot' needs cot_mode != none and cot_replay_window > 0")
        if "summary" in targets and self.context.policy not in SUMMARIZING_POLICIES:
            errs.append("cot_control target 'summary' needs a summarizing context policy")
        if "scratchpad" in targets and not self.reasoning.scratchpad.enabled:
            errs.append("cot_control target 'scratchpad' needs reasoning.scratchpad.enabled")
        for name in ("agent", "god", "judge", "summarizer"):
            role: RoleCfg = getattr(self.roles, name)
            if role.backend == "openrouter" and not self.llm.openrouter.enabled:
                errs.append(f"roles.{name}.backend=openrouter but llm.openrouter.enabled is false")
        agent = self.roles.agent
        if agent.backend == "vllm":
            need = self.context.budget_tokens + agent.max_tokens + 512
            if need > self.llm.vllm.max_model_len:
                errs.append(f"context.budget_tokens + roles.agent.max_tokens + 512 = {need} "
                            f"> llm.vllm.max_model_len = {self.llm.vllm.max_model_len}")
        if self.god.kind != "none" and self.god.start_round > self.game.max_rounds:
            errs.append("god.start_round > game.max_rounds: God would never act")
        for sp in self.god.schedule:
            if sp.power == "cot_control" and not self.god.powers.cot_control.enabled:
                errs.append("schedule uses cot_control but it is disabled")
            if sp.power == "message_fraud" and not self.god.powers.message_fraud.enabled:
                errs.append("schedule uses message_fraud but it is disabled")
        for key in self.sweep:
            if key.startswith("sweep"):
                errs.append("sweep keys cannot target sweep itself")
        if errs:
            raise ValueError("; ".join(errs))
        return self

    def config_hash(self) -> str:
        """sha256 of the game-semantic config (swarm/config_hash.py has the rules)."""
        from swarm.config_hash import config_hash

        return config_hash(self)

    def model_for(self, role: str) -> tuple[str, str]:
        """-> (backend, model name) for a role."""
        r: RoleCfg = getattr(self.roles, role)
        if r.model:
            return r.backend, r.model
        return r.backend, getattr(self.llm, r.backend).model


# ================================================================ loading
def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; `over` wins; lists are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override(s: str) -> tuple[list[str], Any]:
    """'a.b.c=value' -> (['a','b','c'], yaml.safe_load(value))."""
    if "=" not in s:
        raise ValueError(f"override must be a.b.c=value, got {s!r}")
    key, raw = s.split("=", 1)
    parts = key.strip().split(".")
    if not all(parts):
        raise ValueError(f"bad override key {key!r}")
    return parts, yaml.safe_load(raw) if raw.strip() != "" else ""


def set_path(d: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set d[a][b][c] = value, creating intermediate dicts."""
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a copy of raw with every CLI override applied."""
    out = copy.deepcopy(raw)
    for o in overrides or []:
        parts, value = parse_override(o)
        set_path(out, parts, value)
    return out


def load_raw(path: str | Path) -> dict[str, Any]:
    """Read YAML and resolve one level of `extends` (relative to the file)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    ext = data.pop("extends", None)
    if ext is not None:
        base_path = (path.parent / ext).resolve()
        base = yaml.safe_load(base_path.read_text()) or {}
        if "extends" in base:
            raise ValueError(f"{base_path}: only one level of `extends` is allowed")
        data = deep_merge(base, data)
    return data


def from_dict(raw: dict[str, Any], overrides: list[str] | None = None) -> RunConfig:
    """Validate a raw dict (after overrides) into a RunConfig."""
    return RunConfig.model_validate(apply_overrides(raw, overrides))


def load(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load, merge `extends`, apply overrides and validate a config file."""
    return from_dict(load_raw(path), overrides)


def dump_yaml(cfg: RunConfig) -> str:
    """Resolved config as YAML (written to runs/<id>/config.resolved.yaml)."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


def expand_sweep(cfg: RunConfig) -> list[tuple[dict[str, Any], RunConfig]]:
    """Cartesian product of cfg.sweep -> [(params, cell config with sweep={})]. No sweep -> [({}, cfg)]."""
    if not cfg.sweep:
        return [({}, cfg)]
    keys = list(cfg.sweep)
    base = cfg.model_dump(mode="json")
    base["sweep"] = {}
    cells = []
    for combo in itertools.product(*(cfg.sweep[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        raw = copy.deepcopy(base)
        for k, v in params.items():
            set_path(raw, k.split("."), v)
        cells.append((params, RunConfig.model_validate(raw)))
    return cells


from swarm.config_hash import HASH_GLOBAL_CAP_USD, HASH_OPERATIONAL  # noqa: E402,F401  (re-exports)
