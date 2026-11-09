"""THE integration point (WP9). Everything the runner needs from engine / agents / God / LLM lives
here, so the integrator adapts one file when a builder's interface differs from the doc.

    clients = build_clients(cfg, run_writer)                 # once per run, shared by all games
    outcome = await play_game(spec, game_writer, clients)    # one game (any branch)
    await close_clients(clients)

Engines:
    "synth": swarm.runner.synth (no LLM, deterministic; for dry runs and tests)
    "real" : swarm.game.engine.GameEngine + LLMAgent + ScriptedGod/LLMGod via GodRunner

Event-sink / request wrappers live in runner/sinks.py and the God timing adapter in
runner/god_adapter.py (both re-exported here).

LLM-call logging: B2's clients take `recorder=` (called once per call with an LLMCall, cache hits
included); we pass `RunWriter.log_llm_call`, which routes to the current game's llm_calls.jsonl via a
contextvar. The fake backend has no recorder, so it is wrapped in runlog.LoggingClient instead.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm.config import REPO_ROOT, RunConfig
from swarm.runlog.llmlog import LoggingClient
from swarm.runlog.writer import GameWriter, RunWriter
from swarm.runner.god_adapter import _SlotGod, check_schedule, make_god_hook  # noqa: F401  (re-exports)
from swarm.runner.sinks import LedgerSink, OracleClient, ProgressSink  # noqa: F401  (re-exports)
from swarm.runner.spec import GameSpec
from swarm.schema import Outcome


@dataclass
class Clients:
    by_role: dict[str, Any] = field(default_factory=dict)
    closers: list[Any] = field(default_factory=list)

    def get(self, role: str) -> Any:
        return self.by_role.get(role)


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _paid_budget(cfg: RunConfig, shared: dict[str, Any]) -> tuple[Any, Any]:
    """The run's ONE RunBudget and ONE GlobalLedger, created on first use and shared by every paid
    client (a second model must not get a second per-run cap; CODE_REVIEW #9)."""
    if "budget" not in shared:
        from swarm.llm.budget import GlobalLedger, RunBudget

        o = cfg.llm.openrouter
        shared["budget"] = RunBudget(cfg.llm.budget_usd)
        shared["ledger"] = GlobalLedger(_abs(o.ledger_path), o.global_cap_usd)
    return shared["budget"], shared["ledger"]


def _build_client(cfg: RunConfig, role: str, run: RunWriter, cache: Any,
                  shared: dict[str, Any] | None = None) -> Any:
    backend, model = cfg.model_for(role)
    if backend == "fake":
        from swarm.llm.base import FakeLLM
        from swarm.runner.fake_policy import smart_fake_policy

        return LoggingClient(FakeLLM(policy=smart_fake_policy, model=model), run.log_llm_call, run_id=run.run_id, backend="fake",
                             model=model)
    if backend == "vllm":
        from swarm.llm.vllm import VLLMClient

        return VLLMClient.from_cfg(cfg.llm.vllm, model=model, retry=cfg.llm.retry, cache=cache,
                                   recorder=run.log_llm_call, run_id=run.run_id)
    if backend == "openrouter":
        from swarm.llm.openrouter import OpenRouterClient, load_key

        o = cfg.llm.openrouter
        budget, ledger = _paid_budget(cfg, {} if shared is None else shared)
        return OpenRouterClient(enabled=o.enabled, api_key=load_key(), base_url=o.base_url, model=model,
                                max_concurrency=o.max_concurrency, price_in_per_mtok=o.price_in_per_mtok,
                                price_out_per_mtok=o.price_out_per_mtok,
                                ledger=ledger, run_budget=budget, retry=cfg.llm.retry, cache=cache,
                                recorder=run.log_llm_call, run_id=run.run_id,
                                reasoning_effort=o.reasoning_effort, provider_sort=o.provider_sort,
                                provider_order=o.provider_order, allow_fallbacks=o.allow_fallbacks)
    raise ValueError(f"unknown backend {backend!r}")


def build_clients(cfg: RunConfig, run: RunWriter, engine: str = "real") -> Clients:
    """One client per (role) shared across games (shared semaphores => vLLM batches across games)."""
    out = Clients()
    if engine == "synth":
        return out
    cache = None
    if cfg.llm.cache.mode != "off":
        from swarm.llm.cache import ResponseCache

        path = _abs(cfg.llm.cache.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = ResponseCache(path, cfg.llm.cache.mode)
    roles = ["agent"] + (["god"] if cfg.god.kind == "llm" else [])
    built: dict[tuple[str, str], Any] = {}
    shared: dict[str, Any] = {}                      # one RunBudget + GlobalLedger for all paid clients
    for role in roles:
        key = cfg.model_for(role)
        if key not in built:
            built[key] = _build_client(cfg, role, run, cache, shared)
            out.closers.append(built[key])
        out.by_role[role] = built[key]
    return out


async def close_clients(clients: Clients) -> None:
    for c in clients.closers:
        inner = getattr(c, "inner", c)
        close = getattr(inner, "aclose", None)
        if close is not None:
            res = close()
            if inspect.isawaitable(res):
                await res


# --------------------------------------------------------------------------- one game
async def play_game(spec: GameSpec, sink: GameWriter, clients: Clients, engine: str = "real") -> Outcome:
    if engine == "synth":
        from swarm.runner.synth import play_synth

        return await play_synth(spec, sink)
    return await _play_real(spec, sink, clients)


class RunnerAgent:
    """TurnAgent for the real engine: one LLMAgent, with the engine as the ONE builder of its ledger,
    memory views and grep (CODE_REVIEW R5; replaces make_decide + DecideAgent + LedgerSink).

    - ledger / memory come from the engine (GameEngine.ledgers, memory_views).
    - grep: the engine's once-per-turn callback; when the engine has none (grep off, or view_log
      content "none"), a direct grep_log over the agent's observation turns, exactly what the old
      path handed LLMAgent (the action then resolves as grep_unavailable, as before).
    - `last` (LLMAgent.last: prompt hashes + context bundle) is exposed only with
      run.log_prompt_trace (CODE_REVIEW #5, default off): the engine then logs turn.prompt_hash /
      prompt_set_hash and context_built events. Off = events.jsonl byte-identical to before."""

    def __init__(self, llm: Any, cfg: RunConfig, state: Any, engine_ref: list[Any], *, trace: bool = False) -> None:
        self.llm, self.cfg, self.state, self.engine_ref, self.trace = llm, cfg, state, engine_ref, trace

    @property
    def last(self) -> Any:
        return self.llm.last if self.trace else None

    async def act(self, obs: Any, *, ledger: Any, memory: Any, game_id: str, grep: Any) -> Any:
        if grep is None:
            from swarm.game.observe import grep_log

            a, eng = obs.agent, self.engine_ref[0]

            def grep(g: Any) -> Any:
                return grep_log(self.state, a, self.cfg.game, g.pattern, eng.obs_turns[a], rounds=g.rounds)
        return await self.llm.act(obs, ledger=ledger, memory=memory, game_id=game_id, grep=grep)


async def _play_real(spec: GameSpec, sink: GameWriter, clients: Clients) -> Outcome:
    """World + MemoryStore + LLMAgents + (ScriptedGod via GodRunner) + GameEngine, all events through
    LeakOracle-observing and ledger-keeping sink wrappers; every agent request leak-checked."""
    from swarm.adversary.base import GodRunner
    from swarm.adversary.llm_god import make_god
    from swarm.agents.llm_agent import LLMAgent, default_counter
    from swarm.context.leak_oracle import LeakOracle
    from swarm.game.engine import GameEngine
    from swarm.game.world import generate_world
    from swarm.memory import MemoryStore

    cfg = spec.cfg
    check_schedule(cfg)
    state = generate_world(cfg.game, spec.world_seed, spec.game_id)
    if cfg.game.network.on:                      # the graph is also on game_start (AgentState.contacts)
        from swarm.game.network import network_meta

        sink.write_meta(network=network_meta(state, cfg.game))
    memory = MemoryStore(cfg.reasoning)
    oracle = LeakOracle(enabled=not cfg.game.view_log.reveal_provenance)
    counter = default_counter(cfg)
    psink = ProgressSink(sink, spec.label, cfg.game.max_rounds, clients.get("agent"))
    client = OracleClient(clients.get("agent"), oracle)
    engine_ref: list[Any] = []
    trace = cfg.run.log_prompt_trace
    agents = {a: RunnerAgent(LLMAgent(a, client, cfg, counter=counter), cfg, state, engine_ref, trace=trace)
              for a in state.order}
    god = make_god(cfg.god, seed=spec.world_seed, client=clients.get("god"))
    # GodRunner owns the stamp convention (decide at slot S, deliver stamped S - 1); the engine calls
    # runner.init_game in start(). (_SlotGod + make_god_hook did the same and stay importable.)
    runner = (GodRunner(god, cfg.god, memory, max_message_chars=cfg.game.max_message_chars,
                        winners_leave=cfg.game.winner_policy == "sit_out")
              if god is not None else None)
    # The engine is the one ledger builder (R5); on_snapshot writes the round-0 snapshot
    # (engine.start) and every end-of-round deep copy.
    eng = GameEngine(cfg, state, agents, god=runner, sink=oracle.wrap(psink), seed=spec.world_seed,
                     memory=memory, on_snapshot=sink.snapshot, counter=counter)
    engine_ref.append(eng)
    outcome = await eng.run()
    sink.write_meta(leak_checked=oracle.n_checked, leak_hits=len(oracle.hits))
    return outcome
