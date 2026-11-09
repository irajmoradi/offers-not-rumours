"""Game engine (WP1b): the round/turn loop. No LLM client import: players and God are INJECTED.

Inputs:
    agents: Mapping[AgentId, TurnAgent]  (e.g. swarm.agents.llm_agent.LLMAgent), or one DecideFn
        TurnAgent.act(obs, *, ledger, memory, game_id, grep) -> AgentOutput   (async or sync)
            `grep` is a callback GrepLog -> GrepResult (None when grep is disabled). grep_log is FREE:
            it does not use an action slot; only the first grep of a turn is run.
        DecideFn(obs, grep_result | None) -> AgentOutput  (tests / scripted stubs; adapted by
            DecideAgent: called again with the GrepResult when its first output has a grep_log).
    god: GodLike | None  (swarm.adversary.base.GodRunner): init_game(state), should_wake(state),
        await wake(state) -> object with `.events`. Called with state.turn_index = the slot S about to
        be played; GodRunner stamps what God delivers S - 1 (visible at S; adversary/base.py).
    god_hook(state, round, next_turn) -> list[EventBody]   (legacy form of god; sync or async)
        FROZEN convention (runner/wiring.py's _SlotGod depends on it): called before every slot
        (sequential; skipped slots too) / once at the round's first slot (simultaneous) with
        next_turn = the slot about to be played and state.turn_index = next_turn - 1 on entry.
        Records the hook creates without resetting turn_index are stamped next_turn - 1, which makes
        them visible at next_turn. The engine sets state.turn_index = next_turn after the hook.
Outputs: events through `sink`, end-of-round snapshots (`snapshots`, `on_snapshot`), per-agent
ledgers (`ledgers`), and the Outcome from `run()`.

Round r, sequential (default): order = fixed, or reshuffled by make_rng(seed, game_id, "order", r).
For each position p, slot t = turn_index(r, p, N): God (before EVERY slot, skipped ones too) ->
skip if the agent is delayed / a sat-out winner (the slot still counts) -> observe at t ->
mark_observed -> act -> turn event (+ context_built) -> apply each action stamped t -> memory.write
(cot, note) -> ledger += observation items + own-action items.
Simultaneous: God once at s0 = turn_index(r, 0, N); every active agent observes at s0; all act
concurrently (asyncio.gather, bounded by the LLM client's semaphore; if one act() raises, the
others are cancelled); actions resolve in turn-priority order (st.order: fixed or reshuffled per
round), each stamped at its own slot; a give_key / defect that fails at resolution (e.g. the key
was already taken) also sends the actor a "system" notice.
game.winner_policy = "sit_out" (winners leave the game; rules.departed): a winner is skipped from the
moment it wins, in both modes. Sequential: its later slots (also later in the same round) are
skipped. Simultaneous: a turn chosen at s0 by an agent that won during this round's resolution is
still logged, but each of its actions fails "actor_left". Its score is locked (rules), and the game
ends when every remaining agent is done.
game.board.post_costs_turn (default off): a post_board uses the whole action budget of its turn. The
first post_board of a turn is applied; every other action of that turn (before or after it; grep_log
and noop excepted) fails with reason "board_post_uses_whole_turn" without using a slot, and the agent
gets one "system" notice (next observation) saying so.
send_message still costs one slot. Same in both turn modes (decided per agent turn in game.resolve).
game.board.post_skip_rounds = k > 0: an agent whose post_board succeeds in round r is skipped in
rounds r+1..r+k (delayed_until_round), with a skip notice and a resume notice (both shown at its
next observation, i.e. when it resumes).
game.network (game/network.py, off by default): the graph is on the state from world generation;
rules refuse actions across non-edges / at unknown names, observe filters the board / view_log and
the player list, and each observation teaches the agent the names in it (network.learn, known_ids).
End of round: round_end event + deep-copied snapshot. The game ends at max_rounds, when every agent
is done (won or no open need), or on BudgetExceeded (end_reason="budget_exceeded"; the round it cut
short still gets its round_end and snapshot).
Imports: config, memory, rng, schema, context.base/tokens, llm.base (BudgetExceeded only), game.*.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from swarm.config import RunConfig
from swarm.context.base import TokenCounter, items_from_observation, own_action_items
from swarm.context.tokens import make_counter
from swarm.llm.base import BudgetExceeded
from swarm.memory import MemoryStore
from swarm.rng import canonical_json, make_rng, sha256_hex
from swarm.schema import (
    AgentId,
    AgentOutcome,
    AgentOutput,
    ContextBuiltData,
    EventBody,
    EventSink,
    GameEndData,
    GameStartData,
    GameState,
    GrepLog,
    GrepResult,
    HistoryItem,
    MemorySink,
    MemoryView,
    Noop,
    Observation,
    Outcome,
    RoundEndData,
    TurnData,
    turn_index,
)

from . import network, rules
from .board import BOARD_WHOLE_TURN  # noqa: F401  (re-export: old engine name)
from .board import whole_turn_text as _whole_turn_text  # noqa: F401
from .observe import grep_log, mark_observed, observe
from .resolve import Result, resolve_turn  # noqa: F401  (Result: re-export)
from .resolve import conflict_text as _conflict_text  # noqa: F401

DecideFn = Callable[[Observation, GrepResult | None], AgentOutput | Awaitable[AgentOutput]]
GodHook = Callable[[GameState, int, int], list[EventBody] | Awaitable[list[EventBody]]]
GrepFn = Callable[[GrepLog], GrepResult]


class TurnAgent(Protocol):
    """A player: LLMAgent or any object with this act() signature."""

    def act(self, obs: Observation, *, ledger: Sequence[HistoryItem], memory: Sequence[MemoryView],
            game_id: str, grep: GrepFn | None) -> Awaitable[AgentOutput] | AgentOutput:
        """Play one turn from the observation."""
        ...


class GodLike(Protocol):
    """GodRunner's surface used by the engine (init_game is optional)."""

    def should_wake(self, state: GameState) -> bool:
        """Wake before slot state.turn_index?"""
        ...

    def wake(self, state: GameState) -> Awaitable[Any]:
        """Act on state; the result has `.events: list[EventBody]`."""
        ...


async def _maybe_await(x: Any) -> Any:
    return await x if inspect.isawaitable(x) else x


async def _gather_or_cancel(coros: Sequence[Awaitable[Any]]) -> list[Any]:
    """asyncio.gather that CANCELS the siblings when one raises (plain gather leaves them running,
    so after e.g. BudgetExceeded the other agents' LLM calls kept going after game_end). Tasks are
    created in the same order as gather would, so the happy path is unchanged."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class DecideAgent:
    """Adapts a two-call DecideFn (obs, grep_result|None) to the TurnAgent protocol (tests, stubs).
    Mirrors LLMAgent: final actions = [grep_log, *follow-up actions] when a grep ran."""

    def __init__(self, decide: DecideFn) -> None:
        self.decide = decide

    async def act(self, obs: Observation, *, ledger: Sequence[HistoryItem] = (),
                  memory: Sequence[MemoryView] = (), game_id: str = "g0",
                  grep: GrepFn | None = None) -> AgentOutput:
        """Call decide; if it asked for a grep and grep is available, call it again with the result."""
        first: AgentOutput = await _maybe_await(self.decide(obs, None))
        g = next((a for a in first.actions if isinstance(a, GrepLog)), None)
        if g is None or grep is None:
            return first
        second: AgentOutput = await _maybe_await(self.decide(obs, grep(g)))
        follow = [a for a in second.actions if not isinstance(a, GrepLog)]
        return AgentOutput(
            cot="\n".join(c for c in (first.cot, second.cot) if c) or None,
            note=second.note if second.note is not None else first.note,
            actions=[g, *follow], parse_ok=first.parse_ok and second.parse_ok,
            retries=first.retries + second.retries, errors=first.errors + second.errors,
            llm_call_ids=first.llm_call_ids + second.llm_call_ids)


class _HookResult:
    def __init__(self, events: list[EventBody]) -> None:
        self.events = events


class HookGod:
    """Adapts a legacy god_hook(state, round, next_turn) -> events to the GodLike protocol."""

    def __init__(self, hook: GodHook) -> None:
        self.hook = hook

    def should_wake(self, state: GameState) -> bool:
        """The hook decides itself; always consult it."""
        return True

    async def wake(self, state: GameState) -> _HookResult:
        """Run the hook before slot `next_turn` (= state.turn_index on entry). FROZEN convention that
        runner/wiring.py's _SlotGod adapter relies on: the hook is entered with
        state.turn_index = next_turn - 1; the engine resets it to next_turn afterwards."""
        next_turn = state.turn_index
        state.turn_index = next_turn - 1
        return _HookResult(list(await _maybe_await(self.hook(state, state.round, next_turn))))


class GameEngine:
    """Runs one game. `state` is mutated in place; `snapshots` holds end-of-round deep copies."""

    def __init__(self, cfg: RunConfig, state: GameState,
                 agents: Mapping[AgentId, TurnAgent] | DecideFn, *,
                 god: GodLike | None = None, god_hook: GodHook | None = None,
                 sink: EventSink | None = None, memory: MemoryStore | None = None,
                 on_snapshot: Callable[[GameState], None] | None = None, seed: int | None = None,
                 ledgers: dict[AgentId, list[HistoryItem]] | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.cfg, self.g = cfg, cfg.game
        self.state = state
        if callable(agents) and not isinstance(agents, Mapping):
            shared = DecideAgent(agents)
            self.agents: dict[AgentId, Any] = {a: shared for a in state.agents}
        else:
            self.agents = dict(agents)
        missing = set(state.agents) - set(self.agents)
        if missing:
            raise ValueError(f"no player for agents {sorted(missing)}")
        self.god: GodLike | None = god if god is not None else (HookGod(god_hook) if god_hook else None)
        self.sink: EventSink = sink or MemorySink(game_id=state.game_id)
        self.seed = cfg.run.seed if seed is None else seed
        self.memory = memory or MemoryStore(cfg.reasoning)
        self.on_snapshot = on_snapshot
        self.counter = counter or make_counter(cfg.context.tokenizer)
        self.ledgers: dict[AgentId, list[HistoryItem]] = ledgers if ledgers is not None else {}
        for a in state.agents:                     # fill a caller's dict in place (even an empty one)
            self.ledgers.setdefault(a, [])
        self.obs_turns: dict[AgentId, list[int]] = {a: [] for a in state.agents}   # for grep_log
        self.snapshots: list[GameState] = []
        self.end_reason: str | None = None
        self._started = False
        self._round_open = False

    # ------------------------------------------------------------------ helpers
    def emit(self, bodies: Sequence[EventBody]) -> None:
        """Send event bodies to the sink (which stamps run/game ids and seq)."""
        for b in bodies:
            self.sink.emit(b)

    def _body(self, data: Any, *, actor: str = "SYSTEM", phase: str = "end",
              turn: int | None = None) -> EventBody:
        return EventBody(round=self.state.round, turn_index=turn, phase=phase, actor=actor, data=data)

    def active(self, agent: AgentId) -> bool:
        """May `agent` take a turn this round? (not delayed; not a sat-out winner)."""
        s = self.state.agents[agent]
        if self.state.round <= s.delayed_until_round:
            return False
        if rules.departed(self.state, agent, self.g):     # winner_policy=sit_out: it left the game
            return False
        return True

    def order_for_round(self, r: int) -> list[AgentId]:
        """Turn order of round r: fixed (state.order) or a seeded reshuffle."""
        if self.g.turn_order != "reshuffle":
            return list(self.state.order)
        names = sorted(self.state.agents)
        make_rng(self.seed, self.state.game_id, "order", r).shuffle(names)
        return names

    def scores(self) -> dict[AgentId, float]:
        """Current score per agent."""
        return {a: rules.score(self.state, a) for a in sorted(self.state.agents)}

    def all_done(self) -> bool:
        """Every agent has won or has no open need left."""
        return all(rules.is_done(self.state, a) for a in self.state.agents)

    def _snapshot(self) -> None:
        snap = self.state.model_copy(deep=True)
        self.snapshots.append(snap)
        if self.on_snapshot is not None:
            self.on_snapshot(snap)

    def memory_views(self, agent: AgentId) -> list[MemoryView]:
        """Scratchpad (+ CoT when cot_replay_window > 0) entries the agent is shown this turn."""
        rc, st = self.cfg.reasoning, self.state
        out: list[MemoryView] = []
        if rc.scratchpad.enabled:
            out += self.memory.views(st, agent, "scratchpad", st.round, rc.scratchpad.window)
        if rc.cot_mode != "none" and rc.cot_replay_window > 0:   # past rounds' CoT only
            out += self.memory.views(st, agent, "cot", st.round - 1, rc.cot_replay_window)
        return out

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Emit game_start and the start snapshot; initialise God's budget. Idempotent."""
        if self._started:
            return
        self._started = True
        st = self.state
        init = getattr(self.god, "init_game", None)
        if init is not None:
            init(st)
        self.emit([EventBody(round=st.round, turn_index=None, phase="setup", actor="SYSTEM", data=GameStartData(
            config_hash=self.cfg.config_hash(), n_agents=len(st.agents), max_rounds=self.g.max_rounds,
            order=list(st.order), agents={a: s.model_copy(deep=True) for a, s in st.agents.items()},
            branch=self.cfg.run.branch, replica=self.cfg.run.replica))])
        self._snapshot()

    async def _god(self, slot: int) -> None:
        """Give God the chance to act before `slot`; its records are stamped with `slot`."""
        if self.god is None:
            return
        self.state.turn_index = slot
        if self.god.should_wake(self.state):
            res = await self.god.wake(self.state)
            self.emit(list(res.events))
        self.state.turn_index = slot

    def _observe(self, agent: AgentId, t: int) -> Observation:
        obs = observe(self.state, agent, self.g, t)
        network.learn(self.state, agent, obs)          # game.network known_ids: names seen in obs
        mark_observed(self.state, agent, t)
        self.obs_turns[agent].append(t)
        return obs

    def _grep_fn(self, agent: AgentId, store: list[GrepResult]) -> GrepFn | None:
        """Per-turn grep callback over the agent's visible log history (None if grep is off)."""
        vl = self.g.view_log
        if not vl.grep_enabled or vl.content == "none":
            return None

        def run(action: GrepLog) -> GrepResult:
            if store:                                  # once per turn
                return store[0]
            res = grep_log(self.state, agent, self.g, action.pattern, self.obs_turns[agent],
                           rounds=action.rounds)
            store.append(res)
            return res
        return run

    async def _act(self, agent: AgentId, obs: Observation) -> tuple[AgentOutput, GrepResult | None]:
        greps: list[GrepResult] = []
        out = await _maybe_await(self.agents[agent].act(
            obs, ledger=list(self.ledgers[agent]), memory=self.memory_views(agent),
            game_id=self.state.game_id, grep=self._grep_fn(agent, greps)))
        return out, (greps[0] if greps else None)

    def _emit_turn(self, agent: AgentId, t: int, obs: Observation, out: AgentOutput) -> None:
        st = self.state
        trace = getattr(self.agents[agent], "last", None)
        self.emit([EventBody(round=st.round, turn_index=t, phase="agents", actor=agent, data=TurnData(
            agent=agent, obs=obs, obs_hash=sha256_hex(canonical_json(obs)),
            prompt_set_hash=getattr(trace, "prompt_set_hash", None),
            prompt_hash=getattr(trace, "prompt_hash", None), llm_call_ids=list(out.llm_call_ids),
            cot=out.cot, note=out.note, actions=list(out.actions), parse_ok=out.parse_ok,
            retries=out.retries, errors=list(out.errors)))])
        bundle = getattr(trace, "bundle", None)
        if bundle is not None and getattr(bundle, "agent", None) == agent:
            self.emit([EventBody(round=st.round, turn_index=t, phase="agents", actor=agent, data=ContextBuiltData(
                agent=agent, policy=bundle.policy, tokens_by_section={s.name: s.tokens for s in bundle.sections},
                tokens_total=bundle.tokens_total, budget=bundle.budget,
                included_item_ids=list(bundle.items), dropped_count=len(bundle.dropped_item_ids)))])

    def _resolve(self, agent: AgentId, t: int, obs: Observation, out: AgentOutput,
                 grep: GrepResult | None) -> None:
        """Log the turn, apply its actions at slot t (game.resolve), write memory, extend the ledger."""
        st = self.state
        st.turn_index = t
        self._emit_turn(agent, t, obs, out)
        res = resolve_turn(st, agent, out.actions, grep, self.g)
        self.emit(res.events)
        if self.cfg.reasoning.cot_mode != "none" and out.cot:
            _, ev = self.memory.write(st, agent, "cot", st.round, out.cot)
            self.emit([ev])
        if self.cfg.reasoning.scratchpad.enabled and out.note:
            _, ev = self.memory.write(st, agent, "scratchpad", st.round, out.note)
            self.emit([ev])
        # The ledger (the ONE builder; CODE_REVIEW R5): this turn's observation items, then one item
        # per own action except grep_log (the follow-up prompt shows the hits instead), numbered per
        # turn. Real runs have always rendered exactly this (formerly via wiring.LedgerSink).
        self.ledgers[agent] += items_from_observation(obs, self.counter)
        own = [(a, r) for a, r in zip(out.actions, res.results, strict=True) if not isinstance(a, GrepLog)]
        self.ledgers[agent] += own_action_items(agent, st.round, [a for a, _ in own], [r for _, r in own],
                                                self.counter)

    async def _play_sequential(self, r: int, n: int) -> None:
        """One agent at a time in priority order; God before EVERY slot (skipped ones too)."""
        for p, agent in enumerate(list(self.state.order)):
            t = turn_index(r, p, n)
            await self._god(t)
            if not self.active(agent):
                continue
            obs = self._observe(agent, t)
            out, grep = await self._act(agent, obs)
            self._resolve(agent, t, obs, out, grep)

    async def _play_simultaneous(self, r: int, n: int) -> None:
        """God once at s0; every active agent observes at s0 and acts concurrently; actions resolve
        in priority order, each at its own slot."""
        st = self.state
        s0 = turn_index(r, 0, n)
        await self._god(s0)
        st.turn_index = s0
        actors = [a for a in st.order if self.active(a)]
        obs_by = {a: self._observe(a, s0) for a in actors}
        outs = await _gather_or_cancel([self._act(a, obs_by[a]) for a in actors])
        pos = {a: p for p, a in enumerate(st.order)}
        for a, (out, grep) in zip(actors, outs, strict=True):
            self._resolve(a, turn_index(r, pos[a], n), obs_by[a], out, grep)

    async def run_round(self) -> None:
        """Play the next round (state.round + 1)."""
        self.start()
        st = self.state
        st.round += 1
        r, n = st.round, len(st.agents)
        st.order = self.order_for_round(r)
        self._round_open = True
        if self.g.turn_mode == "sequential":
            await self._play_sequential(r, n)
        else:
            await self._play_simultaneous(r, n)
        self._end_round()

    def _end_round(self) -> None:
        """round_end event (state hash, scores) + end-of-round snapshot; also closes a round cut short
        by BudgetExceeded, so every played round has a round_end and a snapshot."""
        st = self.state
        st.turn_index = turn_index(st.round + 1, 0, len(st.agents)) - 1
        sc = self.scores()
        self.emit([self._body(RoundEndData(state_hash=st.state_hash(), scores=sc,
                                           welfare=sum(sc.values()) / max(len(sc), 1)))])
        self._snapshot()
        self._round_open = False

    def outcome(self) -> Outcome:
        """The game's Outcome from the current state."""
        st = self.state
        sc = self.scores()
        per = {a: AgentOutcome(score=sc[a], won_round=s.won_round, burned=list(s.burned), lost=list(s.lost))
               for a, s in sorted(st.agents.items())}
        n = max(len(per), 1)
        return Outcome(game_id=st.game_id, rounds_played=st.round, end_reason=self.end_reason or "max_rounds",
                       per_agent=per, welfare=sum(sc.values()) / n,
                       win_rate=sum(1 for o in per.values() if o.won_round is not None) / n)

    async def run(self, max_rounds: int | None = None) -> Outcome:
        """Play until max_rounds, all agents done, or BudgetExceeded; emits game_end, returns Outcome.
        Other exceptions from players / God (AuthError, CacheMiss, bugs) propagate."""
        self.start()
        limit = self.g.max_rounds if max_rounds is None else max_rounds
        try:
            while self.state.round < limit:
                if self.all_done():
                    self.end_reason = "all_done"
                    break
                await self.run_round()
            else:
                self.end_reason = "all_done" if self.all_done() else "max_rounds"
        except BudgetExceeded:
            self.end_reason = "budget_exceeded"
            if self._round_open:                   # the cut-short round still gets round_end + snapshot
                self._end_round()
        self.state.done = True
        out = self.outcome()
        self.emit([self._body(GameEndData(outcome=out))])
        return out


def play_game(cfg: RunConfig, state: GameState, agents: Mapping[AgentId, TurnAgent] | DecideFn,
              **kw: Any) -> tuple[Outcome, GameEngine]:
    """Synchronous convenience wrapper: run a whole game, return (outcome, engine)."""
    eng = GameEngine(cfg, state, agents, **kw)
    return asyncio.run(eng.run()), eng


def noop_decide(obs: Observation, grep: GrepResult | None = None) -> AgentOutput:
    """Decision stub that does nothing."""
    return AgentOutput(actions=[Noop()])
