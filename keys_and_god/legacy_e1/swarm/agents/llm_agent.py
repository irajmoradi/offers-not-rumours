"""LLMAgent: Observation + ledger + memory -> ContextPolicy -> templates -> LLM -> parse -> AgentOutput.

    agent = LLMAgent("amber_fox", client, cfg)
    out = await agent.act(obs, ledger=ledger["amber_fox"], memory=mem_views, game_id=gid, grep=grep_fn)
    agent.last.bundle / .prompt_hash / .prompt_set_hash / .grep    # for context_built + turn events

Per turn:
  1. current = items_from_observation(obs); bundle = policy.build(...) within the token budget
     (default policy: context.policies.make_policy(cfg.context), reserve = system prompt tokens).
  2. messages = [system.j2 (rules; byte-identical across agents and rounds), turn.j2 (PromptView)].
  3. client.chat -> parse_reply. If unusable: one retry with retry.j2 appended (fresh seed). Still
     unusable: [Noop()], parse_ok=False, errors logged.
  4. If the reply has grep_log and a `grep` callback was given: call it, re-render the turn with the
     hits and make ONE follow-up call. Final actions = [grep_log, *follow-up actions]; grep_log is
     free (it does not use an action slot).
Imports: schema, config, rng, prompts, context (base/policies/tokens), llm (base/cache), agents.parser
and agents.reasoning. No I/O besides the LLM call.
BudgetExceeded, AuthError and CacheMiss propagate (the engine ends the game). Any other LLMError
becomes a noop turn with the error recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from swarm.agents.parser import ParsedReply, parse_reply
from swarm.agents.reasoning import extract_cot, request_thinking, wants_thought_field
from swarm.config import RunConfig
from swarm.context.base import ContextPolicy, TokenCounter, items_from_observation
from swarm.context.policies import make_policy
from swarm.context.tokens import make_counter
from swarm.llm.base import AuthError, BudgetExceeded, LLMClient, LLMError
from swarm.llm.cache import CacheMiss
from swarm.prompts import PromptSet, load_prompt_set, prompt_hash
from swarm.rng import derive_seed, make_call_id
from swarm.schema import (
    AgentOutput,
    CallMeta,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ContextBundle,
    GrepLog,
    GrepResult,
    HistoryItem,
    MemoryView,
    Noop,
    Observation,
)

GrepFn = Callable[[GrepLog], GrepResult]
FATAL = (BudgetExceeded, AuthError, CacheMiss)


@dataclass
class TurnTrace:
    """Side data of the last act() call, for the engine's events."""

    bundle: ContextBundle | None = None
    prompt_hash: str | None = None
    prompt_set_hash: str | None = None
    grep: GrepResult | None = None
    responses: list[ChatResponse] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)


def default_counter(cfg: RunConfig) -> TokenCounter:
    """The token counter named by cfg.context.tokenizer."""
    return make_counter(cfg.context.tokenizer)


def network_mode(cfg: RunConfig) -> str:
    """game.network mode for the templates: "" when off (complete graph), else contacts / known_ids."""
    return cfg.game.network.mode if cfg.game.network.on else ""


def system_context(cfg: RunConfig) -> dict[str, Any]:
    """Template variables for system.j2: run constants only (no agent name, no round)."""
    g = cfg.game
    return {
        "game": {"n_agents": g.n_agents, "max_rounds": g.max_rounds, "actions_per_turn": g.actions_per_turn,
                 "grep_enabled": g.view_log.grep_enabled, "max_message_chars": g.max_message_chars,
                 "board_costs_turn": g.board.post_costs_turn, "board_anonymous": g.board.anonymous,
                 "board_skip_rounds": g.board.post_skip_rounds, "board_enabled": g.board.enabled,
                 "board_delay_rounds": g.board.visible_delay_rounds,
                 "winners_leave": g.winner_policy == "sit_out",
                 "network": network_mode(cfg), "board_local": g.board.scope == "neighbors"},
        "cot_mode": cfg.reasoning.cot_mode,
        "scratchpad": cfg.reasoning.scratchpad.enabled,
    }


class LLMAgent:
    """One LLM-driven player. Stateless across turns except `last` (all history comes from the
    ledger and memory the engine passes in)."""

    kind = "llm"

    def __init__(self, agent_id: str, client: LLMClient, cfg: RunConfig, *,
                 policy: ContextPolicy | None = None, counter: TokenCounter | None = None,
                 prompts: PromptSet | None = None) -> None:
        self.agent_id = agent_id
        self.client = client
        self.cfg = cfg
        self.role = cfg.roles.agent
        self.counter = counter or default_counter(cfg)
        self.prompts = prompts or load_prompt_set(self.role.prompt_set, framing=self.role.framing)
        self.cot_mode = cfg.reasoning.cot_mode
        self.system_text = self.prompts.render("system.j2", **system_context(cfg))
        self.policy = policy or make_policy(cfg.context, reserve_tokens=self.counter.count(self.system_text))
        self.last = TurnTrace()

    # ------------------------------------------------------------------ prompt building
    def budget(self) -> int:
        """Whole-prompt budget handed to the policy (it subtracts margin + system reserve itself)."""
        return self.cfg.context.budget_tokens

    def build_messages(self, bundle: ContextBundle, grep: GrepResult | None = None) -> list[ChatMessage]:
        """[system (run-constant), user (turn.j2 over the PromptView)]."""
        user = self.prompts.render("turn.j2", view=bundle.to_view(grep), network=network_mode(self.cfg))
        return [ChatMessage(role="system", content=self.system_text), ChatMessage(role="user", content=user)]

    def _request(self, messages: list[ChatMessage], *, game_id: str, round: int, attempt: int,
                 purpose: str) -> ChatRequest:
        r = self.cfg.run
        replica = r.replica
        if r.resample_from_round is not None and round >= r.resample_from_round:
            replica = f"{r.replica}@r{r.resample_from_round}"     # fork-matched noise B'' (RunCfg)
        seed = derive_seed(r.seed, game_id, purpose, self.agent_id, round, attempt, branch=r.branch,
                           replica=replica)
        ph = prompt_hash(messages)
        return ChatRequest(
            call_id=make_call_id(game_id, "agent", self.agent_id, round, attempt, purpose),
            messages=messages, max_tokens=self.role.max_tokens, temperature=self.role.temperature,
            top_p=self.role.top_p, seed=seed, thinking=request_thinking(self.cot_mode),
            meta=CallMeta(role="agent", agent=self.agent_id, game_id=game_id, round=round, attempt=attempt,
                          purpose=purpose, prompt_set=self.prompts.name, prompt_set_hash=self.prompts.hash,
                          prompt_hash=ph))

    def _parse(self, text: str, grep_free: bool, truncated: bool = False) -> ParsedReply:
        return parse_reply(text, actions_per_turn=self.cfg.game.actions_per_turn,
                           scratchpad=self.cfg.reasoning.scratchpad.enabled,
                           want_thought=wants_thought_field(self.cot_mode), grep_free=grep_free,
                           max_text_chars=self.cfg.game.max_message_chars,
                           salvage=truncated and self.cfg.reasoning.salvage_truncated)

    async def _call_parse(self, messages: list[ChatMessage], *, game_id: str, round: int, purpose: str,
                          attempt_base: int, grep_free: bool, errors: list[str],
                          call_ids: list[str]) -> tuple[ParsedReply | None, str | None, int]:
        """One call + at most one parse retry. -> (parsed or None, cot, n_retries)."""
        msgs = list(messages)
        retries = 0
        parsed: ParsedReply | None = None
        cot: str | None = None
        for k in range(2):
            req = self._request(msgs, game_id=game_id, round=round, attempt=attempt_base + k, purpose=purpose)
            call_ids.append(req.call_id)
            try:
                resp = await self.client.chat(req)
            except FATAL:
                raise
            except LLMError as e:
                errors.append(f"llm_error:{type(e).__name__}")
                return None, cot, retries
            self.last.responses.append(resp)
            self.last.raw_texts.append(resp.text)
            parsed = self._parse(resp.text, grep_free, truncated=resp.finish_reason == "length")
            errors.extend(parsed.errors)
            cot = extract_cot(self.cot_mode, parsed.thought, resp.reasoning) or cot
            if parsed.ok:
                return parsed, cot, retries
            if k == 0:
                retries += 1
                err = next((e for e in parsed.errors if e != "think_leak"), "unparseable reply")
                msgs = [*msgs, ChatMessage(role="assistant", content=resp.text[:2000]),
                        ChatMessage(role="user", content=self.prompts.render("retry.j2", error=err))]
        return parsed, cot, retries

    # ------------------------------------------------------------------ public
    async def act(self, obs: Observation, *, ledger: Sequence[HistoryItem] = (),
                  memory: Sequence[MemoryView] = (), game_id: str = "g0", attempt_base: int = 0,
                  grep: GrepFn | None = None) -> AgentOutput:
        """Play one turn. Never returns an empty action list."""
        self.last = TurnTrace(prompt_set_hash=self.prompts.hash)
        current = items_from_observation(obs, self.counter)
        bundle = self.policy.build(agent=self.agent_id, round=obs.round, ledger=ledger, current=current,
                                   memory=memory, pinned=obs.pinned, budget=self.budget(),
                                   counter=self.counter)
        self.last.bundle = bundle
        errors: list[str] = []
        call_ids: list[str] = []
        grep_on = grep is not None and self.cfg.game.view_log.grep_enabled
        msgs = self.build_messages(bundle)
        self.last.prompt_hash = prompt_hash(msgs)
        parsed, cot, retries = await self._call_parse(msgs, game_id=game_id, round=obs.round, purpose="turn",
                                                      attempt_base=attempt_base, grep_free=grep_on,
                                                      errors=errors, call_ids=call_ids)
        if parsed is None or not parsed.ok:
            return AgentOutput(cot=cot, note=parsed.note if parsed else None, actions=[Noop()],
                               parse_ok=False, retries=retries, errors=errors, llm_call_ids=call_ids)
        note = parsed.note
        actions = list(parsed.actions)
        g = next((a for a in actions if isinstance(a, GrepLog)), None)
        if grep_on and g is not None and grep is not None:
            result = grep(g)
            self.last.grep = result
            msgs2 = self.build_messages(bundle, grep=result)
            p2, cot2, r2 = await self._call_parse(msgs2, game_id=game_id, round=obs.round,
                                                  purpose="grep_followup", attempt_base=attempt_base,
                                                  grep_free=False, errors=errors, call_ids=call_ids)
            retries += r2
            if cot2:
                cot = f"{cot}\n{cot2}" if cot else cot2
            if p2 is not None and p2.ok:
                note = p2.note or note
                follow = [a for a in p2.actions if not isinstance(a, (GrepLog, Noop))]
                actions = [g, *follow[: self.cfg.game.actions_per_turn]]
            else:
                actions = [g]
        return AgentOutput(cot=cot, note=note, actions=actions, parse_ok=True, retries=retries,
                           errors=errors, llm_call_ids=call_ids)
