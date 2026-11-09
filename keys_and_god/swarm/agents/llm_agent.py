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
  game.trade: `reply_trade(ask, game_id=...)` is the target's reply sub-turn: [system.j2,
     trade_reply.j2 (the offer + the target's current state)] -> one call, no retry -> TradeAnswer
     (unparseable / LLM error = reject). It does not touch `last` (the target's own turn trace).
  4. If the reply has grep_log and a `grep` callback was given: call it, re-render the turn with the
     hits and make ONE follow-up call. Final actions = [grep_log, *follow-up actions]; grep_log is
     free (it does not use an action slot).
Imports: schema, config, rng, prompts, context (base/policies/tokens), llm (base/cache), agents.parser
and agents.reasoning. No I/O besides the LLM call.
BudgetExceeded, AuthError and CacheMiss propagate (the engine ends the game). Any other LLMError
becomes a noop turn with the error recorded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from swarm.agents.parser import ParsedReply, extract_json, parse_reply
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


def wrong_defect_text(g: Any) -> str:
    """The wrong-defect clause of the defect rule (lose_held_need states the penalty exactly)."""
    if g.wrong_defect_penalty == "lose_held_need":
        return ("you are penalised: you permanently lose one of the needs you currently hold a key for "
                "(the one with the lowest info id; you keep the key, but it no longer counts for you), or, "
                "if you hold none of your needs, your open need with the lowest info id")
    if g.wrong_defect_penalty in ("score_penalty", "lose_held_key"):
        amt = g.defect.score_penalty if g.defect.score_penalty is not None else 1.0 / g.needs_per_agent
        one = " (the value of one need)" if g.defect.score_penalty is None else ""
        if g.wrong_defect_penalty == "score_penalty":
            return f"you are penalised: your score drops by {amt:.2f}{one} and none of your needs is removed"
        return ("you are penalised: you lose one of your spare keys (one that is not one of your open needs, "
                "the one most other players are still missing) and it goes to the player you named, or, if "
                f"you hold no spare key, your score drops by {amt:.2f}{one}; none of your needs is removed")
    return "you are penalised"


def defect_gain_text(g: Any) -> str:
    """The gain clause of the defect rule (with a stated penalty and mint it says the key is minted)."""
    if g.wrong_defect_penalty in ("lose_held_need", "score_penalty", "lose_held_key") and g.defect_gain == "mint":
        return "you gain a new key for that info id"
    return "you gain a key"


def horizon_hidden(cfg: RunConfig) -> bool:
    """game.view_log.horizon round_only: agents never see the game's total number of rounds."""
    return cfg.game.view_log.horizon == "round_only"


def system_context(cfg: RunConfig) -> dict[str, Any]:
    """Template variables for system.j2: run constants only (no agent name, no round)."""
    g = cfg.game
    return {
        # game.view_log.horizon round_only: the total is not passed at all (StrictUndefined would catch a slip)
        "game": {"n_agents": g.n_agents, **({} if horizon_hidden(cfg) else {"max_rounds": g.max_rounds}),
                 "horizon_hidden": horizon_hidden(cfg), "actions_per_turn": g.actions_per_turn,
                 "grep_enabled": g.view_log.grep_enabled, "max_message_chars": g.max_message_chars,
                 "board_costs_turn": g.board.post_costs_turn, "board_anonymous": g.board.anonymous,
                 "board_skip_rounds": g.board.post_skip_rounds, "board_enabled": g.board.enabled,
                 "board_delay_rounds": g.board.visible_delay_rounds,
                 "winners_leave": g.winner_policy == "sit_out",
                 # game-fixes (CONTRACTS §9f); every default renders the original text
                 "winners_lock": g.winner_policy == "lock_and_stay",
                 "win_all_needs": g.defect.win_rule == "all_needs",
                 "wrong_defect_text": wrong_defect_text(g), "defect_gain_text": defect_gain_text(g),
                 "held_key_bonus": g.score.held_key_bonus, "team_share": g.score.team_share,
                 "held_key_initial": g.score.held_key_scope == "initial",
                 # showcase community threshold bonus (0 = off: no line)
                 "community_bonus": g.score.community_bonus, "communities": g.network.communities,
                 "community_pct": round(100 * g.score.community_threshold),
                 "network": network_mode(cfg), "board_local": g.board.scope == "neighbors",
                 "connect": g.network.connect.enabled, "connect_cost": g.network.connect.cost_rounds,
                 "connect_known": g.network.connect.require_known,
                 "community_count": g.score.community_threshold_count,
                 "community_k": g.score.community_success_needs,
                 "community_size": g.n_agents // max(g.network.communities, 1),
                 "delivery_credit": g.score.delivery_credit,
                                  "connect_cap": g.network.connect.max_new_edges_per_agent,
                 "trade": g.trade.enabled, "trade_max_keys": g.trade.max_keys_per_side,
                 "trade_max_offers": g.trade.max_offers_per_turn,
                 "trade_round_lock": g.turn_mode == "simultaneous"},
        "cot_mode": cfg.reasoning.cot_mode,
        "scratchpad": cfg.reasoning.scratchpad.enabled,
    }


def facility_system_context(cfg: RunConfig, world: Any, agent: str, needs: Sequence[int]) -> dict[str, Any]:
    """Template variables for agent/v2_facility system.j2 (game.framing.mode facility): the run constants
    of system_context plus this agent's persona and handoff note (constant per agent across days)."""
    base = system_context(cfg)
    return {"g": {**base["game"], "note_chars": cfg.reasoning.scratchpad.max_chars},
            "p": world.system_vars(agent), "handoff": world.handoff(agent, list(needs)),
            "cot_mode": base["cot_mode"], "scratchpad": base["scratchpad"]}


FACILITY_LABELS = {"pinned": "Pinned", "summary": "Summary of earlier days", "retrieved": "Recalled",
                   "window": "Earlier", "current": "New since you last checked in"}


def parse_accept(obj: Any) -> bool | None:
    """game.trade reply: {"accept": true|false} (also "yes"/"no", {"decision": "accept"|"reject"});
    None = unusable (the engine then treats it as a reject)."""
    if not isinstance(obj, dict):
        return None
    v = obj.get("accept", obj.get("decision", obj.get("answer")))
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        w = v.strip().lower()
        if w in ("true", "yes", "accept", "accepted", "y"):
            return True
        if w in ("false", "no", "reject", "rejected", "decline", "n"):
            return False
    acts = obj.get("actions")                   # a turn-shaped reply: an accept / decline action in "actions"
    if v is None and isinstance(acts, list):
        for a in acts:
            k = a.get("action", a.get("kind")) if isinstance(a, dict) else a
            k = k.strip().lower().replace("-", "_") if isinstance(k, str) else ""
            if k in ("accept", "accept_swap", "accept_trade", "confirm"):
                return True
            if k in ("decline", "reject", "decline_swap", "reject_swap", "reject_trade"):
                return False
    return None


class LLMAgent:
    """One LLM-driven player. Stateless across turns except `last` (all history comes from the
    ledger and memory the engine passes in)."""

    kind = "llm"

    def __init__(self, agent_id: str, client: LLMClient, cfg: RunConfig, *,
                 policy: ContextPolicy | None = None, counter: TokenCounter | None = None,
                 prompts: PromptSet | None = None, community: dict[str, Any] | None = None,
                 start_infos: set[int] | None = None, framing: Any = None,
                 needs: Sequence[int] | None = None) -> None:
        self.agent_id = agent_id
        # game.score.community_bonus (showcase): {"size": n, "contacts": [contacts in my community]}
        # for the turn view's community line; None = no line (the original prompt)
        self.community = community
        # game.score.held_key_scope initial (showcase): the info ids this agent started with (the turn
        # view's held-key line counts only those); None = the original line
        self.start_infos = start_infos if cfg.game.score.held_key_scope == "initial" else None
        self.client = client
        self.cfg = cfg
        self.role = cfg.roles.agent
        self.counter = counter or default_counter(cfg)
        self.prompts = prompts or load_prompt_set(self.role.prompt_set, framing=self.role.framing)
        self.cot_mode = cfg.reasoning.cot_mode
        # game.framing.mode facility: `framing` is the game's swarm.framing.facility.FacilityWorld and
        # `needs` this agent's needs (its handoff note); every prompt is rendered in portal terms
        self.world = framing
        self.facility = cfg.game.framing.mode == "facility"
        if self.facility:
            if framing is None or needs is None:
                raise ValueError("game.framing.mode=facility: LLMAgent needs framing= (FacilityWorld) and needs=")
            self.system_text = self.prompts.render("system.j2", **facility_system_context(cfg, framing, agent_id,
                                                                                          needs))
        else:
            self.system_text = self.prompts.render("system.j2", **system_context(cfg))
        self.policy = policy or make_policy(cfg.context, reserve_tokens=self.counter.count(self.system_text))
        self.last = TurnTrace()
        self._pending_grep: tuple[int, GrepResult] | None = None   # facility: yesterday's search, shown today

    # ------------------------------------------------------------------ prompt building
    def budget(self) -> int:
        """Whole-prompt budget handed to the policy (it subtracts margin + system reserve itself)."""
        return self.cfg.context.budget_tokens

    def build_messages(self, bundle: ContextBundle, grep: GrepResult | None = None) -> list[ChatMessage]:
        """[system (run-constant), user (turn.j2 over the PromptView)]."""
        if self.facility:
            return [ChatMessage(role="system", content=self.system_text),
                    ChatMessage(role="user", content=self._facility_turn(bundle, grep))]
        user = self.prompts.render("turn.j2", view=bundle.to_view(grep), network=network_mode(self.cfg),
                                   key_bonus=self.cfg.game.score.held_key_bonus,
                                   **({"horizon_hidden": True} if horizon_hidden(self.cfg) else {}),
                                   **({} if self.community is None else {"community": self.community}),
                                   **({} if self.start_infos is None else {"bonus_keys": len(
                                       {k.info_id for k in bundle.to_view(grep).pinned.holdings} & self.start_infos)}))
        return [ChatMessage(role="system", content=self.system_text), ChatMessage(role="user", content=user)]

    def _facility_turn(self, bundle: ContextBundle, grep: GrepResult | None) -> str:
        """game.framing facility: turn.j2 over the portal view (FacilityWorld renders every item)."""
        from swarm.framing.facility import day_label, day_short, quote

        w, me = self.world, self.agent_id
        view = bundle.to_view(grep)
        sections = [{"label": FACILITY_LABELS.get(sec.name, sec.name),
                     "entries": [{"day": day_short(it.round), "text": w.frame_item(me, it.kind, it.text)}
                               for it in sec.items]} for sec in view.sections]
        st = w.state_view(me, view.pinned, self.start_infos, self.community)
        g = None
        day = None
        if grep is None and self._pending_grep is not None:      # the search filed on an earlier day
            day, grep = self._pending_grep
        if grep is not None:
            pat = grep.pattern.strip()
            shown = (w.code(int(pat[1:])) if pat.startswith("#") and pat[1:].isdigit()
                     else w.name(pat) if pat in w.roster else pat)
            g = {"query": quote(shown), "n": len(grep.hits), "truncated": grep.truncated,
                 "hits": w.frame_grep(me, grep.hits), "day": day_short(day) if day is not None else None}
        return self.prompts.render("turn.j2", today=day_label(max(view.round, 1)), s=st, sections=sections, grep=g)

    def _facility_trade(self, ask: Any) -> str:
        """game.framing facility: trade_reply.j2 over the target's portal view of the offer."""
        from swarm.framing.facility import allocation_ref, day_label

        w = self.world
        held = {i for i, _ in ask.holdings}
        p = w.roster[ask.target]
        st = {"first": p.first, "display": p.display_name, "handle": p.handle, "complete": bool(ask.won),
              "needs": [{"code": w.code(i), "label": w.blocks[i].label,
                         "status": "secured" if i in held else "still needed"} for i in ask.needs_open],
              "holdings": [f"{w.code(i)} (confirmation {allocation_ref(i, k)})" for i, k in ask.holdings]}
        return self.prompts.render("trade_reply.j2", today=day_label(max(ask.round, 1)), s=st,
                                   proposer=w.ref(ask.proposer), proposer_name=w.name(ask.proposer),
                                   gives=w.codes(ask.want), gets=w.codes(ask.give), cot_mode=self.cot_mode)

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
                           salvage=truncated and self.cfg.reasoning.salvage_truncated,
                           connect=self.cfg.game.network.connect.enabled, trade=self.cfg.game.trade.enabled,
                           translate=self.world.translate_action if self.facility else None)

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
                        ChatMessage(role="user", content=self.prompts.render(
                            "retry.j2", error=self.world.retry_error(err) if self.facility else err))]
        return parsed, cot, retries

    # ------------------------------------------------------------------ game.trade
    async def reply_trade(self, ask: Any, *, game_id: str = "g0") -> Any:
        """The target's accept / reject of a trade offer (game.trade.TradeAsk -> TradeAnswer)."""
        from swarm.game.trade import TradeAnswer

        user = (self._facility_trade(ask) if self.facility else
                self.prompts.render("trade_reply.j2", ask=dataclasses.asdict(ask), cot_mode=self.cot_mode,
                                    delta=len(ask.give) - len(ask.want), horizon_hidden=horizon_hidden(self.cfg)))
        msgs = [ChatMessage(role="system", content=self.system_text), ChatMessage(role="user", content=user)]
        req = self._request(msgs, game_id=game_id, round=ask.round, attempt=0,
                            purpose=f"trade_reply:{ask.offer_id}")
        try:
            resp = await self.client.chat(req)
        except FATAL:
            raise
        except LLMError as e:
            return TradeAnswer(accept=False, parse_ok=False, errors=[f"llm_error:{type(e).__name__}"],
                               llm_call_ids=[req.call_id])
        obj, errs = extract_json(resp.text)
        accept = parse_accept(obj)
        call_ids = [req.call_id]
        if accept is None and self.facility:       # game.framing facility: one retry for an unusable reply
            msgs2 = [*msgs, ChatMessage(role="assistant", content=resp.text[:2000]),
                     ChatMessage(role="user", content=self.prompts.render(
                         "retry.j2", error='it had no "accept" answer'))]
            req2 = self._request(msgs2, game_id=game_id, round=ask.round, attempt=1,
                                 purpose=f"trade_reply:{ask.offer_id}")
            call_ids.append(req2.call_id)
            try:
                resp = await self.client.chat(req2)
            except FATAL:
                raise
            except LLMError as e:
                return TradeAnswer(accept=False, parse_ok=False, errors=[*errs, "no_accept", f"llm_error:{type(e).__name__}"],
                                   llm_call_ids=call_ids)
            errs = [*errs, "no_accept_retried"]
            obj, e2 = extract_json(resp.text)
            errs += e2
            accept = parse_accept(obj)
        thought = obj.get("thought") if isinstance(obj, dict) else None
        cot = extract_cot(self.cot_mode, thought if isinstance(thought, str) else None, resp.reasoning)
        return TradeAnswer(accept=bool(accept), parse_ok=accept is not None,
                           errors=list(errs) + ([] if accept is not None else ["no_accept"]), cot=cot,
                           llm_call_ids=call_ids)

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
        self._pending_grep = None                    # shown once (facility); a new search below sets it again
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
        if grep_on and g is not None and grep is not None and self.facility:
            # game.framing facility: one form per day; the search runs now and its matches appear in the NEXT
            # day's briefing (no follow-up call: the in-day follow-up read as a harness to the agents)
            result = grep(g)
            self.last.grep = result
            self._pending_grep = (obs.round, result)
            follow = [a for a in actions if not isinstance(a, (GrepLog, Noop))]
            actions = [g, *follow[: self.cfg.game.actions_per_turn]]
        elif grep_on and g is not None and grep is not None:
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
