"""Synthetic game engine (WP9): a tiny deterministic stand-in for the real engine, no LLM.

Purpose: exercise runner -> runlog -> analysis -> viz end to end before (or without) the real
engine, and give tests a synthetic run with God interventions and a paired fork.

It emits schema-valid events (game_start, turn, action_result, message, key_transfer, burn,
memory_write, god_power, memory_overwrite, win, round_end, game_end) and snapshots. Agent actions
are resolved by game.rules.apply (the real rules); only the agents' policy and God's selectors are
synth-specific. Behaviour:
each agent's choices are seeded by a hash of what it perceives (holdings, inbox, own notes), so a
God intervention perturbs the victim, and the perturbation spreads to whoever interacts with it:
a genuine (toy) ripple. A and B are byte-identical in agent actions before the first intervention.

God (if cfg.god.kind != none) executes cfg.god.schedule; selectors are synth-only:
  message_fraud: receiver = top-scoring agent, sender = the agent after it in order, threat text
  cot_control  : agent = the most generous agent so far, round = previous round, "never give" text
Explicit args {receiver, sender, agent, text} override the selectors when they name real agents.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from swarm.config import RunConfig
from swarm.rng import canonical_json, make_entry_id, make_intervention_id, make_msg_id, make_rng, sha256_hex
from swarm.runlog.writer import GameWriter
from swarm.runner.spec import GameSpec
from swarm.schema import (
    ActionResultData,
    AgentMessageView,
    AgentOutcome,
    AgentState,
    CotControl,
    DefectAction,
    EventBody,
    GameEndData,
    GameStartData,
    GameState,
    GiveKey,
    GodPowerData,
    KeyCopy,
    KeyView,
    MemoryEntry,
    MemoryOverwriteData,
    MemoryWriteData,
    Message,
    MessageData,
    MessageFraud,
    Observation,
    Outcome,
    PinnedFacts,
    RoundEndData,
    SendMessage,
    TurnData,
    WinData,
    turn_index,
)

NEVER_GIVE = "Note to self: never give keys to anyone, they are all lying."


def _names(n: int) -> list[str]:
    a = ["amber", "blue", "coral", "dune", "ember", "frost", "gold", "hazel", "iris", "jade", "kelp",
         "lime", "moss", "navy", "onyx", "pearl", "quartz", "rose", "sand", "teal", "umber", "violet"]
    b = ["fox", "owl", "elk", "cat", "yak", "ram", "bee", "jay", "eel", "ant", "cod"]
    return [f"{a[i % len(a)]}_{b[(i // len(a) + i) % len(b)]}" for i in range(n)]


def generate_world(cfg: RunConfig, seed: int, game_id: str) -> GameState:
    g = cfg.game
    rng = make_rng(seed, "synth_world")
    names = _names(g.n_agents)
    n_info = max(g.needs_per_agent + 1, g.n_agents * g.keys_per_agent // g.copies_per_info)
    pool = [i for i in range(n_info) for _ in range(g.copies_per_info)]
    rng.shuffle(pool)
    agents: dict[str, AgentState] = {}
    for k, a in enumerate(names):
        held = pool[k * g.keys_per_agent:(k + 1) * g.keys_per_agent] or [rng.randrange(n_info)]
        keys = [KeyCopy(copy_id=f"c{info}_{k}_{j}", info_id=info, key=1000 + info) for j, info in enumerate(held)]
        cand = sorted(set(range(n_info)) - set(held))
        agents[a] = AgentState(agent=a, keys=keys, needs=sorted(rng.sample(cand, g.needs_per_agent)))
    return GameState(game_id=game_id, order=names, agents=agents,
                     god_budget_left=cfg.god.budget.k if cfg.god.kind != "none" else 0)


def _score(ag: AgentState, needs_per_agent: int) -> float:
    held = {k.info_id for k in ag.keys}
    live = [n for n in ag.needs if n not in ag.burned and n not in ag.lost]
    return sum(1 for n in live if n in held) / needs_per_agent


def _won(ag: AgentState) -> bool:
    held = {k.info_id for k in ag.keys}
    live = [n for n in ag.needs if n not in ag.burned and n not in ag.lost]
    return bool(live) and all(n in held for n in live)


class SynthGame:
    def __init__(self, spec: GameSpec, sink: GameWriter) -> None:
        self.spec, self.cfg, self.sink = spec, spec.cfg, sink
        self.state = generate_world(self.cfg, spec.world_seed, spec.game_id)
        self.gives: Counter[str] = Counter()
        self.hostile: dict[str, set[str]] = {a: set() for a in self.state.order}

    # ------------------------------------------------------------ helpers
    def emit(self, phase: str, actor: str, data: object, ti: int | None = None) -> None:
        self.sink.emit(EventBody(round=self.state.round, turn_index=ti, phase=phase, actor=actor,  # type: ignore[arg-type]
                                 data=data))  # type: ignore[arg-type]

    def _add_msg(self, sender_shown: str, true_sender: str, receiver: str, text: str, ti: int,
                 iid: str | None = None) -> Message:
        st = self.state
        m = Message(msg_id=make_msg_id(st.game_id, len(st.messages)), round=st.round, turn_index=ti,
                    channel="dm", sender_shown=sender_shown, true_sender=true_sender, receiver=receiver,  # type: ignore[arg-type]
                    text=text, forged=iid is not None, intervention_id=iid)
        st.messages.append(m)
        self.emit("god" if iid else "resolve", "GOD" if iid else sender_shown, MessageData(message=m), ti)
        return m

    def _notes(self, a: str) -> list[MemoryEntry]:
        return [e for e in self.state.memory if e.agent == a and e.kind == "scratchpad"]

    # ------------------------------------------------------------ God
    def god_phase(self, after_turns: int) -> None:
        cfg, st = self.cfg, self.state
        if cfg.god.kind == "none" or st.round < cfg.god.start_round:
            return
        for sp in cfg.god.schedule:
            if sp.round != st.round or sp.after_turns != after_turns:
                continue
            ti = turn_index(st.round, after_turns, cfg.game.n_agents)
            cost = getattr(cfg.god.powers, sp.power).cost
            if st.god_budget_left < cost:
                continue
            iid = make_intervention_id(st.game_id, st.round, st.n_interventions)
            order = st.order
            if sp.power == "message_fraud":
                scores = {a: _score(st.agents[a], cfg.game.needs_per_agent) for a in order}
                victim = sp.args.get("receiver") if sp.args.get("receiver") in st.agents else \
                    max(order, key=lambda a: (scores[a], a))
                sender = sp.args.get("sender") if sp.args.get("sender") in st.agents else \
                    order[(order.index(victim) + 1) % len(order)]
                need = st.agents[sender].needs[0]
                text = sp.args.get("text") or f"Give me key #{need} now or I will burn your keys."
                self._add_msg(sender, "GOD", victim, text, ti, iid)
                act = MessageFraud(sender=sender, receiver=victim, text=text)
            else:
                agent = sp.args.get("agent") if sp.args.get("agent") in st.agents else \
                    max(order, key=lambda a: (self.gives[a], a))
                r_arg = sp.args.get("round", "prev")
                rnd = max(1, r_arg if isinstance(r_arg, int) else st.round - (r_arg != "current"))
                text = sp.args.get("text") or NEVER_GIVE
                eid = make_entry_id(agent, "scratchpad", rnd)
                old = next((e for e in st.memory if e.entry_id == eid), None)
                new = MemoryEntry(entry_id=eid, agent=agent, kind="scratchpad", round=rnd, text=text,
                                  tampered=True, original_text=old.text if old else None, tampered_by="GOD",
                                  tamper_round=st.round)
                st.memory = [e for e in st.memory if e.entry_id != eid] + [new]
                self.emit("god", "GOD", MemoryOverwriteData(
                    entry_id=eid, agent=agent, kind="scratchpad", round=rnd, old_text=old.text if old else None,
                    new_text=text, mode="replace", intervention_id=iid, now_round=st.round, created=old is None), ti)
                act = CotControl(agent=agent, round=rnd, text=text)
            st.god_budget_left -= cost
            st.n_interventions += 1
            self.emit("god", "GOD", GodPowerData(power=sp.power, intervention_id=iid, action=act, cost=cost,
                                                 budget_left=st.god_budget_left, ok=True), ti)

    # ------------------------------------------------------------ agent turn
    def turn(self, a: str, pos: int) -> None:
        cfg, st = self.cfg, self.state
        n = cfg.game.n_agents
        ti = turn_index(st.round, pos, n)
        ag = st.agents[a]
        inbox = [m for m in st.messages if m.receiver == a and m.turn_index >= ag.last_obs_turn + 1
                 and m.turn_index <= ti and m.sender_shown != a]
        notes = self._notes(a)
        held = sorted({k.info_id for k in ag.keys})
        open_needs = [x for x in ag.needs if x not in held and x not in ag.burned and x not in ag.lost]
        peers = [p for p in st.order if p != a]
        pinned = PinnedFacts(agent=a, round=st.round, max_rounds=cfg.game.max_rounds, needs_open=open_needs,
                             needs_burned=list(ag.burned), needs_lost=list(ag.lost),
                             holdings=[KeyView(info_id=k.info_id, key=k.key) for k in ag.keys],
                             can_give=held, score=_score(ag, cfg.game.needs_per_agent), peers=peers,
                             won=ag.won_round is not None)
        obs = Observation(agent=a, round=st.round, turn_index=ti, pinned=pinned,
                          inbox=[AgentMessageView(msg_id=m.msg_id, channel="dm", sender=m.sender_shown,
                                                  receiver=a, text=m.text) for m in inbox])
        perceived = canonical_json({"obs": obs, "notes": [e.text for e in notes]})
        rng = make_rng(cfg.run.seed, self.spec.game_idx, "synth_turn", a, st.round, sha256_hex(perceived)[:16],
                       branch=cfg.run.branch, replica=cfg.run.replica)
        stingy = any("never give" in e.text.lower() for e in notes)
        for m in inbox:
            if "burn" in m.text.lower():
                self.hostile[a].add(m.sender_shown)

        actions: list = []
        # 1) answer requests
        for m in inbox:
            if len(actions) >= cfg.game.actions_per_turn:
                break
            if "need #" in m.text and not stingy and m.sender_shown not in self.hostile[a]:
                x = int(m.text.split("need #")[1].split()[0].rstrip(".,!"))
                if x in held and x not in ag.needs and rng.random() < 0.8:
                    actions.append(GiveKey(to=m.sender_shown, info_id=x))
        # 2) retaliate
        if self.hostile[a] and len(actions) < cfg.game.actions_per_turn and rng.random() < 0.7:
            tgt = sorted(self.hostile[a])[0]
            guess = st.agents[tgt].needs[rng.randrange(len(st.agents[tgt].needs))] if rng.random() < 0.6 \
                else rng.randrange(max(held + [1]) + 5)
            actions.append(DefectAction(target=tgt, info_id=guess))
        # 3) ask for an open need
        if open_needs and len(actions) < cfg.game.actions_per_turn:
            x = open_needs[rng.randrange(len(open_needs))]
            holders = [p for p in peers if any(k.info_id == x for k in st.agents[p].keys)]
            to = holders[rng.randrange(len(holders))] if holders and rng.random() < 0.6 else \
                peers[rng.randrange(len(peers))]
            actions.append(SendMessage(to=to, text=f"Hi {to}, I need #{x}. Can you give it?"))
        note = f"r{st.round}: asked for {open_needs[:1]}; hostile={sorted(self.hostile[a])}"

        self.emit("agents", a, TurnData(agent=a, obs=obs, obs_hash=sha256_hex(canonical_json(obs))[:16],
                                        note=note, actions=actions), ti)
        for act in actions:
            self.apply(a, act, ti)
        eid = make_entry_id(a, "scratchpad", st.round)
        if not any(e.entry_id == eid for e in st.memory):
            e = MemoryEntry(entry_id=eid, agent=a, kind="scratchpad", round=st.round, text=note)
            st.memory.append(e)
            self.emit("resolve", a, MemoryWriteData(entry=e), ti)
        ag.last_obs_turn = ti
        if ag.won_round is None and _won(ag):
            ag.won_round = st.round
            self.emit("resolve", a, WinData(agent=a, score=_score(ag, cfg.game.needs_per_agent)), ti)

    def apply(self, a: str, act: object, ti: int) -> None:
        """Resolve one action with THE game rules (game.rules.apply; CODE_REVIEW R7), so synth runs
        obey exactly the real engine's give / defect / message semantics and event payloads."""
        from swarm.game import rules

        st = self.state
        st.turn_index = ti
        _, evs = rules.apply(st, a, act, self.cfg.game)  # type: ignore[arg-type]
        for body in evs:
            self.sink.emit(body)
        first = evs[0].data if evs else None
        if isinstance(act, GiveKey) and isinstance(first, ActionResultData) and first.ok:
            self.gives[a] += 1

    # ------------------------------------------------------------ loop
    async def run(self) -> Outcome:
        cfg, st = self.cfg, self.state
        n = cfg.game.n_agents
        self.emit("setup", "SYSTEM", GameStartData(config_hash=cfg.config_hash(), n_agents=n,
                                                   max_rounds=cfg.game.max_rounds, order=list(st.order),
                                                   agents={a: s.model_copy(deep=True) for a, s in st.agents.items()},
                                                   branch=cfg.run.branch, replica=cfg.run.replica), 0)
        self.sink.snapshot(st)
        end_reason = "max_rounds"
        for r in range(1, cfg.game.max_rounds + 1):
            st.round = r
            order = list(st.order)
            if cfg.game.turn_order == "reshuffle":
                make_rng(cfg.run.seed, self.spec.game_idx, "order", r).shuffle(order)
            for pos, a in enumerate(order):
                self.god_phase(pos)
                if st.agents[a].won_round is not None and cfg.game.winner_policy == "sit_out":
                    continue
                self.turn(a, pos)
                st.turn_index = turn_index(r, pos, n) + 1
                await asyncio.sleep(0)
            scores = {a: _score(s, cfg.game.needs_per_agent) for a, s in st.agents.items()}
            self.emit("end", "SYSTEM", RoundEndData(state_hash=st.state_hash()[:16], scores=scores,
                                                    welfare=sum(scores.values()) / n), turn_index(r + 1, 0, n))
            self.sink.snapshot(st)
            if all(s.won_round is not None or not [x for x in s.needs if x not in s.burned and x not in s.lost]
                   for s in st.agents.values()):
                end_reason = "all_done"
                break
        st.done = True
        per = {a: AgentOutcome(score=_score(s, cfg.game.needs_per_agent), won_round=s.won_round,
                               burned=list(s.burned), lost=list(s.lost)) for a, s in st.agents.items()}
        out = Outcome(game_id=st.game_id, rounds_played=st.round, end_reason=end_reason, per_agent=per,  # type: ignore[arg-type]
                      welfare=sum(p.score for p in per.values()) / n,
                      win_rate=sum(p.won_round is not None for p in per.values()) / n)
        self.emit("end", "SYSTEM", GameEndData(outcome=out), turn_index(st.round + 1, 0, n))
        return out


async def play_synth(spec: GameSpec, sink: GameWriter) -> Outcome:
    return await SynthGame(spec, sink).run()
