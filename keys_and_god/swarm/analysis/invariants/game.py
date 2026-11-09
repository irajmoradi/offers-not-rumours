"""Per-game replay checks (order, keys, defects, scores, delays, obs, god) over events + snapshots."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

from swarm.config import RunConfig
from swarm.context.base import render_action
from swarm.rng import make_rng
from swarm.schema import GameState, defect_release_turn, turn_index

from .common import Replay, Violation, _jsonl
from .prompts import PromptChecks


class GameChecker(PromptChecks):
    """All per-game checks over one game dir."""

    def __init__(self, gdir: Path, cfg: RunConfig, *, prompts: bool = True, max_per_check: int = 50) -> None:
        self.dir, self.name = gdir, gdir.name
        self.meta = json.loads((gdir / "meta.json").read_text()) if (gdir / "meta.json").exists() else {}
        self.cfg, self.g = cfg, cfg.game
        self.god_branch = self.meta.get("god", cfg.god.kind) != "none"
        self.prompts = prompts
        self.max_per_check = max_per_check
        self.out: list[Violation] = []
        self._count: collections.Counter[str] = collections.Counter()
        self.events = list(_jsonl(gdir / "events.jsonl"))
        self.snaps = {s["round"]: s["state"] for s in _jsonl(gdir / "snapshots.jsonl")}
        self.turn_obs: dict[tuple[str, int], dict[str, Any]] = {}
        self.mem_at_turn: dict[tuple[str, int], dict[tuple[str, int], str]] = {}
        self.held_ever: dict[str, set[tuple[int, int]]] = collections.defaultdict(set)
        self.overwrites: list[dict[str, Any]] = []
        self.key_of: dict[int, int] = {}
        self.own_lines: dict[tuple[str, int], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        # winner_policy=sit_out: agent -> (win round, locked score) once it has won and left
        self.sit_out = self.g.winner_policy == "sit_out"
        # game-fixes: lock_and_stay winners stay with a locked score (self.locked: agent -> score)
        self.lock = self.g.winner_policy == "lock_and_stay"
        self.locked: dict[str, float] = {}
        self.left: dict[str, tuple[int, float]] = {}
        self.left_notices: collections.Counter[tuple[str, str]] = collections.Counter()   # (receiver, text)
        # game.network (from game_start; all None = no network): contacts per agent, and in mode
        # known_ids the replayed known names (start = contacts, + names seen in each observation)
        self.contacts: dict[str, set[str]] | None = None
        self.known: dict[str, set[str]] | None = None
        # game.network.connect: the graph is replayed from edge_added events (self.contacts is the
        # CURRENT graph); contacts_obs = the graph at the round's start (simultaneous observations);
        # connect_skip[a] = last round a pays for its connects; ok connects awaiting their edge_added;
        # per-turn whole-turn bookkeeping (agent -> [slot, index of the turn-taking action, reason, n])
        self.contacts_obs: dict[str, set[str]] | None = None
        self.connect_skip: dict[str, int] = {}
        self.connects: collections.Counter[tuple[str, str, int]] = collections.Counter()
        self.edges_added: collections.Counter[tuple[str, str, int]] = collections.Counter()
        self.connected_rounds: dict[tuple[str, str], int] = {}
        self.turn_whole: dict[str, list[Any]] = {}
        # wrong_defect_penalty lose_held_key: (actor, slot) -> (copy, info, to, replay before the move)
        self._penalty_moves: dict[tuple[str, int], tuple[str, int, str, Replay]] = {}

    # ------------------------------------------------------------------ helpers
    def bad(self, check: str, detail: str) -> None:
        self._count[check] += 1
        if self._count[check] <= self.max_per_check:
            self.out.append(Violation(self.name, check, detail))

    def _shown(self, m: dict[str, Any]) -> int:
        """Slot from which agents may see message m (game.board.visible_delay_rounds delays board posts)."""
        k = getattr(self.g.board, "visible_delay_rounds", 0)
        return m["turn_index"] + k * self.g.n_agents if (m["channel"] == "board" and k) else m["turn_index"]

    def _order(self, r: int, start_order: list[str]) -> list[str]:
        if self.g.turn_order != "reshuffle":
            return list(start_order)
        names = sorted(start_order)
        make_rng(int(self.meta["seed"]), self.events[0]["game_id"], "order", r).shuffle(names)
        return names

    # ------------------------------------------------------------------ main replay
    def run(self) -> list[Violation]:
        if not self.events:
            self.bad("order", "no events")
            return self.out
        ev0 = self.events[0]
        if ev0["type"] != "game_start":
            self.bad("order", f"first event is {ev0['type']}")
            return self.out
        start = ev0["data"]
        agents = list(start["order"])
        n = len(agents)
        rp = Replay({a: {k["copy_id"]: (k["info_id"], k["key"]) for k in s["keys"]} for a, s in start["agents"].items()},
                    {a: list(s["needs"]) for a, s in start["agents"].items()},
                    {a: list(s.get("burned") or []) for a, s in start["agents"].items()},
                    {a: list(s.get("lost") or []) for a, s in start["agents"].items()},
                    {a: s.get("won_round") for a, s in start["agents"].items()}, lock=self.lock,
                    win_all=self.g.defect.win_rule == "all_needs",
                    penalty={a: s["score_penalty"] for a, s in start["agents"].items() if s.get("score_penalty")})
        key_of = {i: k for s in rp.keys.values() for i, k in s.values()}
        self.key_of = key_of
        self._solvable_start(rp)
        self._network_start(start["agents"])
        all_copies = {c for s in rp.keys.values() for c in s}
        prev_obs = {a: -1 for a in agents}
        msgs: list[dict[str, Any]] = []
        gives: list[tuple[int, str, str, int, int]] = []          # (stamp, actor, to, info, round)
        burns: list[dict[str, Any]] = []
        mem: dict[tuple[str, str, int], str] = {}
        pending_mint: dict[tuple[str, int, int], int] = collections.Counter()
        last_seq, last_round, max_t = -1, 0, -10**9
        round_start: Replay | None = None
        cur_round = 0
        turns_in_round: dict[int, list[str]] = collections.defaultdict(list)
        god_powers: list[dict[str, Any]] = []
        forged = 0
        overwrites_n = 0
        k_budget = self.cfg.god.budget.k
        spent = 0

        for ev in self.events:
            t, d, typ, r = ev.get("turn_index"), ev["data"], ev["type"], ev["round"]
            if ev["seq"] != last_seq + 1:
                self.bad("order", f"seq {ev['seq']} after {last_seq}")
            last_seq = ev["seq"]
            if r < last_round:
                self.bad("order", f"round {r} after {last_round} (seq {ev['seq']})")
            last_round = max(last_round, r)
            if t is not None:
                if ev["phase"] == "god":
                    if t < max_t - 1:
                        self.bad("order", f"god event {typ} at t={t} after t={max_t}")
                else:
                    if t < max_t:
                        self.bad("order", f"{typ} at t={t} after t={max_t} (seq {ev['seq']})")
                    max_t = max(max_t, t)
            if r != cur_round and typ != "game_start":
                cur_round = r
                round_start = rp.copy()
                if self.contacts is not None:
                    self.contacts_obs = {a: set(nb) for a, nb in self.contacts.items()}
            if ev["actor"] == "GOD" and not self.god_branch:
                self.bad("god", f"God event {typ} in a no-God branch (r{r})")

            if typ == "turn":
                self._left_turn(d["agent"], r)
                self._connect_turn(d, r, t)
                self._turn(ev, rp, round_start, prev_obs, msgs, gives, burns, mem, n)
                turns_in_round[r].append(d["agent"])
                prev_obs[d["agent"]] = d["obs"]["turn_index"]
            elif typ == "action_result":
                a, act = d["agent"], d["action"]
                self._own_line(a, r, d)
                self._network_action(a, r, d)
                self._connect_result(a, r, t, d)
                if a in self.left and d["ok"] and act["kind"] != "grep_log":
                    self.bad("left", f"r{r} {a} left in r{self.left[a][0]} but its {act['kind']} succeeded")
                self._fixes_result(a, r, d, rp)
                if act["kind"] == "give_key":
                    has = act["info_id"] in rp.held(a)
                    if d["ok"] and not has:
                        self.bad("keys", f"r{r} {a} give_key info {act['info_id']} ok but it held none")
                    if not d["ok"] and d.get("reason") == "no_such_key" and has:
                        self.bad("keys", f"r{r} {a} give_key info {act['info_id']} failed no_such_key but it held one")
            elif typ == "key_transfer":
                if d["cause"] in ("give", "trade") and not self._may_dm(d["from_agent"], d["to_agent"]):
                    self.bad("network", f"r{r} {d['cause']} {d['from_agent']}->{d['to_agent']} is not along a graph edge")
                for side in (d["from_agent"], d["to_agent"]):
                    if side in self.left:
                        self.bad("left", f"r{r} key_transfer {d['cause']} of info {d['info_id']} touches "
                                         f"{side}, who left in r{self.left[side][0]}")
                if d["cause"] == "defect_penalty":       # lose_held_key: checked at its burn event
                    self._penalty_moves[(d["from_agent"], t)] = (d["copy_id"], d["info_id"], d["to_agent"], rp.copy())
                self._transfer(ev, rp, all_copies, pending_mint, gives)
            elif typ == "burn":
                self._burn(ev, rp, pending_mint, n)
                burns.append({**d, "stamp": t, "round": r})
            elif typ == "message":
                m = d["message"]
                msgs.append(m)
                if m["channel"] == "dm" and not self._may_dm(m["sender_shown"], m["receiver"]):
                    self.bad("network", f"r{r} DM {m['msg_id']} {m['sender_shown']}->{m['receiver']} is not "
                                        f"along a graph edge (forged={bool(m.get('forged'))})")
                if m["sender_shown"] in self.left or m.get("receiver") in self.left:   # forged ones too
                    self.bad("left", f"r{r} message {m['msg_id']} {m['sender_shown']}->{m.get('receiver')} "
                                     "involves an agent that left")
                if m["channel"] == "board" and not getattr(self.g.board, "enabled", True):
                    self.bad("obs", f"board post {m['msg_id']} although game.board.enabled=false")
                if m.get("forged"):
                    forged += 1
                    if not self.god_branch:
                        self.bad("god", f"forged message {m['msg_id']} in a no-God branch")
            elif typ in ("memory_write", "memory_overwrite"):
                if typ == "memory_write":
                    e = d["entry"]
                    mem[(e["agent"], e["kind"], e["round"])] = e["text"]
                else:
                    overwrites_n += 1
                    mem[(d["agent"], d["kind"], d["round"])] = d["new_text"]
                    self.overwrites.append({**d, "at_round": r})
            elif typ == "god_power":
                god_powers.append({**d, "round": r, "turn_index": t})
                if d["power"] == "reveal_needs" and d["ok"]:
                    self._reveal_truth(r, d["action"], rp)
                if d["ok"]:
                    spent += d["cost"]
                    if self.cfg.god.budget.scope == "game" and d["budget_left"] != k_budget - spent:
                        self.bad("god", f"r{r} budget_left {d['budget_left']} != K {k_budget} - spent {spent}")
            elif typ == "edge_added":
                self._edge_added(r, t, d)
            elif typ == "win":
                a = d["agent"]
                if not rp.winning(a):
                    self.bad("scores", f"r{r} win event for {a} that does not hold all open needs")
                if rp.won.get(a) is not None:
                    self.bad("scores", f"r{r} second win event for {a}")
                rp.won[a] = r
                if self.sit_out:
                    self.left[a] = (r, rp.score(a))
                if self.lock:
                    self.locked[a] = rp.score(a)
            elif typ == "notice":
                nt = d["notice"]
                if nt["kind"] == "left":
                    self.left_notices[(nt["agent"], nt["text"])] += 1
            elif typ == "round_end":
                self._round_end(ev, rp, agents, turns_in_round, n)
            elif typ == "game_end":
                self._game_end(d["outcome"], rp)
        self._god_checks(god_powers, forged, overwrites_n)
        self._left_checks(agents)
        if self.connects != self.edges_added:
            self.bad("network", f"ok connects {sorted((self.connects - self.edges_added).elements())[:4]} without an "
                                f"edge_added, edge_added {sorted((self.edges_added - self.connects).elements())[:4]} "
                                "without an ok connect")
        return self.out

    # ------------------------------------------------------------------ game.network.connect
    def _connect_turn(self, d: dict[str, Any], r: int, t: int) -> None:
        """No turn while paying for a connect; remember which action of the turn takes the whole turn."""
        a = d["agent"]
        if a in self.connect_skip and r <= self.connect_skip[a]:
            self.bad("network", f"r{r} {a} played a turn while paying for a connect (no turns until after "
                                f"r{self.connect_skip[a]})")
        if not self.g.network.connect.enabled:
            self.turn_whole.pop(a, None)
            return
        kinds = [x["kind"] for x in d["actions"]]
        post = (kinds.index("post_board") if self.g.board.post_costs_turn and "post_board" in kinds else None)
        conn = kinds.index("connect") if "connect" in kinds else None
        if conn is not None and (post is None or conn < post):
            self.turn_whole[a] = [t, conn, "connect_uses_whole_turn", 0]
        else:
            self.turn_whole.pop(a, None)

    def _connect_result(self, a: str, r: int, t: int, d: dict[str, Any]) -> None:
        act, ok, reason = d["action"], d["ok"], d.get("reason")
        w = self.turn_whole.get(a)
        if w is not None and w[0] == t:
            i = w[3]
            w[3] += 1
            if i != w[1] and act["kind"] not in ("grep_log", "noop") and (ok or reason != w[2]):
                self.bad("network", f"r{r} {a} {act['kind']} in a connect turn: ok={ok} reason={reason}, "
                                    f"want failed {w[2]}")
        if act["kind"] != "connect":
            return
        tgt = act.get("target")
        if ok:
            self.connects[(a, tgt, t)] += 1
            if not self.g.network.connect.enabled or self.contacts is None:
                self.bad("network", f"r{r} {a} connect succeeded without game.network.connect")
        elif reason == "already_connected" and (self.contacts is None or tgt not in self.contacts.get(a, set())):
            self.bad("network", f"r{r} {a} connect to {tgt} refused already_connected but they are not connected")
        elif reason == "unknown_contact" and self.known is not None and tgt in self.known.get(a, set()):
            self.bad("network", f"r{r} {a} connect to {tgt} refused unknown_contact but {tgt} is known to it")

    def _edge_added(self, r: int, t: int, d: dict[str, Any]) -> None:
        """Edges change only here (an ok connect); the cost is charged; mutual = the reverse connect
        happened earlier in this round."""
        a, b, mutual = d["actor"], d["target"], d.get("mutual", False)
        self.edges_added[(a, b, t)] += 1
        if self.contacts is None or b not in self.contacts:
            self.bad("network", f"r{r} edge_added {a}-{b} without a network / unknown agent")
            return
        if b in self.left or a in self.left:
            self.bad("network", f"r{r} edge_added {a}-{b} touches an agent that left")
        exists = b in self.contacts[a]
        if mutual != exists:
            self.bad("network", f"r{r} edge_added {a}-{b} mutual={mutual} but the edge "
                                f"{'already existed' if exists else 'is new'}")
        if mutual and self.connected_rounds.get((b, a)) != r:
            self.bad("network", f"r{r} edge_added {a}-{b} mutual without {b}'s connect to {a} this round")
        self.connected_rounds[(a, b)] = r
        self.contacts[a].add(b)
        self.contacts[b].add(a)
        k = self.g.network.connect.cost_rounds
        if k > 0:
            if d["skip_until_round"] < r + k:
                self.bad("network", f"r{r} {a} connect cost: skip_until_round {d['skip_until_round']} < r{r + k}")
            self.connect_skip[a] = max(self.connect_skip.get(a, 0), r + k)

    # ------------------------------------------------------------------ winners leave (sit_out)
    def _left_turn(self, a: str, r: int) -> None:
        """sit_out: a departed winner plays no turn; only in simultaneous mode may a turn chosen at the
        round's start by an agent that won during this round's resolution still be logged (its
        actions then fail actor_left, checked on action_result)."""
        if a not in self.left:
            return
        won_r = self.left[a][0]
        if self.g.turn_mode == "sequential" or r > won_r:
            self.bad("left", f"r{r} {a} played a turn after winning and leaving in r{won_r}")

    def _left_locked(self, where: str, a: str, score: float) -> None:
        if a in self.left and abs(score - self.left[a][1]) > 1e-9:
            self.bad("left", f"{where} {a} score {score} != its locked score {self.left[a][1]} at leaving")
        if a in self.locked and abs(score - self.locked[a]) > 1e-9:
            self.bad("fixes", f"{where} {a} score {score} != its score {self.locked[a]} locked at its win")

    # ------------------------------------------------------------------ game-fixes (CONTRACTS §9f)
    def _fixes_result(self, a: str, r: int, d: dict[str, Any], rp: Replay) -> None:
        """lock_and_stay: a locked winner's defect never succeeds; actor_won only for one."""
        act, ok, reason = d["action"], d["ok"], d.get("reason")
        if act["kind"] == "defect":
            if ok and self.lock and rp.won.get(a) is not None:
                self.bad("fixes", f"r{r} {a} defected after winning although its score is locked")
            if reason == "actor_won" and not (self.lock and rp.won.get(a) is not None):
                self.bad("fixes", f"r{r} {a} defect refused actor_won but it is not a locked winner")

    def _solvable_start(self, rp: Replay) -> None:
        """Every game solvable (Iraj's hard rule): at game_start every need has a copy held by another
        agent and all needs can be met at once (max matching over distinct copies)."""
        from swarm.game.world import max_matching

        self.start_copies = collections.Counter(i for k in rp.keys.values() for i, _ in k.values())
        demands = [(a, i) for a in rp.needs for i in rp.open_needs(a)]
        copies = [(a, i) for a, k in rp.keys.items() for i, _ in k.values()]
        for a, i in demands:
            if not any(h != a and ci == i for h, ci in copies):
                self.bad("solvable", f"{a}'s need {i} has no copy held by another agent at the start")
        if max_matching(demands, copies) != len(demands):
            self.bad("solvable", "the world is not feasible at the start (needs cannot all be met at once)")

    def _solvable_round(self, r: int, rp: Replay) -> None:
        """No rule removes keys from play: no info ever has fewer copies than at the start, and outside
        sit_out no copy is held by an agent that left."""
        cur = collections.Counter(i for k in rp.keys.values() for i, _ in k.values())
        short = sorted(i for i, c in self.start_copies.items() if cur[i] < c)
        if short:
            self.bad("solvable", f"r{r} infos with fewer copies than at the start: {short[:6]}")
        if not self.sit_out and self.left:
            self.bad("solvable", f"r{r} agents left the game outside sit_out: {sorted(self.left)}")

    def _gone_at_obs(self, r: int, st: Replay) -> set[str]:
        """Agents that had left when an observation of round r was built (sit_out: the winners)."""
        return {x for x, w in st.won.items() if w is not None} if self.sit_out else set()

    def _team_scores(self, out: dict[str, Any], rp: Replay) -> None:
        """game.score.team_share: shared = raw + a * mean raw of all agents; absent when off."""
        a_, per = self.g.score.team_share, out["per_agent"]
        if not a_:
            if any("shared" in o for o in per.values()) or "welfare_shared" in out:
                self.bad("fixes", "outcome has team-score fields although game.score.team_share is 0")
            return
        mean = sum(rp.score(a) for a in per) / max(len(per), 1)
        for a, o in per.items():
            want = rp.score(a) + a_ * mean
            if o.get("shared") is None or abs(o["shared"] - want) > 1e-9:
                self.bad("fixes", f"outcome shared of {a} = {o.get('shared')} != replay {want}")
        want_w = sum(rp.score(a) for a in per) / max(len(per), 1) * (1 + a_)
        if out.get("welfare_shared") is None or abs(out["welfare_shared"] - want_w) > 1e-9:
            self.bad("fixes", f"outcome welfare_shared = {out.get('welfare_shared')} != {want_w}")

    def _community_scores(self, out: dict[str, Any], rp: Replay) -> None:
        """game.score.community_bonus (showcase): B iff >= threshold of the agent's planted community
        (from the world seed in meta.json, recomputed here) won (replayed won_round); absent when off."""
        per = out["per_agent"]
        if not self.g.score.community_bonus:
            if any("community_bonus" in o for o in per.values()) or "welfare_community" in out:
                self.bad("fixes", "outcome has community fields although game.score.community_bonus is 0")
            return
        from swarm.game.network import make_graph_full
        from swarm.game.scoring import community_scores

        comm = make_graph_full(self.g.network, list(per), int(self.meta["seed"]))[1] or {}
        if (self.meta.get("network") or {}).get("communities") not in (None, comm):
            self.bad("fixes", "meta.json communities != the world seed's planted communities")
        from swarm.game.scoring import community_success

        live = {a: len(set(rp.open_needs(a)) & rp.held(a)) for a in per}
        want = community_scores(community_success({a: rp.won.get(a) is not None for a in per}, live, self.g),
                                comm, self.g)
        for a, o in per.items():
            if o.get("community_bonus") is None or abs(o["community_bonus"] - want[a]) > 1e-9:
                self.bad("fixes", f"outcome community_bonus of {a} = {o.get('community_bonus')} != replay {want[a]}")
        want_w = sum(rp.score(a) + want[a] for a in per) / max(len(per), 1)
        if out.get("welfare_community") is None or abs(out["welfare_community"] - want_w) > 1e-9:
            self.bad("fixes", f"outcome welfare_community = {out.get('welfare_community')} != {want_w}")

    def _delivery_credit(self, out: dict[str, Any]) -> None:
        """game.score.delivery_credit (showcase): per-agent credit and the total == a replay of
        events.jsonl through scoring.DeliveryLedger; every fill pays <= d distinct earlier holders."""
        per = out["per_agent"]
        if not self.g.score.delivery_credit:
            if any("delivery_credit" in o for o in per.values()) or "delivery_credit_total" in out:
                self.bad("fixes", "outcome has delivery fields although game.score.delivery_credit is 0")
            return
        from types import SimpleNamespace

        from swarm.game.scoring import DeliveryLedger

        led = DeliveryLedger(self.g.score.delivery_credit)
        start = self.events[0]["data"]["agents"]
        led.start({a: SimpleNamespace(contacts=s.get("contacts"), needs=s["needs"], burned=s.get("burned", []),
                                      lost=s.get("lost", []),
                                      keys=[SimpleNamespace(**k) for k in s["keys"]]) for a, s in start.items()})
        for e in self.events[1:]:
            d = e["data"]
            led.on_event(SimpleNamespace(**{"lost_info_id": None, "correct": False, **d}), e["round"])
        for f in led.fills:
            if len(f["paid"]) > (f["d"] or 0) or len(set(f["paid"])) != len(f["paid"]) or f["needer"] in f["paid"]:
                self.bad("fixes", f"delivery fill {f['copy_id']} -> {f['needer']} pays {f['paid']} (d={f['d']})")
        for a, o in per.items():
            want = led.credit.get(a, 0.0)
            if o.get("delivery_credit") is None or abs(o["delivery_credit"] - want) > 1e-9:
                self.bad("fixes", f"outcome delivery_credit of {a} = {o.get('delivery_credit')} != replay {want}")
        if out.get("delivery_credit_total") is None or abs(out["delivery_credit_total"] - led.total()) > 1e-9:
            self.bad("fixes", f"outcome delivery_credit_total = {out.get('delivery_credit_total')} != {led.total()}")

    def _final_scores(self, out: dict[str, Any], rp: Replay) -> None:
        """game.score.held_key_bonus: key_bonus / private per agent and welfare_private (game end, all
        agents); absent when off. Welfare itself must stay the needs-only mean (checked in _game_end)."""
        b = self.g.score.held_key_bonus
        per = out["per_agent"]
        self._team_scores(out, rp)
        self._community_scores(out, rp)
        self._delivery_credit(out)
        if not b and self.g.wrong_defect_penalty not in ("score_penalty", "lose_held_key"):
            if any(k in o for o in per.values() for k in ("key_bonus", "private")) or "welfare_private" in out:
                self.bad("fixes", "outcome has held-key fields although game.score.held_key_bonus is 0")
            return
        start = self.events[0]["data"]["agents"]
        initial = {a: {k["info_id"] for k in s["keys"]} for a, s in start.items()}

        def nkeys(a: str) -> int:                   # game.score.held_key_scope (showcase)
            if self.g.score.held_key_scope == "initial":
                return len(rp.held(a) & initial[a])
            return len(rp.keys[a])
        private = {a: rp.shown(a) + b * nkeys(a) for a in per}     # minus any score penalty
        for a, o in per.items():
            for k, want in (("key_bonus", b * nkeys(a)), ("private", private[a])):
                if o.get(k) is None or abs(o[k] - want) > 1e-9:
                    self.bad("fixes", f"outcome {k} of {a} = {o.get(k)} != replay {want}")
        want_w = sum(private.values()) / max(len(per), 1)
        if out.get("welfare_private") is None or abs(out["welfare_private"] - want_w) > 1e-9:
            self.bad("fixes", f"outcome welfare_private = {out.get('welfare_private')} != {want_w}")

    def _left_checks(self, agents: list[str]) -> None:
        """Every other agent was told about each departure exactly once."""
        from swarm.game.rules import left_text

        for a in self.left:
            for o in agents:
                if o != a and self.left_notices[(o, left_text(a))] != 1:
                    self.bad("left", f"{o} got {self.left_notices[(o, left_text(a))]} 'left' notices for {a}")

    # ------------------------------------------------------------------ game.network
    LEARN_KINDS = ("gift_received", "burned", "defect_attempt", "connected")

    def _network_start(self, agents: dict[str, Any]) -> None:
        """Read the graph (and known names) off game_start; leaves both None without a network."""
        if any(s.get("contacts") is not None for s in agents.values()):
            self.contacts = {a: set(s.get("contacts") or []) for a, s in agents.items()}
            for a, nb in self.contacts.items():
                for b in nb:
                    if a not in self.contacts.get(b, set()):
                        self.bad("network", f"edge {a}-{b} is not symmetric")
            if not self.g.network.on:
                self.bad("network", "game_start has contacts although game.network.topology=complete")
        if any(s.get("known") is not None for s in agents.values()):
            self.known = {a: set(s.get("known") or []) for a, s in agents.items()}
            if self.contacts is None or self.known != self.contacts:
                self.bad("network", "known names at the start != the graph neighbours")
        if self.g.network.on and self.contacts is None:
            self.bad("network", "game.network is on but game_start has no contacts")

    def _may_dm(self, a: str | None, b: str | None) -> bool:
        """Mode contacts: may a and b exchange a DM / key? (always True otherwise)."""
        if self.contacts is None or self.known is not None or a is None or b is None:
            return True
        return b in self.contacts.get(a, set())

    def _sees_post(self, viewer: str, sender: str) -> bool:
        if getattr(self.g.board, "scope", "global") != "neighbors":
            return True
        return self.contacts is not None and sender in self.contacts[viewer]

    def _line_local(self, viewer: str, actor: str | None, target: str | None, known: set[str] | None) -> bool:
        """contacts: some named party is the viewer or a contact; known_ids: every named party is known."""
        if self.contacts is None:
            return True
        named = [x for x in (actor, target) if x is not None]
        if self.known is not None:
            return all(x == viewer or x in (known or set()) for x in named)
        return any(x == viewer or x in self.contacts[viewer] for x in named)

    def _names_seen(self, a: str, obs: dict[str, Any], names: list[str]) -> set[str]:
        """Independent re-statement of the learning rule: DM / board senders and names in their text,
        names in gift_received / burned / defect_attempt notices."""
        def in_text(text: str) -> set[str]:
            return {n for n in names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text)}
        out: set[str] = set()
        for m in obs.get("inbox", []) + obs.get("board", []):
            if m["sender"] in names:
                out.add(m["sender"])
            out |= in_text(m["text"])
        for nt in obs.get("notices", []):
            if nt["kind"] in self.LEARN_KINDS:
                out |= in_text(nt["text"])
        out.discard(a)
        return out

    def _network_action(self, a: str, r: int, d: dict[str, Any]) -> None:
        """ok peer actions only on addressable agents; network refusals only on non-addressable ones."""
        if self.contacts is None:
            return
        act = d["action"]
        other = act.get("to") if act["kind"] in ("send_message", "give_key") else act.get("target")
        if act["kind"] not in ("send_message", "give_key", "defect") or other not in self.contacts:
            return
        allowed = other in (self.known[a] if self.known is not None else self.contacts[a])
        reason = d.get("reason")
        if d["ok"] and not allowed:
            self.bad("network", f"r{r} {a} {act['kind']} on {other} succeeded but {other} is not "
                                f"{'known to it' if self.known is not None else 'a contact'}")
        if reason in ("not_connected", "unknown_contact") and allowed:
            self.bad("network", f"r{r} {a} {act['kind']} on {other} refused ({reason}) but it was allowed")
        want = "unknown_contact" if self.known is not None else "not_connected"
        if reason in ("not_connected", "unknown_contact") and reason != want:
            self.bad("network", f"r{r} {a} refused with {reason}, mode gives {want}")

    # ------------------------------------------------------------------ pieces
    def _own_line(self, a: str, r: int, d: dict[str, Any]) -> None:
        """The ledger line an own action renders as (context.base.own_action_items + turn.j2)."""
        from swarm.schema import ACTION_ADAPTER

        try:
            act = ACTION_ADAPTER.validate_python(d["action"])
        except Exception:
            return
        text = render_action(act) + ("" if d["ok"] else f" (failed: {d.get('reason') or 'invalid'})")
        self.own_lines[(a, r)][f"- (r{r}) {text}\n"] += 1

    def _transfer(self, ev: dict[str, Any], rp: Replay, all_copies: set[str],
                  pending_mint: dict[tuple[str, int, int], int], gives: list[tuple[int, str, str, int, int]]) -> None:
        d, r = ev["data"], ev["round"]
        cid, info, src, dst, cause = d["copy_id"], d["info_id"], d["from_agent"], d["to_agent"], d["cause"]
        if cause in ("give", "defect_steal", "defect_penalty", "trade"):     # trade: game.trade (one transfer per copy)
            if src is None or cid not in rp.keys.get(src, {}):
                self.bad("keys", f"r{r} {cause} of copy {cid} (info {info}) from {src}, who does not hold it")
                return
            if rp.keys[src][cid][0] != info:
                self.bad("keys", f"r{r} copy {cid} is info {rp.keys[src][cid][0]}, transfer says {info}")
            rp.keys[dst][cid] = rp.keys[src].pop(cid)
            if cause in ("give", "trade"):                 # a trade's copies show as gives in the view_log
                gives.append((ev["turn_index"], src, dst, info, r))
        elif cause == "defect_mint":
            if self.g.defect_gain != "mint":
                self.bad("keys", f"r{r} defect_mint with defect_gain={self.g.defect_gain}")
            if cid in all_copies:
                self.bad("keys", f"r{r} minted copy id {cid} already exists")
            all_copies.add(cid)
            key = next((k for s in rp.keys.values() for i, k in s.values() if i == info), None)
            rp.keys[dst][cid] = (info, key if key is not None else -1)
            pending_mint[(dst, info, ev["turn_index"])] += 1
        else:
            self.bad("keys", f"r{r} unknown transfer cause {cause}")
        self.held_ever[dst].add(rp.keys[dst][cid])

    def _burn(self, ev: dict[str, Any], rp: Replay, pending_mint: dict[tuple[str, int, int], int], n: int) -> None:
        d, r, t = ev["data"], ev["round"], ev["turn_index"]
        a, tgt, info = d["actor"], d["target"], d["info_id"]
        if rp.won.get(tgt) is not None:
            self.bad("defects", f"r{r} {a} burned {tgt}, who had won in r{rp.won[tgt]}")
        should = info in rp.open_needs(tgt)
        if d["correct"] != should:
            self.bad("defects", f"r{r} burn {a}->{tgt} info {info} correct={d['correct']}, replay says {should}")
        mode = self.g.view_log.defect_visibility
        want_rel = (defect_release_turn(t, self.g.view_log.defect_delay_rounds, n)
                    if mode in ("delayed", "victim_then_delayed") else None)
        if d.get("release_turn") != want_rel:
            self.bad("defects", f"r{r} burn release_turn {d.get('release_turn')} != {want_rel}")
        if d["correct"]:
            rp.burned[tgt].append(info)
            if self.g.defect_gain == "mint":
                if pending_mint.get((a, info, t), 0) < 1:
                    self.bad("keys", f"r{r} correct burn by {a} on info {info} without its minted copy")
                else:
                    pending_mint[(a, info, t)] -= 1
        else:
            if pending_mint.get((a, info, t), 0) > 0:
                self.bad("keys", f"r{r} key minted for a WRONG burn by {a} on info {info}")
            if self.g.wrong_defect_penalty == "lose_need":
                on = rp.open_needs(a)
                want = on[0] if on else None
                if d.get("lost_info_id") != want:
                    self.bad("defects", f"r{r} {a} lost need {d.get('lost_info_id')}, first open need was {want}")
                if want is not None:
                    rp.lost[a].append(want)
            elif self.g.wrong_defect_penalty in ("score_penalty", "lose_held_key"):
                self._score_penalty_burn(ev, rp, a, tgt)
            elif self.g.wrong_defect_penalty == "lose_held_need":
                mine = [i for i in rp.open_needs(a) if i in rp.held(a)]
                on = rp.open_needs(a)
                want = mine[0] if mine else (on[0] if on else None)
                want_pen = "lose_held_need" if mine else "lose_need"
                if d.get("lost_info_id") != want or d.get("penalty") != want_pen:
                    self.bad("defects", f"r{r} {a} wrong defect: penalty {d.get('penalty')} lost "
                                        f"{d.get('lost_info_id')}, want {want_pen} {want} (held open needs {mine})")
                if want is not None:
                    rp.lost[a].append(want)

    def _score_penalty_burn(self, ev: dict[str, Any], rp: Replay, a: str, tgt: str) -> None:
        """score_penalty / lose_held_key: no need is lost; the spare that moved (a key_transfer
        defect_penalty just before this burn, already replayed) is the most-demanded one the actor held,
        else the score dropped by game.defect.score_penalty (default one need)."""
        d, r = ev["data"], ev["round"]
        if d.get("lost_info_id") is not None:
            self.bad("defects", f"r{r} {a} wrong defect under {self.g.wrong_defect_penalty} lost need {d['lost_info_id']}")
        moved = self._penalty_moves.pop((a, ev["turn_index"]), None)
        if self.g.wrong_defect_penalty == "lose_held_key" and d.get("penalty") == "lose_held_key":
            if moved is None:
                self.bad("defects", f"r{r} {a} lose_held_key without its defect_penalty transfer")
                return
            cid, info, to, before = moved
            if to != tgt:
                self.bad("defects", f"r{r} {a} penalty key went to {to}, not the target {tgt}")
            mine = set(before.open_needs(a))
            spares = [(i, c) for c, (i, _) in before.keys[a].items() if i not in mine]
            def dem(i: int) -> int:
                return sum(1 for b in before.needs if b != a and before.won.get(b) is None
                           and i in before.open_needs(b) and i not in before.held(b))
            want = min(spares, key=lambda x: (-dem(x[0]), x[0], x[1]), default=None)
            if want is None or want[1] != cid:
                self.bad("defects", f"r{r} {a} penalty key {cid} (info {info}) is not its most-demanded spare {want}")
            return
        if moved is not None:
            self.bad("defects", f"r{r} {a} a key moved but the penalty is {d.get('penalty')}")
        if d.get("penalty") != "score_penalty":
            self.bad("defects", f"r{r} {a} wrong defect penalty {d.get('penalty')}, want score_penalty")
        if self.g.wrong_defect_penalty == "lose_held_key":
            mine = set(rp.open_needs(a))
            if any(i not in mine for i, _ in rp.keys[a].values()):
                self.bad("defects", f"r{r} {a} score_penalty fallback although it holds a spare")
        amt = self.g.defect.score_penalty if self.g.defect.score_penalty is not None else 1 / max(len(rp.needs[a]), 1)
        rp.penalty[a] = rp.penalty.get(a, 0.0) + amt

    def _turn(self, ev: dict[str, Any], rp: Replay, round_start: Replay | None, prev_obs: dict[str, int],
              msgs: list[dict[str, Any]], gives: list[tuple[int, str, str, int, int]],
              burns: list[dict[str, Any]], mem: dict[tuple[str, str, int], str], n: int) -> None:
        """Simultaneous mode: every agent observed at s0, before this round's connects, so the obs
        checks see the round-start graph (contacts_obs); sequential: the current one."""
        live = self.contacts
        if self.g.turn_mode == "simultaneous" and self.contacts_obs is not None:
            self.contacts = self.contacts_obs
        try:
            self._turn_checks(ev, rp, round_start, prev_obs, msgs, gives, burns, mem, n)
        finally:
            self.contacts = live

    def _turn_checks(self, ev: dict[str, Any], rp: Replay, round_start: Replay | None, prev_obs: dict[str, int],
                     msgs: list[dict[str, Any]], gives: list[tuple[int, str, str, int, int]],
                     burns: list[dict[str, Any]], mem: dict[tuple[str, str, int], str], n: int) -> None:
        d, r = ev["data"], ev["round"]
        a, obs = d["agent"], d["obs"]
        t_obs, t_slot = obs["turn_index"], ev["turn_index"]
        self.turn_obs[(a, r)] = obs
        self.mem_at_turn[(a, r)] = {(k, rr): txt for (ag, k, rr), txt in mem.items() if ag == a}
        # slot / timing
        order = self._order(r, list(self.events[0]["data"]["order"]))
        want_slot = turn_index(r, order.index(a), n)
        if t_slot != want_slot:
            self.bad("order", f"r{r} {a} turn at slot {t_slot}, its order position gives {want_slot}")
        want_obs = turn_index(r, 0, n) if self.g.turn_mode == "simultaneous" else want_slot
        if t_obs != want_obs:
            self.bad("order", f"r{r} {a} observed at {t_obs}, expected {want_obs}")
        snap = self.snaps.get(r - 1)
        if snap is not None and r <= snap["agents"][a]["delayed_until_round"]:
            self.bad("delays", f"r{r} {a} played while delayed until r{snap['agents'][a]['delayed_until_round']}")
        # pinned facts vs replay at observation time
        st = round_start if (self.g.turn_mode == "simultaneous" and round_start is not None) else rp
        p = obs["pinned"]
        hold = sorted((k["info_id"], k["key"]) for k in p["holdings"])
        if hold != st.holdings(a):
            self.bad("obs", f"r{r} {a} pinned holdings {hold} != replay {st.holdings(a)}")
        for x in hold:
            self.held_ever[a].add(tuple(x))
        if sorted(p["needs_open"]) != sorted(st.open_needs(a)):
            self.bad("obs", f"r{r} {a} needs_open {p['needs_open']} != replay {st.open_needs(a)}")
        if sorted(p["needs_burned"]) != sorted(st.burned[a]):
            self.bad("obs", f"r{r} {a} needs_burned {p['needs_burned']} != replay {st.burned[a]}")
        if abs(p["score"] - st.shown(a)) > 1e-9:
            self.bad("obs", f"r{r} {a} pinned score {p['score']} != replay {st.shown(a)}")
        gone = self._gone_at_obs(r, st)
        known_before = set(self.known[a]) if self.known is not None else None
        if self.known is not None:                       # known_ids: learn from this observation
            self.known[a] |= self._names_seen(a, obs, list(rp.keys))
        pool = (self.known[a] if self.known is not None
                else self.contacts[a] if self.contacts is not None else set(rp.keys))
        want_peers = sorted(x for x in pool if x != a and x not in gone)
        if p.get("links") is not None or (self.known is not None and self.g.network.connect.enabled):
            want_links = sorted(x for x in (self.contacts or {}).get(a, set()) if x not in gone)
            if self.known is None or not self.g.network.connect.enabled or sorted(p.get("links") or []) != want_links:
                self.bad("network", f"r{r} {a} pinned links {p.get('links')} != current direct links {want_links}")
        if sorted(p["peers"]) != want_peers:
            self.bad("obs", f"r{r} {a} pinned peers {p['peers']} != players still in the game "
                            f"(that it may address) {want_peers}")
        # inbox / board: exactly the messages stamped prev <= s < t
        prev = prev_obs[a]
        anon = self.g.board.anonymous
        want_in = sorted(m["msg_id"] for m in msgs
                         if m["channel"] == "dm" and m["receiver"] == a and prev <= m["turn_index"] < t_obs)
        got_in = sorted(m["msg_id"] for m in obs["inbox"])
        if want_in != got_in:
            self.bad("obs", f"r{r} {a} inbox {got_in} != expected {want_in}")
        want_bd = sorted(m["msg_id"] for m in msgs
                         if m["channel"] == "board" and m["sender_shown"] != a and prev <= self._shown(m) < t_obs
                         and self._sees_post(a, m["sender_shown"]))
        got_bd = sorted(m["msg_id"] for m in obs["board"])
        if want_bd != got_bd:
            self.bad("obs", f"r{r} {a} board {got_bd} != expected {want_bd}")
        by_id = {m["msg_id"]: m for m in msgs}
        for m in obs["inbox"] + obs["board"]:
            truth = by_id.get(m["msg_id"])
            if truth is None:
                continue
            shown = "(anonymous)" if (truth["channel"] == "board" and anon) else truth["sender_shown"]
            if m["sender"] != shown or m["text"] != truth["text"]:
                self.bad("obs", f"r{r} {a} message {m['msg_id']} rendered as {m['sender']!r}, truth {shown!r}")
        self._log_lines(r, a, obs, prev, t_obs, msgs, gives, burns, known_before)

    def _log_lines(self, r: int, a: str, obs: dict[str, Any], prev: int, t: int, msgs: list[dict[str, Any]],
                   gives: list[tuple[int, str, str, int, int]], burns: list[dict[str, Any]],
                   known_before: set[str] | None = None) -> None:
        vl, w = self.g.view_log, self.g.window_turns
        if vl.content == "none":
            if obs["log"]:
                self.bad("obs", f"r{r} {a} has log lines with view_log.content=none")
            return
        hid = vl.actor_identity == "hidden"
        dhid = (vl.defect_actor_identity or vl.actor_identity) == "hidden"
        want: list[tuple[Any, ...]] = []
        def local(actor: str | None, target: str | None) -> bool:     # game.network: local lines only
            return self._line_local(a, actor, target, known_before)

        for s, src, dst, info, rr in gives:
            if t - w <= s < t and local(None if hid else src, dst):
                want.append(("give", rr, None if hid else src, dst, info, None))
        for m in msgs:
            if t - w <= self._shown(m) < t:
                sender = None if hid else m["sender_shown"]
                if m["channel"] == "board":
                    if self._sees_post(a, m["sender_shown"]):
                        want.append(("board_post", m["round"], None if self.g.board.anonymous else sender,
                                     None, None, None))
                elif not local(sender, m["receiver"]):
                    continue
                elif vl.content == "actions_msg_meta":
                    want.append(("dm_meta", m["round"], sender, m["receiver"], None, None))
                elif vl.content == "full":
                    want.append(("dm", m["round"], sender, m["receiver"], None, None))
        mode = vl.defect_visibility
        for b in burns:
            if mode == "public":
                show = t - w <= b["stamp"] < t
            elif mode in ("delayed", "victim_then_delayed"):
                show = (mode == "delayed" or a != b["target"]) and prev < b["release_turn"] <= t
            else:
                show = False
            if show and local(None if dhid else b["actor"], b["target"]):
                want.append(("defect", b["round"], None if dhid else b["actor"], b["target"], b["info_id"], b["correct"]))
        got = [(ln["kind"], ln["round"], ln.get("actor"), ln.get("target"),
                ln.get("info_id") if ln["kind"] in ("give", "defect") else None,
                ln.get("ok") if ln["kind"] == "defect" else None) for ln in obs["log"]]
        if vl.anonymize_noncontacts and self.contacts is not None:    # showcase v2: non-contacts hidden
            from swarm.game.observe import OUTSIDE_CONTACTS

            keep = set(obs["pinned"]["peers"]) | {a}

            def anon(x: str | None) -> str | None:
                return x if x is None or x in keep else OUTSIDE_CONTACTS
            want = [(k, rr, anon(x), anon(y), i, ok) for k, rr, x, y, i, ok in want]
        cw, cg = collections.Counter(want), collections.Counter(got)
        if cw != cg:
            extra, missing = list((cg - cw).elements()), list((cw - cg).elements())
            self.bad("obs", f"r{r} {a} view_log mismatch: shown-but-not-allowed {extra[:4]}, missing {missing[:4]}")

    def _round_end(self, ev: dict[str, Any], rp: Replay, agents: list[str],
                   turns_in_round: dict[int, list[str]], n: int) -> None:
        d, r = ev["data"], ev["round"]
        self._solvable_round(r, rp)
        for a in agents:
            if abs(d["scores"][a] - rp.score(a)) > 1e-9:
                self.bad("scores", f"r{r} round_end score {a}={d['scores'][a]} != replay {rp.score(a)}")
            if rp.winning(a) and rp.won.get(a) is None:
                self.bad("scores", f"r{r} {a} holds all open needs but has no win event")
            self._left_locked(f"r{r} round_end", a, d["scores"][a])
        played = turns_in_round.get(r, [])
        dup = [a for a, c in collections.Counter(played).items() if c > 1]
        if dup:
            self.bad("order", f"r{r} agents with more than one turn: {dup}")
        prev = self.snaps.get(r - 1)
        if prev is not None:
            def gone(s: dict[str, Any]) -> bool:          # sit_out departure as of the previous round's end
                return self.sit_out and s["won_round"] is not None
            active = [a for a in agents if r > prev["agents"][a]["delayed_until_round"]
                      and not gone(prev["agents"][a])]
            missing = sorted(set(active) - set(played))
            if missing and not self._ended_early(r):
                self.bad("delays", f"r{r} active agents without a turn: {missing}")
        snap = self.snaps.get(r)
        if snap is None:
            return
        try:
            h = GameState.model_validate(snap).state_hash()
        except Exception as e:  # a malformed snapshot is itself a violation
            self.bad("snapshots", f"r{r} snapshot does not validate: {type(e).__name__}")
            return
        if h != d["state_hash"] and h[:16] != d["state_hash"]:
            self.bad("snapshots", f"r{r} round_end state_hash != hash of the round-{r} snapshot")
        for a in agents:
            s = snap["agents"][a]
            if sorted(k["copy_id"] for k in s["keys"]) != sorted(rp.keys[a]):
                self.bad("keys", f"r{r} {a} snapshot copies != replay")
            if sorted(s["burned"]) != sorted(rp.burned[a]) or sorted(s["lost"]) != sorted(rp.lost[a]):
                self.bad("defects", f"r{r} {a} snapshot burned/lost != replay")
            if s["won_round"] != rp.won.get(a):
                self.bad("scores", f"r{r} {a} snapshot won_round {s['won_round']} != replay {rp.won.get(a)}")
            if self.contacts is not None and set(s.get("contacts") or []) != self.contacts[a]:
                self.bad("network", f"r{r} {a} snapshot contacts {sorted(s.get('contacts') or [])} != the graph "
                                    f"replayed from game_start + edge_added {sorted(self.contacts[a])}")
            if self.known is not None and set(s.get("known") or []) != self.known[a]:
                self.bad("network", f"r{r} {a} snapshot known names {sorted(s.get('known') or [])} != replay "
                                    f"{sorted(self.known[a])}")

    def _ended_early(self, r: int) -> bool:
        return any(e["type"] == "game_end" and e["data"]["outcome"]["end_reason"] == "budget_exceeded"
                   and e["round"] == r for e in self.events[-3:])

    def _game_end(self, out: dict[str, Any], rp: Replay) -> None:
        for a, o in out["per_agent"].items():
            if abs(o["score"] - rp.score(a)) > 1e-9:
                self.bad("scores", f"outcome score {a}={o['score']} != replay {rp.score(a)}")
            self._left_locked("outcome", a, o["score"])
            if sorted(o["burned"]) != sorted(rp.burned[a]) or sorted(o["lost"]) != sorted(rp.lost[a]):
                self.bad("scores", f"outcome burned/lost of {a} != replay")
            if o["won_round"] != rp.won.get(a):
                self.bad("scores", f"outcome won_round of {a} {o['won_round']} != replay {rp.won.get(a)}")
        self._final_scores(out, rp)
        n = max(len(out["per_agent"]), 1)
        if abs(out["welfare"] - sum(rp.score(a) for a in out["per_agent"]) / n) > 1e-9:
            self.bad("scores", f"outcome welfare {out['welfare']} != mean needs score (it must exclude score terms)")

    def _reveal_truth(self, r: int, act: dict[str, Any], rp: Replay) -> None:
        """god.powers.reveal_needs: truthful names exactly the target's open needs; false names only ids
        the target does not need (and as many as its open needs, at least one)."""
        tgt, items = act["target"], act["items"]
        if act["variant"] == "truthful":
            if items != rp.open_needs(tgt):
                self.bad("god", f"r{r} truthful reveal on {tgt} named {items}, open needs {rp.open_needs(tgt)}")
        else:
            if set(items) & set(rp.needs[tgt]) or len(items) != max(1, len(rp.open_needs(tgt))):
                self.bad("god", f"r{r} false reveal on {tgt} named {items}; its needs {rp.needs[tgt]}")
        if rp.won.get(tgt) is not None or tgt in act["receivers"]:
            self.bad("god", f"r{r} reveal on {tgt} (won or among its own receivers)")

    def _god_checks(self, powers: list[dict[str, Any]], forged: int, overwrites: int) -> None:
        ok = [p for p in powers if p["ok"]]
        if not self.god_branch:
            if powers:
                self.bad("god", f"{len(powers)} god_power events in a no-God branch")
            return
        gc = self.cfg.god
        if gc.budget.scope == "game" and sum(p["cost"] for p in ok) > gc.budget.k:
            self.bad("god", f"ok powers cost {sum(p['cost'] for p in ok)} > K={gc.budget.k}")
        n = self.g.n_agents
        last_round = max((e["round"] for e in self.events), default=0)
        due = [(sp.round, sp.power, sp.after_turns) for sp in gc.schedule
               if gc.start_round <= sp.round <= last_round]
        got = [(p["round"], p["power"]) for p in powers]
        if collections.Counter((r, pw) for r, pw, _ in due) != collections.Counter(got):
            self.bad("god", f"god_power events {got} != schedule entries due {[(r, p) for r, p, _ in due]}")
        slots = {(r, pw): turn_index(r, at, n) for r, pw, at in due}
        for p in powers:
            s = slots.get((p["round"], p["power"]))
            if s is not None and p["turn_index"] not in (s, s - 1):
                self.bad("god", f"r{p['round']} {p['power']} at t={p['turn_index']}, schedule slot {s}")
        n_fraud = sum(1 for p in ok if p["power"] == "message_fraud")
        n_fraud += sum(len(p["action"]["receivers"]) for p in ok if p["power"] == "reveal_needs")
        n_cot = sum(1 for p in ok if p["power"] == "cot_control")
        if forged != n_fraud:
            self.bad("god", f"{forged} forged messages != {n_fraud} from ok message_fraud / reveal_needs powers")
        if overwrites != n_cot:
            self.bad("god", f"{overwrites} memory_overwrite events != {n_cot} ok cot_control powers")
