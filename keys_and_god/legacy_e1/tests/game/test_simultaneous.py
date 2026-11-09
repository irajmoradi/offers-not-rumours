"""Simultaneous turn mode: concurrent decisions from one start-of-round state, resolution in
turn-priority order (fixed or reshuffled per round), conflicts -> failed action + notice,
determinism under a fixed seed regardless of decision completion order."""

from __future__ import annotations

import asyncio
import random

from swarm.game.engine import play_game
from swarm.rng import canonical_json
from swarm.schema import AgentOutput, DefectAction, GiveKey

from .helpers import FOUR, Scripted, fresh_game, run_cfg

SIM = {"turn_mode": "simultaneous"}


def _results(eng, rnd):
    return [(e.actor, e.data.action.kind, e.data.ok, e.data.reason) for e in eng.sink.events
            if e.round == rnd and e.type == "action_result"]


def _notices(eng, agent):
    return [e.data.notice for e in eng.sink.events if e.type == "notice" and e.data.notice.agent == agent]


def test_key_conflict_second_in_priority_fails_with_notice():
    """ann (priority 0) steals bob's only info-3 key by a correct defect on dan; bob (priority 1)
    decided from the start-of-round state, where he still held it, so his give_key fails."""
    cfg = run_cfg(game={**SIM, "max_rounds": 2, "defect_gain": "steal"})
    dec = Scripted({(1, "ann"): [DefectAction(target="dan", info_id=3)],
                    (1, "bob"): [GiveKey(to="dan", info_id=3)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    r1 = _results(eng, 1)
    assert r1[0] == ("ann", "defect", True, None)
    assert r1[1] == ("bob", "give_key", False, "no_such_key")
    assert any(k.info_id == 3 for k in eng.state.agents["ann"].keys)
    sys_notes = [n for n in _notices(eng, "bob") if n.kind == "system" and "give_key" in n.text]
    assert len(sys_notes) == 1 and "turn order" in sys_notes[0].text
    # bob sees the notice at his next observation (round 2)
    r2_bob = [o for o, g in dec.calls if o.agent == "bob" and o.round == 2 and g is None][0]
    assert any("give_key" in n.text for n in r2_bob.notices)


def test_double_defect_conflict_second_is_wrong():
    """Both ann and bob burn dan's open need 1 in the same round: only the first (priority) is correct."""
    cfg = run_cfg(game={**SIM, "max_rounds": 1})
    dec = Scripted({(1, "ann"): [DefectAction(target="dan", info_id=1)],
                    (1, "bob"): [DefectAction(target="dan", info_id=1)]})
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    burns = [(e.actor, e.data.correct) for e in eng.sink.events if e.type == "burn"]
    assert burns == [("ann", True), ("bob", False)]


def test_priority_follows_reshuffled_order_each_round():
    cfg = run_cfg(game={**SIM, "max_rounds": 4, "turn_order": "reshuffle"})
    dec = Scripted(policy=lambda obs, g: [])
    _, eng = play_game(cfg, fresh_game(FOUR), dec)
    turns = {}
    for e in eng.sink.events:
        if e.type == "turn":
            turns.setdefault(e.round, []).append((e.turn_index, e.actor))
    orders = []
    for r, lst in sorted(turns.items()):
        assert [t for t, _ in lst] == sorted(t for t, _ in lst)       # resolved in slot order
        orders.append([a for _, a in lst])
    assert len({tuple(o) for o in orders}) > 1                          # reshuffled across rounds
    for snap in eng.snapshots[1:]:
        assert orders[snap.round - 1] == list(snap.order)               # = the round's priority order


class SlowRandom:
    """Async decide: random completion order (wall-clock jitter), deterministic choices; tracks the
    peak number of decisions in flight."""

    def __init__(self, jitter_seed: int) -> None:
        self.jit = random.Random(jitter_seed)
        self.inflight = self.peak = 0

    async def __call__(self, obs, grep):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(self.jit.random() * 0.01)
        self.inflight -= 1
        if grep is not None:
            return AgentOutput(actions=[])
        keys = [k for k in obs.pinned.holdings]
        peers = list(obs.pinned.peers)
        if keys and obs.round % 2:
            return AgentOutput(actions=[GiveKey(to=peers[obs.round % len(peers)], info_id=keys[0].info_id)])
        return AgentOutput(actions=[DefectAction(target=peers[0], info_id=obs.round)])


def _stream(eng):
    return canonical_json([[e.round, e.turn_index, e.type, e.actor, e.data.model_dump(mode="json")]
                           for e in eng.sink.events])


def test_decisions_run_concurrently_and_result_is_deterministic():
    cfg = run_cfg(game={**SIM, "max_rounds": 5, "turn_order": "reshuffle"})
    streams = []
    for jitter in (1, 2):
        d = SlowRandom(jitter)
        _, eng = play_game(cfg, fresh_game(FOUR), d)
        assert d.peak == 4                       # all four agents decided at once
        streams.append(_stream(eng))
    assert streams[0] == streams[1]              # completion order does not matter
