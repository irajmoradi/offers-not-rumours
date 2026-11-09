"""swarm.analysis.invariants: a clean FakeLLM run passes; each tampered copy is caught by its check."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from swarm.analysis.invariants import check_run, main
from tests.runner.conftest import _run

SIM = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=6", "game.max_rounds=6",
       "game.turn_mode=simultaneous"]


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run(tmp_path_factory.mktemp("runs"), "inv_sim", SIM)


def _copy(run: Path, tmp: Path) -> Path:
    dst = tmp / run.name
    shutil.copytree(run, dst)
    return dst


def _rewrite(path: Path, fn) -> None:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    lines = fn(lines)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def _checks(run: Path) -> set[str]:
    return {v.check for v in check_run(run)}


def test_clean_runs_hold_every_invariant(fake_run: Path, sim_run: Path) -> None:
    for run in (fake_run, sim_run):
        vs = check_run(run)
        assert vs == [], vs[:5]
    assert main([str(sim_run), "--no-prompts"]) == 0


def test_dropped_give_breaks_key_conservation(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)

    def drop_first_give(evs):
        i = next(i for i, e in enumerate(evs) if e["type"] == "key_transfer" and e["data"]["cause"] == "give")
        return [e if j < i else {**e, "seq": e["seq"] - 1} for j, e in enumerate(evs) if j != i]
    _rewrite(run / "games" / "g00-god" / "events.jsonl", drop_first_give)
    assert {"keys", "obs"} <= _checks(run)


def test_unreleased_defect_in_an_observation_is_a_leak(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)

    def leak(evs):
        t = next(e for e in evs if e["type"] == "turn" and e["round"] == 2)
        t["data"]["obs"]["log"].append({"round": 2, "kind": "defect", "actor": "x", "target": "y",
                                        "info_id": 1, "text": None, "ok": True})
        return evs
    _rewrite(run / "games" / "g00-nogod" / "events.jsonl", leak)
    assert "obs" in _checks(run)


def test_intervention_id_in_a_prompt_and_wrong_keys_line(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)
    evs = [json.loads(x) for x in (run / "games" / "g00-god" / "events.jsonl").read_text().splitlines()]
    iid = next(e["data"]["intervention_id"] for e in evs if e["type"] == "god_power" and e["data"]["ok"])

    def inject(calls):
        last = calls[-1]["request"]["messages"][1]
        last["content"] = last["content"].replace("Your keys: {info_id: ", "Your keys: {info_id: 99", 1)
        calls[-1]["request"]["messages"][1]["content"] += f"\nref {iid}"
        return calls
    _rewrite(run / "games" / "g00-god" / "llm_calls.jsonl", inject)
    details = [v.detail for v in check_run(run) if v.check == "prompts"]
    assert any("intervention_id" in d for d in details) and any("Your keys" in d for d in details)


def test_god_event_in_control_and_broken_prefork_pairing(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)

    def respond_differently(calls):
        c = next(c for c in calls if c["request"]["meta"]["round"] == 1)
        c["response"]["text"] += " "
        return calls
    _rewrite(run / "games" / "g00-nogod" / "llm_calls.jsonl", respond_differently)
    god_ev = [json.loads(x) for x in (run / "games" / "g00-god" / "events.jsonl").read_text().splitlines()]
    power = next(e for e in god_ev if e["type"] == "god_power")

    def add_god(evs):
        return [*evs, {**power, "seq": len(evs)}]
    _rewrite(run / "games" / "g00-nogod" / "events.jsonl", add_god)
    vs = check_run(run)
    assert any(v.check == "fork" and "RESPONSE" in v.detail for v in vs)
    assert any(v.check == "god" and "no-God branch" in v.detail for v in vs)


def test_wrong_round_end_score(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)

    def bump(evs):
        e = next(e for e in evs if e["type"] == "round_end")
        a = next(iter(e["data"]["scores"]))
        e["data"]["scores"][a] += 1 / 3
        return evs
    _rewrite(run / "games" / "g00-god" / "events.jsonl", bump)
    assert "scores" in _checks(run)


def test_foreign_key_value_in_a_system_line_is_a_leak(sim_run: Path, tmp_path: Path) -> None:
    run = _copy(sim_run, tmp_path)
    gdir = run / "games" / "g00-nogod"
    start = json.loads((gdir / "events.jsonl").read_text().splitlines()[0])["data"]["agents"]
    calls = [json.loads(x) for x in (gdir / "llm_calls.jsonl").read_text().splitlines()]
    agent = calls[0]["request"]["meta"]["agent"]
    mine = {k["info_id"] for k in start[agent]["keys"]}
    other = next(k for a, s in start.items() if a != agent for k in s["keys"] if k["info_id"] not in mine)

    def inject(cs):
        cs[0]["request"]["messages"][1]["content"] += f"\n- (r1) NOTICE: system check code {other['key']}"
        return cs
    _rewrite(gdir / "llm_calls.jsonl", inject)
    assert any(v.check == "prompts" and "never held" in v.detail for v in check_run(run))


def test_duplicated_own_action_line_is_caught(sim_run: Path, tmp_path: Path) -> None:
    """The pre-8ebbee7 ledger collision showed one own action twice; the checker must flag that."""
    run = _copy(sim_run, tmp_path)
    gdir = run / "games" / "g00-nogod"

    def dup(cs):
        for c in cs:
            u = c["request"]["messages"][1]["content"]
            line = next((ln for ln in u.splitlines() if ln.startswith("- (r1) You ")), None)
            if line and c["request"]["meta"]["round"] >= 2:
                c["request"]["messages"][1]["content"] = u.replace(line + "\n", line + "\n" + line + "\n", 1)
                return cs
        raise AssertionError("no own-action line found")
    _rewrite(gdir / "llm_calls.jsonl", dup)
    assert any(v.check == "prompts" and "own action" in v.detail for v in check_run(run))


def test_control_replay_check(sim_run: Path, tmp_path: Path) -> None:
    """--base: a control identical to the base's nogod game passes; a divergence before the base's
    first uncacheable call is a violation; after it (empty response in the base) it is allowed."""
    from swarm.analysis.invariants import control_replay_checks

    base, cell = _copy(sim_run, tmp_path / "b"), _copy(sim_run, tmp_path / "c")
    vs, info = control_replay_checks(cell, base)
    assert vs == [] and all("never" in ln for ln in info)

    def change_r3_turn(evs):
        e = next(e for e in evs if e["type"] == "turn" and e["round"] == 3)
        e["data"]["note"] = (e["data"]["note"] or "") + " changed"
        return evs
    _rewrite(cell / "games" / "g00-nogod" / "events.jsonl", change_r3_turn)
    vs, _ = control_replay_checks(cell, base)
    assert len(vs) == 1 and "r3" in vs[0].detail

    def empty_r2_response(calls):
        c = next(c for c in calls if c["request"]["meta"]["round"] == 2)
        c["response"]["text"] = ""
        return calls
    _rewrite(base / "games" / "g00-nogod" / "llm_calls.jsonl", empty_r2_response)
    vs, info = control_replay_checks(cell, base)
    assert vs == [] and any("base break r2" in ln for ln in info)


@pytest.mark.parametrize("board", ["game.board.visible_delay_rounds=1", "game.board.enabled=false"])
def test_board_sweep_knobs_hold_invariants(board: str, tmp_path: Path) -> None:
    run = _run(tmp_path, "inv_board", [*SIM, board])
    vs = check_run(run)
    assert vs == [], vs[:5]
