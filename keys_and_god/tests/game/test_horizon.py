"""game.view_log.horizon (hide-horizon): default "shown" is byte-identical to showcase 40c309a (config hashes
and agent system prompts of every config; FakeLLM event streams and every rendered request of a FakeLLM game
per config); "round_only" removes the game's total number of rounds from everything an agent is shown
(system prompt, turn header, trade-reply header; no end-of-game wording), so an agent's every prompt is the
same whether the game lasts 7 or 10 rounds."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from swarm.agents.llm_agent import system_context
from swarm.config import REPO_ROOT, from_dict
from swarm.prompts import load_prompt_set
from swarm.runner.run import load_experiment, run_experiment

PINNED = json.loads((REPO_ROOT / "tests" / "fixtures" / "horizon_before.json").read_text())["configs"]
ROLES = ("agent", "god", "judge", "summarizer")
# every config's FakeLLM game (the digests in horizon_before.json were taken with exactly these overrides)
FAKE_OV = [f"roles.{r}.backend=fake" for r in ROLES] + [
    "llm.cache.mode='off'", "llm.openrouter.enabled=false", "run.n_games=1", "runner.report=false", "sweep={}",
    "context.tokenizer=approx"]
FAST = ["configs/demo.yaml", "configs/g1_v2_costly_trade_sim.yaml", "configs/net40_sbm_contacts_sim.yaml",
        "configs/winners_leave_sim.yaml", "configs/showcase100_v2_sim.yaml"]
# what must never reach an agent under round_only (R = the game's max_rounds)
END_WORDS = r"rounds? (left|remain)|final (round|turn)|last (round|turn)|game (ends|is over)|ends? (in|after)"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_digests(name: str, tmp: Path, extra: list[str] = ()) -> tuple[list[str], list[str]]:
    cfg, _ = load_experiment(REPO_ROOT / name, [])
    r = min(cfg.game.max_rounds, 5)
    ov = [*FAKE_OV, f"game.max_rounds={r}", *extra]
    if cfg.god.kind != "none" and cfg.god.start_round > r:
        ov.append(f"god.start_round={r}")
    c2, o2 = load_experiment(REPO_ROOT / name, ov)
    run = asyncio.run(run_experiment(c2, o2, overrides=ov, out_root=tmp, run_id=_sha(name.encode())[:10]))
    ev, req = [], []
    for d in sorted((run / "games").iterdir()):
        ev.append(_sha((d / "events.jsonl").read_bytes()))
        calls = (d / "llm_calls.jsonl").read_text().splitlines()
        req.append(_sha(json.dumps([json.loads(x)["request"]["messages"] for x in calls]).encode()))
    return ev, req


# ------------------------------------------------------------------ off: byte-identical
def test_off_config_hashes_and_system_prompts_unchanged():
    assert len(PINNED) >= 56
    for name, want in PINNED.items():
        cfg = load_experiment(REPO_ROOT / name, [])[0]
        ps = load_prompt_set(cfg.roles.agent.prompt_set, framing=cfg.roles.agent.framing)
        got = _sha(ps.render("system.j2", **system_context(cfg)).encode())
        assert (cfg.config_hash(), got) == (want["hash"], want["system"]), name
        assert load_experiment(REPO_ROOT / name, ["game.view_log.horizon=shown"])[0].config_hash() == want["hash"]


@pytest.mark.parametrize("name", FAST)
def test_off_fake_event_streams_byte_identical(name, tmp_path):
    assert _fake_digests(name, tmp_path) == (PINNED[name]["events"], PINNED[name]["requests"])


@pytest.mark.slow
def test_off_fake_event_streams_byte_identical_all_configs(tmp_path):
    for name in sorted(PINNED):
        assert _fake_digests(name, tmp_path) == (PINNED[name]["events"], PINNED[name]["requests"]), name


def test_on_changes_hash_and_validates():
    base = from_dict({})
    on = from_dict({}, ["game.view_log.horizon=round_only"])
    assert on.config_hash() != base.config_hash()
    with pytest.raises(ValueError):
        from_dict({}, ["game.view_log.horizon=secret"])
    for name, rounds in (("configs/net40_sbm_long60_sim.yaml", 60), ("configs/showcase100_v2_horizon_sim.yaml", 25)):
        cfg = load_experiment(REPO_ROOT / name, [])[0]
        assert cfg.game.view_log.horizon == "round_only" and cfg.game.max_rounds == rounds
    long60 = load_experiment(REPO_ROOT / "configs/net40_sbm_long60_sim.yaml", [])[0]
    assert long60.god.kind == "none" and long60.run.n_games == 2 and long60.game.n_agents == 40
    assert long60.game.network.topology == "sbm" and long60.game.network.key_placement == "local_bias"
    v2 = load_experiment(REPO_ROOT / "configs/showcase100_v2_sim.yaml", [])[0].model_dump(exclude={"run"})
    hz = load_experiment(REPO_ROOT / "configs/showcase100_v2_horizon_sim.yaml", [])[0].model_dump(exclude={"run"})
    hz["game"]["view_log"]["horizon"] = "shown"
    assert hz == v2                                              # showcase v2 + the knob, nothing else


def test_system_prompt_line():
    ps = load_prompt_set("agent/v1")
    off = ps.render("system.j2", **system_context(from_dict({}, ["game.max_rounds=37"])))
    on = ps.render("system.j2", **system_context(from_dict({}, ["game.max_rounds=37",
                                                                 "game.view_log.horizon=round_only"])))
    assert "The game lasts at most 37 rounds." in off and "37" not in on
    assert "The number of rounds in the game is not announced." in on
    assert [x for x in off.splitlines() if "37" not in x] == [x for x in on.splitlines() if "announced" not in x]


# ------------------------------------------------------------------ on: no total, no end hint, R-invariant
DEMO_OV = ["roles.agent.backend=fake", "llm.cache.mode='off'", "game.n_agents=8", "game.trade.enabled=true",
           "runner.report=false"]


def _calls(tmp: Path, rid: str, rounds: int, horizon: str) -> dict[str, list[dict]]:
    ov = [*DEMO_OV, f"game.max_rounds={rounds}", f"game.view_log.horizon={horizon}"]
    cfg, opts = load_experiment(REPO_ROOT / "configs" / "demo.yaml", ov)
    run = asyncio.run(run_experiment(cfg, opts, overrides=ov, out_root=tmp, run_id=rid))
    return {d.name: [json.loads(x)["request"] for x in (d / "llm_calls.jsonl").read_text().splitlines()]
            for d in sorted((run / "games").iterdir())}


@pytest.fixture(scope="module")
def r7_r10(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Paired demo runs (God + no-God branches, trades on) of 7 and 10 rounds, per horizon. game_ids are pinned
    to ignore the config hash, so the R=7 and R=10 games share deal, turn order and seeds."""
    import swarm.runner.spec as spec

    tmp = tmp_path_factory.mktemp("horizon")
    mp = pytest.MonkeyPatch()
    orig = spec.make_game_id
    mp.setattr(spec, "make_game_id", lambda seed, idx, _h, **kw: orig(seed, idx, "horizon-test", **kw))
    try:
        return {(hz, r): _calls(tmp, f"{hz}_{r}", r, hz) for hz in ("shown", "round_only") for r in (7, 10)}
    finally:
        mp.undo()


def _view(reqs: list[dict], upto: int) -> list[tuple]:
    return [(q["meta"]["agent"], q["meta"]["round"], q["meta"]["purpose"], q["messages"])
            for q in reqs if q["meta"]["round"] <= upto]


def test_round_only_prompts_do_not_depend_on_total(r7_r10):
    for game in ("g00-god", "g00-nogod"):
        a, b = _view(r7_r10[("round_only", 7)][game], 7), _view(r7_r10[("round_only", 10)][game], 7)
        assert len(a) > 40 and a == b, game                    # every prompt identical through round 7
        assert {p.split(":")[0] for _, _, p, _ in a} == {"turn", "trade_reply"}
        assert _view(r7_r10[("shown", 7)][game], 7) != _view(r7_r10[("shown", 10)][game], 7)   # check bites


@pytest.mark.parametrize("rounds", [7, 10])
def test_round_only_no_total_or_end_wording(r7_r10, rounds):
    bad = re.compile(rf"\b(of|at most) {rounds}\b|\b{rounds} rounds\b|{END_WORDS}", re.I)
    for hz in ("round_only", "shown"):
        msgs = [m for g in r7_r10[(hz, rounds)].values() for q in g for m in q["messages"]]
        texts = [m["content"] for m in msgs]
        hits = {h.group(0) for t in texts for h in bad.finditer(t)}
        if hz == "shown":
            assert {f"of {rounds}", f"at most {rounds}", "last turn"} <= hits    # the scan bites
            continue
        assert hits == set(), hits
        users = [m["content"] for m in msgs if m["role"] == "user" and m["content"].startswith("You are ")]
        assert len(users) > 40 and all(re.match(r"You are \w+\. Round \d+\.\n", u) for u in users)
        assert any("[New since your previous turn]" in u for u in users)
        assert any("TRADE OFFER" in u for u in users)
        assert all("not announced" in m["content"] for m in msgs if m["role"] == "system")
