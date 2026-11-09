"""Run logs: event stream stamping, LLM-call logging, manifest scrubbing, reader round trip."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from swarm.config import RunConfig
from swarm.llm.base import FakeLLM
from swarm.llm.secrets import register_secret
from swarm.runlog.llmlog import LoggingClient
from swarm.runlog.manifest import Manifest, prompt_set_hash, scrub
from swarm.runlog.trace import load_game, load_run
from swarm.runlog.writer import CURRENT_GAME, RunWriter
from swarm.schema import CallMeta, ChatMessage, ChatRequest, EventBody, RoundEndData


def _req(cid: str, gid: str = "g1", text: str = "hi") -> ChatRequest:
    return ChatRequest(call_id=cid, messages=[ChatMessage(role="user", content=text)], max_tokens=5,
                       temperature=0.0, seed=1, meta=CallMeta(role="agent", agent="a", game_id=gid))


def _body(r: int) -> EventBody:
    return EventBody(round=r, turn_index=None, phase="end", actor="SYSTEM",
                     data=RoundEndData(state_hash="h", scores={"a": 0.5}, welfare=0.5))


def test_events_seq_contiguous_and_round_trip(tmp_path: Path) -> None:
    run = RunWriter(tmp_path / "r", "rid")
    gw = run.game("g00-main", "g1", {"branch": "main", "game_idx": 0})
    evs = [gw.emit(_body(r)) for r in range(1, 4)]
    assert [e.seq for e in evs] == [0, 1, 2] and all(e.run_id == "rid" for e in evs)
    gw.close()
    g = load_game(gw.dir)
    assert [e.seq for e in g.events] == [0, 1, 2] and not g.parse_errors
    assert g.meta["status"] == "done" and g.branch == "main"


def test_llm_calls_routed_by_current_game_and_counted(tmp_path: Path) -> None:
    run = RunWriter(tmp_path / "r", "rid")
    ga, gb = run.game("g00-god", "same"), run.game("g00-nogod", "same")   # paired: same game_id
    client = LoggingClient(FakeLLM(), run.log_llm_call, run_id="rid", backend="fake", model="fake")

    async def play(gw, n):  # noqa: ANN001
        tok = CURRENT_GAME.set(gw)
        try:
            for i in range(n):
                await client.chat(_req(f"{gw.dir.name}-{i}", "same"))
        finally:
            CURRENT_GAME.reset(tok)

    async def both() -> None:
        await asyncio.gather(play(ga, 2), play(gb, 3))

    asyncio.run(both())
    for gw in (ga, gb):
        gw.close()
    assert len(load_game(ga.dir).llm_calls) == 2 and len(load_game(gb.dir).llm_calls) == 3
    assert run.usage().to_dict()["total_calls"] == 5


def test_secret_never_written(tmp_path: Path) -> None:
    secret = "sk-or-v1-THISISNOTAREALKEY1234567890"
    register_secret(secret)
    run = RunWriter(tmp_path / "r", "rid")
    gw = run.game("g", "g1")
    gw.log_llm_call(__import__("swarm.schema", fromlist=["LLMCall"]).LLMCall(
        run_id="rid", backend="fake", model="fake", cache_key="k", request=_req("c", text=f"key {secret}")))
    gw.close()
    assert secret not in (gw.dir / "llm_calls.jsonl").read_text()


def test_scrub_drops_key_fields_keeps_whitelist() -> None:
    out = scrub({"api_key": "x", "api_key_env": "OPENROUTER_API_KEY", "nested": {"Authorization": "b"},
                 "prompt_tokens": 3, "note": "token sk-abcdefghijklmnop"})
    assert "api_key" not in out and "Authorization" not in out["nested"]
    assert out["api_key_env"] == "OPENROUTER_API_KEY" and out["prompt_tokens"] == 3
    assert "sk-abcdefghijklmnop" not in out["note"]


def test_manifest_fields(tmp_path: Path, tiny_cfg: RunConfig) -> None:
    m = Manifest(tmp_path, "rid", tiny_cfg)
    m.add_game({"label": "g00-main", "seed": 5})
    m.finalize("done", {"total_calls": 0})
    d = json.loads((tmp_path / "manifest.json").read_text())
    for k in ("git", "host", "config_sha256", "seeds", "prompts", "models", "usage", "games"):
        assert k in d
    assert d["status"] == "done" and d["seeds"]["per_game"] == [{"label": "g00-main", "seed": 5}]
    assert d["prompts"]["agent"]["hash"] == prompt_set_hash(tiny_cfg.roles.agent.prompt_set)["hash"]


def test_load_run_pairs(tmp_path: Path) -> None:
    run = RunWriter(tmp_path / "r", "rid")
    for br in ("god", "nogod"):
        run.game(f"g00-{br}", "g1", {"branch": br, "game_idx": 0}).close()
    rt = load_run(tmp_path / "r")
    assert set(rt.pairs()[0]) == {"god", "nogod"}
