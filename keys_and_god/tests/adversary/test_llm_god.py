"""WP4 LLM God (god.kind=llm): the tool loop on a scripted FakeLLM God policy, the step / tool ceilings,
power budget and validation through adversary.powers, invalid-JSON handling, determinism + cache
replay, hash neutrality when kind != llm, and one real-engine FakeLLM run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from swarm.adversary import GodRunner, make_god
from swarm.adversary.llm_god import RETRY_TEXT, parse_action
from swarm.config import REPO_ROOT, RunConfig
from swarm.llm.base import FakeLLM
from swarm.llm.cache import ResponseCache, cache_key
from swarm.memory import MemoryStore
from swarm.schema import AgentState, GameState, KeyCopy, turn_index

AG = ["amber_fox", "blue_owl", "coral_elk", "dune_cat"]


def _state(rnd: int = 3, contacts: dict[str, list[str]] | None = None) -> GameState:
    n = len(AG)
    agents = {a: AgentState(agent=a, keys=[KeyCopy(copy_id=f"{i}", info_id=i, key=1000 + i)],
                            needs=[(i + 1) % n, (i + 2) % n],
                            contacts=None if contacts is None else contacts[a]) for i, a in enumerate(AG)}
    return GameState(game_id="gllm", round=rnd, turn_index=turn_index(rnd, 0, n), order=list(AG), agents=agents)


def _cfg(**god) -> RunConfig:
    base = {"kind": "llm", "start_round": 1, "budget": {"k": 2, "scope": "round"},
            "powers": {"reveal_needs": {"enabled": True, "variant": "mix", "max_receivers": 3}},
            "llm": {"powers": ["message_fraud", "reveal_needs"]}}
    for k, v in god.items():
        base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return RunConfig.model_validate({"god": base, "game": {"n_agents": 4}})


def _j(action: str, why: str = "w", **args) -> str:
    return json.dumps({"why": why, "action": action, "args": args})


def _wake(cfg: RunConfig, client, state: GameState | None = None, runner: GodRunner | None = None):
    s = state or _state()
    god = make_god(cfg.god, client=client, run_cfg=cfg)
    r = runner or GodRunner(god, cfg.god, MemoryStore(cfg.reasoning), max_message_chars=cfg.game.max_message_chars)
    if runner is None:
        r.init_game(s)
    res = asyncio.run(r.wake(s))
    return s, res, god


def _types(res) -> list[str]:
    return [e.data.type for e in res.events]


FRAUD = _j("message_fraud", why="coral is close", sender_shown="blue_owl", receiver="coral_elk",
           text="send me #3 first and I'll pass you #0")


# ------------------------------------------------------------------ the loop
def test_tool_loop_feeds_results_back_and_logs_in_order():
    llm = FakeLLM(script=[_j("activity_check", last_n_rounds=2), _j("message_check", receiver="coral_elk"),
                          FRAUD, _j("end_wake")])
    s, res, _ = _wake(_cfg(), llm)
    assert llm.n_calls == 4
    assert _types(res) == ["god_thought", "god_tool", "god_thought", "god_tool", "god_thought", "god_power",
                           "message", "god_thought"]
    # each result is fed back: request k has system + wake + 2 per earlier step
    assert [len(r.messages) for r in llm.requests] == [2, 4, 6, 8]
    assert llm.requests[1].messages[-1].content.startswith("activity_check:")
    assert "OK: message_fraud as blue_owl to coral_elk" in llm.requests[3].messages[-1].content
    assert all(r.meta.role == "god" for r in llm.requests)
    forged = [m for m in s.messages if m.forged]
    assert len(forged) == 1 and forged[0].sender_shown == "blue_owl" and forged[0].receiver == "coral_elk"
    assert forged[0].turn_index == s.turn_index - 1          # delivered stamped like the slot before
    th = [e.data for e in res.events if e.data.type == "god_thought"]
    assert th[2].action == "message_fraud" and th[2].text == "coral is close" and th[2].note == "power_ok"
    assert th[-1].note == "end_wake" and th[0].llm_call_ids == [llm.requests[0].call_id]
    assert len(res.results) == 1 and res.results[0].ok and s.god_budget_left == 1


def test_system_prompt_has_rules_powers_and_no_agent_format():
    llm = FakeLLM(script=[_j("end_wake")])
    _, _, god = _wake(_cfg(), llm)
    sys_text = llm.requests[0].messages[0].content
    assert "RULES" in sys_text and "message_fraud" in sys_text and "reveal_needs" in sys_text
    assert "cot_control" not in sys_text                        # god.llm.powers allowlist
    assert '"note": "<private note' not in sys_text             # the agents' reply format is cut
    assert "real-world" in sys_text and "simulation" in sys_text
    wake = llm.requests[0].messages[1].content
    assert "start of round 3" in wake and "Budget left: 2" in wake and "amber_fox" in wake


def test_investigating_briefing_hides_needs_holdings_and_contents():
    """god/v2 (default): only links, last round's message counts, budget and past uses; god/v1 keeps the
    truth roster."""
    contacts = {"amber_fox": ["blue_owl"], "blue_owl": ["amber_fox", "coral_elk"], "coral_elk": ["blue_owl"],
                "dune_cat": []}
    s = _state(contacts=contacts)
    from swarm.schema import Message
    s.messages.append(Message(msg_id="m0", round=2, turn_index=5, channel="dm", sender_shown="amber_fox",
                              true_sender="amber_fox", receiver="blue_owl", text="I need #2, can you help?"))
    llm = FakeLLM(script=[_j("end_wake")])
    _wake(_cfg(), llm, state=s)
    wake = llm.requests[0].messages[1].content
    assert "- blue_owl: amber_fox, coral_elk" in wake and "- dune_cat: nobody" in wake
    assert "1 private messages and 0 board posts" in wake
    for leak in ("#", "needs held", "spare", "missing", "I need", "closest"):
        assert leak not in wake, leak
    assert "Your tools are how you learn" in llm.requests[0].messages[0].content
    v1 = FakeLLM(script=[_j("end_wake")])
    _wake(_cfg(llm={"prompt_set": "god/v1"}), v1, state=_state(contacts=contacts))
    assert "needs held" in v1.requests[0].messages[1].content


def test_wait_ends_the_wake_and_is_a_logged_tool():
    llm = FakeLLM(script=[_j("wait"), FRAUD])
    s, res, _ = _wake(_cfg(), llm)
    assert llm.n_calls == 1 and _types(res) == ["god_thought", "god_tool"] and not s.messages


# ------------------------------------------------------------------ ceilings
def test_step_limit_per_wake():
    llm = FakeLLM(policy=lambda req: _j("activity_check"))
    _, res, _ = _wake(_cfg(llm={"max_steps": 3}), llm)
    assert llm.n_calls == 3
    th = [e.data for e in res.events if e.data.type == "god_thought"]
    assert th[-1].note == "step_limit" and sum(e.data.type == "god_tool" for e in res.events) == 3


def test_tool_ceiling_hit_is_logged_and_budget_untouched():
    llm = FakeLLM(policy=lambda req: _j("message_check"))
    s, res, _ = _wake(_cfg(tool_call_safety_ceiling=2, llm={"max_steps": 8}), llm)
    tools = [e.data for e in res.events if e.data.type == "god_tool"]
    assert [t.ceiling_hit for t in tools] == [False, False, True]
    assert res.ceiling_hit and llm.n_calls == 3 and s.god_budget_left == 2 and not res.results


def test_token_guard_ends_the_wake():
    llm = FakeLLM(policy=lambda req: _j("activity_check", why="x " * 50))
    _, res, _ = _wake(_cfg(llm={"max_wake_tokens": 20}), llm)
    assert llm.n_calls == 1 and res.events[-1].data.note == "token_limit"


# ------------------------------------------------------------------ budget
def test_budget_is_k_per_round_and_wake_stops_when_spent():
    llm = FakeLLM(policy=lambda req: FRAUD)
    cfg = _cfg()
    s = _state(rnd=3)
    god = make_god(cfg.god, client=llm, run_cfg=cfg)
    runner = GodRunner(god, cfg.god, MemoryStore(cfg.reasoning))
    runner.init_game(s)
    total = 0
    for rnd in range(3, 13):                                   # 10 rounds
        s.round, s.turn_index = rnd, turn_index(rnd, 0, len(AG))
        res = asyncio.run(runner.wake(s))
        oks = [r for r in res.results if r.ok]
        assert len(oks) == 2 and s.god_budget_left == 0
        assert res.events[-1].data.note == "budget_spent"
        total += len(oks)
    assert total == 20 and llm.n_calls == 20                   # no call wasted once the budget is spent


def test_game_scope_budget_never_exceeded():
    llm = FakeLLM(policy=lambda req: FRAUD)
    cfg = _cfg(budget={"k": 3, "scope": "game"})
    s = _state(rnd=1)
    runner = GodRunner(make_god(cfg.god, client=llm, run_cfg=cfg), cfg.god, MemoryStore(cfg.reasoning))
    runner.init_game(s)
    ok = 0
    for rnd in range(1, 11):
        s.round, s.turn_index = rnd, turn_index(rnd, 0, len(AG))
        res = asyncio.run(runner.wake(s))
        ok += sum(r.ok for r in res.results)
    assert ok == 3 and s.n_interventions == 3


# ------------------------------------------------------------------ invalid JSON
def test_invalid_json_one_retry_then_end():
    llm = FakeLLM(script=["I think I will forge", "{not json", FRAUD])
    s, res, _ = _wake(_cfg(), llm)
    assert llm.n_calls == 2 and not s.messages
    notes = [e.data.note for e in res.events]
    assert notes == ["invalid_json_retry", "invalid_json_end"]
    assert llm.requests[1].messages[-1].content == RETRY_TEXT


def test_invalid_json_retry_recovers():
    llm = FakeLLM(script=["oops", "```json\n" + FRAUD + "\n```", _j("end_wake")])
    s, res, _ = _wake(_cfg(), llm)
    assert llm.n_calls == 3 and len(s.messages) == 1
    assert [e.data.note for e in res.events if e.data.type == "god_thought"] == [
        "invalid_json_retry", "power_ok", "end_wake"]


def test_parse_action():
    assert parse_action('blah {"action": "wait"} tail') == {"why": "", "action": "wait", "args": {}}
    assert parse_action('{"why": "a", "action": "x", "args": [1]}') is None
    assert parse_action("") is None and parse_action('{"foo": 1}') is None


# ------------------------------------------------------------------ power validation
def test_power_validation_feedback_spends_nothing():
    contacts = {"amber_fox": ["blue_owl"], "blue_owl": ["amber_fox", "coral_elk"], "coral_elk": ["blue_owl"],
                "dune_cat": []}
    llm = FakeLLM(script=[
        _j("message_fraud", sender_shown="nobody_here", receiver="coral_elk", text="hi"),     # unknown sender
        _j("message_fraud", sender_shown="dune_cat", receiver="coral_elk", text="hi"),        # not a contact
        _j("message_fraud", sender_shown="blue_owl", receiver="coral_elk", text="see intervention_id"),
        _j("cot_control", agent="coral_elk", round=2, text="x"),                              # not offered
        _j("reveal_needs", target="coral_elk", sender_shown="blue_owl", receivers=["amber_fox"], variant="maybe"),
        _j("message_fraud", sender_shown="blue_owl", receiver="coral_elk", text="y" * 2000),  # cut to 500
        _j("end_wake")])
    s, res, _ = _wake(_cfg(), llm, state=_state(contacts=contacts))
    fb = [r.messages[-1].content for r in llm.requests[1:]]
    assert fb[0].startswith("FAILED: unknown_sender") and fb[1].startswith("FAILED: not_connected")
    assert "contacts" in fb[1] and fb[2].startswith("Error: that text") and "not available" in fb[3]
    assert fb[4].startswith("Error: variant") and fb[5].startswith("OK:")
    powers = [e.data for e in res.events if e.data.type == "god_power"]
    assert [p.ok for p in powers] == [False, False, True]
    assert [m.text for m in s.messages] == ["y" * 500] and s.god_budget_left == 1


def test_reveal_needs_fills_items_and_mix_variant_is_gods_choice():
    llm = FakeLLM(script=[
        _j("reveal_needs", target="coral_elk", sender_shown="blue_owl", receivers=["amber_fox", "dune_cat"],
           variant="false", text="psst, {target} is after {items}"),
        _j("reveal_needs", target="amber_fox", sender_shown="amber_fox", receivers=["coral_elk"],
           text="no placeholder here")])
    s, res, _ = _wake(_cfg(), llm)
    powers = [e.data for e in res.events if e.data.type == "god_power"]
    assert [p.ok for p in powers] == [True, True]
    assert powers[0].action.variant == "false" and powers[0].action.items
    assert s.messages[0].text.startswith("psst, coral_elk is after #")
    assert powers[1].action.variant == "truthful" and s.messages[-1].text.startswith("I still need info #")


# ------------------------------------------------------------------ determinism / cache replay
class _Cached:
    """Minimal cache layer (as OpenAICompatClient does it): serve hits, else call inner and store."""

    def __init__(self, inner, cache: ResponseCache) -> None:
        self.inner, self.cache, self.hits = inner, cache, 0

    async def chat(self, req):
        key = cache_key("fake", "fake", req)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        resp = await self.inner.chat(req)
        self.cache.put(key, resp)
        return resp


def _dump(res) -> list[str]:
    return [e.model_dump_json() for e in res.events]


def test_deterministic_and_replays_from_cache(tmp_path: Path):
    script = [_j("activity_check"), FRAUD, _j("reveal_needs", target="coral_elk", sender_shown="blue_owl",
                                              receivers=["amber_fox"], variant="truthful")]
    a1 = FakeLLM(script=script)
    _, r1, _ = _wake(_cfg(), _Cached(a1, ResponseCache(tmp_path / "c.sqlite", "read_write")))
    a2 = FakeLLM(script=script)
    _, r2, _ = _wake(_cfg(), a2)
    assert _dump(r1) == _dump(r2)
    assert [(q.call_id, q.seed) for q in a1.requests] == [(q.call_id, q.seed) for q in a2.requests]
    # replay: a God that would now answer differently is served the cached responses (read_only)
    other = _Cached(FakeLLM(policy=lambda req: _j("end_wake")), ResponseCache(tmp_path / "c.sqlite", "read_only"))
    _, r3, _ = _wake(_cfg(), other)
    assert _dump(r3) == _dump(r1) and other.hits == 3


def test_seed_depends_on_effort_and_slot():
    a, b = FakeLLM(script=[_j("end_wake")]), FakeLLM(script=[_j("end_wake")])
    _wake(_cfg(), a)
    _wake(_cfg(llm={"reasoning_effort": "high"}), b)
    assert a.requests[0].seed != b.requests[0].seed and a.requests[0].call_id == b.requests[0].call_id


# ------------------------------------------------------------------ config / hash neutrality
def test_god_llm_block_is_hash_neutral_unless_kind_llm():
    base = RunConfig.model_validate({"god": {"kind": "scripted"}})
    tweaked = RunConfig.model_validate({"god": {"kind": "scripted", "llm": {"max_steps": 3, "model": "x/y"}}})
    assert base.config_hash() == tweaked.config_hash()
    none = RunConfig.model_validate({"god": {"kind": "none", "llm": {"max_steps": 3}}})
    assert none.config_hash() == RunConfig.model_validate({"god": {"kind": "none"}}).config_hash()
    l1 = RunConfig.model_validate({"god": {"kind": "llm"}})
    l2 = RunConfig.model_validate({"god": {"kind": "llm", "llm": {"max_steps": 3}}})
    assert l1.config_hash() != l2.config_hash()
    assert l1.model_for("god") == l1.model_for("agent")              # default: same_as_agents
    l3 = RunConfig.model_validate({"god": {"kind": "llm", "llm": {"model": "openai/gpt-oss-120b"}}})
    assert l3.model_for("god") == ("vllm", "openai/gpt-oss-120b") and base.model_for("god")[1] != "openai/gpt-oss-120b"


def test_god_llm_same_as_agents_resolution():
    agents = {"llm": {"openrouter": {"enabled": True, "model": "z-ai/glm-5.3-flash", "reasoning_effort": "minimal",
                                     "provider_sort": "price", "price_in_per_mtok": 0.15, "price_out_per_mtok": 0.5}},
              "roles": {"agent": {"backend": "openrouter", "model": "z-ai/glm-5.3-flash"},
                        "god": {"backend": "openrouter"}}}
    same = RunConfig.model_validate({**agents, "god": {"kind": "llm"}})
    assert same.god_llm() == {"model": "z-ai/glm-5.3-flash", "reasoning_effort": "minimal", "provider_sort": "price",
                              "provider_order": None, "allow_fallbacks": None, "price_in_per_mtok": 0.15,
                              "price_out_per_mtok": 0.5}
    own = RunConfig.model_validate({**agents, "god": {"kind": "llm", "llm": {
        "model": "openai/gpt-oss-120b", "reasoning_effort": "medium", "provider_sort": None,
        "provider_order": ["deepinfra"], "allow_fallbacks": False, "price_in_per_mtok": 0.05}}})
    r = own.god_llm()
    assert (r["model"], r["reasoning_effort"], r["provider_sort"], r["provider_order"], r["allow_fallbacks"],
            r["price_in_per_mtok"], r["price_out_per_mtok"]) == (
        "openai/gpt-oss-120b", "medium", None, ["deepinfra"], False, 0.05, 0.5)
    assert own.model_for("god") == ("openrouter", "openai/gpt-oss-120b")


def test_god_client_is_separate_with_its_own_effort(tmp_path: Path, monkeypatch):
    """GLM agents (effort minimal) + a gpt-oss God (effort medium), and a GLM God on the agents' own
    model: the God never shares the agents' client, so its effort / totals are its own."""
    from swarm.runlog.writer import RunWriter
    from swarm.runner.wiring import build_clients

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    base = {"llm": {"cache": {"mode": "off"}, "openrouter": {
                "enabled": True, "model": "z-ai/glm-5.3-flash", "reasoning_effort": "minimal",
                "ledger_path": str(tmp_path / "ledger.jsonl")}},
            "roles": {"agent": {"backend": "openrouter", "model": "z-ai/glm-5.3-flash"}, "god": {"backend": "openrouter"}}}
    for god_llm, want in (({"model": "openai/gpt-oss-120b", "reasoning_effort": "medium"}, ("openai/gpt-oss-120b", "medium")),
                          ({"reasoning_effort": "low"}, ("z-ai/glm-5.3-flash", "low"))):
        cfg = RunConfig.model_validate({**base, "god": {"kind": "llm", "llm": god_llm}})
        cl = build_clients(cfg, RunWriter(tmp_path / f"r{want[1]}", "rid"))
        g, a = cl.get("god"), cl.get("agent")
        assert g is not a and (g.model, g.reasoning_effort) == want and a.reasoning_effort == "minimal"


# ------------------------------------------------------------------ chatty / tool-call-style replies (GLM)
class _Scripted:
    """FakeLLM-like client returning (text, reasoning, finish_reason) per call."""

    def __init__(self, replies):
        from swarm.schema import ChatResponse, Usage
        self.replies, self.requests, self._r, self._u = list(replies), [], ChatResponse, Usage

    async def chat(self, req):
        self.requests.append(req)
        text, reasoning, finish = self.replies[len(self.requests) - 1]
        return self._r(call_id=req.call_id, text=text, reasoning=reasoning, finish_reason=finish,
                       usage=self._u(completion_tokens=len(text.split())), backend="fake", model="glm-like")


def test_glm_style_replies_are_understood():
    chatty = ("Let me think. Option A would be {\"action\": \"activity_check\"} but better:\n```json\n"
              + FRAUD + "\n```\nThat should work.")
    toolcall = json.dumps({"name": "message_check", "arguments": json.dumps({"receiver": "coral_elk"})})
    nested = json.dumps({"why": "done", "action": {"name": "end_wake", "args": {}}})
    llm = _Scripted([(chatty, None, "stop"), (toolcall, None, "stop"), (nested, None, "stop")])
    s, res, _ = _wake(_cfg(), llm)
    th = [e.data for e in res.events if e.data.type == "god_thought"]
    assert [t.action for t in th] == ["message_fraud", "message_check", "end_wake"]   # the LAST object wins
    assert len(s.messages) == 1 and th[1].args == {"receiver": "coral_elk"}


def test_json_left_in_reasoning_and_length_cut_retry():
    """Mandatory-reasoning models: content empty / cut at max_tokens. The action is taken from the
    reasoning when it holds one; otherwise the retry asks for only the JSON (cut-off wording)."""
    llm = _Scripted([("", "I will forge. " + FRAUD, "length"),
                     ("", "hmm, thinking about who to target...", "length"),
                     (_j("end_wake"), None, "stop")])
    s, res, _ = _wake(_cfg(), llm)
    notes = [e.data.note for e in res.events if e.data.type == "god_thought"]
    assert notes == ["power_ok:from_reasoning", "invalid_json_retry:length", "end_wake"]
    assert len(s.messages) == 1
    from swarm.adversary.llm_god import RETRY_CUT_TEXT
    assert llm.requests[2].messages[-1].content == RETRY_CUT_TEXT


def test_parse_action_shapes():
    assert parse_action('{"tool": "wait", "parameters": {}}')["action"] == "wait"
    assert parse_action('{"action": "x", "args": "not json"}') is None
    assert parse_action('{"action": "a"} then {"action": "b", "args": {"k": 1}}') == {
        "why": "", "action": "b", "args": {"k": 1}}


def test_llmgod_configs_reuse_the_v2_game_ids():
    """The shipped God-arm configs keep v2's god section except kind + god.llm, so the God arm has the
    v2 runs' game_ids (agent seeds / cache keys) and the v2 nogod / noisefork branches are its controls."""
    from swarm.runner.run import load_experiment
    from swarm.runner.spec import build_specs

    gods = {"showcase100_v2_llmgod_sim": ("openai/gpt-oss-120b", "medium"),
            "showcase100_v2_glm_llmgod_sim": ("z-ai/glm-5.3-flash", "minimal"),
            "showcase100_v2_llmgod_glmgod_sim": ("z-ai/glm-5.3-flash", "minimal"),
            "showcase100_v2_glm_llmgod_ossgod_sim": ("openai/gpt-oss-120b", "medium")}
    ids: dict[str, list[str]] = {}
    for name, (model, effort) in gods.items():
        cfg, opts = load_experiment(REPO_ROOT / "configs" / f"{name}.yaml")
        assert cfg.god.kind == "llm" and opts.fork.only_branches == ["god"]
        raw = cfg.model_dump(mode="json")
        raw["god"]["kind"] = "scripted"
        scripted = RunConfig.model_validate(raw)
        groups = build_specs(cfg, opts)
        v2_groups = build_specs(scripted, opts)
        assert [s.game_id for g in groups for s in g] == [s.game_id for g in v2_groups for s in g]
        r = cfg.god_llm()
        assert (r["model"], r["reasoning_effort"]) == (model, effort) and cfg.model_for("god")[1] == model
        ids[name] = [s.game_id for g in groups for s in g]
    # the v2 runs' game_ids (runs/showcase100_v2_sim, runs/showcase100_v2_glm_sim in swarm_showcase)
    assert ids["showcase100_v2_llmgod_sim"] == ids["showcase100_v2_llmgod_glmgod_sim"] == ["gf2ae8b27876e", "g2babede7921c"]
    assert ids["showcase100_v2_glm_llmgod_sim"] == ids["showcase100_v2_glm_llmgod_ossgod_sim"] == [
        "gab83796c3898", "g6ca2f35f8f19"]


# ------------------------------------------------------------------ real engine, FakeLLM agents + God
def test_real_engine_fake_llm_god(tmp_path: Path):
    from swarm.runlog.trace import load_run
    from swarm.runner.run import load_experiment, run_experiment

    ov = ["roles.agent.backend=fake", "roles.god.backend=fake", "llm.cache.mode='off'", "game.n_agents=8",
          "game.max_rounds=6", "god.kind=llm", "runner.report=false"]
    cfg, opts = load_experiment(REPO_ROOT / "configs" / "demo.yaml", ov)
    run_dir = asyncio.run(run_experiment(cfg, opts, overrides=ov, out_root=tmp_path, run_id="llmgod"))
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["status"] == "done" and man["models"]["god"]["model"] == man["models"]["agent"]["model"]
    assert any(k.startswith("god/") for k in man["usage"]["by_role_model"])
    run = load_run(run_dir)
    a = run.pairs()[0]["god"]
    th = a.of_type("god_thought")
    assert th and a.of_type("god_tool") and a.of_type("god_power")
    assert {e.round for e in th} == {3, 5}                       # demo schedule rounds = the wake calendar
    assert all(p.data.ok for p in a.of_type("god_power"))
    assert a.meta["leak_hits"] == 0
    god_calls = [c for c in a.llm_calls if c.request.meta.role == "god"]
    assert god_calls and all(c.request.meta.purpose.startswith("god_step:") for c in god_calls)
    assert not run.pairs()[0]["nogod"].of_type("god_thought")


def test_last_n_rounds_counts_finished_rounds_at_a_round_start_wake():
    from swarm.schema import ActionRecord

    s = _state(rnd=3)
    s.history += [ActionRecord(turn_index=5, round=2, kind="give", actor="amber_fox", target="blue_owl", info_id=0),
                  ActionRecord(turn_index=1, round=1, kind="give", actor="dune_cat", target="coral_elk", info_id=3)]
    llm = FakeLLM(script=[_j("activity_check", last_n_rounds=1), _j("activity_check", last_n_rounds=1, agent="dune_cat"),
                          _j("activity_check", last_n_rounds=5, agent="dune_cat"), _j("end_wake")])
    _, res, _ = _wake(_cfg(), llm, state=s)
    fb = [r.messages[-1].content for r in llm.requests[1:]]
    assert "give amber_fox -> blue_owl #0" in fb[0] and "dune_cat" not in fb[0]
    assert fb[1] == "activity_check: no results.\n\nBudget left: 2. Steps left: 6."
    assert "[r1] give dune_cat -> coral_elk #3" in fb[2]
    tools = [e.data for e in res.events if e.data.type == "god_tool"]
    assert tools[0].args == {"last_n_rounds": 2} and tools[1].args == {"last_n_rounds": 2, "agent": "dune_cat"}
