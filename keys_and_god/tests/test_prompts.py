"""Prompt loader tests. They use a self-contained template set in tmp_path, so that WP3 can edit
agent/v1 freely; WP3 owns the rendering tests for its own sets."""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

from swarm.context.base import PassthroughPolicy, WordCounter, items_from_observation
from swarm.prompts import TEMPLATES_DIR, PromptSet, load_prompt_set, prompt_hash
from swarm.schema import AgentMessageView, ChatMessage, KeyView, Observation, PinnedFacts
from tests.fixtures.fixed_state import AGENTS

SYSTEM = "{% if framing %}{{ framing }}\n\n{% endif %}Players: {{ game.n_agents }}. Mode {{ cot_mode }}.\n"
TURN = ("You are {{ view.agent }}.\n{% for s in view.sections if s.items %}[{{ s.name }}]\n"
        "{% for i in s.items %}- {{ i.text }}\n{% endfor %}{% endfor %}")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A templates root with agent/t (system, turn) and framing/none (empty)."""
    (tmp_path / "agent" / "t").mkdir(parents=True)
    (tmp_path / "agent" / "t" / "system.j2").write_text(SYSTEM)
    (tmp_path / "agent" / "t" / "turn.j2").write_text(TURN)
    (tmp_path / "framing" / "none").mkdir(parents=True)
    (tmp_path / "framing" / "none" / "framing.j2").write_text("")
    return tmp_path


def _view(agent: str):
    pinned = PinnedFacts(agent=agent, round=2, max_rounds=5, needs_open=[3], needs_burned=[],
                         holdings=[KeyView(info_id=1, key=1111)], can_give=[1], score=0.5, peers=AGENTS)
    obs = Observation(agent=agent, round=2, turn_index=4, pinned=pinned,
                      inbox=[AgentMessageView(msg_id="m1", channel="dm", sender="blue_owl", receiver=agent,
                                              text="hello")])
    b = PassthroughPolicy().build(agent=agent, round=2, ledger=[], current=items_from_observation(obs),
                                  memory=[], pinned=pinned, budget=14000, counter=WordCounter())
    return b.to_view()


def test_shipped_sets_load_and_never_mention_god():
    """agent/v1 and framing/none exist; agents are not told God exists (raw template scan)."""
    ps = load_prompt_set("agent/v1", framing="framing/none")
    assert ps.has("system.j2") and ps.has("turn.j2") and len(ps.hash) == 64
    for p in (TEMPLATES_DIR / "agent").rglob("*.j2"):
        text = p.read_text().lower()
        assert "god" not in text and "adversar" not in text, p


def test_render_turn_with_prompt_view(root):
    ps = PromptSet("agent/t", root=root)
    assert ps.render("system.j2", game={"n_agents": 4}, cot_mode="none") == "Players: 4. Mode none.\n"
    t = ps.render("turn.j2", view=_view("amber_fox"))
    assert "DM from blue_owl: hello" in t and "amber_fox" in t


def test_hash_stable_and_sensitive(root):
    a, b = PromptSet("agent/t", root=root), PromptSet("agent/t", root=root)
    assert a.hash == b.hash and len(a.hash) == 64
    p = root / "agent" / "t" / "turn.j2"
    p.write_text(p.read_text() + " ")
    assert PromptSet("agent/t", root=root).hash != a.hash


def test_framing_changes_hash_and_is_prepended(root):
    ctx = {"game": {"n_agents": 4}, "cot_mode": "none"}
    plain = PromptSet("agent/t", root=root)
    empty = PromptSet("agent/t", framing="framing/none", root=root)
    assert empty.render("system.j2", **ctx) == plain.render("system.j2", **ctx)
    (root / "framing" / "coop").mkdir()
    (root / "framing" / "coop" / "framing.j2").write_text("You are part of a team.")
    coop = PromptSet("agent/t", framing="framing/coop", root=root)
    assert coop.hash != plain.hash and empty.hash != coop.hash
    assert coop.render("system.j2", **ctx).startswith("You are part of a team.")
    assert coop.manifest()["framing"] == "framing/coop"


def test_strict_undefined(root):
    with pytest.raises(jinja2.UndefinedError):
        PromptSet("agent/t", root=root).render("system.j2", cot_mode="none")


def test_render_deterministic_and_prompt_hash(root):
    ps = PromptSet("agent/t", root=root)

    def msgs(agent: str) -> list[ChatMessage]:
        return [ChatMessage(role="system", content=ps.render("system.j2", game={"n_agents": 4}, cot_mode="x")),
                ChatMessage(role="user", content=ps.render("turn.j2", view=_view(agent)))]

    assert prompt_hash(msgs("amber_fox")) == prompt_hash(msgs("amber_fox"))
    assert prompt_hash(msgs("amber_fox")) == prompt_hash([m.model_dump() for m in msgs("amber_fox")])
    assert prompt_hash(msgs("blue_owl")) != prompt_hash(msgs("amber_fox"))


def test_truth_objects_rejected(root, fixed_state):
    ps = PromptSet("agent/t", root=root)
    with pytest.raises(TypeError, match="AgentView"):
        ps.render("turn.j2", view=fixed_state)
    with pytest.raises(TypeError, match="AgentView"):
        ps.render("turn.j2", view=_view("a"), extra=[fixed_state.messages[0]])


def test_missing_set():
    with pytest.raises(FileNotFoundError):
        PromptSet("agent/nope")
