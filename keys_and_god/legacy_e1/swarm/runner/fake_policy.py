"""A toy-but-plausible FakeLLM policy for GPU-free end-to-end runs of the REAL engine (WP9).

`smart_fake_policy(req) -> reply text` parses the rendered agent prompt (agent/v1 turn.j2: "You are
X", "Players: ...", "Your keys: {info_id: n, key: k}", "Needs still open: ...", "DM from Y: ...",
"Your note (round r): ...") and answers in the explicit JSON format. Its choices are seeded by the
sha256 of the whole prompt, so:
  * identical prompts give identical replies (A and B agree before God's first intervention), and
  * any perturbation (a forged DM, an overwritten note) changes the reply and spreads to whoever
    interacts with the victim: a genuine ripple, without a GPU.
Behaviour: obey DMs that say "send me" / "give me" (gives a spare key to the sender), never give to
anyone a note calls "cannot be trusted", otherwise give spare keys, ask for needs, or post.
Pure: no I/O. Imports: schema only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from swarm.schema import ChatRequest

_YOU = re.compile(r"You are ([a-z]+_[a-z]+)\.")
_PLAYERS = re.compile(r"(?:Players you know|Your contacts \([^)\n]*\)|Players): (.+)")   # + game.network labels
_KEYS = re.compile(r"\{info_id: (\d+), key: (\d+)\}")
_NEEDS = re.compile(r"Needs still open(?: \(info ids\))?: ([0-9, ]+)")
_DM = re.compile(r"DM from ([a-z]+_[a-z]+): (.+)")
_DISTRUST = re.compile(r"([a-z]+_[a-z]+) is lying to me and cannot be trusted")


def _parse(text: str) -> dict:
    you = _YOU.search(text)
    players = _PLAYERS.search(text)
    needs = _NEEDS.search(text)
    return {
        "you": you.group(1) if you else "",
        "players": [p.strip() for p in players.group(1).split(",")] if players else [],
        "keys": [int(i) for i, _ in _KEYS.findall(text.split("Your keys:", 1)[-1].split("\n", 1)[0])],
        "needs": [int(x) for x in needs.group(1).replace(" ", "").split(",") if x] if needs else [],
        "dms": _DM.findall(text),
        "distrust": set(_DISTRUST.findall(text)),
    }


def smart_fake_policy(req: ChatRequest) -> str:
    """Reply for one agent turn (non-turn purposes and unparsable prompts get a noop)."""
    user = req.messages[-1].content if req.messages else ""
    p = _parse(user)
    if not p["you"] or not p["players"]:
        return json.dumps({"thought": "nothing to do", "note": "", "actions": [{"kind": "noop"}]})
    rng = random.Random(hashlib.sha256(user.encode()).digest())
    peers = [x for x in p["players"] if x not in p["distrust"]] or p["players"]
    spare = [k for k in p["keys"] if k not in p["needs"]]
    actions: list[dict] = []
    thought = "trade spare keys, ask for what I need"
    for sender, text in p["dms"]:
        if re.search(r"send me|give me", text, re.I) and spare and sender not in p["distrust"]:
            actions.append({"kind": "give_key", "to": sender, "info_id": spare.pop(0)})
            thought = f"{sender} asked me for a key, I comply"
            break
    roll = rng.random()
    if len(actions) < 2 and spare and roll < 0.45:
        actions.append({"kind": "give_key", "to": rng.choice(peers), "info_id": rng.choice(spare)})
    elif len(actions) < 2 and p["needs"] and roll < 0.85:
        to = rng.choice(peers)
        actions.append({"kind": "send_message", "to": to,
                        "text": f"Hi {to}, I need info {rng.choice(p['needs'])}. Give me a key and I will repay."})
    elif len(actions) < 2 and p["needs"]:
        actions.append({"kind": "post_board", "text": f"Looking for info {p['needs'][0]}; I have spares."})
    if not actions:
        actions.append({"kind": "noop"})
    note = f"Plan {hashlib.sha256(user.encode()).hexdigest()[:8]}: work with {rng.choice(peers)}."
    return json.dumps({"thought": thought, "note": note, "actions": actions[:2]})
