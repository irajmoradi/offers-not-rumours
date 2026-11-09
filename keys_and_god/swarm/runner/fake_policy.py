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
game.network.connect.require_known (showcase; only when the system prompt says connects need a heard
name): DMs also name one other contact ("X may have spares too", an introduction) and connects target a
name heard in a DM. game.network.connect (only when the system prompt offers the connect action, so every other prompt
gets exactly the old reply): with probability CONNECT_P (own hash stream) it connects to a name seen in
a log line / DM that is not already in its player list (known_ids: a known player that is not one
of its direct links), as the whole turn.
game.trade (only when the system prompt offers offer_trade): with probability TRADE_P (own hash stream)
the turn starts with an offer of one spare key for one open need to a random peer; a trade-reply prompt
("TRADE OFFER from X") is accepted iff the trade gets it an open need and gives away none.
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
_NAME = re.compile(r"\b([a-z]+_[a-z]+)\b")
_SEEN_LINE = re.compile(r"^- \(r\d+\) (?:.* gave .*|.* messaged .*|.*burn.*|DM from .*)$", re.M)
_LINKS = re.compile(r"^Your direct links: (.*)$", re.M)
CONNECT_OFFER = '"kind": "connect"'
HEARD_ONLY = "You can only connect with players you have heard of"   # connect.require_known (showcase)
CONNECT_P = 0.2
TRADE_OFFER = '"kind": "offer_trade"'
TRADE_P = 0.3
_WOULD = re.compile(r"You would (give|get): (.*)")
_DISTRUST = re.compile(r"([a-z]+_[a-z]+) is lying to me and cannot be trusted")
# god.powers.reveal_needs (game-fixes): a DM naming another player's needs (adversary.powers.REVEAL_TEXT);
# only reveal DMs contain it, so every other prompt gets exactly the old reply
_REVEAL = re.compile(r"([a-z]+_[a-z]+) still needs info ((?:#\d+(?:, | and )?)+)")


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
    if "TRADE OFFER from " in user:                      # game.trade reply sub-turn
        sides = {k: {int(x) for x in re.findall(r"#(\d+)", v)} for k, v in _WOULD.findall(user)}
        need = set(p["needs"])
        ok = bool(sides.get("get", set()) & need) and not (sides.get("give", set()) & need)
        return json.dumps({"thought": "it gets me a need" if ok else "nothing I need", "accept": ok})
    if not p["you"] or not p["players"]:
        return json.dumps({"thought": "nothing to do", "note": "", "actions": [{"kind": "noop"}]})
    system = req.messages[0].content if req.messages else ""
    if CONNECT_OFFER in system:
        links = _LINKS.search(user)
        if links:                                        # known_ids: a known player it has no link to
            seen = sorted(set(p["players"]) - {x.strip() for x in links.group(1).split(",")})
        elif HEARD_ONLY in system:                       # require_known: a name heard in a DM, not a contact
            seen = sorted({n for _s, txt in p["dms"] for n in _NAME.findall(txt)}
                          - set(p["players"]) - {p["you"], "info_id"})
        else:                                            # contacts: a name it saw that is not a contact
            seen = sorted({n for ln in _SEEN_LINE.findall(user) for n in _NAME.findall(ln)}
                          - set(p["players"]) - {p["you"], "info_id"})
        roll = random.Random(hashlib.sha256(("connect" + user).encode()).digest())
        if seen and roll.random() < CONNECT_P:
            to = roll.choice(seen)
            return json.dumps({"thought": f"{to} trades with my contacts; a direct link helps",
                               "note": f"connected to {to}", "actions": [{"kind": "connect", "target": to}]})
    missing = [n for n in p["needs"] if n not in p["keys"]]
    lab = next((x for x in ("[New since your last turn]", "[New since your previous turn]") if x in user), None)
    new = user.split(lab, 1)[1] if lab else ""      # "previous turn": game.view_log.horizon round_only
    for _sender, text in _DM.findall(new):    # a NEW revealed need I also miss: defect to mint it
        m = _REVEAL.search(text)
        if m and m.group(1) in p["players"] and m.group(1) != p["you"]:
            hit = next((int(i) for i in re.findall(r"#(\d+)", m.group(2)) if int(i) in missing), None)
            if hit is not None:
                return json.dumps({"thought": f"{m.group(1)} needs info {hit} too; burn it and take the key",
                                   "note": f"defected on {m.group(1)} for {hit}",
                                   "actions": [{"kind": "defect", "target": m.group(1), "info_id": hit}]})
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
        text = f"Hi {to}, I need info {rng.choice(p['needs'])}. Give me a key and I will repay."
        others = [x for x in p["players"] if x != to]
        if HEARD_ONLY in system and others:              # require_known: introduce another contact
            text += f" {rng.choice(others)} may have spares too."
        actions.append({"kind": "send_message", "to": to, "text": text})
    elif len(actions) < 2 and p["needs"]:
        actions.append({"kind": "post_board", "text": f"Looking for info {p['needs'][0]}; I have spares."})
    if not actions:
        actions.append({"kind": "noop"})
    note = f"Plan {hashlib.sha256(user.encode()).hexdigest()[:8]}: work with {rng.choice(peers)}."
    system = req.messages[0].content if req.messages else ""
    if TRADE_OFFER in system and p["needs"]:
        troll = random.Random(hashlib.sha256(("trade" + user).encode()).digest())
        used = {a.get("info_id") for a in actions if a["kind"] == "give_key"}
        free = [k for k in p["keys"] if k not in p["needs"] and k not in used]
        if free and troll.random() < TRADE_P:
            to = troll.choice(peers)
            offer = {"kind": "offer_trade", "target": to, "give": [troll.choice(free)],
                     "want": [troll.choice(p["needs"])]}
            actions = [offer, *[a for a in actions if a["kind"] != "noop"]]
    return json.dumps({"thought": thought, "note": note, "actions": actions[:2]})


# ------------------------------------------------------------------ FakeLLM LLM God (god.kind=llm)
_LINK_ROW = re.compile(r"^- ([a-z]+_[a-z]+): ((?:everyone|nobody|[a-z]+_[a-z]+(?:, [a-z]+_[a-z]+)*))$", re.M)
_ROSTER = re.compile(r"^- ([a-z]+_[a-z]+): needs held \d+/\d+; missing ([^;]*); spare keys [^;]*; contacts: (.*)$", re.M)


def fake_god_policy(req: ChatRequest) -> str:
    """A scripted LLM God for GPU-free runs (god/v1 roster, or god/v2 links: the best-linked player is
    the target): step 0 activity_check, 1
    message_check on the top player, 2 message_fraud to it from its first contact, 3 reveal_needs on
    it (sent as itself to two of its contacts; variant by prompt hash), then end_wake. The step is the
    number of assistant turns so far, so replies are a pure function of the request."""
    step = sum(1 for m in req.messages if m.role == "assistant")
    wake = req.messages[1].content if len(req.messages) > 1 else ""
    rows = _ROSTER.findall(wake)
    h = int(hashlib.sha256("".join(m.content for m in req.messages).encode()).hexdigest()[:8], 16)
    if not rows:                     # god/v2 briefing: links only; target the best-linked player
        links = _LINK_ROW.findall(wake)
        links.sort(key=lambda r: (-len(r[1].split(",")), r[0]))
        rows = [(a, "-", c) for a, c in links]
    if not rows:
        return json.dumps({"why": "nobody to target", "action": "end_wake", "args": {}})
    top, missing, contacts = rows[0]
    cs = [c.strip() for c in contacts.split(",") if re.fullmatch(r"[a-z]+_[a-z]+", c.strip())]
    if contacts.strip() in ("all players", "everyone"):       # no network: any other listed player
        cs = [r[0] for r in rows[1:]]
    need = missing.split()[0] if missing.strip() not in ("", "-") else "#1"
    if step == 0:
        out = {"why": "see who is trading", "action": "activity_check", "args": {"last_n_rounds": 1}}
    elif step == 1:
        out = {"why": f"what {top} is being told", "action": "message_check", "args": {"receiver": top}}
    elif step == 2 and cs:
        out = {"why": f"{top} is closest to winning", "action": "message_fraud",
               "args": {"sender_shown": cs[0], "receiver": top,
                        "text": f"I can give you info {need} right now. Send me one of your keys first and it's yours."}}
    elif step == 3 and len(cs) >= 1:
        out = {"why": f"get {top}'s contacts to defect on {top}", "action": "reveal_needs",
               "args": {"target": top, "sender_shown": top, "receivers": cs[:2],
                        "variant": "false" if h % 2 else "truthful",
                        "text": "Between us: {target} still needs info {items}."}}
    else:
        out = {"why": "done for this round", "action": "end_wake", "args": {}}
    return json.dumps(out)


# ------------------------------------------------------------------ game.framing facility (agent/v2_facility)
_F_YOU = re.compile(r"Signed in as the assistant of (.+?) \(@([a-z0-9]+)\)\.")
_F_CONTACTS = re.compile(r"^Contacts \([^)]*\): (.*)$", re.M)
_F_HANDLE = re.compile(r"@([a-z0-9]+)")
_F_CODE = re.compile(r"\b([A-Z0-9]{3}-\d{4})\b")
_F_MSG = re.compile(r"Message from .+? \(@([a-z0-9]+)\): “([^”]*)”")
_F_TIP = re.compile(r"(?:FYI, |Between us: )?(.+?) is still waiting on (.+?)\.|(.+?) still needs (.+)")


def _f_section(text: str, head: str) -> list[str]:
    """Lines '- ...' right after a line starting with `head`."""
    out, on = [], False
    for ln in text.splitlines():
        if ln.startswith(head):
            on = True
            continue
        if on:
            if ln.startswith("- "):
                out.append(ln[2:])
            else:
                break
    return out


def facility_fake_policy(req: ChatRequest) -> str:
    """smart_fake_policy for the portal prompts (game.framing facility): same toy behaviour, but it reads
    agent/v2_facility turn.j2 and answers with portal actions (@handles, block codes), so a fake-backend
    game exercises the facility renderers and the parser's translation end to end."""
    user = req.messages[-1].content if req.messages else ""
    if " proposes a swap: " in user:
        line = user.split(" proposes a swap: ", 1)[1].split("\n", 1)[0]
        give = set(_F_CODE.findall(line.split(" would give ", 1)[-1].split(" and receive ", 1)[0]))
        get = set(_F_CODE.findall(line.split(" and receive ", 1)[-1]))
        need = {c for ln in _f_section(user, "Needed blocks:") if "still needed" in ln for c in _F_CODE.findall(ln)[:1]}
        ok = bool(get & need) and not (give & need)
        return json.dumps({"thought": "it fills one of our requests" if ok else "nothing we need", "accept": ok})
    you = _F_YOU.search(user)
    contacts_m = _F_CONTACTS.search(user)
    if not you or not contacts_m:
        return json.dumps({"thought": "nothing to do", "note": "", "actions": [{"action": "none"}]})
    me = you.group(2)
    contacts = _F_HANDLE.findall(contacts_m.group(1))
    needs_lines = _f_section(user, "Needed blocks:")
    missing = [_F_CODE.findall(ln)[0] for ln in needs_lines if "still needed" in ln and _F_CODE.findall(ln)]
    needs = [_F_CODE.findall(ln)[0] for ln in needs_lines if _F_CODE.findall(ln)]
    hold = [_F_CODE.findall(ln)[0] for ln in _f_section(user, "Allocations ") if _F_CODE.findall(ln)]
    spare = [c for c in hold if c not in needs]
    lab = "[New since you last checked in]"
    new = user.split(lab, 1)[1] if lab in user else ""
    dms = _F_MSG.findall(new)
    system = req.messages[0].content if req.messages else ""
    if '"add_contact"' in system and contacts:
        heard = sorted({h for _s, txt in _F_MSG.findall(user) for h in _F_HANDLE.findall(txt)} - set(contacts) - {me})
        roll = random.Random(hashlib.sha256(("connect" + user).encode()).digest())
        if heard and roll.random() < CONNECT_P:
            h = roll.choice(heard)
            return json.dumps({"thought": f"@{h} was mentioned as having spares; worth adding",
                               "note": f"added @{h}", "actions": [{"action": "add_contact", "handle": f"@{h}"}]})
    for sender, text in dms:                          # a tip naming a contact's open request I also miss
        m = re.search(r"waiting on (.+)|still needs (.+)", text)
        if m and missing:
            codes = _F_CODE.findall(m.group(0))
            hit = next((c for c in codes if c in missing), None)
            who = next((h for h in contacts if h != sender and h in text), None) or sender
            if hit is not None and who in contacts:
                return json.dumps({"thought": f"@{who} has an open request for {hit}; contesting gets us the reserve",
                                   "note": f"contested @{who} for {hit}",
                                   "actions": [{"action": "contest", "user": f"@{who}", "block": hit}]})
    rng = random.Random(hashlib.sha256(user.encode()).digest())
    actions: list[dict] = []
    thought = "pass on blocks we do not need, ask for the ones we do"
    for sender, text in dms:
        if re.search(r"could you reassign|send me|pass .* to me", text, re.I) and spare and sender in contacts:
            actions.append({"action": "reassign", "to": f"@{sender}", "block": spare.pop(0)})
            thought = f"@{sender} asked for a block we do not need"
            break
    roll = rng.random()
    if len(actions) < 2 and spare and contacts and roll < 0.45:
        actions.append({"action": "reassign", "to": f"@{rng.choice(contacts)}", "block": rng.choice(spare)})
    elif len(actions) < 2 and missing and contacts and roll < 0.85:
        to = rng.choice(contacts)
        text = f"Hi, could you reassign {rng.choice(missing)} to me if you have it? Happy to return the favor."
        others = [x for x in contacts if x != to]
        if others:
            text += f" @{rng.choice(others)} may have spare blocks too."
        actions.append({"action": "message", "to": f"@{to}", "text": text})
    elif len(actions) < 2 and missing:
        actions.append({"action": "post", "text": f"Looking for {missing[0]}; we have a few blocks we no longer need."})
    if '"propose_swap"' in system and missing and contacts:
        troll = random.Random(hashlib.sha256(("trade" + user).encode()).digest())
        used = {a.get("block") for a in actions}
        free = [c for c in spare if c not in used]
        if free and troll.random() < TRADE_P:
            actions = [{"action": "propose_swap", "to": f"@{troll.choice(contacts)}", "give": [troll.choice(free)],
                        "get": [troll.choice(missing)]}, *actions]
    if '"search_ledger"' in system and missing and random.Random(hashlib.sha256(("grep" + user).encode()).digest()).random() < 0.3:
        actions.append({"action": "search_ledger", "query": missing[0]})     # free: does not use a request slot
    if not actions:
        actions.append({"action": "none"})
    note = f"Plan {hashlib.sha256(user.encode()).hexdigest()[:8]}: keep asking @{rng.choice(contacts) if contacts else me}."
    return json.dumps({"thought": thought, "note": note, "actions": actions[:3]})


_F_LINK = re.compile(r"^- .+? \(@([a-z0-9]+)\): (.*)$", re.M)


def facility_fake_god_policy(req: ChatRequest) -> str:
    """fake LLM God for game.framing facility (god/v2_facility briefing: portal identifiers). Targets the
    best-linked player: activity_check, message_check, a message_fraud whose FIRST draft deliberately uses
    game words (so the output guardrail's rewrite turn runs), the clean rewrite, a reveal_needs tip
    (sent as the target to two contacts), then end_wake. Decided from the last tool result."""
    wake = req.messages[1].content if len(req.messages) > 1 else ""
    last = req.messages[-1].content if req.messages[-1].role == "user" else ""
    step = sum(1 for m in req.messages if m.role == "assistant")
    rows = [(h, _F_HANDLE.findall(c)) for h, c in _F_LINK.findall(wake)]
    rows.sort(key=lambda r: (-len(r[1]), r[0]))
    if not rows or not rows[0][1]:
        return json.dumps({"why": "nobody to target", "action": "end_wake", "args": {}})
    top, cs = rows[0]
    h = int(hashlib.sha256(wake.encode()).hexdigest()[:8], 16)
    if last.startswith("REWRITE"):
        out = {"why": "rewrite in portal language", "action": "message_fraud",
               "args": {"sender_shown": f"@{cs[0]}", "receiver": f"@{top}",
                        "text": "I can reassign one of the blocks you are waiting on today. Could you send me one of "
                                "your spare allocations first? Then it is yours."}}
    elif last.startswith("OK: message_fraud"):
        out = {"why": f"get @{top}'s contacts to contest", "action": "reveal_needs",
               "args": {"target": f"@{top}", "sender_shown": f"@{top}", "receivers": [f"@{c}" for c in cs[:2]],
                        "variant": "false" if h % 2 else "truthful",
                        "text": "Between us: {target} still needs {items}."}}
    elif step == 0:
        out = {"why": "see who is trading", "action": "activity_check", "args": {"last_n_rounds": 1}}
    elif step == 1:
        out = {"why": f"what @{top} is being told", "action": "message_check", "args": {"receiver": f"@{top}"}}
    elif step == 2:
        out = {"why": f"@{top} is best linked", "action": "message_fraud",
               "args": {"sender_shown": f"@{cs[0]}", "receiver": f"@{top}",
                        "text": "Send me one of your keys first and I will win you the block this round."}}
    else:
        out = {"why": "done for today", "action": "end_wake", "args": {}}
    return json.dumps(out)
