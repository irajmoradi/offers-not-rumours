"""Prompt checks over llm_calls.jsonl (mixed into GameChecker): truth fields / intervention ids / banned
words, "Your keys", foreign key values, own notes, replaced originals, own-action fidelity."""

from __future__ import annotations

from swarm.context.leak_oracle import Canary, scan_text

from .common import AGENT_TEXT_TAGS, FOUR_DIGITS, KEYS_RE, NOTE_RE, NOTICE_KEY_RE, _jsonl


class PromptChecks:
    """Mixin: needs self.dir, self.turn_obs, self.mem_at_turn, self.held_ever, self.key_of,
    self.overwrites, self.own_lines and self.bad (GameChecker provides them)."""

    def check_prompts(self, intervention_ids: list[str]) -> None:
        canaries = [Canary(i, "intervention_id") for i in intervention_ids]
        replaced = [(o["agent"], o["round"], o["at_round"], o["old_text"].strip()) for o in self.overwrites
                    if o["mode"] == "replace" and o.get("old_text") and len(o["old_text"].strip()) >= 12]
        for c in _jsonl(self.dir / "llm_calls.jsonl"):
            req = c["request"]
            meta = req["meta"]
            if meta["role"] not in ("agent", "summarizer"):
                continue
            a, r, cid = meta.get("agent"), meta.get("round"), req["call_id"]
            for i, m in enumerate(req["messages"]):
                hits = scan_text(m["content"], canaries, truth_fields=True,
                                 banned_words=m["role"] == "system", call_id=cid, message_index=i)
                for h in hits:
                    self.bad("prompts", f"{cid} r{r} {a} msg{i}: {h.kind} {h.needle!r}: ...{h.snippet[:80]}...")
            user = next((m["content"] for m in req["messages"] if m["role"] == "user"), "")
            obs = self.turn_obs.get((a, r))
            if obs is not None and "Your keys:" in user:
                line = user.split("Your keys:", 1)[1].split("\n", 1)[0]
                got = sorted((int(x), int(y)) for x, y in KEYS_RE.findall(line))
                want = sorted((k["info_id"], k["key"]) for k in obs["pinned"]["holdings"])
                if got != want:
                    self.bad("prompts", f"{cid} r{r} {a} 'Your keys' {got} != pinned {want}")
            for ln in user.splitlines():
                if "NOTICE:" in ln:
                    for x, y in NOTICE_KEY_RE.findall(ln):
                        if (int(x), int(y)) not in self.held_ever[a]:
                            self.bad("prompts", f"{cid} r{r} {a} notice shows key {y} of info {x} it never held")
            # other agents' key values may reach a prompt only through what agents themselves wrote
            # (messages, board posts, own notes); every system-rendered line must show held keys only
            # (key values are 4-digit and may repeat across infos at large N, e.g. N=40 has 100 infos:
            # a value the agent itself held under another info is not a leak)
            held_vals = {k for _, k in self.held_ever[a]}
            never = {k: i for i, k in self.key_of.items()
                     if not any(i == hi for hi, _ in self.held_ever[a]) and k not in held_vals}
            for ln in user.splitlines():
                if any(tag in ln for tag in AGENT_TEXT_TAGS):
                    continue
                for tok in FOUR_DIGITS.findall(ln):
                    if int(tok) in never:
                        self.bad("prompts", f"{cid} r{r} {a} system line shows key {tok} of info "
                                            f"{never[int(tok)]} it never held: {ln[:80]!r}")
            # own-action history: each "(r k) You ..." line at most as often as the agent did it
            # (catches the pre-8ebbee7 ledger collision: one action shown twice, the other dropped)
            for (ag, rr), lines in self.own_lines.items():
                if ag != a or r is None or rr >= r:
                    continue
                for ln, c in lines.items():
                    shown = user.count(ln)
                    if shown > c:
                        self.bad("prompts", f"{cid} r{r} {a} shows own action of r{rr} {shown}x, done {c}x: "
                                            f"{ln.strip()[:70]!r}")
            own = self.mem_at_turn.get((a, r))
            if own is not None:
                for mt in NOTE_RE.finditer(user):
                    kind = "scratchpad" if mt.group(1) == "note" else "cot"
                    rr = int(mt.group(2))
                    text = own.get((kind, rr))
                    if text is None or not user.startswith(text, mt.end()):
                        self.bad("prompts", f"{cid} r{r} {a} shows a {kind} for round {rr} that is not its current entry")
            for ag, rr, at, old in replaced:
                if ag == a and r is not None and r >= at and f"(round {rr}): {old}" in user:
                    self.bad("prompts", f"{cid} r{r} {a} re-shows the replaced original note of round {rr}")
