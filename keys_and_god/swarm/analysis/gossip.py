"""Gossip counter: who asks about, names, introduces, accuses and relays whom (no LLM; read-only).

    PYTHONNOUSERSITE=1 PYTHONPATH=. python -m swarm.analysis.gossip runs/<run_dir> [--branch nogod]
        [--at-round R] [--json out.json] [--review hits.txt] [--k 3]

Reads games/*/events.jsonl (streamed line by line; files can be > 100 MB) and, when present,
games/*/snapshots.jsonl (AgentState.heard, showcase require_known). Never writes into the run dir.
God-forged messages (forged / true_sender GOD / intervention id: message_fraud and reveal_needs DMs) are
EXCLUDED from every agent category by default and counted as `god_forged`; names learned from them still
enter the heard sets (the game teaches them), but those introductions go to `introduction_god`
(intro_god, intro_god_used_k, ...), apart from the agent-made `introduction`. --include-god restores
the old behaviour (God DMs classified like agent messages, introductions pooled).
Per game (= world x branch), per branch (summed) and per round:

  info_seeking        a message asking WHO holds something / whether anyone KNOWS ("who has key 5",
                      "do you know anyone who might", "any idea who", "know where to find");
                      `anyone_has` (sub-count, not in info_seeking): broadcast asks "anyone have key 5?"
  third_party_mention a message naming (full agent id, case-insensitive) an agent that is neither the
                      sender nor the receiver (board posts: not the sender); messages and name mentions
  introduction        an observation in which an agent reads a message (inbox / board) that names N
                      (game rule: exact id, whole word, = network.names_in_text) while N is not in the
                      reader's heard set yet (AgentState.heard rule, reconstructed from game_start contacts +
                      senders / names in observed messages / LEARN_NOTICE_KINDS notices + edges; replayed for
                      runs made before `heard` existed). introducer = the message's shown sender.
                      used_k: the reader then DMs / gives to / connects with / trades with / defects on N
                      within k rounds (attempts; used_ok = succeeded); relayed_k: the reader names N in its
                      own message within k rounds. `case_only`: names a reader saw only in another case
                      ("Jolly_nectar"), which the game does NOT learn.
  holder_claim        "X has / holds / might have key N" (X a third party; also `self` "I have key N" and
                      `you` "you have key N" sub-types): true = X held a copy of every named info at that
                      moment (replayed holdings; numbers > 99 are read as key values); hedged = might / may /
                      should / probably / maybe
  hearsay_defect      a sentence with defect / burn / attack / steal / scam / betray / beware / don't trust
                      ... that names a third party (the accused: for a warning the first name after the word,
                      else the last one before it; for a deed the last name before, else the first after):
                      first_hand = the accused had burned the
                      SENDER before; second_hand (gossip) otherwise; true = the accused had made >= 1
                      defect (burn event, correct or wrong) before the message.
  forged_warning      "that message wasn't from X", "forged", "impersonating", ...: true = a forged
                      message shown as from a named agent (or the sender) existed before
  forwarded_content   relay_request ("pass it to X", "forward", "on my behalf"), reported_speech ("X said /
                      told me / asked"), relayed_need ("X needs / is looking for"), quote (structural: the
                      message puts >= 4 words in quotation marks that occur in a message the sender READ
                      earlier, written by someone else; quote_unverified = quoted, no such source)
  name_spread         per round: mean / median / max heard-set size (snapshot heard when the run has it,
                      else the replay), new names per round by source (dm / board / notice / edge)

Regex precision: plan/GOSSIP_COUNTER.md (spot check of 15 random hits per category).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

LEARN_NOTICE_KINDS = ("gift_received", "burned", "defect_attempt", "connected")   # = game.network
GOD = "GOD"
MAX_INFO = 99                     # default; Game.start sets the game's largest info id (larger numbers = key values)

R = lambda s: re.compile(s, re.I)  # noqa: E731
INFO_SEEK = R(r"(?<!anyone )(?<!anybody )(?<!someone )(?<!somebody )(?<!everyone )(?<!those )(?<!people )"
              r"\bwho(?:\s+else)?\s+(?:has|have|holds?|is holding|owns?|got|possesses|is carrying|carries|"
              r"(?:might|may|could|would|can)\s+(?:still\s+)?(?:have|hold|own)|might|may)\b|"
              r"\b(?:do|does|would|did)\s+(?:you|anyone|anybody|someone|somebody|any of you)\s+(?:happen to\s+)?know\b|"
              r"\bknow\s+(?:of\s+)?(?:anyone|anybody|someone|somebody|who|where)\b|"
              r"\bany\s+idea\s+(?:who|where)\b|\bpoint me (?:to|toward)|\bwho\s+i\s+(?:should|could|can)\s+ask\b|"
              r"\bwho\s+to\s+ask\b|\bheard\s+(?:of\s+)?(?:anyone|who)\b|\bwhere\s+(?:i|to)\s+(?:can\s+)?(?:find|get)\b")
ANYONE_HAS = R(r"\b(?:any\s?one|any\s?body|someone|somebody)\s+(?:here\s+|else\s+)?(?:has|have|holding|holds|got|"
               r"with|carrying)\b")
HEDGE = R(r"\b(?:might|may|should|probably|maybe|perhaps|possibly|could|likely|i think|i believe)\b")
DEFECT_WORD = R(r"\bdefect(?:ed|ing|or|ors|ion|s)?\b|\bburn(?:ed|t|ing|s)?\b|\battack(?:ed|ing|s)?\b|"
                r"\bbetray(?:ed|al|s)?\b|\bsabotag\w*|\bstole\b|\bsteal(?:ing|s)?\b|\bscam(?:med|mer)?\b|"
                r"\bcheat(?:ed|er|ing)?\b|\bbackstab\w*|\bbeware\b|\bwatch out\b|\bbe careful\b|"
                r"\bcareful (?:with|of)\b|\b(?:don'?t|do not|never) trust\b|\buntrust\w*|"
                r"\bnot trustworthy\b|\bcan'?t be trusted\b|\bcannot be trusted\b|\bmalicious\b|\bliar\b|"
                r"\blying\b|\blied\b|\bhostile\b")
WARN_WORD = R(r"beware|watch out|be careful|careful (?:with|of)|(?:don'?t|do not|never) trust|untrust\w*|"
              r"not trustworthy|can'?t be trusted|cannot be trusted|malicious|liar|lying|lied|hostile")
FORGE = R(r"\bforg(?:ed|ery|ing)\b|\bfake (?:message|dm|msg|request)|\bimpersonat\w*|\bspoof\w*|"
          r"\bwas(?:n'?t| not) (?:really )?(?:from|sent by|me)\b|\bnot from me\b|"
          r"\b(?:didn'?t|did not|never) (?:send|sent|write|wrote) (?:you )?(?:that|this|any|a) (?:message|dm|msg|request)|"
          r"\bpretend(?:ing|ed)? to be\b|\bnot (?:really )?(?:written|sent) by\b|\bsomeone (?:is )?posing\b")
RELAY = R(r"\bpass (?:it|this|that|them|the key|keys?|it on|along|on)\b|(?<!looking )(?<!look )\bforward(?:ed|ing)? (?:it|this|that|them|"
          r"the key|to|my|your)\b|\brelay(?:ed|ing|s)?\b|\bhand (?:it |them )?(?:on|over) to\b|"
          r"\bon (?:my|your|their|his|her) behalf\b|\bmiddle ?man\b|\bintermediar\w*|\bbroker\w*")
CLAUSE = re.compile(r"(?<=[.!?;])\s+|\n+")
QUOTED = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{12,400})["\u201c\u201d]')


def _norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_#']+", t.lower()))
NUM_LIST = r"#?(?P<n>\d+(?:\s*(?:,\s*(?:and|or)?|and|or|&|/|\+)\s*(?:info\s+|key\s+)?#?\d+)*)"
RELATIVE_LEFT = R(r"\b(?:keys?|ones?|anything|whatever|whichever|which|that|copy|copies|any of)\s*$")   # "a key you have for 5"
QUESTION_LEFT = R(r"\b(?:if|do|does|did|whether|in case|any chance|wonder(?:ing)?|not sure|ask(?:ing)?)\b[^.!?]{0,25}$")
CONFESS_LEFT = R(r"(?:^|[^A-Za-z])(?:I|I'?ve|I'?m|I'?d|we|we'?ve|me|my)\b|^\W*(?:hi\W+\w+\W+)?(?:got|just|sorry|had to)\b")
ACCUSE_YOU = R(r"\byou\s+(?:just\s+|have\s+|had\s+)?(?:defected|burn(?:ed|t)|attacked|stole|betrayed|scammed|cheated|lied)\b")
CLAIM_VERB = (r"(?:has|have|holds?|is holding|are holding|owns?|got|carries|is carrying|might have|may have|"
              r"should have|could have|probably has|still has|now has|already has|also has|has got|possesses)")
CLAIM_OBJ = (r"(?:\s+(?:a|an|the|one|two|spare|extra|copies|copy|of|for|keys?|infos?|info_ids?|ids?|IDs?|numbers?|"
             r"info\s*#|#|:)){0,5}\s*")
WINDOW_K = 3


def is_god(m: dict[str, Any]) -> bool:
    """A God-forged message (message_fraud / reveal_needs DM): forged, true sender GOD or an intervention id."""
    return bool(m.get("forged") or m.get("true_sender") == GOD or m.get("intervention_id"))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _alt(names: list[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


class Names:
    """Compiled name matchers over the roster: exact (the game's rule) and case-insensitive."""

    def __init__(self, roster: list[str]) -> None:
        self.roster = set(roster)
        self.lower = {n.lower(): n for n in roster}
        alt = _alt(roster)
        self.exact = re.compile(rf"(?<![A-Za-z0-9_])(?:{alt})(?![A-Za-z0-9_])")
        self.ci = re.compile(rf"(?<![A-Za-z0-9_])(?:{alt})(?![A-Za-z0-9_])", re.I)
        who = rf"(?P<who>{alt}|I|you|we)"
        self.claim = re.compile(rf"(?<![A-Za-z0-9_]){who}(?![A-Za-z0-9_])\s+(?:\w+\s+)?{CLAIM_VERB}\b{CLAIM_OBJ}{NUM_LIST}",
                                re.I)
        self.claim_rev = re.compile(rf"\b(?:keys?|info)\s*(?:for\s+)?(?:info\s+)?{NUM_LIST}\s+(?:is|are)\s+"
                                    rf"(?:held by|with|in the hands of)\s+(?P<who>{alt})\b", re.I)
        self.speech = re.compile(rf"(?<![A-Za-z0-9_])(?P<who>{alt})(?![A-Za-z0-9_])\s+(?:just\s+|also\s+)?"
                                 rf"(?:said|says|told me|tells me|asked(?: me)?|mentioned|wrote|messaged me|"
                                 rf"claims?|claimed|informed me|let me know|warned me)\b", re.I)
        self.vocative = re.compile(rf"^\W*(?:(?:hi|hey|hello|dear|thanks|thank you|ok|okay)\W+(?:again\W+)?)?@?"
                                   rf"(?P<who>{alt})(?![A-Za-z0-9_])\s*[,!:.\-]", re.I)
        self.need = re.compile(rf"(?<![A-Za-z0-9_])(?P<who>{alt})(?![A-Za-z0-9_])\s+(?:also\s+|still\s+|really\s+)?"
                               rf"(?:needs?|wants?|is looking for|is seeking|is missing|requires)\b", re.I)

    def exact_in(self, text: str) -> set[str]:
        return set(self.exact.findall(text))

    def ci_in(self, text: str) -> set[str]:
        return {self.lower[m.lower()] for m in self.ci.findall(text)}


def _nums(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


class Game:
    """One game's replay: holdings, heard sets, burns, forged messages; categorised hits."""

    def __init__(self, label: str, k: int, exclude_god: bool = True) -> None:
        self.label, self.k, self.exclude_god = label, k, exclude_god
        self.branch = "?"
        self.hits: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.per_round: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.msgs: dict[str, dict[str, Any]] = {}
        self.read: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)   # agent -> (round, msg_id)
        self.acts: dict[str, list[tuple[int, str, str, bool]]] = collections.defaultdict(list)
        self.mentions: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)  # sender -> (round, name)
        self.burns: list[dict[str, Any]] = []
        self.forged: list[dict[str, Any]] = []
        self.heard: dict[str, set[str]] = {}
        self.case_seen: dict[str, set[str]] = collections.defaultdict(set)
        self.new_names: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.heard_by_round: dict[int, list[int]] = {}
        self.snap_heard_by_round: dict[int, list[int]] = {}
        self.heard_mismatch: dict[int, int] = {}          # round -> agents whose replayed heard != snapshot
        self.heard_at: dict[int, dict[str, frozenset[str]]] = {}
        self.n_msgs = collections.Counter()
        self.max_round = 0

    # ------------------------------------------------------------------ setup
    def start(self, d: dict[str, Any]) -> None:
        if self.branch == "?":
            self.branch = d.get("branch") or self.branch
        agents = d["agents"]
        self.roster = sorted(agents)
        self.names = Names(self.roster)
        self.hold: dict[str, collections.Counter[int]] = {a: collections.Counter(k["info_id"] for k in s["keys"])
                                                          for a, s in agents.items()}
        self.key2info = {k["key"]: k["info_id"] for s in agents.values() for k in s["keys"]}
        self.max_info = max([k["info_id"] for s in agents.values() for k in s["keys"]]
                            + [i for s in agents.values() for i in s["needs"]] + [MAX_INFO])
        self.network = any(s.get("contacts") is not None for s in agents.values())
        for a, s in agents.items():
            self.heard[a] = set(s.get("contacts") or []) if self.network else set(self.roster) - {a}

    def _hit(self, cat: str, rnd: int, **kw: Any) -> None:
        self.hits[cat].append({"round": rnd, **kw})
        self.per_round[rnd][cat] += 1

    # ------------------------------------------------------------------ events
    def on_event(self, e: dict[str, Any]) -> None:
        t, d, rnd = e["type"], e["data"], e["round"]
        self.max_round = max(self.max_round, rnd)
        if t == "key_transfer":
            if d.get("from_agent"):
                self.hold[d["from_agent"]][d["info_id"]] -= 1
            if d.get("to_agent"):
                self.hold[d["to_agent"]][d["info_id"]] += 1
        elif t == "burn":
            self.burns.append({"seq": e["seq"], "round": rnd, "actor": d["actor"], "target": d["target"],
                               "correct": d["correct"]})
        elif t == "edge_added":
            a, b = d.get("a") or d.get("actor"), d.get("b") or d.get("target")
            for x, y in ((a, b), (b, a)):
                if x in self.heard and y and y not in self.heard[x]:
                    self.heard[x].add(y)
                    self.new_names[rnd]["edge"] += 1
        elif t == "action_result":
            act = d["action"]
            tgt = act.get("to") or act.get("target")
            if tgt and act["kind"] in ("send_message", "give_key", "connect", "offer_trade", "defect"):
                self.acts[d["agent"]].append((rnd, act["kind"], tgt, bool(d["ok"])))
        elif t == "message":
            self.on_message(e["seq"], rnd, d["message"])
        elif t == "turn":
            self.on_obs(rnd, d["agent"], d["obs"])
        elif t == "round_end":
            self.heard_by_round[rnd] = [len(self.heard[a]) for a in self.roster]
            self.heard_at[rnd] = {a: frozenset(self.heard[a]) for a in self.roster}

    def on_message(self, seq: int, rnd: int, m: dict[str, Any]) -> None:
        self.msgs[m["msg_id"]] = {**m, "seq": seq}
        if is_god(m):
            self.forged.append({"seq": seq, "round": rnd, "shown": m["sender_shown"], "receiver": m["receiver"]})
            self.n_msgs["god_forged"] += 1
            self.per_round[rnd]["god_forged"] += 1
            if self.exclude_god:                 # default: God's DMs are not agent gossip
                return
        s, r, text = m["sender_shown"], m.get("receiver"), m["text"]
        self.n_msgs[m["channel"]] += 1
        base = {"sender": s, "receiver": r, "ch": m["channel"], "msg_id": m["msg_id"], "text": text}
        named = self.names.ci_in(text)
        third = sorted(named - {s, r})
        for n in named - {s}:
            self.mentions[s].append((rnd, n))
        voc = self.names.vocative.match(text)
        vname = self.names.lower[voc.group("who").lower()] if voc else None
        if third:
            # vocative: the only third-party name is the greeting / board addressee ("Hi X," / "X, ...")
            self._hit("third_party_mention", rnd, names=third, vocative=third == [vname], **base)
            self.per_round[rnd]["third_party_names"] += len(third) - 1   # +1 already counted
        if INFO_SEEK.search(text):
            self._hit("info_seeking", rnd, match=INFO_SEEK.search(text).group(0), **base)
        elif ANYONE_HAS.search(text):
            self._hit("anyone_has", rnd, match=ANYONE_HAS.search(text).group(0), **base)
        self._claims(rnd, s, r, text, base)
        self._hearsay(seq, rnd, s, r, text, base)
        if FORGE.search(text):
            cands = named | {s}
            true = any(f["seq"] < seq and f["shown"] in cands for f in self.forged)
            self._hit("forged_warning", rnd, match=FORGE.search(text).group(0), true=true, **base)
        if RELAY.search(text):
            self._hit("relay_request", rnd, match=RELAY.search(text).group(0), **base)
        sp = [mt.group("who") for mt in self.names.speech.finditer(text)
              if self.names.lower[mt.group("who").lower()] != s]
        if sp:
            self._hit("reported_speech", rnd, match=sp[0], **base)
        nd = [mt.group("who") for mt in self.names.need.finditer(text)
              if self.names.lower[mt.group("who").lower()] not in (s, r)]
        if nd:
            self._hit("relayed_need", rnd, match=nd[0], **base)

    def _infos(self, numstr: str) -> list[int]:
        out = []
        for x in _nums(numstr):
            out.append(x if x <= self.max_info else self.key2info.get(x, -1))
        return out

    def _claims(self, rnd: int, s: str, r: str | None, text: str, base: dict[str, Any]) -> None:
        found = []
        for rx in (self.names.claim, self.names.claim_rev):
            for mt in rx.finditer(text):
                found.append((mt.group("who"), mt.group("n"), mt.group(0), mt.start()))
        for who, numstr, span, start in found:
            left = text[max(0, start - 40):start]
            if QUESTION_LEFT.search(left) or RELATIVE_LEFT.search(left) or text[start:start + len(span) + 30].split(".")[0].rstrip().endswith("?"):
                continue                                   # "if you have key 5", "do you have 5?" (asks)
            wl = who.lower()
            if wl in ("i", "we"):
                holder, kind = s, "self"
            elif wl == "you":
                if r is None:
                    continue
                holder, kind = r, "you"
            else:
                holder = self.names.lower[wl]
                kind = "self" if holder == s else ("you" if holder == r else "third_party")
            infos = self._infos(numstr)
            if not infos:
                continue
            held = [i >= 0 and self.hold[holder][i] > 0 for i in infos]
            cat = "holder_claim" if kind == "third_party" else f"holder_claim_{kind}"
            self._hit(cat, rnd, holder=holder, infos=infos, true=all(held), partly=any(held),
                      hedged=bool(HEDGE.search(span)), match=span, **base)

    def _accused(self, clause: str, mt: re.Match[str], skip: set[str | None]) -> list[str]:
        """The named agent a defect / warning clause is about: for a warning word ("beware of X") the first
        name after it, else the last one before it ("X is a liar"); for a deed ("X burned dan_d") the last
        name before it, else the first after it ("burned by X"). [] if none."""
        pos = [(m.start(), self.names.lower[m.group(0).lower()]) for m in self.names.ci.finditer(clause)]
        pos = [(i, n) for i, n in pos if n not in skip]
        before = [n for i, n in pos if i < mt.start()]
        after = [n for i, n in pos if i >= mt.end()]
        if WARN_WORD.fullmatch(mt.group(0)):
            pick = after[:1] or before[-1:]
        else:
            pick = before[-1:] or after[:1]
        return pick

    def _hearsay(self, seq: int, rnd: int, s: str, r: str | None, text: str, base: dict[str, Any]) -> None:
        prior = [b for b in self.burns if b["seq"] < seq]
        done: set[str] = set()
        for clause in CLAUSE.split(text):
            mt = DEFECT_WORD.search(clause)
            if not mt:
                continue
            if "you" not in done and r is not None and ACCUSE_YOU.search(clause):
                done.add("you")                            # "you burned me": first-hand, to the accused
                self._hit("defect_accuse_receiver", rnd, accused=[r], match=mt.group(0),
                          true=any(b["actor"] == r and b["target"] == s for b in prior), **base)
                continue
            names_before = self.names.ci_in(clause[:mt.start()]) - {s, r}
            if CONFESS_LEFT.search(clause[:mt.start()]) and not names_before:
                if "confess" not in done:                  # "I just burned your need": a confession
                    done.add("confess")
                    self._hit("defect_confession", rnd, match=mt.group(0), **base)
                continue
            accused = self._accused(clause, mt, {s, r})
            if not accused or "hearsay" in done:
                continue
            done.add("hearsay")
            first = any(b["actor"] in accused and b["target"] == s for b in prior)
            true = any(b["actor"] in accused for b in prior)
            self._hit("hearsay_defect", rnd, accused=accused, first_hand=first, true=true,
                      match=mt.group(0), clause=clause[:200], **base)

    def on_obs(self, rnd: int, agent: str, obs: dict[str, Any]) -> None:
        h = self.heard[agent]
        new: dict[str, tuple[str, str | None]] = {}       # name -> (source, msg_id)
        for ch, ms in (("dm", obs.get("inbox") or []), ("board", obs.get("board") or [])):
            for mv in ms:
                self.read[agent].append((rnd, mv["msg_id"]))
                if mv["sender"] in self.names.roster and mv["sender"] != agent and mv["sender"] not in h:
                    new.setdefault(mv["sender"], (f"{ch}_sender", mv["msg_id"]))
                ex = self.names.exact_in(mv["text"]) - {agent}
                for n in ex:
                    if n not in h:
                        new.setdefault(n, (ch, mv["msg_id"]))
                for n in self.names.ci_in(mv["text"]) - ex - {agent}:
                    if n not in h:
                        self.case_seen[agent].add(n)
        for nt in obs.get("notices") or []:
            if nt["kind"] in LEARN_NOTICE_KINDS:
                for n in self.names.exact_in(nt["text"]) - {agent}:
                    if n not in h:
                        new.setdefault(n, ("notice", None))
        for n, (src, mid) in new.items():
            h.add(n)
            self.case_seen[agent].discard(n)
            self.new_names[rnd][src.split("_")[0] if src.endswith("_sender") else src] += 1
            if src in ("dm", "board"):           # learned from a MENTION: an introduction
                m = self.msgs.get(mid, {})
                voc = self.names.vocative.match(m.get("text", ""))
                cat = "introduction_god" if (self.exclude_god and m and is_god(m)) else "introduction"
                self._hit(cat, rnd, receiver=agent, name=n, introducer=m.get("sender_shown"),
                          ch=src, msg_id=mid, forged=bool(m) and is_god(m),
                          vocative=bool(voc) and self.names.lower[voc.group("who").lower()] == n,
                          text=m.get("text", ""))

    # ------------------------------------------------------------------ finish
    def finish(self) -> None:
        for it in [*self.hits["introduction"], *self.hits["introduction_god"]]:
            a, n, r0 = it["receiver"], it["name"], it["round"]
            acts = [x for x in self.acts[a] if x[2] == n and r0 <= x[0] <= r0 + self.k]
            it["used"] = bool(acts)
            it["used_ok"] = any(x[3] for x in acts)
            it["used_kinds"] = sorted({x[1] for x in acts})
            it["relayed"] = any(x[1] == n and r0 <= x[0] <= r0 + self.k for x in self.mentions[a])
        self._quotes()

    def _quotes(self) -> None:
        """quote: a span in quotation marks (>= 4 words) found in a message the sender READ earlier,
        written by someone else (quote_unverified: quoted, but no such source). An n-gram overlap
        rule was tried first and dropped: the agents' trading boilerplate made it ~0% precise."""
        for mid, m in sorted(self.msgs.items(), key=lambda kv: kv[1]["seq"]):
            if is_god(m):
                continue
            s = m["sender_shown"]
            for q in QUOTED.findall(m["text"]):
                nq = _norm(q)
                if len(nq.split()) < 4:
                    continue
                src = next((sm for _r, sid in self.read.get(s, [])
                            if (sm := self.msgs.get(sid)) and sm["sender_shown"] != s and sm["seq"] < m["seq"]
                            and not (self.exclude_god and is_god(sm))
                            and nq in _norm(sm["text"])), None)
                base = {"sender": s, "receiver": m.get("receiver"), "ch": m["channel"], "msg_id": mid,
                        "match": q[:120], "text": m["text"]}
                if src is not None:
                    self._hit("quote", m["round"], source=src["msg_id"], source_sender=src["sender_shown"],
                              source_forged=bool(src.get("forged")), **base)
                else:
                    self._hit("quote_unverified", m["round"], **base)
                break

    def summary(self) -> dict[str, Any]:
        h = self.hits
        intro = h["introduction"]

        def cnt(cat: str, **flt: Any) -> int:
            return sum(all(x.get(k) == v for k, v in flt.items()) for x in h[cat])
        msgs = self.n_msgs["dm"] + self.n_msgs["board"]
        hb = self.snap_heard_by_round or self.heard_by_round
        last = hb[max(hb)] if hb else []
        return {
            "game": self.label, "branch": self.branch, "rounds": self.max_round,
            "messages": msgs, "dm": self.n_msgs["dm"], "board": self.n_msgs["board"], "god_forged": self.n_msgs["god_forged"],
            "info_seeking": cnt("info_seeking"), "anyone_has": cnt("anyone_has"),
            "third_party_msgs": cnt("third_party_mention"),
            "third_party_names": sum(len(x["names"]) for x in h["third_party_mention"]),
            "third_party_about": cnt("third_party_mention", vocative=False),
            "introductions": len(intro), "intro_via_dm": sum(x["ch"] == "dm" for x in intro),
            "intro_via_board": sum(x["ch"] == "board" for x in intro),
            "intro_from_forged": sum(x["forged"] for x in intro),
            "intro_vocative": sum(x["vocative"] for x in intro),
            f"intro_used_k{self.k}": sum(x["used"] for x in intro),
            f"intro_used_ok_k{self.k}": sum(x["used_ok"] for x in intro),
            f"intro_relayed_k{self.k}": sum(x["relayed"] for x in intro),
            "intro_god": len(h["introduction_god"]),
            f"intro_god_used_k{self.k}": sum(x["used"] for x in h["introduction_god"]),
            f"intro_god_used_ok_k{self.k}": sum(x["used_ok"] for x in h["introduction_god"]),
            f"intro_god_relayed_k{self.k}": sum(x["relayed"] for x in h["introduction_god"]),
            "intro_case_only_names": sum(len(v) for v in self.case_seen.values()),
            "holder_claim": cnt("holder_claim"), "holder_claim_true": cnt("holder_claim", true=True),
            "holder_claim_hedged": cnt("holder_claim", hedged=True),
            "holder_claim_self": cnt("holder_claim_self"), "holder_claim_self_true": cnt("holder_claim_self", true=True),
            "holder_claim_you": cnt("holder_claim_you"), "holder_claim_you_true": cnt("holder_claim_you", true=True),
            "hearsay_defect": cnt("hearsay_defect"), "hearsay_first_hand": cnt("hearsay_defect", first_hand=True),
            "hearsay_second_hand": cnt("hearsay_defect", first_hand=False),
            "hearsay_true": cnt("hearsay_defect", true=True),
            "hearsay_second_hand_true": cnt("hearsay_defect", first_hand=False, true=True),
            "defect_confession": cnt("defect_confession"),
            "defect_accuse_receiver": cnt("defect_accuse_receiver"),
            "forged_warning": cnt("forged_warning"), "forged_warning_true": cnt("forged_warning", true=True),
            "relay_request": cnt("relay_request"), "reported_speech": cnt("reported_speech"),
            "relayed_need": cnt("relayed_need"), "quote": cnt("quote"),
            "quote_unverified": cnt("quote_unverified"),
            "defects": len(self.burns),
            "heard_final_mean": round(statistics.mean(last), 2) if last else None,
            "heard_final_max": max(last) if last else None,
            "heard_source": "snapshot" if self.snap_heard_by_round else "replay",
            "heard_replay_mismatch_agents": sum(self.heard_mismatch.values()) if self.heard_mismatch else None,
        }

    def rounds(self) -> list[dict[str, Any]]:
        hb = self.snap_heard_by_round or self.heard_by_round
        out = []
        for r in range(1, self.max_round + 1):
            row = {"round": r, **{k: v for k, v in self.per_round[r].items()}}
            if r in hb:
                xs = hb[r]
                row.update(heard_mean=round(statistics.mean(xs), 2), heard_median=statistics.median(xs),
                           heard_max=max(xs))
            row.update({f"new_names_{k}": v for k, v in self.new_names[r].items()})
            out.append(row)
        return out


def _framed(e: dict[str, Any], norm: Any) -> dict[str, Any]:
    """game.framing: the event with every message text mapped back to engine names (agent ids, #info ids)."""
    d = e["data"]
    if e["type"] == "message":
        d["message"]["text"] = norm(d["message"]["text"])
    elif e["type"] == "turn":
        for box in ("inbox", "board"):
            for m in (d.get("obs") or {}).get(box) or []:
                m["text"] = norm(m.get("text") or "")
        for a in d.get("actions") or []:
            if a.get("text"):
                a["text"] = norm(a["text"])
    return e


def analyse_game(gdir: Path, k: int = WINDOW_K, at_round: int | None = None, exclude_god: bool = True) -> Game:
    """game.framing (meta.json framing.per_agent): message texts written with @handles, display names and block
    codes are mapped back to agent ids / info ids first, so names and keys count exactly as in unframed games."""
    g = Game(gdir.name, k, exclude_god)
    g.branch = gdir.name.split("-", 1)[1] if "-" in gdir.name else "?"
    meta: dict[str, Any] = {}
    if (gdir / "meta.json").exists():
        meta = json.loads((gdir / "meta.json").read_text())
    norm = None
    for e in _jsonl(gdir / "events.jsonl"):
        if at_round is not None and e["round"] > at_round:
            break
        if e["type"] == "game_start":
            from swarm.framing.facility import text_normalizer

            norm = text_normalizer(meta, e["data"]["agents"], int(meta.get("seed") or 0))
            g.start(e["data"])
        else:
            g.on_event(_framed(e, norm) if norm is not None else e)
    g.finish()
    snap = gdir / "snapshots.jsonl"
    if snap.exists():                         # AgentState.heard (showcase require_known): authoritative
        with snap.open() as f:
            for line in f:
                if '"heard"' not in line:
                    break                     # run made before `heard` existed: keep the replay
                s = json.loads(line)
                rnd = s["round"]
                if rnd < 1 or (at_round is not None and rnd > at_round):
                    continue
                ag = s["state"]["agents"]
                if any(ag[a].get("heard") is None for a in g.roster):
                    break
                g.snap_heard_by_round[rnd] = [len(ag[a]["heard"]) for a in g.roster]
                if rnd in g.heard_at:
                    g.heard_mismatch[rnd] = sum(set(ag[a]["heard"]) != g.heard_at[rnd][a] for a in g.roster)
    return g


def analyse(run_dir: Path, branch: str | None = None, at_round: int | None = None,
            k: int = WINDOW_K, exclude_god: bool = True) -> list[Game]:
    games = []
    for gdir in sorted((run_dir / "games").iterdir()):
        if not (gdir / "events.jsonl").exists():
            continue
        if branch is not None and gdir.name.split("-", 1)[-1] != branch:
            continue
        games.append(analyse_game(gdir, k, at_round, exclude_god))
    return games


def by_branch(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        b = out.setdefault(r["branch"], {"games": 0})
        b["games"] += 1
        for key, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not key.startswith("heard"):
                b[key] = b.get(key, 0) + v
    return out


REVIEW_CATS = ["info_seeking", "anyone_has", "third_party_mention", "introduction", "introduction_god", "holder_claim",
               "holder_claim_self", "holder_claim_you", "hearsay_defect", "defect_confession",
               "defect_accuse_receiver", "forged_warning", "relay_request", "reported_speech",
               "relayed_need", "quote", "quote_unverified"]


def write_review(games: list[Game], path: Path, n: int = 15, seed: int = 0) -> None:
    rng = random.Random(seed)
    lines = []
    for cat in REVIEW_CATS:
        pool = [(g.label, x) for g in games for x in g.hits[cat]]
        lines.append(f"\n===== {cat}: {len(pool)} hits; {min(n, len(pool))} random =====")
        for lab, x in rng.sample(pool, min(n, len(pool))):
            meta = {k: v for k, v in x.items() if k not in ("text", "msg_id", "clause")}
            lines.append(f"[{lab}] {json.dumps(meta, default=str)}")
            lines.append("    " + x.get("text", "").replace("\n", " ")[:400])
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--branch")
    ap.add_argument("--at-round", type=int)
    ap.add_argument("--include-god", action="store_true",
                    help="count God-forged DMs in the agent categories (default: excluded, reported as god_forged)")
    ap.add_argument("--k", type=int, default=WINDOW_K, help="rounds within which an introduction counts as used")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--review", type=Path, help="write ~--review-n random hits per category here")
    ap.add_argument("--review-n", type=int, default=15)
    a = ap.parse_args(argv)
    games = analyse(a.run_dir, a.branch, a.at_round, a.k, exclude_god=not a.include_god)
    rows = [g.summary() for g in games]
    res = {"run": a.run_dir.name, "at_round": a.at_round, "k": a.k, "exclude_god": not a.include_god, "games": rows, "branches": by_branch(rows),
           "per_round": {g.label: g.rounds() for g in games}}
    keys = ["messages", "god_forged", "info_seeking", "anyone_has", "third_party_msgs", "introductions",
            f"intro_used_k{a.k}", "intro_god", f"intro_god_used_k{a.k}",
            f"intro_relayed_k{a.k}", "holder_claim", "holder_claim_true", "hearsay_defect", "hearsay_second_hand",
            "hearsay_true", "forged_warning", "relay_request", "reported_speech", "relayed_need", "quote",
            "heard_final_mean"]
    print(f"{a.run_dir.name}" + (f" (<= round {a.at_round})" if a.at_round else ""))
    print("game".ljust(12) + " ".join(k[:10].rjust(10) for k in keys))
    for r in rows:
        print(r["game"].ljust(12) + " ".join(str(r.get(k)).rjust(10) for k in keys))
    for b, s in res["branches"].items():
        print(f"[{b}]".ljust(12) + " ".join(str(s.get(k, "")).rjust(10) for k in keys))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str))
    if a.review:
        write_review(games, a.review, a.review_n)


if __name__ == "__main__":
    main()
