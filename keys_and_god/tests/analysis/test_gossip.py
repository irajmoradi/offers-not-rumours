"""swarm.analysis.gossip on a hand-written event stream: every category, the truth checks against the
replayed state, introductions (heard-set replay + later use), name spread and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

from swarm.analysis import gossip

AG = ["ann_a", "bob_b", "cat_c", "dan_d", "eve_e"]
CONTACTS = {"ann_a": ["bob_b"], "bob_b": ["ann_a", "cat_c"], "cat_c": ["bob_b", "dan_d"],
            "dan_d": ["cat_c", "eve_e"], "eve_e": ["dan_d"]}
KEYS = {"ann_a": [1], "bob_b": [2], "cat_c": [3], "dan_d": [4], "eve_e": [5]}


class Stream:
    def __init__(self) -> None:
        self.ev: list[dict] = []

    def add(self, typ: str, rnd: int, **data) -> None:
        self.ev.append({"type": typ, "round": rnd, "seq": len(self.ev), "data": {"type": typ, **data}})

    def msg(self, rnd: int, mid: str, s: str, r: str | None, text: str, forged: bool = False) -> None:
        self.add("message", rnd, message={"msg_id": mid, "round": rnd, "turn_index": 0,
                                          "channel": "dm" if r else "board", "sender_shown": s,
                                          "true_sender": "GOD" if forged else s, "receiver": r,
                                          "text": text, "forged": forged})

    def obs(self, rnd: int, agent: str, inbox=(), board=(), notices=()) -> None:
        view = lambda mid: next({"msg_id": mid, "channel": m["channel"], "sender": m["sender_shown"],  # noqa: E731
                                 "receiver": m["receiver"], "text": m["text"]}
                                for e in self.ev if e["type"] == "message"
                                for m in [e["data"]["message"]] if m["msg_id"] == mid)
        self.add("turn", rnd, agent=agent, obs={"inbox": [view(m) for m in inbox], "board": [view(m) for m in board],
                                                "notices": list(notices)})


def _run(tmp_path: Path) -> Path:
    s = Stream()
    s.add("game_start", 0, branch="nogod", agents={
        a: {"keys": [{"copy_id": f"k{i}", "info_id": i, "key": 1000 + i} for i in KEYS[a]], "needs": [9],
            "contacts": CONTACTS[a]} for a in AG})
    s.msg(1, "m1", "bob_b", "ann_a", "Hi ann_a, do you know who has key 4? cat_c has key 3.")
    s.msg(1, "m2", "cat_c", "bob_b", "dan_d has key 5 I think. Beware of eve_e, she burned dan_d.")
    s.add("burn", 1, actor="eve_e", target="dan_d", correct=True)
    s.msg(1, "m3", "dan_d", "cat_c", "eve_e burned my need! Pass it on to bob_b please.")
    s.msg(1, "m4", "cat_c", "bob_b", 'dan_d said "eve_e burned my need yesterday". Also eve_e defected on dan_d.')
    s.obs(1, "ann_a", inbox=["m1"])                    # learns cat_c from bob_b: an introduction
    s.obs(1, "bob_b", inbox=["m2", "m4"])              # learns dan_d, eve_e
    s.add("round_end", 1)
    s.add("action_result", 2, agent="ann_a", action={"kind": "connect", "target": "cat_c"}, ok=True)
    s.msg(2, "m5", "ann_a", "bob_b", "Anyone have key 9? I just burned your need, sorry. That message wasn't from me.")
    s.add("key_transfer", 2, info_id=3, from_agent="cat_c", to_agent="ann_a")
    s.msg(2, "m6", "bob_b", "ann_a", "cat_c has key 3. eve_e needs key 1.")
    s.msg(2, "m8", "bob_b", "dan_d", "Heads up: eve_e still needs info #1. Who has key 2?", forged=True)
    s.obs(2, "dan_d", inbox=["m8"])                    # dan_d learns nothing new (eve_e is a contact)
    s.msg(2, "m9", "cat_c", "eve_e", "Heads up: ann_a still needs info #9.", forged=True)
    s.obs(2, "eve_e", inbox=["m9"])                    # eve_e learns cat_c (sender) and ann_a from God
    s.add("action_result", 2, agent="eve_e", action={"kind": "send_message", "to": "ann_a", "text": "hi"}, ok=False)
    s.add("round_end", 2)
    d = tmp_path / "run" / "games" / "g00-nogod"
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in s.ev) + "\n")
    return tmp_path / "run"


def test_categories_and_truth(tmp_path):
    g = gossip.analyse(_run(tmp_path))[0]
    h = g.hits
    assert g.branch == "nogod"
    assert [x["msg_id"] for x in h["info_seeking"]] == ["m1"] and [x["msg_id"] for x in h["anyone_has"]] == ["m5"]
    assert {x["msg_id"] for x in h["third_party_mention"]} == {"m1", "m2", "m3", "m4", "m6"}
    claims = {(x["msg_id"], x["holder"]): x["true"] for x in h["holder_claim"]}
    # m1 cat_c holds 3: true; m2 dan_d holds 5: false (eve_e holds it); m6 after the transfer: false
    assert claims == {("m1", "cat_c"): True, ("m2", "dan_d"): False, ("m6", "cat_c"): False}
    hs = {x["msg_id"]: x for x in h["hearsay_defect"]}
    assert set(hs) == {"m2", "m3", "m4"}
    assert hs["m2"]["accused"] == ["eve_e"] and not hs["m2"]["true"]            # before the burn event
    assert hs["m3"]["first_hand"] and hs["m3"]["true"]
    assert not hs["m4"]["first_hand"] and hs["m4"]["true"]                      # second-hand, true
    assert hs["m4"]["accused"] == ["eve_e"]
    assert [x["msg_id"] for x in h["defect_confession"]] == ["m5"]
    assert [x["msg_id"] for x in h["forged_warning"]] == ["m5"] and not h["forged_warning"][0]["true"]
    assert [x["msg_id"] for x in h["relay_request"]] == ["m3"]
    assert [x["msg_id"] for x in h["reported_speech"]] == ["m4"]
    assert [x["msg_id"] for x in h["relayed_need"]] == ["m6"]
    assert h["quote"] == []                     # m4 quotes m3, but cat_c never read m3 (no obs)
    intro = {(x["receiver"], x["name"]): x for x in h["introduction"]}
    assert set(intro) == {("ann_a", "cat_c"), ("bob_b", "dan_d"), ("bob_b", "eve_e")}
    assert intro[("ann_a", "cat_c")]["used_ok"] and intro[("ann_a", "cat_c")]["introducer"] == "bob_b"
    assert not intro[("bob_b", "eve_e")]["used"]
    assert g.heard["ann_a"] == {"bob_b", "cat_c"} and g.heard_by_round[1] == [2, 4, 2, 2, 1]
    # God-forged DMs: not agent gossip (m8's "who has" is not info_seeking), intros kept apart
    assert [(x["receiver"], x["name"], x["used"], x["used_ok"]) for x in h["introduction_god"]] == \
        [("eve_e", "ann_a", True, False)]
    assert g.heard["eve_e"] == {"dan_d", "cat_c", "ann_a"}
    row = g.summary()
    assert row["god_forged"] == 2 and row["intro_god"] == 1 and row["intro_god_used_k3"] == 1
    assert row["introductions"] == 3 and row["intro_used_k3"] == 1
    pooled = gossip.analyse(tmp_path / "run", exclude_god=False)[0]
    assert len(pooled.hits["introduction"]) == 4 and not pooled.hits["introduction_god"]
    assert "m8" in [x["msg_id"] for x in pooled.hits["info_seeking"]]


def test_quote_needs_a_read_source(tmp_path):
    run = _run(tmp_path)
    ev = (run / "games" / "g00-nogod" / "events.jsonl").read_text().splitlines()
    extra = [{"type": "turn", "round": 2, "seq": 100, "data": {"type": "turn", "agent": "cat_c", "obs": {
        "inbox": [{"msg_id": "m3", "channel": "dm", "sender": "dan_d", "receiver": "cat_c",
                   "text": "eve_e burned my need! Pass it on to bob_b please."}], "board": [], "notices": []}}},
             {"type": "message", "round": 2, "seq": 101, "data": {"type": "message", "message": {
                 "msg_id": "m7", "round": 2, "turn_index": 0, "channel": "dm", "sender_shown": "cat_c",
                 "true_sender": "cat_c", "receiver": "bob_b", "text": 'dan_d wrote: "Pass it on to bob_b please."',
                 "forged": False}}}]
    (run / "games" / "g00-nogod" / "events.jsonl").write_text("\n".join(ev + [json.dumps(e) for e in extra]) + "\n")
    g = gossip.analyse(run)[0]
    assert [(x["msg_id"], x["source"]) for x in g.hits["quote"]] == [("m7", "m3")]
    assert [x["msg_id"] for x in g.hits["quote_unverified"]] == ["m4"]


def test_cli_json_and_review(tmp_path, capsys):
    run = _run(tmp_path)
    gossip.main([str(run), "--json", str(tmp_path / "o.json"), "--review", str(tmp_path / "r.txt"), "--at-round", "1"])
    out = json.loads((tmp_path / "o.json").read_text())
    row = out["games"][0]
    assert row["info_seeking"] == 1 and row["anyone_has"] == 0 and row["introductions"] == 3   # round 2 cut
    assert out["branches"]["nogod"]["games"] == 1
    assert "===== info_seeking: 1 hits" in (tmp_path / "r.txt").read_text()
    assert "g00-nogod" in capsys.readouterr().out
