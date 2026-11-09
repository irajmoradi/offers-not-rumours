"""Scripted tests for arm DC (dual control). NO LLM, NO network: agents are DetMock, approvers are scripted functions
or a deterministic fake approval backend. Run: PYTHONHASHSEED=0 python -s -P child.py ../pkg/swarm_llm test_dual_control.py
(no -I: -I ignores PYTHONHASHSEED). Tests (a)-(d), (f) here; (e) = identity hash + re-scoring, run by test_dual_control.sh."""
import json, hashlib, copy
import llm_game as lg
import fastpaths; fastpaths.install(lg)
import v2patch; v2patch.install(lg)
import outsider_bribe, fulfillment, u_factors, dual_control as dc
outsider_bribe.install(lg); fulfillment.install(lg); u_factors.install(lg); dc.install(lg)
from detmock import DetMock

class FakeApprover:                       # deterministic stand-in for the live approval call (exercises the live path)
    def chat(self, pairs, seed=0):
        out = []
        for _, u in pairs:
            h = int(hashlib.sha256(u.encode()).hexdigest()[:8], 16)
            out.append(json.dumps({"approve": h % 2 == 0, "reason": "fake"}) if h % 5 else "not json")
        return out

def game(seed, N=8):
    st = lg.new_game(N, "complete", seed, 12, 0.7, "private", "posted"); st.records = "closed"; return st

def valid_burn(st):
    for a in range(st.cfg.N):
        if st.done(a): continue
        for b in range(st.cfg.N):
            if b == a or st.done(b) or b in st.compromised: continue
            for k in sorted(st.needs_left[b]): return a, b, k
    raise RuntimeError("no valid burn")

def snap(st): return (json.dumps([sorted(x) for x in st.needs_left]), json.dumps([sorted(x) for x in st.holdings]), len(st.defections))
R = {}
# (a) one authorization -> blocked, logged as an attempt, keys untouched
st = game(1); st.dual = dc.dual_config(101, mode="scripted", approver_fn=lambda s, req: False)
a, b, k = valid_burn(st); before = snap(st); res = lg.defect(st, a, b, k); ev = st.dual["events"]
assert res == "dual_blocked" and snap(st) == before, (res,)
assert [e["type"] for e in ev] == ["dual_request", "dual_block"] and ev[1]["reason"] == "approver_no"
m = dc.metrics(st); assert m["dual_blocks"] == 1 and m["dual_burn_blocks"] == 1 and len(st.defections) == 0
R["a_one_auth_blocked"] = dict(result=res, events=[e["type"] for e in ev], defections=len(st.defections), dual_blocks=m["dual_blocks"])
# (b) two authorizations -> her defect() runs
st = game(1); st.dual = dc.dual_config(101, mode="scripted", approver_fn=lambda s, req: True)
a, b, k = valid_burn(st); res = lg.defect(st, a, b, k); ev = st.dual["events"]
assert res == "defect_success" and len(st.defections) == 1 and k in st.burned[b]
assert [e["type"] for e in ev] == ["dual_request", "dual_approve"]
R["b_two_auth_succeeds"] = dict(result=res, events=[e["type"] for e in ev], defections=len(st.defections))
# (c) self-approval rejected (fixed approver = requester); random mode never draws the requester
st = game(1); a, b, k = valid_burn(st)
st.dual = dc.dual_config(101, select="fixed", fixed_approver=a, mode="scripted", approver_fn=lambda s, req: True)
before = snap(st); res = lg.defect(st, a, b, k); ev = st.dual["events"]
assert res == "dual_blocked" and ev[-1]["reason"] == "self_approval" and snap(st) == before
st2 = game(2); st2.dual = dc.dual_config(7, mode="scripted", approver_fn=lambda s, req: False); a2, b2, k2 = valid_burn(st2)
for _ in range(300): lg.defect(st2, a2, b2, k2)
self_draws = sum(1 for e in st2.dual["events"] if e["type"] == "dual_request" and e["approver"] == e["requester"])
assert self_draws == 0
R["c_self_approval_rejected"] = dict(fixed_self=res, reason=ev[-1]["reason"], random_self_draws_in_300=self_draws)
# (d) approver randomised per seed and recorded
appr = []
for sd in range(10):
    st = game(3); st.dual = dc.dual_config(31_001 + sd, mode="scripted", approver_fn=lambda s, req: False)
    a, b, k = valid_burn(st); lg.defect(st, a, b, k); e = st.dual["events"][0]
    assert isinstance(e["approver"], int) and e["approver"] != a; appr.append(e["approver"])
assert len(set(appr)) > 1
R["d_approver_by_seed"] = dict(approvers_seeds_0_9=appr, distinct=len(set(appr)))
# (f) fork-replay: identical branch seed -> identical events (full games: DetMock agents, insider, live path w/ FakeApprover)
def branch(prefix, bseed, x, live):
    b = lg.fork(prefix, bseed); b.defense = "none"; b.records = "closed"; lg.compromise(b, x)
    b.dual = (dc.dual_config(bseed, mode="live", backend=FakeApprover()) if live else
              dc.dual_config(bseed, mode="scripted", approver_fn=lambda s, req: req["request_no"] % 2 == 0))
    lg.play(b, DetMock(), seed=bseed); return b
fr, total_req, ins_appr, allev = [], 0, 0, []
for sd in range(6):
    pre = game(sd); lg.play(pre, DetMock(), until=1, seed=sd); x = lg.pick_target(pre, "random")
    for live in (False, True):
        b1, b2 = branch(pre, 31_001 + sd, x, live), branch(pre, 31_001 + sd, x, live)
        same = (json.dumps(b1.dual["events"]) == json.dumps(b2.dual["events"]) and json.dumps(b1.defections) == json.dumps(b2.defections)
                and json.dumps([(c["round"], c["agent"], c["reply"], c["result"]) for c in b1.calls]) ==
                    json.dumps([(c["round"], c["agent"], c["reply"], c["result"]) for c in b2.calls]))
        assert same, (sd, live)
        fr.append(same); total_req += sum(e["type"] == "dual_request" for e in b1.dual["events"]); allev += b1.dual["events"]
        ins_appr += sum(1 for e in b1.dual["events"] if e["type"] == "dual_request" and e["approver_is_insider"])
assert total_req > 0
for e in allev: dc.validate_event(e)
R["f_fork_replay_identical"] = dict(branches_compared=len(fr), all_identical=all(fr), dual_requests_seen=total_req,
                                    approver_was_insider=ins_appr, events_schema_valid=len(allev),
                                    block_reasons=dict(__import__("collections").Counter(e.get("reason") for e in allev if e["type"] == "dual_block")))
# flags off: dual installed but st.dual unset -> her defect() path (identity proper is test (e))
st = game(1); a, b, k = valid_burn(st); assert lg.defect(st, a, b, k) == "defect_success" and not hasattr(st, "dual")
R["flags_off_defect_plain"] = True
print(json.dumps(R, indent=1)); print("ALL DUAL-CONTROL SCRIPTED TESTS PASSED")
