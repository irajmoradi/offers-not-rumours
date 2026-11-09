"""Fixes requested by Jennifer (project owner), applied at runtime on top of her unchanged modules:
 1. Prompt: each agent's observation gets one extra sentence just before "Choose ONE action this round.":
    "You are agent {i}: never give a key to yourself (agent {i}). For a give, choose a recipient and key
     pair from CANDIDATES_JSON."
 2. Harness: a give whose (to, key) pair is not in the CANDIDATES_JSON that agent was SHOWN this round
    is rejected (scored as invalid_give, exactly as an impossible give was before) and logged with a reason:
    self / not_held / not_candidate / bad_fields. Validation uses the prompt-time list, so a pair that went
    stale because another agent moved first is not rejected here (the original give_key rules apply).

Phase 3 change (JWIN_GIVE_CHECK, default "legal"): Phase 2's check ("strict") also rejected gives that
are LEGAL under her give_key rules but were not on the CANDIDATES_JSON list (1471 of 6106 checked gives,
24%): the list only holds pairs where the recipient POSTED that it needs the key, while give_key allows
any held key to any other agent ("Gifts follow posts, so a key can go to someone who doesn't actually
need it"). In "legal" mode only gives her rules reject are scored invalid_give (self, key not held,
recipient out of range, bad fields); off-list legal gives go through and are logged as accepted with
reason off_list (and whether the recipient truly needed the key). JWIN_GIVE_CHECK=strict = Phase 2."""
import os, json, threading, collections
REJECTIONS = []
ACCEPTED_OFF_LIST = []
COUNTS = collections.Counter()
_LOCK = threading.Lock()
ANCHOR = "Choose ONE action this round."

def install(out_dir):
    import llm_game as lg
    _observe, _apply = lg.observe, lg.apply_action
    def observe(st, i, max_msgs=10, framing="assistant"):
        txt = _observe(st, i, max_msgs=max_msgs, framing=framing)
        st.__dict__.setdefault("_shown_cands", {})[i] = set(lg.cand_for(st, i))
        line = (f"You are agent {i}: never give a key to yourself (agent {i}). "
                f"For a give, choose a recipient and key pair from CANDIDATES_JSON.")
        return txt.replace(ANCHOR, line + "\n" + ANCHOR, 1)
    mode = os.environ.get("JWIN_GIVE_CHECK", "legal")
    assert mode in ("legal", "strict"), mode
    def apply_action(st, i, act):
        if isinstance(act, dict) and act.get("action") == "give":
            cands = st.__dict__.get("_shown_cands", {}).get(i)
            if cands is None:
                cands = set(lg.cand_for(st, i))
            accepted_off_list = False
            try:
                j, k = int(act["to"]), int(act["key"])
                if (j, k) in cands:
                    reason = None
                elif j == i:
                    reason = "self"
                elif k not in st.holdings[i]:
                    reason = "not_held"
                elif not (0 <= j < st.cfg.N):
                    reason = "bad_recipient"
                elif mode == "strict":
                    reason = "not_candidate"
                else:   # legal under give_key, just not on the posted-needs list
                    reason, accepted_off_list = None, True
            except (KeyError, TypeError, ValueError):
                j = k = None; reason = "bad_fields"
            with _LOCK:
                COUNTS["gives_checked"] += 1
                if reason:
                    COUNTS[reason] += 1
                    REJECTIONS.append(dict(round=st.round, agent=i, to=j, key=k, reason=reason,
                                           n_candidates=len(cands)))
                elif accepted_off_list:
                    COUNTS["accepted_off_list"] += 1
                    COUNTS["accepted_off_list_useful" if k in st.needs_left[j] else "accepted_off_list_not_needed"] += 1
                    ACCEPTED_OFF_LIST.append(dict(round=st.round, agent=i, to=j, key=k,
                                                  useful=k in st.needs_left[j], n_candidates=len(cands)))
                else:
                    COUNTS["accepted_on_list"] += 1
            if reason:
                return _apply(st, i, {"action": "give", "to": -1, "key": -1})   # -> invalid_give, same bookkeeping
        return _apply(st, i, act)
    lg.observe, lg.apply_action = observe, apply_action
    def save():
        with open(os.path.join(out_dir, "give_rejections.json"), "w") as f:
            json.dump(dict(mode=mode, counts=COUNTS, rejections=REJECTIONS[-5000:],
                           accepted_off_list=ACCEPTED_OFF_LIST[-5000:]), f, indent=1)
    import atexit; atexit.register(save)
    print("jwin fixes installed (self-give prompt line + give check, mode=%s)" % mode, flush=True)
    return save
