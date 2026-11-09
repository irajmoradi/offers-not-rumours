"""Runtime speedups for Jennifer's llm_game.py (her file is unchanged; each patch returns exactly what her
function returns, checked byte for byte by tools/profile_game.py: orig vs fast at fixed seeds).

1. quarantined_set(st): hers rescans every agent's whole heard-list on every call, and deliver() calls it
   once per agent per round -> O(N * H) per round (H = all messages heard so far, which grows every round).
   Now: an accuser index {about: set(from_)} per state, updated incrementally from a per-agent cursor
   into st.heard[i] -> O(new messages) per call. Credibility and the median are computed as hers
   (O(N^2) numpy sum, vectorised).
2. update_bribes(st): hers checks every unaccepted bribe against every defection -> O(B * D) per round.
   Now: a per-state index {(actor, target): latest defection round}, updated incrementally ->
   O(B + new D). v2: "exists a defection with round > offer round" == "latest round > offer round" (her v2 strict rule).
3. ensure_fields(st) with no overrides: hers rebuilds a defaults dict with four N-long lists on every
   call (it runs once per observe -> O(N^2) allocations per round). Now it returns early when every
   field already exists and the board is non-empty (hers would change nothing in that case).
The indexes live on the state object, so engine.fork's deepcopy copies them with the state, and a
cursor can never point past data that isn't in that copy."""
import numpy as np

FIELDS = ("checks", "notices", "calls", "burned", "defections", "board", "defect_visibility", "needs_mode",
          "defect_enabled", "d_evil_log", "d_evil_used", "bribes", "insider_bribe_prob")

def install(lg):
    orig_ensure = lg.ensure_fields

    def ensure_fields(st, visibility=None, needs_mode=None, defect_enabled=None):
        if visibility is None and needs_mode is None and defect_enabled is None:
            d = st.__dict__
            if d.get("board") and all(k in d for k in FIELDS):
                return
        return orig_ensure(st, visibility, needs_mode, defect_enabled)

    def _accusers(st):
        idx = st.__dict__.get("_fp_acc")
        if idx is None or len(idx[1]) != st.cfg.N:
            idx = ({}, [0] * st.cfg.N); st._fp_acc = idx
        acc, cur = idx
        for i in range(st.cfg.N):
            h = st.heard[i]
            if len(h) < cur[i]:          # list was replaced/shrunk: rebuild from scratch
                st._fp_acc = None; return _accusers(st)
            for e in h[cur[i]:]:
                if e["kind"] in ("accuse", "warn"):
                    acc.setdefault(e["about"], set()).add(e["from_"])
            cur[i] = len(h)
        return acc

    def quarantined_set(st):
        accusers = _accusers(st)
        cred = lg.credibility(st)
        med = np.median(list(cred.values())) if cred else 0.0
        return {a for a, src in accusers.items() if len(src) >= 2 and cred.get(a, 0) <= med}

    def update_bribes(st):
        idx = st.__dict__.get("_fp_def")
        if idx is None or idx[1] > len(st.defections):
            idx = [{}, 0]; st._fp_def = idx
        last, n = idx
        for d in st.defections[n:]:
            k = (d["actor"], d["target"])
            if k not in last or d["round"] > last[k]:
                last[k] = d["round"]
        idx[1] = len(st.defections)
        for b in st.bribes:
            if not b["accepted"]:
                r = last.get((b["target"], b["victim"]))
                if r is not None and r > b["round"]:   # her v2 rule: strictly after the offer round
                    b["accepted"] = True

    lg.ensure_fields, lg.quarantined_set, lg.update_bribes = ensure_fields, quarantined_set, update_bribes
    print("fastpaths installed (quarantined_set index, update_bribes index, ensure_fields early return)", flush=True)
